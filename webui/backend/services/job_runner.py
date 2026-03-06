import asyncio
import contextlib
import logging
import os
import signal
from datetime import UTC, datetime
from pathlib import Path

from webui.backend.database import SessionLocal
from webui.backend.models import JobRecord

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_active_job_ids: set[str] = set()
_job_subscribers: dict[str, list[asyncio.Queue]] = {}


def subscribe(job_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _job_subscribers.setdefault(job_id, []).append(q)
    return q


def unsubscribe(job_id: str, q: asyncio.Queue) -> None:
    if job_id in _job_subscribers:
        _job_subscribers[job_id] = [x for x in _job_subscribers[job_id] if x is not q]


async def _broadcast(job_id: str, msg: dict) -> None:
    for q in _job_subscribers.get(job_id, []):
        await q.put(msg)


def _log_file(job_id: str) -> Path:
    p = PROJECT_ROOT / "runs" / f"job_{job_id}.log"
    p.parent.mkdir(exist_ok=True)
    return p


async def _run_subprocess(job_id: str, cmd: list[str], env: dict | None = None) -> int:
    """Run cmd as subprocess, tail log file to subscribers. Returns exit code."""
    log_file = _log_file(job_id)
    log_file.write_text("")

    proc_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if env:
        proc_env.update(env)

    with open(log_file, "w") as lf:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=lf,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=proc_env,
        )

    # Update pid in DB
    db = SessionLocal()
    job = db.get(JobRecord, job_id)
    if job:
        job.pid = process.pid
        db.commit()
    db.close()

    # Tail log and broadcast
    wait_task = asyncio.create_task(process.wait())
    prev_size = 0
    while not wait_task.done():
        try:
            content = log_file.read_text()
        except OSError:
            content = ""
        if len(content) > prev_size:
            new_lines = content[prev_size:]
            prev_size = len(content)
            for line in new_lines.splitlines():
                if line:
                    await _broadcast(job_id, {"type": "log", "line": line})
                    if line.startswith("MLFLOW_RUN_ID="):
                        run_id = line.split("=", 1)[1].strip()
                        _db = SessionLocal()
                        _job = _db.get(JobRecord, job_id)
                        if _job:
                            _job.mlflow_run_id = run_id
                            _db.commit()
                        _db.close()
        await asyncio.sleep(1)

    # Final log flush
    try:
        final_log = log_file.read_text()
    except OSError:
        final_log = ""

    db = SessionLocal()
    job = db.get(JobRecord, job_id)
    if job:
        job.log = final_log
        job.pid = None
        db.commit()
    db.close()

    return process.returncode


async def run_train_job(
    job_id: str,
    py_config: str,
    log_dir: str,
    manifest_train: str = "",
    manifest_val: str = "",
) -> None:
    db = SessionLocal()
    job = db.get(JobRecord, job_id)
    if not job:
        db.close()
        return

    job.status = "running"
    job.started_at = datetime.now(UTC)
    db.commit()
    db.close()
    _active_job_ids.add(job_id)
    await _broadcast(job_id, {"type": "status", "status": "running"})

    cmd = ["uv", "run", "python", "-u", "train.py", "--py-config", py_config, "--log-dir", log_dir]
    if manifest_train and manifest_val:
        cmd += ["--manifest-train", manifest_train, "--manifest-val", manifest_val]

    env = {"MLFLOW_EXPERIMENT_NAME": log_dir}

    try:
        returncode = await _run_subprocess(job_id, cmd, env)
    except Exception as e:
        logger.exception("Train job %s failed", job_id)
        _finalize_job(job_id, "failed", str(e))
        await _broadcast(job_id, {"type": "status", "status": "failed"})
        _active_job_ids.discard(job_id)
        return

    status = "completed" if returncode == 0 else "failed"
    error = None if returncode == 0 else f"Process exited with code {returncode}"
    _finalize_job(job_id, status, error)
    await _broadcast(job_id, {"type": "status", "status": status})
    _active_job_ids.discard(job_id)


async def run_eval_job(
    job_id: str,
    resume_from: str,
    manifest_val: str,
    py_config: str,
    log_dir: str,
) -> None:
    db = SessionLocal()
    job = db.get(JobRecord, job_id)
    if not job:
        db.close()
        return

    job.status = "running"
    job.started_at = datetime.now(UTC)
    db.commit()
    db.close()
    _active_job_ids.add(job_id)
    await _broadcast(job_id, {"type": "status", "status": "running"})

    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "eval.py",
        "--py-config",
        py_config,
        "--resume-from",
        resume_from,
        "--log-dir",
        log_dir,
        "--depth",
        "--img-gt",
    ]
    if manifest_val:
        cmd += ["--manifest-val", manifest_val]

    try:
        returncode = await _run_subprocess(job_id, cmd)
    except Exception as e:
        logger.exception("Eval job %s failed", job_id)
        _finalize_job(job_id, "failed", str(e))
        await _broadcast(job_id, {"type": "status", "status": "failed"})
        _active_job_ids.discard(job_id)
        return

    status = "completed" if returncode == 0 else "failed"
    error = None if returncode == 0 else f"Process exited with code {returncode}"
    _finalize_job(job_id, status, error)
    await _broadcast(job_id, {"type": "status", "status": status})
    _active_job_ids.discard(job_id)


def _finalize_job(job_id: str, status: str, error: str | None) -> None:
    db = SessionLocal()
    job = db.get(JobRecord, job_id)
    if job:
        job.status = status
        job.error = error
        job.completed_at = datetime.now(UTC)
        db.commit()
    db.close()


def kill_job(job_id: str) -> bool:
    db = SessionLocal()
    job = db.get(JobRecord, job_id)
    if not job:
        db.close()
        return False
    if job.pid:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(job.pid, signal.SIGTERM)
    job.status = "cancelled"
    job.completed_at = datetime.now(UTC)
    job.pid = None
    db.commit()
    db.close()
    return True


def mark_active_jobs_failed() -> None:
    if not _active_job_ids:
        return
    db = SessionLocal()
    for jid in list(_active_job_ids):
        job = db.get(JobRecord, jid)
        if job and job.status == "running":
            job.status = "failed"
            job.error = "Server restarted"
            job.completed_at = datetime.now(UTC)
            job.pid = None
    db.commit()
    db.close()
    _active_job_ids.clear()
