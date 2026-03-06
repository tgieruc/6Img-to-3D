import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from webui.backend.database import get_db
from webui.backend.models import JobRecord
from webui.backend.services import job_runner

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class TrainJobCreate(BaseModel):
    name: str
    py_config: str = "config/config.py"
    manifest_train: str = ""
    manifest_val: str = ""


class EvalJobCreate(BaseModel):
    name: str
    resume_from: str
    manifest_val: str = ""
    py_config: str = "config/config.py"


def _job_dict(j: JobRecord) -> dict:
    return {
        "id": j.id,
        "name": j.name,
        "type": j.job_type,
        "status": j.status,
        "mlflow_run_id": j.mlflow_run_id,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        "error": j.error,
    }


@router.post("/train", status_code=201)
def create_train_job(
    payload: TrainJobCreate,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    job = JobRecord(job_type="train", name=payload.name)
    db.add(job)
    db.commit()
    db.refresh(job)
    background.add_task(
        asyncio.ensure_future,
        job_runner.run_train_job(
            job.id,
            payload.py_config,
            payload.name,
            payload.manifest_train,
            payload.manifest_val,
        ),
    )
    return {"id": job.id, "status": "queued"}


@router.post("/eval", status_code=201)
def create_eval_job(
    payload: EvalJobCreate,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    job = JobRecord(job_type="eval", name=payload.name)
    db.add(job)
    db.commit()
    db.refresh(job)
    background.add_task(
        asyncio.ensure_future,
        job_runner.run_eval_job(
            job.id,
            payload.resume_from,
            payload.manifest_val,
            payload.py_config,
            payload.name,
        ),
    )
    return {"id": job.id, "status": "queued"}


@router.get("")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(JobRecord).order_by(JobRecord.created_at.desc()).all()
    return [_job_dict(j) for j in jobs]


@router.get("/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    j = db.get(JobRecord, job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    d = _job_dict(j)
    d["log"] = j.log
    return d


@router.delete("/{job_id}")
def cancel_job(job_id: str):
    job_runner.kill_job(job_id)
    return {"status": "cancelled"}


@router.get("/{job_id}/stream")
async def stream_job_log(job_id: str):
    queue = job_runner.subscribe(job_id)

    async def event_gen() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=25)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("type") == "status" and msg.get("status") in ("completed", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    yield 'data: {"type":"ping"}\n\n'
        finally:
            job_runner.unsubscribe(job_id, queue)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/{job_id}/metrics")
def get_metrics(job_id: str, db: Session = Depends(get_db)):
    """Return metric history from MLflow for this job's run."""
    import mlflow

    j = db.get(JobRecord, job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")

    client = mlflow.MlflowClient()
    try:
        if j.mlflow_run_id:
            run_id = j.mlflow_run_id
        else:
            # Try to find by experiment name = job name
            experiments = client.search_experiments(filter_string=f"name = '{j.name}'")
            if not experiments:
                return {}
            runs = client.search_runs(
                experiment_ids=[experiments[0].experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs:
                return {}
            run_id = runs[0].info.run_id

        run = client.get_run(run_id)
        keys = list(run.data.metrics.keys())
        return {
            key: [{"step": m.step, "value": m.value} for m in client.get_metric_history(run_id, key)] for key in keys
        }
    except Exception:
        return {}


@router.get("/{job_id}/renders")
def list_renders(job_id: str, db: Session = Depends(get_db)):
    j = db.get(JobRecord, job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    renders_dir = PROJECT_ROOT / j.name / "renders"
    if not renders_dir.exists():
        return []
    return sorted(p.name for p in renders_dir.glob("*.png"))


@router.get("/{job_id}/renders/{filename}")
def get_render(job_id: str, filename: str, db: Session = Depends(get_db)):
    j = db.get(JobRecord, job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    img_path = PROJECT_ROOT / j.name / "renders" / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(img_path))
