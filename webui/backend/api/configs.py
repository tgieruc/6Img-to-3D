import os
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from webui.backend.database import get_db
from webui.backend.models import ConfigRecord
from webui.backend.schemas.config_schema import FullConfig
from webui.backend.services.config_io import export_to_py, load_from_py

router = APIRouter(prefix="/api/configs", tags=["configs"])


@router.post("", status_code=201)
def create_config(cfg: FullConfig, name: str, db: Session = Depends(get_db)):
    record = ConfigRecord(name=name, data=cfg.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "name": record.name, "created_at": str(record.created_at)}


@router.get("")
def list_configs(db: Session = Depends(get_db)):
    return [
        {"id": r.id, "name": r.name, "created_at": str(r.created_at)}
        for r in db.query(ConfigRecord).order_by(ConfigRecord.created_at.desc()).all()
    ]


@router.post("/import", status_code=201)
async def import_config(name: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="wb") as f:
        f.write(content)
        tmp = f.name
    try:
        cfg = load_from_py(tmp)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse config: {e}")
    finally:
        os.unlink(tmp)
    record = ConfigRecord(name=name, data=cfg.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "name": record.name, "data": cfg.model_dump()}


@router.get("/{config_id}")
def get_config(config_id: str, db: Session = Depends(get_db)):
    r = db.get(ConfigRecord, config_id)
    if not r:
        raise HTTPException(status_code=404, detail="Config not found")
    return {"id": r.id, "name": r.name, "data": r.data, "created_at": str(r.created_at)}


@router.put("/{config_id}")
def update_config(
    config_id: str,
    cfg: FullConfig,
    name: str | None = None,
    db: Session = Depends(get_db),
):
    r = db.get(ConfigRecord, config_id)
    if not r:
        raise HTTPException(status_code=404, detail="Config not found")
    r.data = cfg.model_dump()
    if name:
        r.name = name
    db.commit()
    return {"id": r.id, "name": r.name}


@router.delete("/{config_id}", status_code=204)
def delete_config(config_id: str, db: Session = Depends(get_db)):
    r = db.get(ConfigRecord, config_id)
    if not r:
        raise HTTPException(status_code=404, detail="Config not found")
    db.delete(r)
    db.commit()


@router.post("/{config_id}/clone", status_code=201)
def clone_config(config_id: str, new_name: str, db: Session = Depends(get_db)):
    r = db.get(ConfigRecord, config_id)
    if not r:
        raise HTTPException(status_code=404, detail="Config not found")
    clone = ConfigRecord(name=new_name, data=dict(r.data))
    db.add(clone)
    db.commit()
    db.refresh(clone)
    return {"id": clone.id, "name": clone.name, "created_at": str(clone.created_at)}


@router.get("/{config_id}/export")
def export_config(config_id: str, db: Session = Depends(get_db)):
    r = db.get(ConfigRecord, config_id)
    if not r:
        raise HTTPException(status_code=404, detail="Config not found")
    cfg = FullConfig(**r.data)
    py_str = export_to_py(cfg)
    safe_name = re.sub(r"[^\w\-.]", "_", r.name)
    return PlainTextResponse(
        content=py_str,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.py"'},
    )


@router.post("/{config_id}/write")
def write_config_to_disk(config_id: str, db: Session = Depends(get_db)):
    r = db.get(ConfigRecord, config_id)
    if not r:
        raise HTTPException(status_code=404, detail="Config not found")
    cfg = FullConfig(**r.data)
    py_str = export_to_py(cfg)
    out_dir = Path(__file__).parents[3] / "runs" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{r.id}.py"
    out_path.write_text(py_str)
    return {"path": str(out_path)}
