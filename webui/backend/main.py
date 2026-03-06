from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webui.backend.api.configs import router as configs_router
from webui.backend.api.data import router as data_router
from webui.backend.api.jobs import router as jobs_router
from webui.backend.api.recipes import router as recipes_router
from webui.backend.database import Base, SessionLocal, engine
from webui.backend.models import JobRecord
from webui.backend.services.job_runner import mark_active_jobs_failed


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    orphaned = db.query(JobRecord).filter(JobRecord.status.in_(["queued", "running"])).all()
    for job in orphaned:
        job.status = "failed"
        job.error = "Server restarted"
        job.completed_at = datetime.now(UTC)
    if orphaned:
        db.commit()
    db.close()
    yield
    mark_active_jobs_failed()


app = FastAPI(title="6Img-to-3D Web UI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(configs_router)
app.include_router(data_router)
app.include_router(jobs_router)
app.include_router(recipes_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
