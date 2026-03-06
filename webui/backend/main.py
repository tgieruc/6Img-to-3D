from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webui.backend.database import Base, SessionLocal, engine
from webui.backend.models import JobRecord


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


app = FastAPI(title="6Img-to-3D Web UI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}
