# Web UI & Data Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a full-pipeline web app (Split Manager + Training Dashboard + Evaluation Viewer) with MLflow experiment tracking, on top of fixed manifest-based dataloader.

**Architecture:** FastAPI + SQLite backend in `6Img-to-3D/webui/`, React+Vite frontend (same stack as seed4d webui), MLflow for experiment tracking. Dataloader fixes make `ManifestDataset` fully wire up `RaysDataset` from the target path so `train.py` can use JSONL manifests instead of hardcoded config hierarchies.

**Tech Stack:** Python/FastAPI/SQLAlchemy/MLflow, React/Vite/TypeScript/Tailwind/TanStack Query/Recharts, uv for deps.

**Design doc:** `docs/plans/2026-03-06-webui-data-pipeline-design.md`

---

## Phase 1 — Dataloader Fixes

### Task 1: Fix ManifestDataset to load RaysDataset from target path

The current `ManifestDataset.__getitem__` returns `entry["target"]` as a raw string.
`train.py` expects `(input_rgb, img_meta, sphere_dataloader)` where `sphere_dataloader`
is a `DataLoader` wrapping a `RaysDataset` (for val) or a numpy array (for train with pickles).
We wire up a `RaysDataset` from the target `transforms_ego.json` directory.

**Files:**
- Modify: `dataloader/manifest_dataset.py`
- Test: `tests/test_manifest_dataset.py`

**Step 1: Write the failing test**

```python
# tests/test_manifest_dataset.py
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def make_fake_transforms(tmp_path: Path, name: str = "transforms_ego.json") -> Path:
    """Write a minimal transforms_ego.json with 1 frame."""
    sensor_dir = tmp_path / "transforms"
    sensor_dir.mkdir(parents=True)
    # create a tiny 4x4 black image
    img_path = sensor_dir / "frame_0.png"
    import cv2, numpy as np
    cv2.imwrite(str(img_path), np.zeros((48, 64, 3), dtype=np.uint8))
    tf = {
        "fl_x": 100.0, "fl_y": 100.0, "cx": 32.0, "cy": 24.0,
        "w": 64, "h": 48,
        "frames": [{"file_path": "frame_0.png", "transform_matrix": np.eye(4).tolist()}],
    }
    tf_path = sensor_dir / name
    tf_path.write_text(json.dumps(tf))
    return tf_path


def make_manifest(tmp_path: Path, input_tf: Path, target_tf: Path) -> Path:
    entry = {
        "input": str(input_tf),
        "target": str(target_tf),
        "town": "Town02", "weather": "ClearNoon",
        "vehicle": "vehicle.mini.cooper_s", "spawn_point": 1, "step": 0,
    }
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")
    return manifest


def test_manifest_dataset_returns_rays_dataset(tmp_path):
    """ManifestDataset.__getitem__ must return (imgs, meta, DataLoader) not (imgs, meta, str)."""
    from dataloader.manifest_dataset import ManifestDataset
    from torch.utils.data import DataLoader

    input_tf = make_fake_transforms(tmp_path / "input")
    target_tf = make_fake_transforms(tmp_path / "target")
    manifest = make_manifest(tmp_path, input_tf, target_tf)

    cfg = MagicMock()
    cfg.decoder.whiteout = False
    dataset_cfg = MagicMock()
    dataset_cfg.depth = False
    dataset_cfg.phase = "val"

    ds = ManifestDataset(manifest, config=cfg, dataset_config=dataset_cfg)
    assert len(ds) == 1
    imgs, meta, loader = ds[0]
    assert hasattr(loader, "__iter__"), "third element must be iterable (DataLoader or list)"
    assert imgs.shape[0] == 1  # 1 frame stacked
    assert "K" in meta
```

**Step 2: Run test to confirm it fails**

```bash
cd /home/bdw/Documents/6Img-to-3D
uv run pytest tests/test_manifest_dataset.py::test_manifest_dataset_returns_rays_dataset -v
```
Expected: FAIL — `ManifestDataset.__init__` missing `config`/`dataset_config` params or returns string.

**Step 3: Implement the fix**

Replace `dataloader/manifest_dataset.py` entirely:

```python
"""Dataset that loads scenes from a JSONL manifest.

Each manifest line:
  {"input": "/abs/path/.../transforms_ego.json",
   "target": "/abs/path/.../transforms_ego.json",
   "town": ..., "weather": ..., "vehicle": ..., "spawn_point": ..., "step": ...}
"""

import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader

from dataloader.dataset import build_pose_intrinsics_vector, img_norm_cfg
from dataloader.rays_dataset import RaysDataset
from dataloader.transform_3d import NormalizeMultiviewImage


def _load_transforms(tf_path: Path) -> dict:
    with open(tf_path) as f:
        return json.load(f)


def _build_K_and_c2w(tf: dict) -> tuple[np.ndarray, list]:
    frames = tf["frames"]
    n = len(frames)
    K = np.zeros((n, 3, 4))
    c2ws = []
    for i, frame in enumerate(frames):
        K[i, 0, 0] = frame.get("fl_x", tf.get("fl_x", 0))
        K[i, 1, 1] = frame.get("fl_y", tf.get("fl_y", 0))
        K[i, 2, 2] = 1.0
        K[i, 0, 2] = frame.get("cx", tf.get("cx", 0))
        K[i, 1, 2] = frame.get("cy", tf.get("cy", 0))
        c2ws.append(frame["transform_matrix"])
    return K, c2ws


class ManifestDataset(data.Dataset):
    """Load scenes from a JSONL manifest file."""

    def __init__(self, manifest_path: str | Path, config, dataset_config):
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.dataset_config = dataset_config
        self.transforms = NormalizeMultiviewImage(**img_norm_cfg)
        self.entries = []
        with open(self.manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        input_tf_path = Path(entry["input"])
        tf = _load_transforms(input_tf_path)

        K, c2ws = _build_K_and_c2w(tf)
        sensor_dir = input_tf_path.parent

        imgs = []
        for frame in tf["frames"]:
            img_path = (sensor_dir / frame["file_path"]).resolve()
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)[:, :, :3].astype(np.float32)
            imgs.append(img)

        imgs = self.transforms(imgs)
        img_shape = [img.shape for img in imgs]

        img_meta = dict(
            K=K,
            c2w=c2ws,
            img_shape=img_shape,
            pose_intrinsics=build_pose_intrinsics_vector(c2ws, K),
            num_cams=len(c2ws),
            town=entry.get("town"),
            weather=entry.get("weather"),
            vehicle=entry.get("vehicle"),
            spawn_point=entry.get("spawn_point"),
            step=entry.get("step"),
        )

        # Load target RaysDataset from the sphere/transforms directory
        target_tf_path = Path(entry["target"])
        target_dir = target_tf_path.parent.parent  # .../sphere/
        mode = "test" if self.dataset_config.phase == "val" else self.dataset_config.phase
        sphere_dataset = RaysDataset(
            str(target_dir),
            config=self.config,
            dataset_config=self.dataset_config,
            mode=mode,
            factor=getattr(self.dataset_config, "factor", 1.0),
        )

        batch_size = getattr(self.dataset_config, "batch_size", 1)
        if self.dataset_config.phase == "train":
            sphere_dataloader = DataLoader(
                sphere_dataset, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True,
            )
        else:
            sphere_dataloader = DataLoader(
                sphere_dataset, batch_size=batch_size, shuffle=False,
            )

        return np.stack(imgs), img_meta, sphere_dataloader
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_manifest_dataset.py::test_manifest_dataset_returns_rays_dataset -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add dataloader/manifest_dataset.py tests/test_manifest_dataset.py
git commit -m "fix(dataloader): ManifestDataset wires up RaysDataset from target path"
```

---

### Task 2: Add manifest support to data_builder and train.py

`data_builder.build()` currently only handles `CarlaDataset`/`PickledCarlaDataset`.
We add a manifest branch and a `--manifest-train` / `--manifest-val` flag to `train.py`.

**Files:**
- Modify: `builder/data_builder.py`
- Modify: `train.py:445-462` (argparse section)
- Test: `tests/test_data_builder.py`

**Step 1: Write the failing test**

```python
# tests/test_data_builder.py
import json, tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np


def _make_manifest(tmp_path, name="train.jsonl"):
    import cv2
    sensor_dir = tmp_path / "input" / "transforms"
    sensor_dir.mkdir(parents=True)
    target_dir = tmp_path / "target" / "transforms"
    target_dir.mkdir(parents=True)
    for d in [sensor_dir, target_dir]:
        cv2.imwrite(str(d / "frame_0.png"), np.zeros((48, 64, 3), dtype=np.uint8))
    tf = {
        "fl_x": 100.0, "fl_y": 100.0, "cx": 32.0, "cy": 24.0, "w": 64, "h": 48,
        "frames": [{"file_path": "frame_0.png", "transform_matrix": np.eye(4).tolist()}],
    }
    for d in [sensor_dir, target_dir]:
        (d / "transforms_ego.json").write_text(json.dumps(tf))
    entry = {
        "input": str(sensor_dir / "transforms_ego.json"),
        "target": str(target_dir / "transforms_ego.json"),
        "town": "Town02", "weather": "ClearNoon",
        "vehicle": "v", "spawn_point": 1, "step": 0,
    }
    m = tmp_path / name
    m.write_text(json.dumps(entry) + "\n")
    return str(m)


def test_data_builder_manifest_branch(tmp_path):
    """build_from_manifests returns two DataLoaders."""
    from builder.data_builder import build_from_manifests
    from torch.utils.data import DataLoader

    train_m = _make_manifest(tmp_path / "train", "train.jsonl")
    val_m = _make_manifest(tmp_path / "val", "val.jsonl")

    cfg = MagicMock()
    cfg.decoder.whiteout = False
    train_dl_cfg = MagicMock()
    train_dl_cfg.depth = False
    train_dl_cfg.phase = "train"
    train_dl_cfg.batch_size = 1
    train_dl_cfg.factor = 1.0
    val_dl_cfg = MagicMock()
    val_dl_cfg.depth = False
    val_dl_cfg.phase = "val"
    val_dl_cfg.batch_size = 1
    val_dl_cfg.factor = 1.0

    train_loader, val_loader = build_from_manifests(
        train_manifest=train_m,
        val_manifest=val_m,
        config=cfg,
        train_dataset_config=train_dl_cfg,
        val_dataset_config=val_dl_cfg,
    )
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert len(train_loader.dataset) == 1
```

**Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_data_builder.py::test_data_builder_manifest_branch -v
```
Expected: FAIL — `build_from_manifests` does not exist.

**Step 3: Add `build_from_manifests` to data_builder.py**

Add after the existing `build()` function in `builder/data_builder.py`:

```python
from dataloader.manifest_dataset import ManifestDataset
from dataloader.dataset_wrapper import custom_collate_fn


def build_from_manifests(
    train_manifest: str,
    val_manifest: str,
    config,
    train_dataset_config,
    val_dataset_config,
):
    """Build train/val DataLoaders from JSONL manifest files."""
    train_dataset = ManifestDataset(train_manifest, config=config, dataset_config=train_dataset_config)
    val_dataset = ManifestDataset(val_manifest, config=config, dataset_config=val_dataset_config)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader
```

**Step 4: Add `--manifest-train` / `--manifest-val` to train.py**

In `train.py` at line 448, add after `--py-config`:

```python
parser.add_argument("--manifest-train", type=str, default="", help="Path to train.jsonl manifest")
parser.add_argument("--manifest-val", type=str, default="", help="Path to val.jsonl manifest")
```

In `main()` at line 137, replace:

```python
train_dataset_loader, val_dataset_loader = data_builder.build(cfg)
```

with:

```python
if args.manifest_train and args.manifest_val:
    from types import SimpleNamespace
    train_dl_cfg = SimpleNamespace(
        depth=cfg.dataset_params.train_data_loader.get("depth", False),
        phase="train",
        batch_size=cfg.dataset_params.train_data_loader.get("batch_size", 1),
        factor=cfg.dataset_params.train_data_loader.get("factor", 1.0),
    )
    val_dl_cfg = SimpleNamespace(
        depth=cfg.dataset_params.val_data_loader.get("depth", False),
        phase="val",
        batch_size=cfg.dataset_params.val_data_loader.get("batch_size", 1),
        factor=cfg.dataset_params.val_data_loader.get("factor", 0.25),
    )
    train_dataset_loader, val_dataset_loader = data_builder.build_from_manifests(
        train_manifest=args.manifest_train,
        val_manifest=args.manifest_val,
        config=cfg,
        train_dataset_config=train_dl_cfg,
        val_dataset_config=val_dl_cfg,
    )
else:
    train_dataset_loader, val_dataset_loader = data_builder.build(cfg)
```

**Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_data_builder.py::test_data_builder_manifest_branch -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add builder/data_builder.py train.py tests/test_data_builder.py
git commit -m "feat(dataloader): add manifest-based dataloader and --manifest-train/val flags"
```

---

## Phase 2 — Backend Scaffold

### Task 3: Create webui package structure

**Files:**
- Create: `webui/__init__.py`
- Create: `webui/backend/__init__.py`
- Create: `webui/backend/database.py`
- Create: `webui/backend/models.py`
- Create: `webui/backend/main.py`

**Step 1: Create package files**

`webui/__init__.py` — empty

`webui/backend/__init__.py` — empty

`webui/backend/database.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = "sqlite:///./webui.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

`webui/backend/models.py`:

```python
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from webui.backend.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class RecipeRecord(Base):
    __tablename__ = "recipes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255))
    yaml_content: Mapped[str] = mapped_column(Text)
    train_manifest: Mapped[str | None] = mapped_column(Text, nullable=True)
    val_manifest: Mapped[str | None] = mapped_column(Text, nullable=True)
    test_manifest: Mapped[str | None] = mapped_column(Text, nullable=True)
    scene_counts: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class JobRecord(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_type: Mapped[str] = mapped_column(String(20), default="train")  # "train" | "eval"
    name: Mapped[str] = mapped_column(String(255), default="")
    status: Mapped[str] = mapped_column(String(20), default="queued")
    mlflow_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    log: Mapped[str] = mapped_column(Text, default="")
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
```

`webui/backend/main.py`:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from webui.backend.database import Base, engine, SessionLocal
from webui.backend.models import JobRecord
from datetime import UTC, datetime


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


# Routers added in later tasks
```

**Step 2: Verify app starts**

```bash
cd /home/bdw/Documents/6Img-to-3D
uv run uvicorn webui.backend.main:app --reload --port 8001
```
Expected: server starts, `GET http://localhost:8001/api/health` returns `{"status":"ok"}`. Ctrl-C.

**Step 3: Commit**

```bash
git add webui/
git commit -m "feat(webui): scaffold FastAPI backend with DB models"
```

---

### Task 4: Data scan API

Returns the seed4d data directory hierarchy so the frontend can populate dropdowns.

**Files:**
- Create: `webui/backend/api/__init__.py`
- Create: `webui/backend/api/data.py`
- Modify: `webui/backend/main.py`

**Step 1: Write the failing test**

```python
# tests/test_webui_data_api.py
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json


@pytest.fixture
def fake_data_dir(tmp_path):
    """Create a minimal seed4d-style directory tree."""
    step = tmp_path / "Town02" / "ClearNoon" / "vehicle.mini.cooper_s" / "spawn_point_1" / "step_0" / "ego_vehicle"
    for sensor in ["nuscenes", "sphere"]:
        tf_dir = step / sensor / "transforms"
        tf_dir.mkdir(parents=True)
        tf = {"frames": [], "fl_x": 100, "fl_y": 100, "cx": 32, "cy": 24, "w": 64, "h": 48}
        (tf_dir / "transforms_ego.json").write_text(json.dumps(tf))
    return tmp_path


def test_scan_returns_hierarchy(fake_data_dir, monkeypatch):
    from webui.backend.api import data as data_api
    monkeypatch.setattr(data_api, "DATA_DIR", fake_data_dir)

    from webui.backend.main import app
    client = TestClient(app)
    resp = client.get("/api/data/scan")
    assert resp.status_code == 200
    result = resp.json()
    assert result["towns"] == ["Town02"]
    assert "ClearNoon" in result["weathers"]
    assert "nuscenes" in result["sensors"]
    assert "sphere" in result["sensors"]


def test_options_returns_available_filters(fake_data_dir, monkeypatch):
    from webui.backend.api import data as data_api
    monkeypatch.setattr(data_api, "DATA_DIR", fake_data_dir)

    from webui.backend.main import app
    client = TestClient(app)
    resp = client.get("/api/data/options")
    assert resp.status_code == 200
    opts = resp.json()
    assert "Town02" in opts["towns"]
    assert "vehicle.mini.cooper_s" in opts["vehicles"]
```

**Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_webui_data_api.py -v
```
Expected: FAIL — route not found.

**Step 3: Implement data scan API**

`webui/backend/api/__init__.py` — empty

`webui/backend/api/data.py`:

```python
from pathlib import Path
from fastapi import APIRouter

router = APIRouter(prefix="/api/data", tags=["data"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT.parent / "seed4d" / "data"


def _iter_scenes(data_dir: Path):
    """Yield dicts describing each scene (ego_vehicle dir)."""
    for ego_dir in sorted(data_dir.rglob("ego_vehicle")):
        parts = ego_dir.parts
        if len(parts) < 7:
            continue
        step_part = parts[-2]
        sp_part = parts[-3]
        vehicle = parts[-4]
        weather = parts[-5]
        town = parts[-6]
        sensors = sorted(d.name for d in ego_dir.iterdir() if d.is_dir())
        yield {
            "town": town,
            "weather": weather,
            "vehicle": vehicle,
            "spawn_point": sp_part,
            "step": step_part,
            "sensors": sensors,
            "path": str(ego_dir.relative_to(data_dir)),
        }


@router.get("/scan")
def scan():
    """Return aggregated hierarchy: available towns, weathers, vehicles, sensors."""
    if not DATA_DIR.exists():
        return {"towns": [], "weathers": [], "vehicles": [], "sensors": [], "scenes": []}
    towns, weathers, vehicles, sensors = set(), set(), set(), set()
    scenes = []
    for scene in _iter_scenes(DATA_DIR):
        towns.add(scene["town"])
        weathers.add(scene["weather"])
        vehicles.add(scene["vehicle"])
        sensors.update(scene["sensors"])
        scenes.append(scene)
    return {
        "towns": sorted(towns),
        "weathers": sorted(weathers),
        "vehicles": sorted(vehicles),
        "sensors": sorted(sensors),
        "scenes": scenes,
    }


@router.get("/options")
def options():
    """Same as scan but without per-scene detail — just the filter dropdowns."""
    result = scan()
    return {k: result[k] for k in ("towns", "weathers", "vehicles", "sensors")}
```

Register in `webui/backend/main.py` — add after the health route:

```python
from webui.backend.api.data import router as data_router
app.include_router(data_router)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_webui_data_api.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add webui/backend/api/ webui/backend/main.py tests/test_webui_data_api.py
git commit -m "feat(webui): add data scan API"
```

---

### Task 5: Recipe CRUD + export API

**Files:**
- Create: `webui/backend/api/recipes.py`
- Modify: `webui/backend/main.py`
- Test: `tests/test_webui_recipes_api.py`

**Step 1: Write the failing test**

```python
# tests/test_webui_recipes_api.py
import pytest
import json
from fastapi.testclient import TestClient
from pathlib import Path


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBUI_DB_URL", f"sqlite:///{tmp_path}/test.db")
    # Re-create engine for this test DB
    from webui.backend import database
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False})
    database.engine = engine
    database.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    from webui.backend.database import Base
    Base.metadata.create_all(bind=engine)
    from webui.backend.main import app
    return TestClient(app)


RECIPE = {
    "name": "test-recipe",
    "data_dir": "/tmp/seed4d/data",
    "output_dir": "/tmp/seed4d",
    "global_filters": {
        "vehicles": ["vehicle.mini.cooper_s"],
        "weathers": ["ClearNoon"],
        "input_sensor": "nuscenes",
        "target_sensor": "sphere",
    },
    "splits": {
        "train": {"towns": ["Town01"], "spawn_points": "all", "steps": "all"},
        "val": {"towns": ["Town02"], "spawn_points": "all", "steps": "all"},
        "test": {"towns": ["Town05"], "spawn_points": "all", "steps": "all"},
    },
}


def test_create_and_list_recipe(client):
    resp = client.post("/api/recipes", json=RECIPE)
    assert resp.status_code == 201
    created = resp.json()
    assert created["name"] == "test-recipe"
    assert "id" in created

    resp2 = client.get("/api/recipes")
    assert resp2.status_code == 200
    assert len(resp2.json()) == 1


def test_overlap_validation_fails(client):
    """If train and val share a town, creation should fail."""
    bad = dict(RECIPE)
    bad["splits"] = {
        "train": {"towns": ["Town02"], "spawn_points": "all", "steps": "all"},
        "val": {"towns": ["Town02"], "spawn_points": "all", "steps": "all"},
        "test": {"towns": ["Town05"], "spawn_points": "all", "steps": "all"},
    }
    resp = client.post("/api/recipes", json=bad)
    assert resp.status_code == 422
    assert "overlap" in resp.json()["detail"].lower()
```

**Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_webui_recipes_api.py -v
```
Expected: FAIL — route not found.

**Step 3: Implement recipes API**

`webui/backend/api/recipes.py`:

```python
import json
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from webui.backend.database import get_db
from webui.backend.models import RecipeRecord

router = APIRouter(prefix="/api/recipes", tags=["recipes"])


class SplitRule(BaseModel):
    towns: list[str] | str = "all"
    spawn_points: list[int] | str = "all"
    steps: list[int] | str = "all"


class GlobalFilters(BaseModel):
    vehicles: list[str] = []
    weathers: list[str] = []
    input_sensor: str = "nuscenes"
    target_sensor: str = "sphere"


class RecipeCreate(BaseModel):
    name: str
    data_dir: str
    output_dir: str
    global_filters: GlobalFilters
    splits: dict[str, SplitRule]  # keys: "train", "val", "test"


def _validate_no_overlap(splits: dict[str, SplitRule]) -> None:
    """Raise if any town appears in more than one split."""
    assigned: dict[str, str] = {}  # town -> split name
    for split_name, rule in splits.items():
        towns = rule.towns if isinstance(rule.towns, list) else []
        for town in towns:
            if town in assigned:
                raise HTTPException(
                    status_code=422,
                    detail=f"Town overlap: '{town}' appears in both '{assigned[town]}' and '{split_name}'",
                )
            assigned[town] = split_name


def _to_yaml(recipe: RecipeCreate) -> str:
    data = {
        "data_dir": recipe.data_dir,
        "output_dir": recipe.output_dir,
        "global": recipe.global_filters.model_dump(),
        "splits": {k: v.model_dump() for k, v in recipe.splits.items()},
    }
    return yaml.dump(data, default_flow_style=False)


@router.post("", status_code=201)
def create_recipe(recipe: RecipeCreate, db: Session = Depends(get_db)):
    _validate_no_overlap(recipe.splits)
    record = RecipeRecord(
        name=recipe.name,
        yaml_content=_to_yaml(recipe),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "name": record.name, "created_at": str(record.created_at)}


@router.get("")
def list_recipes(db: Session = Depends(get_db)):
    return [
        {"id": r.id, "name": r.name, "created_at": str(r.created_at),
         "train_manifest": r.train_manifest, "val_manifest": r.val_manifest,
         "scene_counts": r.scene_counts}
        for r in db.query(RecipeRecord).order_by(RecipeRecord.created_at.desc()).all()
    ]


@router.get("/{recipe_id}")
def get_recipe(recipe_id: str, db: Session = Depends(get_db)):
    r = db.get(RecipeRecord, recipe_id)
    if not r:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return {"id": r.id, "name": r.name, "yaml_content": r.yaml_content,
            "train_manifest": r.train_manifest, "val_manifest": r.val_manifest,
            "test_manifest": r.test_manifest, "scene_counts": r.scene_counts}
```

Register in `webui/backend/main.py`:

```python
from webui.backend.api.recipes import router as recipes_router
app.include_router(recipes_router)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_webui_recipes_api.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add webui/backend/api/recipes.py webui/backend/main.py tests/test_webui_recipes_api.py
git commit -m "feat(webui): recipe CRUD API with overlap validation"
```

---

### Task 6: Recipe export — generate JSONL manifests from recipe

**Files:**
- Create: `webui/backend/services/manifest_exporter.py`
- Modify: `webui/backend/api/recipes.py`
- Test: `tests/test_manifest_exporter.py`

**Step 1: Write the failing test**

```python
# tests/test_manifest_exporter.py
import json
from pathlib import Path
import pytest


@pytest.fixture
def fake_data(tmp_path):
    """Seed4d-style tree: Town01 (train) + Town02 (val)."""
    for town, sp, step in [("Town01", 1, 0), ("Town01", 2, 0), ("Town02", 1, 0)]:
        for sensor in ["nuscenes", "sphere"]:
            d = tmp_path / town / "ClearNoon" / "vehicle.mini.cooper_s" / f"spawn_point_{sp}" / f"step_{step}" / "ego_vehicle" / sensor / "transforms"
            d.mkdir(parents=True)
            tf = {"fl_x": 100, "fl_y": 100, "cx": 32, "cy": 24, "w": 64, "h": 48, "frames": []}
            (d / "transforms_ego.json").write_text(json.dumps(tf))
    return tmp_path


def test_export_creates_correct_manifests(fake_data, tmp_path):
    from webui.backend.services.manifest_exporter import export_manifests

    recipe = {
        "data_dir": str(fake_data),
        "output_dir": str(tmp_path / "out"),
        "global": {
            "vehicles": ["vehicle.mini.cooper_s"],
            "weathers": ["ClearNoon"],
            "input_sensor": "nuscenes",
            "target_sensor": "sphere",
        },
        "splits": {
            "train": {"towns": ["Town01"], "spawn_points": "all", "steps": "all"},
            "val": {"towns": ["Town02"], "spawn_points": "all", "steps": "all"},
        },
    }

    result = export_manifests(recipe)
    assert result["scene_counts"]["train"] == 2
    assert result["scene_counts"]["val"] == 1

    train_lines = Path(result["train"]).read_text().strip().splitlines()
    assert len(train_lines) == 2
    entry = json.loads(train_lines[0])
    assert "input" in entry and "target" in entry
    assert entry["town"] == "Town01"
    assert "nuscenes" in entry["input"]
    assert "sphere" in entry["target"]
```

**Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_manifest_exporter.py::test_export_creates_correct_manifests -v
```
Expected: FAIL

**Step 3: Implement manifest_exporter**

`webui/backend/services/__init__.py` — empty

`webui/backend/services/manifest_exporter.py`:

```python
import json
from pathlib import Path


def _discover_scenes(data_dir: Path, global_filters: dict) -> list[dict]:
    """Walk data_dir, yield entries matching global filters."""
    vehicles = set(global_filters.get("vehicles") or [])
    weathers = set(global_filters.get("weathers") or [])
    input_sensor = global_filters["input_sensor"]
    target_sensor = global_filters["target_sensor"]

    entries = []
    for ego_dir in sorted(data_dir.rglob("ego_vehicle")):
        parts = ego_dir.parts
        if len(parts) < 7:
            continue
        step_part = parts[-2]
        sp_part = parts[-3]
        vehicle = parts[-4]
        weather = parts[-5]
        town = parts[-6]

        if vehicles and vehicle not in vehicles:
            continue
        if weathers and weather not in weathers:
            continue

        input_tf = ego_dir / input_sensor / "transforms" / "transforms_ego.json"
        target_tf = ego_dir / target_sensor / "transforms" / "transforms_ego.json"
        if not input_tf.exists() or not target_tf.exists():
            continue

        entries.append({
            "input": str(input_tf),
            "target": str(target_tf),
            "town": town,
            "weather": weather,
            "vehicle": vehicle,
            "spawn_point": int(sp_part.split("_")[-1]),
            "step": int(step_part.split("_")[-1]),
        })
    return entries


def _matches_split_rule(entry: dict, rule: dict) -> bool:
    towns = rule.get("towns", "all")
    spawn_points = rule.get("spawn_points", "all")
    steps = rule.get("steps", "all")

    if towns != "all" and entry["town"] not in towns:
        return False
    if spawn_points != "all" and entry["spawn_point"] not in spawn_points:
        return False
    if steps != "all" and entry["step"] not in steps:
        return False
    return True


def export_manifests(recipe: dict) -> dict:
    """Generate JSONL manifests from a recipe dict. Returns paths + scene counts."""
    data_dir = Path(recipe["data_dir"])
    output_dir = Path(recipe["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries = _discover_scenes(data_dir, recipe["global"])
    splits = recipe.get("splits", {})

    result = {"scene_counts": {}}
    for split_name, rule in splits.items():
        matched = [e for e in all_entries if _matches_split_rule(e, rule)]
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for entry in matched:
                f.write(json.dumps(entry) + "\n")
        result[split_name] = str(out_path)
        result["scene_counts"][split_name] = len(matched)

    return result
```

Add export endpoint to `webui/backend/api/recipes.py`:

```python
import yaml as _yaml
from webui.backend.services.manifest_exporter import export_manifests as _export


@router.post("/{recipe_id}/export")
def export_recipe(recipe_id: str, db: Session = Depends(get_db)):
    r = db.get(RecipeRecord, recipe_id)
    if not r:
        raise HTTPException(status_code=404, detail="Recipe not found")
    recipe_dict = _yaml.safe_load(r.yaml_content)
    result = _export(recipe_dict)
    r.train_manifest = result.get("train")
    r.val_manifest = result.get("val")
    r.test_manifest = result.get("test")
    r.scene_counts = result["scene_counts"]
    db.commit()
    return result
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_manifest_exporter.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add webui/backend/services/ webui/backend/api/recipes.py tests/test_manifest_exporter.py
git commit -m "feat(webui): recipe export generates JSONL manifests from data dir"
```

---

## Phase 3 — Split Manager UI

### Task 7: Frontend scaffold

**Files:**
- Create: `webui/frontend/` (Vite + React + Tailwind — same setup as seed4d webui)

**Step 1: Scaffold frontend**

```bash
cd /home/bdw/Documents/6Img-to-3D/webui
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install react-router-dom @tanstack/react-query recharts axios
npm install -D @types/recharts
```

Configure `tailwind.config.js` (content: `["./index.html", "./src/**/*.{ts,tsx}"]`).

Replace `src/index.css` with:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Step 2: Create App shell (3 pages)**

`src/App.tsx`:

```tsx
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import SplitManager from './pages/SplitManager'
import Training from './pages/Training'
import Evaluation from './pages/Evaluation'

const queryClient = new QueryClient()

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <nav className="border-b border-gray-800 px-6 py-3 flex gap-6 items-center">
        <span className="font-bold text-lg tracking-tight">6Img-to-3D</span>
        {[
          { to: '/splits', label: 'Split Manager' },
          { to: '/training', label: 'Training' },
          { to: '/eval', label: 'Evaluation' },
        ].map(({ to, label }) => (
          <NavLink key={to} to={to} className={({ isActive }) =>
            isActive ? 'text-blue-400' : 'text-gray-400 hover:text-gray-200'
          }>{label}</NavLink>
        ))}
      </nav>
      <main className="p-6">{children}</main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<SplitManager />} />
            <Route path="/splits" element={<SplitManager />} />
            <Route path="/training" element={<Training />} />
            <Route path="/eval" element={<Evaluation />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
```

Create stub pages `src/pages/SplitManager.tsx`, `Training.tsx`, `Evaluation.tsx` — each just returns `<div>Coming soon</div>`.

**Step 3: Verify frontend builds**

```bash
cd /home/bdw/Documents/6Img-to-3D/webui/frontend
npm run build
```
Expected: build succeeds with no errors.

**Step 4: Commit**

```bash
cd /home/bdw/Documents/6Img-to-3D
git add webui/frontend/
git commit -m "feat(webui): scaffold React frontend with 3-page shell"
```

---

### Task 8: Split Manager — API client + Global Filters panel

**Files:**
- Create: `webui/frontend/src/api.ts`
- Create: `webui/frontend/src/pages/SplitManager.tsx` (full implementation)

**Step 1: API client**

`src/api.ts`:

```ts
import axios from 'axios'

const api = axios.create({ baseURL: 'http://localhost:8001' })

export async function getDataOptions() {
  const res = await api.get('/api/data/options')
  return res.data as {
    towns: string[]; weathers: string[]; vehicles: string[]; sensors: string[]
  }
}

export async function listRecipes() {
  const res = await api.get('/api/recipes')
  return res.data as Array<{ id: string; name: string; scene_counts: Record<string, number> | null }>
}

export async function createRecipe(payload: unknown) {
  const res = await api.post('/api/recipes', payload)
  return res.data
}

export async function exportRecipe(id: string) {
  const res = await api.post(`/api/recipes/${id}/export`)
  return res.data
}

export async function previewSplit(payload: unknown) {
  const res = await api.post('/api/data/preview', payload)
  return res.data as Record<string, number>
}
```

Add preview endpoint to `webui/backend/api/data.py`:

```python
from pydantic import BaseModel

class PreviewRequest(BaseModel):
    data_dir: str
    global_filters: dict
    splits: dict

@router.post("/preview")
def preview(req: PreviewRequest):
    from webui.backend.services.manifest_exporter import _discover_scenes, _matches_split_rule
    data_dir = Path(req.data_dir)
    if not data_dir.exists():
        return {k: 0 for k in req.splits}
    all_entries = _discover_scenes(data_dir, req.global_filters)
    return {
        split_name: sum(1 for e in all_entries if _matches_split_rule(e, rule))
        for split_name, rule in req.splits.items()
    }
```

**Step 2: Implement SplitManager page**

`src/pages/SplitManager.tsx` — three-column layout with:
- Left: Global Filters (vehicle multi-select, weather multi-select, input/target sensor dropdowns)
- Center: Split Rules (Train/Val/Test panels each with town checkboxes; already-checked towns greyed in other panels)
- Right: Live preview counts (polls `/api/data/preview` on any filter change), Export button

Key state:

```ts
const [globalFilters, setGlobalFilters] = useState({
  vehicles: [] as string[],
  weathers: [] as string[],
  input_sensor: 'nuscenes',
  target_sensor: 'sphere',
})
const [splits, setSplits] = useState({
  train: { towns: [] as string[], spawn_points: 'all', steps: 'all' },
  val: { towns: [] as string[], spawn_points: 'all', steps: 'all' },
  test: { towns: [] as string[], spawn_points: 'all', steps: 'all' },
})
```

Mutual exclusivity: when a town is checked in `train`, disable it in `val` and `test` checkboxes.

Export flow: POST `/api/recipes` (create), then POST `/api/recipes/{id}/export`.

**Step 3: Verify UI works end-to-end**

```bash
# Terminal 1
cd /home/bdw/Documents/6Img-to-3D
uv run uvicorn webui.backend.main:app --reload --port 8001

# Terminal 2
cd /home/bdw/Documents/6Img-to-3D/webui/frontend
npm run dev
```
Open `http://localhost:5173/splits` — verify dropdowns populate from seed4d data, towns can be assigned to splits, preview counts update live.

**Step 4: Commit**

```bash
git add webui/frontend/src/ webui/backend/api/data.py
git commit -m "feat(webui): Split Manager UI with live preview and export"
```

---

## Phase 4 — MLflow Integration

### Task 9: Switch train.py from tensorboardX to MLflow

**Files:**
- Modify: `train.py`
- Modify: `pyproject.toml` (add mlflow dep)

**Step 1: Add MLflow dependency**

```bash
cd /home/bdw/Documents/6Img-to-3D
uv add mlflow
```

**Step 2: Replace tensorboardX logging in train.py**

At the top of `train.py`, replace:

```python
from tensorboardX import SummaryWriter
```

with:

```python
import mlflow
import mlflow.pytorch
```

Replace the `SummaryWriter` setup block (lines 39-43):

```python
    mlflow.set_experiment(args.log_dir or "6img-to-3d")
    mlflow.start_run(run_name=args.log_dir or None)
    mlflow.log_params({
        "config": args.py_config,
        "lr": cfg.optimizer.lr,
        "num_epochs": cfg.optimizer.num_epochs,
        "manifest_train": args.manifest_train or "config",
    })
    logdir = f"runs/{time.strftime('%b%d_%H-%M-%S', time.localtime())}_{args.log_dir}"
    os.makedirs(logdir, exist_ok=True)
    save_dir = os.path.join(logdir, "models")
```

Replace `writer.add_scalars("Loss/train", loss_dict, epoch)` (line 291) with:

```python
    mlflow.log_metrics({f"train/{k}": v for k, v in loss_dict.items()}, step=epoch)
```

Replace `writer.add_scalar("val/psnr", ...)` and `writer.add_scalar("val/lpips", ...)` (lines 405-406) with:

```python
    mlflow.log_metrics({"val/psnr": float(np.mean(psnr_list)), "val/lpips": float(np.mean(lpips_list))}, step=epoch)
```

Replace `writer.add_figure(...)` calls with no-ops (images logged as artifacts at eval time).

At the end of `main()`, before the function closes, add:

```python
    mlflow.end_run()
```

**Step 3: Verify training still runs (smoke test)**

```bash
uv run python train.py --py-config config/config.py --log-dir test-mlflow --num-scenes 1
```
Expected: completes 1 epoch without error. `mlruns/` directory created.

**Step 4: Commit**

```bash
git add train.py pyproject.toml uv.lock
git commit -m "feat(train): switch logging from tensorboardX to MLflow"
```

---

### Task 10: Training job runner + metrics API

**Files:**
- Create: `webui/backend/services/job_runner.py`
- Create: `webui/backend/api/jobs.py`
- Modify: `webui/backend/main.py`

**Step 1: Implement job runner**

`webui/backend/services/job_runner.py` — adapted from seed4d's version but simpler
(no Docker, runs `train.py` directly as a subprocess):

```python
import asyncio
import contextlib
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from webui.backend.database import SessionLocal
from webui.backend.models import JobRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_active_job_ids: set[str] = set()
_job_subscribers: dict[str, list[asyncio.Queue]] = {}


def subscribe(job_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _job_subscribers.setdefault(job_id, []).append(q)
    return q


def unsubscribe(job_id: str, q: asyncio.Queue):
    if job_id in _job_subscribers:
        _job_subscribers[job_id] = [x for x in _job_subscribers[job_id] if x is not q]


async def _broadcast(job_id: str, msg: dict):
    for q in _job_subscribers.get(job_id, []):
        await q.put(msg)


async def run_train_job(job_id: str, py_config: str, log_dir: str,
                        manifest_train: str = "", manifest_val: str = ""):
    db = SessionLocal()
    job = db.get(JobRecord, job_id)
    if not job:
        db.close()
        return
    job.status = "running"
    job.started_at = datetime.now(UTC)
    db.commit()
    _active_job_ids.add(job_id)

    log_file = PROJECT_ROOT / "runs" / f"job_{job_id}.log"
    log_file.parent.mkdir(exist_ok=True)
    log_file.write_text("")

    cmd = [
        "uv", "run", "python", "-u", "train.py",
        "--py-config", py_config,
        "--log-dir", log_dir,
    ]
    if manifest_train:
        cmd += ["--manifest-train", manifest_train, "--manifest-val", manifest_val]

    env = {**os.environ, "MLFLOW_EXPERIMENT_NAME": log_dir, "PYTHONUNBUFFERED": "1"}

    with open(log_file, "w") as lf:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=lf, stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT), env=env,
        )

    job.pid = process.pid
    db.commit()
    db.close()

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
        await asyncio.sleep(2)

    db = SessionLocal()
    job = db.get(JobRecord, job_id)
    if job:
        job.log = log_file.read_text() if log_file.exists() else ""
        job.pid = None
        if process.returncode == 0:
            job.status = "completed"
        else:
            job.status = "failed"
            job.error = f"exit code {process.returncode}"
        job.completed_at = datetime.now(UTC)
        db.commit()
    db.close()
    _active_job_ids.discard(job_id)
    await _broadcast(job_id, {"type": "status", "status": job.status})


def kill_job(job_id: str):
    db = SessionLocal()
    job = db.get(JobRecord, job_id)
    if job and job.pid:
        with contextlib.suppress(ProcessLookupError):
            os.kill(job.pid, 15)
        job.status = "cancelled"
        job.completed_at = datetime.now(UTC)
        db.commit()
    db.close()


def mark_active_jobs_failed():
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
```

**Step 2: Jobs API**

`webui/backend/api/jobs.py`:

```python
import asyncio
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from webui.backend.database import get_db
from webui.backend.models import JobRecord
from webui.backend.services import job_runner

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class TrainJobCreate(BaseModel):
    name: str
    py_config: str = "config/config.py"
    manifest_train: str = ""
    manifest_val: str = ""


@router.post("/train", status_code=201)
def create_train_job(payload: TrainJobCreate, background: BackgroundTasks, db: Session = Depends(get_db)):
    job = JobRecord(job_type="train", name=payload.name)
    db.add(job)
    db.commit()
    db.refresh(job)
    background.add_task(
        asyncio.ensure_future,
        job_runner.run_train_job(
            job.id, payload.py_config, payload.name,
            payload.manifest_train, payload.manifest_val,
        ),
    )
    return {"id": job.id, "status": "queued"}


@router.get("")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(JobRecord).order_by(JobRecord.created_at.desc()).all()
    return [
        {"id": j.id, "name": j.name, "type": j.job_type, "status": j.status,
         "mlflow_run_id": j.mlflow_run_id, "created_at": str(j.created_at),
         "completed_at": str(j.completed_at) if j.completed_at else None}
        for j in jobs
    ]


@router.get("/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    j = db.get(JobRecord, job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return {"id": j.id, "name": j.name, "status": j.status, "log": j.log,
            "mlflow_run_id": j.mlflow_run_id, "error": j.error}


@router.get("/{job_id}/stream")
async def stream_job_log(job_id: str):
    queue = job_runner.subscribe(job_id)
    async def event_generator():
        try:
            while True:
                msg = await asyncio.wait_for(queue.get(), timeout=30)
                yield f"data: {msg}\n\n"
                if msg.get("type") == "status" and msg.get("status") in ("completed", "failed", "cancelled"):
                    break
        except asyncio.TimeoutError:
            yield "data: {\"type\": \"ping\"}\n\n"
        finally:
            job_runner.unsubscribe(job_id, queue)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.delete("/{job_id}")
def cancel_job(job_id: str):
    job_runner.kill_job(job_id)
    return {"status": "cancelled"}
```

Register in `webui/backend/main.py`:

```python
from webui.backend.api.jobs import router as jobs_router
from webui.backend.services.job_runner import mark_active_jobs_failed

# Add to lifespan cleanup:
# yield
# mark_active_jobs_failed()

app.include_router(jobs_router)
```

**Step 3: MLflow metrics API**

Add to `webui/backend/api/jobs.py`:

```python
@router.get("/{job_id}/metrics")
def get_metrics(job_id: str, db: Session = Depends(get_db)):
    """Return metric history from MLflow for a run."""
    import mlflow
    j = db.get(JobRecord, job_id)
    if not j or not j.mlflow_run_id:
        # Try to find by experiment name = job name
        client = mlflow.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[],
            filter_string=f"tags.mlflow.runName = '{j.name}'" if j else "",
        )
        if not runs:
            return {}
        run_id = runs[0].info.run_id
    else:
        run_id = j.mlflow_run_id

    client = mlflow.MlflowClient()
    metric_keys = [m.key for m in client.get_run(run_id).data.metrics.keys()]
    # Actually get metric keys properly:
    run = client.get_run(run_id)
    keys = list(run.data.metrics.keys())
    history = {}
    for key in keys:
        history[key] = [
            {"step": m.step, "value": m.value}
            for m in client.get_metric_history(run_id, key)
        ]
    return history
```

**Step 4: Commit**

```bash
git add webui/backend/services/job_runner.py webui/backend/api/jobs.py webui/backend/main.py
git commit -m "feat(webui): training job runner + MLflow metrics API"
```

---

## Phase 5 — Training Dashboard UI

### Task 11: Training page — experiment list + run detail with live charts

**Files:**
- Create: `webui/frontend/src/pages/Training.tsx`
- Modify: `webui/frontend/src/api.ts`

**Step 1: Add training API calls to api.ts**

```ts
export async function listJobs() {
  const res = await api.get('/api/jobs')
  return res.data as Array<{ id: string; name: string; status: string; created_at: string }>
}

export async function getJob(id: string) {
  const res = await api.get(`/api/jobs/${id}`)
  return res.data
}

export async function getMetrics(id: string) {
  const res = await api.get(`/api/jobs/${id}/metrics`)
  return res.data as Record<string, Array<{ step: number; value: number }>>
}

export async function createTrainJob(payload: {
  name: string; py_config: string; manifest_train: string; manifest_val: string
}) {
  const res = await api.post('/api/jobs/train', payload)
  return res.data
}

export async function cancelJob(id: string) {
  await api.delete(`/api/jobs/${id}`)
}
```

**Step 2: Implement Training.tsx**

Two-panel layout:
- Left: job list with status badges (running=blue dot, done=green, failed=red), click to select
- Right: selected job detail
  - Tabs: Loss chart | Val Metrics | Log
  - Loss chart: Recharts `LineChart` with lines for `train/loss`, `train/mse_loss`, `train/lpips_loss`
  - Val metrics: `val/psnr` and `val/lpips` as numbers + sparkline
  - Log: `<pre>` with last 50 lines, auto-scrolled
  - "New Run" button opens a modal dialog (name, manifest dropdown from recipes, config path)

Metrics polling: use `useQuery` with `refetchInterval: 5000` when job status is "running".

Log streaming: `EventSource` on `/api/jobs/{id}/stream`, append lines to local state.

**Step 3: Verify UI works**

```bash
# Start backend and frontend as in Task 8
```
Open `http://localhost:5173/training` — verify job list loads, new run dialog works, metrics chart updates.

**Step 4: Commit**

```bash
git add webui/frontend/src/pages/Training.tsx webui/frontend/src/api.ts
git commit -m "feat(webui): Training Dashboard with live MLflow charts"
```

---

## Phase 6 — Evaluation Viewer

### Task 12: Eval job runner + eval.py manifest flag

**Files:**
- Modify: `eval.py` (add `--manifest-val` flag)
- Modify: `webui/backend/services/job_runner.py`
- Modify: `webui/backend/api/jobs.py`

**Step 1: Add `--manifest-val` flag to eval.py**

Read `eval.py` first, then add after `--resume-from`:

```python
parser.add_argument("--manifest-val", type=str, default="", help="Path to val.jsonl manifest")
```

And in the dataset building section, add the same manifest branch as in `train.py` Task 2.

**Step 2: Add `run_eval_job` to job_runner.py**

```python
async def run_eval_job(job_id: str, resume_from: str, manifest_val: str,
                       py_config: str, log_dir: str):
    db = SessionLocal()
    job = db.get(JobRecord, job_id)
    if not job:
        db.close(); return
    job.status = "running"
    job.started_at = datetime.now(UTC)
    db.commit()
    _active_job_ids.add(job_id)

    log_file = PROJECT_ROOT / "runs" / f"job_{job_id}.log"
    log_file.parent.mkdir(exist_ok=True)

    cmd = [
        "uv", "run", "python", "-u", "eval.py",
        "--py-config", py_config,
        "--resume-from", resume_from,
        "--log-dir", log_dir,
        "--depth", "--img-gt",
    ]
    if manifest_val:
        cmd += ["--manifest-val", manifest_val]

    with open(log_file, "w") as lf:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=lf, stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    # ... same log-tail loop as run_train_job ...
```

**Step 3: Add eval job endpoint to jobs.py**

```python
class EvalJobCreate(BaseModel):
    name: str
    resume_from: str
    manifest_val: str = ""
    py_config: str = "config/config.py"


@router.post("/eval", status_code=201)
def create_eval_job(payload: EvalJobCreate, background: BackgroundTasks, db: Session = Depends(get_db)):
    job = JobRecord(job_type="eval", name=payload.name)
    db.add(job)
    db.commit()
    db.refresh(job)
    background.add_task(
        asyncio.ensure_future,
        job_runner.run_eval_job(
            job.id, payload.resume_from, payload.manifest_val,
            payload.py_config, payload.name,
        ),
    )
    return {"id": job.id, "status": "queued"}
```

**Step 4: Commit**

```bash
git add eval.py webui/backend/services/job_runner.py webui/backend/api/jobs.py
git commit -m "feat(webui): eval job runner with --manifest-val support"
```

---

### Task 13: Evaluation Viewer UI

**Files:**
- Create: `webui/frontend/src/pages/Evaluation.tsx`
- Add rendered image serving to `webui/backend/api/jobs.py`

**Step 1: Add rendered image API**

```python
@router.get("/{job_id}/renders")
def list_renders(job_id: str):
    """List rendered images saved to runs/{name}/renders/."""
    from pathlib import Path
    job_dir = PROJECT_ROOT / "runs"
    # Find dir by job name
    db = SessionLocal()
    j = db.get(JobRecord, job_id)
    db.close()
    if not j:
        raise HTTPException(404, "Job not found")
    renders_dir = PROJECT_ROOT / j.name / "renders"
    if not renders_dir.exists():
        return []
    return sorted(str(p.relative_to(PROJECT_ROOT)) for p in renders_dir.glob("*.png"))


@router.get("/{job_id}/renders/{filename}")
def get_render(job_id: str, filename: str):
    from fastapi.responses import FileResponse
    db = SessionLocal()
    j = db.get(JobRecord, job_id)
    db.close()
    if not j:
        raise HTTPException(404)
    img_path = PROJECT_ROOT / j.name / "renders" / filename
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(img_path)
```

**Step 2: Implement Evaluation.tsx**

Two-panel layout:
- Left: list of completed eval jobs (filtered to `job_type == "eval"`)
- Right: selected eval detail
  - Metrics summary card: PSNR / LPIPS / SSIM parsed from job log
  - Scene browser: grid of rendered images loaded from `/api/jobs/{id}/renders`
  - Each image cell shows GT (from manifest target) | Rendered | toggle Diff heatmap
  - Compare mode: two job selectors, same scene side by side
  - "Run Eval" button opens dialog (pick a training run, pick checkpoint, pick manifest)

**Step 3: Verify end-to-end**

Run a full training cycle (even 1 epoch), then eval, then view in UI.

```bash
uv run python train.py --py-config config/config.py --log-dir e2e-test \
    --manifest-train /home/bdw/Documents/seed4d/train.jsonl \
    --manifest-val /home/bdw/Documents/seed4d/val.jsonl \
    --num-scenes 2
```

**Step 4: Commit**

```bash
git add webui/frontend/src/pages/Evaluation.tsx webui/backend/api/jobs.py
git commit -m "feat(webui): Evaluation Viewer with scene browser and compare mode"
```

---

### Task 14: Add dev.sh + pyproject.toml entries for webui

**Files:**
- Create: `webui/dev.sh`
- Modify: `pyproject.toml`

**Step 1: dev.sh**

```bash
#!/usr/bin/env bash
# Start backend and frontend in parallel
set -e
cd "$(dirname "$0")/.."
uv run uvicorn webui.backend.main:app --reload --port 8001 &
BACKEND_PID=$!
cd webui/frontend && npm run dev &
FRONTEND_PID=$!
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
```

```bash
chmod +x webui/dev.sh
```

**Step 2: Add webui deps to pyproject.toml**

```bash
uv add fastapi uvicorn[standard] sqlalchemy pyyaml mlflow
```

**Step 3: Final smoke test**

```bash
cd /home/bdw/Documents/6Img-to-3D
uv run pytest tests/ -v
```
Expected: all tests pass.

**Step 4: Final commit**

```bash
git add webui/dev.sh pyproject.toml uv.lock
git commit -m "feat(webui): add dev.sh launcher and webui deps to pyproject.toml"
```
