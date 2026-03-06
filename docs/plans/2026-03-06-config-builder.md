# Config Builder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "Config" tab to the webui that lets users create, edit, import, and export `config.py` files through a structured form with live preview.

**Architecture:** Pydantic models define the config schema on the backend; FastAPI routes handle CRUD + import/export; `config_io.py` handles parsing existing `config.py` files via `mmengine.Config` and rendering new ones as Python strings. The frontend is a two-panel React page: a sectioned form on the left and a live `config.py` preview on the right. `train.py` is untouched — the webui generates a valid `config.py` as output.

**Tech Stack:** FastAPI, SQLAlchemy (SQLite), Pydantic v2, mmengine, React 18, TypeScript, TanStack Query, Tailwind CSS

**Design doc:** `docs/plans/2026-03-06-config-builder-design.md`

---

## Task 1: Pydantic schema

**Files:**
- Create: `webui/backend/schemas/config_schema.py`
- Create: `webui/backend/schemas/__init__.py`

**Step 1: Create the schema package**

```bash
mkdir -p webui/backend/schemas
touch webui/backend/schemas/__init__.py
```

**Step 2: Write `config_schema.py`**

```python
# webui/backend/schemas/config_schema.py
from pydantic import BaseModel, Field


class PIFConfig(BaseModel):
    enabled: bool = False
    factor: float = 0.125
    transforms_path: str = ""


class EncoderConfig(BaseModel):
    dim: int = 128
    num_heads: int = 8
    num_levels: int = 4
    max_cams: int = 6
    min_cams_train: int = 1
    tpv_h: int = 200
    tpv_w: int = 200
    tpv_z: int = 16
    num_encoder_layers: int = 5
    scene_contraction: bool = True
    scene_contraction_factor: list[float] = Field(default_factory=lambda: [0.5, 0.1, 0.1])
    offset: list[float] = Field(default_factory=lambda: [-4.0, 0.0, 0.0])
    scale: list[float] = Field(default_factory=lambda: [0.25, 0.25, 0.25])
    num_points_in_pillar: list[int] = Field(default_factory=lambda: [4, 32, 32])
    num_points: list[int] = Field(default_factory=lambda: [8, 64, 64])
    hybrid_attn_anchors: int = 16
    hybrid_attn_points: int = 32


class DecoderConfig(BaseModel):
    hidden_dim: int = 128
    hidden_layers: int = 5
    density_activation: str = "trunc_exp"
    nb_bins: int = 64
    nb_bins_sample: int = 64
    hn: float = 0.0
    hf: float = 60.0
    train_stratified: bool = True
    white_background: bool = False
    whiteout: bool = False
    testing_batch_size: int = 8192


class OptimizerConfig(BaseModel):
    lr: float = 5e-5
    num_epochs: int = 100
    num_warmup_steps: int = 1000
    lpips_loss_weight: float = 0.2
    tv_loss_weight: float = 0.0
    dist_loss_weight: float = 1e-3
    depth_loss_weight: float = 1.0
    clip_grad_norm: float = 1.5


class TrainLoaderConfig(BaseModel):
    pickled: bool = True
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 12
    towns: list[str] = Field(default_factory=lambda: [
        "Town01", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"
    ])
    weather: list[str] = Field(default_factory=lambda: ["ClearNoon"])
    vehicle: list[str] = Field(default_factory=lambda: ["vehicle.tesla.invisible"])
    factor: float = 0.08
    num_imgs: int = 3
    depth: bool = True
    min_cams_train: int = 1
    max_cams_train: int = 6


class ValLoaderConfig(BaseModel):
    batch_size: int = 1
    num_workers: int = 12
    towns: list[str] = Field(default_factory=lambda: ["Town02"])
    weather: list[str] = Field(default_factory=lambda: ["ClearNoon"])
    vehicle: list[str] = Field(default_factory=lambda: ["vehicle.tesla.invisible"])
    spawn_point: list[int] = Field(default_factory=lambda: [3, 7, 12, 48, 98, 66])
    factor: float = 0.25
    depth: bool = True


class DatasetConfig(BaseModel):
    data_path: str = "/app/data/"
    train: TrainLoaderConfig = Field(default_factory=TrainLoaderConfig)
    val: ValLoaderConfig = Field(default_factory=ValLoaderConfig)


class FullConfig(BaseModel):
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    pif: PIFConfig = Field(default_factory=PIFConfig)
```

**Step 3: Write a quick sanity test**

```bash
# From repo root
uv run python -c "
from webui.backend.schemas.config_schema import FullConfig
c = FullConfig()
print('encoder dim:', c.encoder.dim)
print('decoder hidden_layers:', c.decoder.hidden_layers)
print('OK')
"
```

Expected output ends with `OK`.

**Step 4: Commit**

```bash
git add webui/backend/schemas/
git commit -m "feat(webui): add Pydantic config schema (FullConfig)"
```

---

## Task 2: config_io.py — import and export

**Files:**
- Create: `webui/backend/services/config_io.py`
- Create: `tests/webui/test_config_io.py`

**What this module does:**
- `load_from_py(path: str) -> FullConfig` — uses `mmengine.Config.fromfile()` then maps keys to the Pydantic schema
- `export_to_py(cfg: FullConfig) -> str` — renders a `config.py` string that `mmengine.Config.fromfile()` can reload

**Step 1: Create test file**

```python
# tests/webui/test_config_io.py
import os, textwrap, tempfile, pytest
from webui.backend.services.config_io import load_from_py, export_to_py
from webui.backend.schemas.config_schema import FullConfig

FIXTURE = "config/config.py"  # the existing config in the repo

def test_load_from_existing_config():
    if not os.path.exists(FIXTURE):
        pytest.skip("config/config.py not present")
    cfg = load_from_py(FIXTURE)
    assert isinstance(cfg, FullConfig)
    assert cfg.encoder.dim == 128
    assert cfg.encoder.tpv_h == 200
    assert cfg.encoder.num_encoder_layers == 5
    assert cfg.pif.enabled is True
    assert cfg.pif.factor == 0.125
    assert cfg.decoder.hidden_dim == 128
    assert cfg.optimizer.lr == pytest.approx(5e-5)

def test_export_roundtrip():
    """Export default config to .py, reload it, check key values survived."""
    cfg = FullConfig()
    py_str = export_to_py(cfg)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(py_str)
        tmp = f.name
    try:
        reloaded = load_from_py(tmp)
        assert reloaded.encoder.dim == cfg.encoder.dim
        assert reloaded.encoder.tpv_h == cfg.encoder.tpv_h
        assert reloaded.decoder.hidden_layers == cfg.decoder.hidden_layers
        assert reloaded.optimizer.lr == pytest.approx(cfg.optimizer.lr)
        assert reloaded.pif.enabled == cfg.pif.enabled
    finally:
        os.unlink(tmp)
```

**Step 2: Run tests — expect failures**

```bash
mkdir -p tests/webui && touch tests/webui/__init__.py
uv run pytest tests/webui/test_config_io.py -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError` for `config_io`.

**Step 3: Implement `config_io.py`**

```python
# webui/backend/services/config_io.py
import textwrap
from mmengine.config import Config as MMConfig
from webui.backend.schemas.config_schema import (
    FullConfig, EncoderConfig, DecoderConfig, OptimizerConfig,
    DatasetConfig, TrainLoaderConfig, ValLoaderConfig, PIFConfig,
)


def load_from_py(path: str) -> FullConfig:
    raw = MMConfig.fromfile(path)

    train_dl = raw.get("dataset_params", {}).get("train_data_loader", {})
    val_dl = raw.get("dataset_params", {}).get("val_data_loader", {})

    encoder = EncoderConfig(
        dim=raw.get("_dim_", 128),
        num_heads=raw.get("num_heads", 8),
        num_levels=raw.get("_num_levels_", 4),
        max_cams=raw.get("_max_cams_", 6),
        min_cams_train=raw.get("_min_cams_train_", 1),
        tpv_h=raw.get("N_h_", 200),
        tpv_w=raw.get("N_w_", 200),
        tpv_z=raw.get("N_z_", 16),
        num_encoder_layers=raw.get("tpv_encoder_layers", 5),
        scene_contraction=raw.get("scene_contraction", True),
        scene_contraction_factor=raw.get("scene_contraction_factor", [0.5, 0.1, 0.1]),
        offset=raw.get("offset", [-4.0, 0.0, 0.0]),
        scale=raw.get("scale", [0.25, 0.25, 0.25]),
        num_points_in_pillar=raw.get("num_points_in_pillar", [4, 32, 32]),
        num_points=raw.get("num_points", [8, 64, 64]),
        hybrid_attn_anchors=raw.get("hybrid_attn_anchors", 16),
        hybrid_attn_points=raw.get("hybrid_attn_points", 32),
    )

    dec_raw = raw.get("decoder", {})
    decoder = DecoderConfig(
        hidden_dim=dec_raw.get("hidden_dim", 128),
        hidden_layers=dec_raw.get("hidden_layers", 5),
        density_activation=dec_raw.get("density_activation", "trunc_exp"),
        nb_bins=dec_raw.get("nb_bins", 64),
        nb_bins_sample=dec_raw.get("nb_bins_sample", 64),
        hn=dec_raw.get("hn", 0.0),
        hf=dec_raw.get("hf", 60.0),
        train_stratified=dec_raw.get("train_stratified", True),
        white_background=dec_raw.get("white_background", False),
        whiteout=dec_raw.get("whiteout", False),
        testing_batch_size=dec_raw.get("testing_batch_size", 8192),
    )

    opt_raw = raw.get("optimizer", {})
    optimizer = OptimizerConfig(
        lr=opt_raw.get("lr", 5e-5),
        num_epochs=opt_raw.get("num_epochs", 100),
        num_warmup_steps=opt_raw.get("num_training_steps", 1000),
        lpips_loss_weight=opt_raw.get("lpips_loss_weight", 0.2),
        tv_loss_weight=opt_raw.get("tv_loss_weight", 0.0),
        dist_loss_weight=opt_raw.get("dist_loss_weight", 1e-3),
        depth_loss_weight=opt_raw.get("depth_loss_weight", 1.0),
        clip_grad_norm=opt_raw.get("clip_grad_norm", 1.5),
    )

    ds_raw = raw.get("dataset_params", {})
    dataset = DatasetConfig(
        data_path=ds_raw.get("data_path", "/app/data/"),
        train=TrainLoaderConfig(
            pickled=train_dl.get("pickled", True),
            batch_size=train_dl.get("batch_size", 1),
            shuffle=train_dl.get("shuffle", True),
            num_workers=train_dl.get("num_workers", 12),
            towns=train_dl.get("town", ["Town01"]),
            weather=train_dl.get("weather", ["ClearNoon"]),
            vehicle=train_dl.get("vehicle", ["vehicle.tesla.invisible"]),
            factor=train_dl.get("factor", 0.08),
            num_imgs=train_dl.get("num_imgs", 3),
            depth=train_dl.get("depth", True),
            min_cams_train=train_dl.get("min_cams_train", 1),
            max_cams_train=train_dl.get("max_cams_train", 6),
        ),
        val=ValLoaderConfig(
            batch_size=val_dl.get("batch_size", 1),
            num_workers=val_dl.get("num_workers", 12),
            towns=val_dl.get("town", ["Town02"]),
            weather=val_dl.get("weather", ["ClearNoon"]),
            vehicle=val_dl.get("vehicle", ["vehicle.tesla.invisible"]),
            spawn_point=val_dl.get("spawn_point", [3, 7, 12, 48, 98, 66]),
            factor=val_dl.get("factor", 0.25),
            depth=val_dl.get("depth", True),
        ),
    )

    pif = PIFConfig(
        enabled=raw.get("pif", False),
        factor=raw.get("pif_factor", 0.125),
        transforms_path=raw.get("pif_transforms", ""),
    )

    return FullConfig(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        dataset=dataset,
        pif=pif,
    )


def export_to_py(cfg: FullConfig) -> str:
    e = cfg.encoder
    d = cfg.decoder
    o = cfg.optimizer
    ds = cfg.dataset
    p = cfg.pif

    train = ds.train
    val = ds.val

    return textwrap.dedent(f"""\
        # Generated by 6Img-to-3D Config Builder
        _base_ = [
            "./_base_/dataset.py",
            "./_base_/optimizer.py",
            "./_base_/triplane_decoder.py",
        ]

        _dim_ = {e.dim}
        num_heads = {e.num_heads}
        _pos_dim_ = [48, 48, 32]
        _ffn_dim_ = _dim_ * 2
        _num_levels_ = {e.num_levels}
        _max_cams_ = {e.max_cams}
        _min_cams_train_ = {e.min_cams_train}

        N_h_ = {e.tpv_h}
        N_w_ = {e.tpv_w}
        N_z_ = {e.tpv_z}

        offset_h = {e.offset[1]}
        offset_w = {e.offset[2]}
        offset_z = {e.offset[0]}
        offset = [offset_z, offset_h, offset_w]
        scale_h = {e.scale[1]}
        scale_w = {e.scale[2]}
        scale_z = {e.scale[0]}
        scale = [scale_z, scale_h, scale_w]

        scene_contraction = {e.scene_contraction}
        scene_contraction_factor = {e.scene_contraction_factor}

        pif = {p.enabled}
        pif_factor = {p.factor}
        pif_transforms = "{p.transforms_path}"

        tpv_encoder_layers = {e.num_encoder_layers}
        num_points_in_pillar = {e.num_points_in_pillar}
        num_points = {e.num_points}
        hybrid_attn_anchors = {e.hybrid_attn_anchors}
        hybrid_attn_points = {e.hybrid_attn_points}
        hybrid_attn_init = 0

        self_cross_layer = dict(
            type="TPVFormerLayer",
            attn_cfgs=[
                dict(
                    type="TPVImageCrossAttention",
                    max_cams=_max_cams_,
                    deformable_attention=dict(
                        type="TPVMSDeformableAttention3D",
                        embed_dims=_dim_,
                        num_heads=num_heads,
                        num_points=num_points,
                        num_z_anchors=num_points_in_pillar,
                        num_levels=_num_levels_,
                        floor_sampling_offset=False,
                        tpv_h=N_h_,
                        tpv_w=N_w_,
                        tpv_z=N_z_,
                    ),
                    embed_dims=_dim_,
                    tpv_h=N_h_,
                    tpv_w=N_w_,
                    tpv_z=N_z_,
                ),
                dict(
                    type="TPVCrossViewHybridAttention",
                    tpv_h=N_h_,
                    tpv_w=N_w_,
                    tpv_z=N_z_,
                    num_anchors=hybrid_attn_anchors,
                    embed_dims=_dim_,
                    num_heads=num_heads,
                    num_points=hybrid_attn_points,
                    init_mode=hybrid_attn_init,
                ),
            ],
            feedforward_channels=_ffn_dim_,
            ffn_dropout=0.1,
            operation_order=("cross_attn", "norm", "self_attn", "norm", "ffn", "norm"),
        )

        self_layer = dict(
            type="TPVFormerLayer",
            attn_cfgs=[
                dict(
                    type="TPVCrossViewHybridAttention",
                    tpv_h=N_h_,
                    tpv_w=N_w_,
                    tpv_z=N_z_,
                    num_anchors=hybrid_attn_anchors,
                    embed_dims=_dim_,
                    num_heads=num_heads,
                    num_points=hybrid_attn_points,
                    init_mode=hybrid_attn_init,
                )
            ],
            feedforward_channels=_ffn_dim_,
            ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "ffn", "norm"),
        )

        _cross_layers = {e.num_encoder_layers - 2}
        _self_layers = 2

        model = dict(
            type="TPVFormer",
            output_features=True,
            img_backbone=dict(
                type="ResNet",
                depth=101,
                num_stages=4,
                out_indices=(1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type="BN2d", requires_grad=False),
                norm_eval=True,
                style="caffe",
                dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True),
            ),
            img_neck=dict(
                type="FPN",
                in_channels=[512, 1024, 2048],
                out_channels=_dim_,
                start_level=0,
                add_extra_convs="on_output",
                num_outs=4,
                relu_before_extra_convs=True,
            ),
            tpv_head=dict(
                type="TPVFormerHead",
                tpv_h=N_h_,
                tpv_w=N_w_,
                tpv_z=N_z_,
                num_feature_levels=_num_levels_,
                max_cams=_max_cams_,
                embed_dims=_dim_,
                encoder=dict(
                    type="TPVFormerEncoder",
                    tpv_h=N_h_,
                    tpv_w=N_w_,
                    tpv_z=N_z_,
                    offset=[offset_z, offset_h, offset_w],
                    scale=[scale_z, scale_h, scale_w],
                    intrin_factor=pif_factor,
                    scene_contraction=scene_contraction,
                    scene_contraction_factor=scene_contraction_factor,
                    num_layers=tpv_encoder_layers,
                    num_points_in_pillar=num_points_in_pillar,
                    num_points_in_pillar_cross_view=[16, 16, 16],
                    return_intermediate=False,
                    transformerlayers=[self_cross_layer] * _cross_layers + [self_layer] * _self_layers,
                ),
                positional_encoding=dict(
                    type="CustomPositionalEncoding",
                    num_feats=_pos_dim_,
                    h=N_h_,
                    w=N_w_,
                    z=N_z_,
                ),
            ),
        )

        dataset_params = dict(
            data_path="{ds.data_path}",
            version="triplane",
            train_data_loader=dict(
                pickled={train.pickled},
                phase="train",
                batch_size={train.batch_size},
                shuffle={train.shuffle},
                num_workers={train.num_workers},
                town={train.towns},
                weather={train.weather},
                vehicle={train.vehicle},
                selection=["input_images", "sphere_dataset"],
                factor={train.factor},
                whole_image=True,
                num_imgs={train.num_imgs},
                depth={train.depth},
                min_cams_train={train.min_cams_train},
                max_cams_train={train.max_cams_train},
            ),
            val_data_loader=dict(
                pickled=False,
                phase="test",
                batch_size={val.batch_size},
                shuffle=False,
                num_workers={val.num_workers},
                town={val.towns},
                weather={val.weather},
                vehicle={val.vehicle},
                spawn_point={val.spawn_point},
                step=["all"],
                selection=["input_images", "sphere_dataset"],
                factor={val.factor},
                depth={val.depth},
            ),
        )

        optimizer = dict(
            lr={o.lr},
            num_training_steps={o.num_warmup_steps},
            num_epochs={o.num_epochs},
            lpips_loss_weight={o.lpips_loss_weight},
            tv_loss_weight={o.tv_loss_weight},
            dist_loss_weight={o.dist_loss_weight},
            clip_grad_norm={o.clip_grad_norm},
            depth_loss_weight={o.depth_loss_weight},
        )

        decoder = dict(
            whiteout={d.whiteout},
            white_background={d.white_background},
            density_activation="{d.density_activation}",
            hidden_dim={d.hidden_dim},
            hidden_layers={d.hidden_layers},
            hn={d.hn},
            hf={d.hf},
            nb_bins={d.nb_bins},
            nb_bins_sample={d.nb_bins_sample},
            train_stratified={d.train_stratified},
            testing_batch_size={d.testing_batch_size},
        )
    """)
```

**Step 4: Run tests**

```bash
uv run pytest tests/webui/test_config_io.py -v
```

Expected: both tests PASS.

**Step 5: Commit**

```bash
git add webui/backend/services/config_io.py tests/webui/
git commit -m "feat(webui): add config_io load_from_py and export_to_py"
```

---

## Task 3: ConfigRecord DB model + migration

**Files:**
- Modify: `webui/backend/models.py` (add `ConfigRecord`)
- Modify: `webui/backend/database.py` (ensure `create_all` creates the new table)

**Step 1: Add `ConfigRecord` to `models.py`**

Open `webui/backend/models.py`. After the existing `JobRecord` class, add:

```python
class ConfigRecord(Base):
    __tablename__ = "config_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255))
    data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)
```

**Step 2: Ensure table is created on startup**

Find where `Base.metadata.create_all(engine)` is called (it's in the FastAPI app startup or `database.py`). Verify `ConfigRecord` is imported there. Check the main app entrypoint:

```bash
grep -r "create_all" webui/
```

If it's in an `app.py` or similar, add the import `from webui.backend.models import ConfigRecord` (or confirm all models are imported via the existing wildcard).

**Step 3: Verify table creation**

```bash
# Delete the dev DB to start fresh (safe — it's only dev data)
rm -f webui.db
uv run python -c "
from webui.backend.database import engine, Base
from webui.backend.models import ConfigRecord
Base.metadata.create_all(engine)
import sqlite3
conn = sqlite3.connect('webui.db')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
print([t[0] for t in tables])
conn.close()
"
```

Expected: output includes `config_records`.

**Step 4: Commit**

```bash
git add webui/backend/models.py
git commit -m "feat(webui): add ConfigRecord SQLAlchemy model"
```

---

## Task 4: API routes `/api/configs`

**Files:**
- Create: `webui/backend/api/configs.py`
- Modify: `webui/backend/api/__init__.py` (register new router)

**Step 1: Find how existing routers are registered**

```bash
grep -r "include_router" webui/
```

Look at how `recipes.py` router is included to follow the same pattern.

**Step 2: Create `configs.py`**

```python
# webui/backend/api/configs.py
import tempfile, os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from webui.backend.database import get_db
from webui.backend.models import ConfigRecord
from webui.backend.schemas.config_schema import FullConfig
from webui.backend.services.config_io import load_from_py, export_to_py

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


@router.get("/{config_id}")
def get_config(config_id: str, db: Session = Depends(get_db)):
    r = db.get(ConfigRecord, config_id)
    if not r:
        raise HTTPException(status_code=404, detail="Config not found")
    return {"id": r.id, "name": r.name, "data": r.data, "created_at": str(r.created_at)}


@router.put("/{config_id}")
def update_config(config_id: str, cfg: FullConfig, name: str | None = None, db: Session = Depends(get_db)):
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


@router.post("/import", status_code=201)
async def import_config(name: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a config.py file and parse it into a FullConfig."""
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


@router.get("/{config_id}/export")
def export_config(config_id: str, db: Session = Depends(get_db)):
    """Return the config as a downloadable config.py string."""
    r = db.get(ConfigRecord, config_id)
    if not r:
        raise HTTPException(status_code=404, detail="Config not found")
    cfg = FullConfig(**r.data)
    py_str = export_to_py(cfg)
    return PlainTextResponse(
        content=py_str,
        headers={"Content-Disposition": f'attachment; filename="{r.name}.py"'},
    )
```

**Step 3: Register the router**

Find the FastAPI app file (check `webui/backend/api/__init__.py` or wherever `app = FastAPI()` is defined):

```bash
grep -r "FastAPI\|app.include_router\|APIRouter" webui/ --include="*.py" -l
```

Add to that file:
```python
from webui.backend.api.configs import router as configs_router
app.include_router(configs_router)
```

**Step 4: Start the backend and smoke-test**

```bash
# In one terminal:
uv run uvicorn webui.backend.main:app --port 8001 --reload &
sleep 2

# Create a default config:
curl -s -X POST "http://localhost:8001/api/configs?name=default" \
  -H "Content-Type: application/json" \
  -d '{"encoder":{},"decoder":{},"optimizer":{},"dataset":{},"pif":{}}' | python3 -m json.tool

# List configs:
curl -s http://localhost:8001/api/configs | python3 -m json.tool
```

Expected: config created with an `id`, appears in list.

**Step 5: Test export**

```bash
CONFIG_ID=$(curl -s http://localhost:8001/api/configs | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")
curl -s "http://localhost:8001/api/configs/$CONFIG_ID/export"
```

Expected: valid Python config file printed to stdout.

**Step 6: Stop background server, commit**

```bash
kill %1 2>/dev/null || true
git add webui/backend/api/configs.py webui/backend/api/__init__.py
git commit -m "feat(webui): add /api/configs CRUD, import, export endpoints"
```

---

## Task 5: Frontend — API client functions + TypeScript types

**Files:**
- Modify: `webui/frontend/src/api.ts`

**Step 1: Add types and functions to `api.ts`**

Append to `webui/frontend/src/api.ts`:

```typescript
// ── Config Builder ─────────────────────────────────────────────

export interface PIFConfig {
  enabled: boolean
  factor: number
  transforms_path: string
}

export interface EncoderConfig {
  dim: number
  num_heads: number
  num_levels: number
  max_cams: number
  min_cams_train: number
  tpv_h: number
  tpv_w: number
  tpv_z: number
  num_encoder_layers: number
  scene_contraction: boolean
  scene_contraction_factor: number[]
  offset: number[]
  scale: number[]
  num_points_in_pillar: number[]
  num_points: number[]
  hybrid_attn_anchors: number
  hybrid_attn_points: number
}

export interface DecoderConfig {
  hidden_dim: number
  hidden_layers: number
  density_activation: string
  nb_bins: number
  nb_bins_sample: number
  hn: number
  hf: number
  train_stratified: boolean
  white_background: boolean
  whiteout: boolean
  testing_batch_size: number
}

export interface OptimizerConfig {
  lr: number
  num_epochs: number
  num_warmup_steps: number
  lpips_loss_weight: number
  tv_loss_weight: number
  dist_loss_weight: number
  depth_loss_weight: number
  clip_grad_norm: number
}

export interface TrainLoaderConfig {
  pickled: boolean
  batch_size: number
  shuffle: boolean
  num_workers: number
  towns: string[]
  weather: string[]
  vehicle: string[]
  factor: number
  num_imgs: number
  depth: boolean
  min_cams_train: number
  max_cams_train: number
}

export interface ValLoaderConfig {
  batch_size: number
  num_workers: number
  towns: string[]
  weather: string[]
  vehicle: string[]
  spawn_point: number[]
  factor: number
  depth: boolean
}

export interface DatasetConfig {
  data_path: string
  train: TrainLoaderConfig
  val: ValLoaderConfig
}

export interface FullConfig {
  encoder: EncoderConfig
  decoder: DecoderConfig
  optimizer: OptimizerConfig
  dataset: DatasetConfig
  pif: PIFConfig
}

export interface ConfigRecord {
  id: string
  name: string
  created_at: string
  data?: FullConfig
}

export async function listConfigs(): Promise<ConfigRecord[]> {
  const res = await api.get('/api/configs')
  return res.data
}

export async function getConfig(id: string): Promise<ConfigRecord & { data: FullConfig }> {
  const res = await api.get(`/api/configs/${id}`)
  return res.data
}

export async function createConfig(name: string, data: FullConfig): Promise<ConfigRecord> {
  const res = await api.post(`/api/configs?name=${encodeURIComponent(name)}`, data)
  return res.data
}

export async function updateConfig(id: string, name: string, data: FullConfig): Promise<ConfigRecord> {
  const res = await api.put(`/api/configs/${id}?name=${encodeURIComponent(name)}`, data)
  return res.data
}

export async function deleteConfig(id: string): Promise<void> {
  await api.delete(`/api/configs/${id}`)
}

export async function cloneConfig(id: string, newName: string): Promise<ConfigRecord> {
  const res = await api.post(`/api/configs/${id}/clone?new_name=${encodeURIComponent(newName)}`)
  return res.data
}

export async function importConfig(name: string, file: File): Promise<ConfigRecord & { data: FullConfig }> {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post(`/api/configs/import?name=${encodeURIComponent(name)}`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export function configExportUrl(id: string): string {
  return `http://localhost:8001/api/configs/${id}/export`
}
```

**Step 2: Check TypeScript compiles**

```bash
cd webui/frontend && npx tsc --noEmit 2>&1 | head -30
```

Expected: no errors.

**Step 3: Commit**

```bash
cd ../..
git add webui/frontend/src/api.ts
git commit -m "feat(webui): add config builder TypeScript API client"
```

---

## Task 6: Frontend — Config Builder page

**Files:**
- Create: `webui/frontend/src/pages/ConfigBuilder.tsx`
- Modify: `webui/frontend/src/App.tsx` (add nav link + route)

**Overview:** Two-panel layout. Left: config list sidebar + form. Right: live `config.py` preview. Color-coded collapsible sections. Font: JetBrains Mono for code, DM Sans for headers (loaded via Google Fonts in `index.html`).

**Step 1: Add fonts to `index.html`**

In `webui/frontend/index.html`, add inside `<head>`:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

**Step 2: Create `ConfigBuilder.tsx`**

This is the main page component. Create `webui/frontend/src/pages/ConfigBuilder.tsx`:

```tsx
import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  listConfigs, getConfig, createConfig, updateConfig, deleteConfig,
  cloneConfig, importConfig, configExportUrl,
  type FullConfig, type ConfigRecord,
} from '../api'

// ── Default config matching backend defaults ─────────────────────────────────

const DEFAULT_CONFIG: FullConfig = {
  encoder: {
    dim: 128, num_heads: 8, num_levels: 4, max_cams: 6, min_cams_train: 1,
    tpv_h: 200, tpv_w: 200, tpv_z: 16, num_encoder_layers: 5,
    scene_contraction: true, scene_contraction_factor: [0.5, 0.1, 0.1],
    offset: [-4.0, 0.0, 0.0], scale: [0.25, 0.25, 0.25],
    num_points_in_pillar: [4, 32, 32], num_points: [8, 64, 64],
    hybrid_attn_anchors: 16, hybrid_attn_points: 32,
  },
  decoder: {
    hidden_dim: 128, hidden_layers: 5, density_activation: 'trunc_exp',
    nb_bins: 64, nb_bins_sample: 64, hn: 0, hf: 60,
    train_stratified: true, white_background: false, whiteout: false,
    testing_batch_size: 8192,
  },
  optimizer: {
    lr: 5e-5, num_epochs: 100, num_warmup_steps: 1000,
    lpips_loss_weight: 0.2, tv_loss_weight: 0.0,
    dist_loss_weight: 1e-3, depth_loss_weight: 1.0, clip_grad_norm: 1.5,
  },
  dataset: {
    data_path: '/app/data/',
    train: {
      pickled: true, batch_size: 1, shuffle: true, num_workers: 12,
      towns: ['Town01', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD'],
      weather: ['ClearNoon'], vehicle: ['vehicle.tesla.invisible'],
      factor: 0.08, num_imgs: 3, depth: true, min_cams_train: 1, max_cams_train: 6,
    },
    val: {
      batch_size: 1, num_workers: 12, towns: ['Town02'],
      weather: ['ClearNoon'], vehicle: ['vehicle.tesla.invisible'],
      spawn_point: [3, 7, 12, 48, 98, 66], factor: 0.25, depth: true,
    },
  },
  pif: { enabled: false, factor: 0.125, transforms_path: '' },
}

// ── Config.py preview generator (mirrors backend export_to_py logic) ─────────

function generatePreview(cfg: FullConfig): string {
  const e = cfg.encoder
  const d = cfg.decoder
  const o = cfg.optimizer
  const ds = cfg.dataset
  const p = cfg.pif
  return `# Generated by 6Img-to-3D Config Builder
_base_ = [
    "./_base_/dataset.py",
    "./_base_/optimizer.py",
    "./_base_/triplane_decoder.py",
]

_dim_ = ${e.dim}
num_heads = ${e.num_heads}
_num_levels_ = ${e.num_levels}
_max_cams_ = ${e.max_cams}

N_h_ = ${e.tpv_h}
N_w_ = ${e.tpv_w}
N_z_ = ${e.tpv_z}

offset = [${e.offset.join(', ')}]
scale = [${e.scale.join(', ')}]

scene_contraction = ${e.scene_contraction}
scene_contraction_factor = [${e.scene_contraction_factor.join(', ')}]

pif = ${e.toString ? p.enabled : p.enabled}
pif_factor = ${p.factor}
pif_transforms = "${p.transforms_path}"

tpv_encoder_layers = ${e.num_encoder_layers}
num_points_in_pillar = [${e.num_points_in_pillar.join(', ')}]
num_points = [${e.num_points.join(', ')}]
hybrid_attn_anchors = ${e.hybrid_attn_anchors}
hybrid_attn_points = ${e.hybrid_attn_points}

decoder = dict(
    hidden_dim=${d.hidden_dim},
    hidden_layers=${d.hidden_layers},
    density_activation="${d.density_activation}",
    nb_bins=${d.nb_bins},
    hn=${d.hn},
    hf=${d.hf},
    testing_batch_size=${d.testing_batch_size},
)

optimizer = dict(
    lr=${o.lr},
    num_epochs=${o.num_epochs},
    lpips_loss_weight=${o.lpips_loss_weight},
    tv_loss_weight=${o.tv_loss_weight},
    dist_loss_weight=${o.dist_loss_weight},
    depth_loss_weight=${o.depth_loss_weight},
    clip_grad_norm=${o.clip_grad_norm},
)

dataset_params = dict(
    data_path="${ds.data_path}",
    train_data_loader=dict(
        towns=${JSON.stringify(ds.train.towns)},
        weather=${JSON.stringify(ds.train.weather)},
        factor=${ds.train.factor},
        num_imgs=${ds.train.num_imgs},
    ),
    val_data_loader=dict(
        towns=${JSON.stringify(ds.val.towns)},
        spawn_point=${JSON.stringify(ds.val.spawn_point)},
        factor=${ds.val.factor},
    ),
)
`
}

// ── Small UI primitives ───────────────────────────────────────────────────────

function Section({
  title, accent, children, defaultOpen = true,
}: {
  title: string; accent: string; children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`border-l-2 ${accent} mb-4`}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-white/5 transition-colors"
      >
        <span className="text-xs font-semibold uppercase tracking-widest text-gray-300" style={{ fontFamily: 'DM Sans, sans-serif' }}>
          {title}
        </span>
        <span className="ml-auto text-gray-600 text-xs">{open ? '▲' : '▼'}</span>
      </button>
      {open && <div className="px-3 pb-3 grid grid-cols-2 gap-x-4 gap-y-2">{children}</div>}
    </div>
  )
}

function Field({ label, children, wide }: { label: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className={wide ? 'col-span-2' : ''}>
      <label className="block text-xs text-gray-500 mb-0.5">{label}</label>
      {children}
    </div>
  )
}

function NumInput({ value, onChange, step, min }: { value: number; onChange: (v: number) => void; step?: number; min?: number }) {
  return (
    <input
      type="number" value={value} step={step ?? 'any'} min={min}
      onChange={e => onChange(Number(e.target.value))}
      className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs font-mono text-gray-200 focus:border-gray-500 outline-none"
      style={{ fontFamily: 'JetBrains Mono, monospace' }}
    />
  )
}

function TextInput({ value, onChange, placeholder }: { value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <input
      type="text" value={value} placeholder={placeholder}
      onChange={e => onChange(e.target.value)}
      className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs font-mono text-gray-200 focus:border-gray-500 outline-none"
      style={{ fontFamily: 'JetBrains Mono, monospace' }}
    />
  )
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className={`relative w-9 h-5 rounded-full transition-colors ${value ? 'bg-blue-600' : 'bg-gray-700'}`}
    >
      <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${value ? 'translate-x-4' : ''}`} />
    </button>
  )
}

function TagInput({ values, onChange }: { values: string[] | number[]; onChange: (v: any[]) => void }) {
  const [draft, setDraft] = useState('')
  const isNum = typeof values[0] === 'number' || values.length === 0

  function add() {
    if (!draft.trim()) return
    const val = isNum ? Number(draft.trim()) : draft.trim()
    onChange([...values, val])
    setDraft('')
  }

  return (
    <div className="flex flex-wrap gap-1 bg-gray-900 border border-gray-700 rounded p-1.5 min-h-[30px]">
      {values.map((v, i) => (
        <span key={i} className="flex items-center gap-1 bg-gray-800 rounded px-1.5 py-0.5 text-xs font-mono text-gray-300">
          {String(v)}
          <button onClick={() => onChange(values.filter((_, j) => j !== i))} className="text-gray-600 hover:text-red-400 leading-none">×</button>
        </span>
      ))}
      <input
        value={draft} onChange={e => setDraft(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); add() } }}
        placeholder="add…"
        className="bg-transparent outline-none text-xs text-gray-300 min-w-[40px] flex-1"
        style={{ fontFamily: 'JetBrains Mono, monospace' }}
      />
    </div>
  )
}

// ── Config form ───────────────────────────────────────────────────────────────

function ConfigForm({ cfg, onChange }: { cfg: FullConfig; onChange: (c: FullConfig) => void }) {
  function setEncoder(key: string, val: any) {
    onChange({ ...cfg, encoder: { ...cfg.encoder, [key]: val } })
  }
  function setDecoder(key: string, val: any) {
    onChange({ ...cfg, decoder: { ...cfg.decoder, [key]: val } })
  }
  function setOptimizer(key: string, val: any) {
    onChange({ ...cfg, optimizer: { ...cfg.optimizer, [key]: val } })
  }
  function setDataset(key: string, val: any) {
    onChange({ ...cfg, dataset: { ...cfg.dataset, [key]: val } })
  }
  function setTrain(key: string, val: any) {
    onChange({ ...cfg, dataset: { ...cfg.dataset, train: { ...cfg.dataset.train, [key]: val } } })
  }
  function setVal(key: string, val: any) {
    onChange({ ...cfg, dataset: { ...cfg.dataset, val: { ...cfg.dataset.val, [key]: val } } })
  }
  function setPif(key: string, val: any) {
    onChange({ ...cfg, pif: { ...cfg.pif, [key]: val } })
  }

  return (
    <div className="overflow-y-auto flex-1 pr-1">

      <Section title="Dataset" accent="border-cyan-500">
        <Field label="data_path" wide>
          <TextInput value={cfg.dataset.data_path} onChange={v => setDataset('data_path', v)} placeholder="/app/data/" />
        </Field>
        <Field label="train towns" wide>
          <TagInput values={cfg.dataset.train.towns} onChange={v => setTrain('towns', v)} />
        </Field>
        <Field label="train factor">
          <NumInput value={cfg.dataset.train.factor} onChange={v => setTrain('factor', v)} step={0.01} />
        </Field>
        <Field label="train num_imgs">
          <NumInput value={cfg.dataset.train.num_imgs} onChange={v => setTrain('num_imgs', v)} step={1} min={1} />
        </Field>
        <Field label="val towns" wide>
          <TagInput values={cfg.dataset.val.towns} onChange={v => setVal('towns', v)} />
        </Field>
        <Field label="val spawn_points" wide>
          <TagInput values={cfg.dataset.val.spawn_point} onChange={v => setVal('spawn_point', v)} />
        </Field>
        <Field label="val factor">
          <NumInput value={cfg.dataset.val.factor} onChange={v => setVal('factor', v)} step={0.01} />
        </Field>
        <Field label="train depth">
          <Toggle value={cfg.dataset.train.depth} onChange={v => setTrain('depth', v)} />
        </Field>
      </Section>

      <Section title="Encoder" accent="border-violet-500">
        <Field label="dim (features)">
          <NumInput value={cfg.encoder.dim} onChange={v => setEncoder('dim', v)} step={1} min={1} />
        </Field>
        <Field label="num_heads">
          <NumInput value={cfg.encoder.num_heads} onChange={v => setEncoder('num_heads', v)} step={1} min={1} />
        </Field>
        <Field label="tpv_h">
          <NumInput value={cfg.encoder.tpv_h} onChange={v => setEncoder('tpv_h', v)} step={1} min={1} />
        </Field>
        <Field label="tpv_w">
          <NumInput value={cfg.encoder.tpv_w} onChange={v => setEncoder('tpv_w', v)} step={1} min={1} />
        </Field>
        <Field label="tpv_z">
          <NumInput value={cfg.encoder.tpv_z} onChange={v => setEncoder('tpv_z', v)} step={1} min={1} />
        </Field>
        <Field label="encoder layers">
          <NumInput value={cfg.encoder.num_encoder_layers} onChange={v => setEncoder('num_encoder_layers', v)} step={1} min={2} />
        </Field>
        <Field label="scene_contraction">
          <Toggle value={cfg.encoder.scene_contraction} onChange={v => setEncoder('scene_contraction', v)} />
        </Field>
        <Field label="contraction_factor" wide>
          <TagInput values={cfg.encoder.scene_contraction_factor} onChange={v => setEncoder('scene_contraction_factor', v)} />
        </Field>
        <Field label="offset [z,h,w]" wide>
          <TagInput values={cfg.encoder.offset} onChange={v => setEncoder('offset', v)} />
        </Field>
        <Field label="scale [z,h,w]" wide>
          <TagInput values={cfg.encoder.scale} onChange={v => setEncoder('scale', v)} />
        </Field>
      </Section>

      <Section title="Decoder" accent="border-emerald-500">
        <Field label="hidden_dim">
          <NumInput value={cfg.decoder.hidden_dim} onChange={v => setDecoder('hidden_dim', v)} step={1} />
        </Field>
        <Field label="hidden_layers">
          <NumInput value={cfg.decoder.hidden_layers} onChange={v => setDecoder('hidden_layers', v)} step={1} min={1} />
        </Field>
        <Field label="nb_bins">
          <NumInput value={cfg.decoder.nb_bins} onChange={v => setDecoder('nb_bins', v)} step={1} min={1} />
        </Field>
        <Field label="nb_bins_sample">
          <NumInput value={cfg.decoder.nb_bins_sample} onChange={v => setDecoder('nb_bins_sample', v)} step={1} min={1} />
        </Field>
        <Field label="near (hn)">
          <NumInput value={cfg.decoder.hn} onChange={v => setDecoder('hn', v)} step={0.1} />
        </Field>
        <Field label="far (hf)">
          <NumInput value={cfg.decoder.hf} onChange={v => setDecoder('hf', v)} step={1} />
        </Field>
        <Field label="density_activation" wide>
          <TextInput value={cfg.decoder.density_activation} onChange={v => setDecoder('density_activation', v)} />
        </Field>
        <Field label="white_background">
          <Toggle value={cfg.decoder.white_background} onChange={v => setDecoder('white_background', v)} />
        </Field>
        <Field label="testing_batch_size">
          <NumInput value={cfg.decoder.testing_batch_size} onChange={v => setDecoder('testing_batch_size', v)} step={512} min={64} />
        </Field>
      </Section>

      <Section title="Optimizer" accent="border-amber-500">
        <Field label="lr">
          <NumInput value={cfg.optimizer.lr} onChange={v => setOptimizer('lr', v)} step={1e-6} />
        </Field>
        <Field label="num_epochs">
          <NumInput value={cfg.optimizer.num_epochs} onChange={v => setOptimizer('num_epochs', v)} step={1} min={1} />
        </Field>
        <Field label="warmup_steps">
          <NumInput value={cfg.optimizer.num_warmup_steps} onChange={v => setOptimizer('num_warmup_steps', v)} step={100} min={0} />
        </Field>
        <Field label="clip_grad_norm">
          <NumInput value={cfg.optimizer.clip_grad_norm} onChange={v => setOptimizer('clip_grad_norm', v)} step={0.1} />
        </Field>
        <Field label="lpips_weight">
          <NumInput value={cfg.optimizer.lpips_loss_weight} onChange={v => setOptimizer('lpips_loss_weight', v)} step={0.01} />
        </Field>
        <Field label="tv_weight">
          <NumInput value={cfg.optimizer.tv_loss_weight} onChange={v => setOptimizer('tv_loss_weight', v)} step={0.001} />
        </Field>
        <Field label="dist_weight">
          <NumInput value={cfg.optimizer.dist_loss_weight} onChange={v => setOptimizer('dist_loss_weight', v)} step={0.0001} />
        </Field>
        <Field label="depth_weight">
          <NumInput value={cfg.optimizer.depth_loss_weight} onChange={v => setOptimizer('depth_loss_weight', v)} step={0.1} />
        </Field>
      </Section>

      <Section title="PIF" accent="border-rose-500">
        <Field label="enabled">
          <Toggle value={cfg.pif.enabled} onChange={v => setPif('enabled', v)} />
        </Field>
        <Field label="factor">
          <NumInput value={cfg.pif.factor} onChange={v => setPif('factor', v)} step={0.001} />
        </Field>
        <Field label="transforms_path" wide>
          <TextInput value={cfg.pif.transforms_path} onChange={v => setPif('transforms_path', v)} placeholder="/app/data/Town02/.../nuscenes/" />
        </Field>
      </Section>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ConfigBuilder() {
  const qc = useQueryClient()
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [cfg, setCfg] = useState<FullConfig>(DEFAULT_CONFIG)
  const [name, setName] = useState('untitled')
  const [importName, setImportName] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)

  const { data: configs = [] } = useQuery({ queryKey: ['configs'], queryFn: listConfigs })

  // Load config when selecting from list
  const { data: selected } = useQuery({
    queryKey: ['config', selectedId],
    queryFn: () => getConfig(selectedId!),
    enabled: !!selectedId,
  })

  // Sync editor when selection changes
  const prevId = useState<string | null>(null)[0]
  if (selected && selected.id !== prevId && selected.data) {
    setCfg(selected.data)
    setName(selected.name)
  }

  const deleteMutation = useMutation({
    mutationFn: deleteConfig,
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['configs'] }); setSelectedId(null) },
  })

  const cloneMutation = useMutation({
    mutationFn: ({ id, n }: { id: string; n: string }) => cloneConfig(id, n),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['configs'] }),
  })

  const preview = useMemo(() => generatePreview(cfg), [cfg])

  async function handleSave() {
    setSaving(true); setSaveError(null)
    try {
      if (selectedId) {
        await updateConfig(selectedId, name, cfg)
      } else {
        const r = await createConfig(name, cfg)
        setSelectedId(r.id)
      }
      qc.invalidateQueries({ queryKey: ['configs'] })
    } catch (e: unknown) {
      setSaveError(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  async function handleImport(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const n = importName || file.name.replace('.py', '')
    try {
      const r = await importConfig(n, file)
      qc.invalidateQueries({ queryKey: ['configs'] })
      setSelectedId(r.id)
      if (r.data) { setCfg(r.data); setName(r.name) }
    } catch {
      setSaveError('Import failed — check that the file is a valid config.py')
    }
  }

  return (
    <div className="flex gap-0 h-[calc(100vh-5rem)]" style={{ fontFamily: 'DM Sans, sans-serif' }}>

      {/* ── Left sidebar: config list ── */}
      <div className="w-52 shrink-0 border-r border-gray-800 flex flex-col">
        <div className="p-3 border-b border-gray-800">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase tracking-widest text-gray-400">Configs</span>
            <button
              onClick={() => { setSelectedId(null); setCfg(DEFAULT_CONFIG); setName('untitled') }}
              className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded px-2 py-0.5"
            >
              + New
            </button>
          </div>
          {/* Import */}
          <div className="flex gap-1">
            <input
              value={importName} onChange={e => setImportName(e.target.value)}
              placeholder="name"
              className="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-300 outline-none min-w-0"
            />
            <label className="cursor-pointer text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded px-2 py-0.5 whitespace-nowrap">
              Import
              <input type="file" accept=".py" onChange={handleImport} className="hidden" />
            </label>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {(configs as ConfigRecord[]).map(r => (
            <button
              key={r.id}
              onClick={() => setSelectedId(r.id)}
              className={`w-full text-left rounded px-2 py-1.5 mb-0.5 text-xs transition-colors group ${selectedId === r.id ? 'bg-gray-800 text-gray-100' : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'}`}
            >
              <div className="font-medium truncate">{r.name}</div>
              <div className="flex gap-1 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={e => { e.stopPropagation(); cloneMutation.mutate({ id: r.id, n: r.name + '-copy' }) }}
                  className="text-gray-600 hover:text-gray-300 text-xs"
                >clone</button>
                <button
                  onClick={e => { e.stopPropagation(); deleteMutation.mutate(r.id) }}
                  className="text-gray-600 hover:text-red-400 text-xs"
                >del</button>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* ── Center: form ── */}
      <div className="w-96 shrink-0 border-r border-gray-800 flex flex-col">
        <div className="p-3 border-b border-gray-800 flex items-center gap-2">
          <input
            value={name} onChange={e => setName(e.target.value)}
            className="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100 outline-none focus:border-gray-500"
          />
          <button
            onClick={handleSave} disabled={saving}
            className="text-xs bg-blue-700 hover:bg-blue-600 text-white rounded px-3 py-1 disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
          {selectedId && (
            <a
              href={configExportUrl(selectedId)}
              download
              className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded px-2 py-1"
            >
              Export
            </a>
          )}
        </div>
        {saveError && <p className="text-red-400 text-xs px-3 pt-1">{saveError}</p>}
        <ConfigForm cfg={cfg} onChange={setCfg} />
      </div>

      {/* ── Right: live preview ── */}
      <div className="flex-1 flex flex-col bg-gray-950 min-w-0">
        <div className="px-4 py-2 border-b border-gray-800 flex items-center gap-2">
          <span className="text-xs text-gray-500 uppercase tracking-widest font-semibold">config.py preview</span>
          <span className="ml-auto text-xs text-gray-700">live</span>
        </div>
        <pre
          className="flex-1 overflow-auto p-4 text-xs leading-relaxed text-gray-300"
          style={{ fontFamily: 'JetBrains Mono, monospace' }}
        >
          {preview}
        </pre>
      </div>
    </div>
  )
}
```

**Step 3: Register in `App.tsx`**

In `webui/frontend/src/App.tsx`:

1. Add import at top: `import ConfigBuilder from './pages/ConfigBuilder'`
2. Add nav link: `{ to: '/configs', label: 'Config Builder' }`
3. Add route: `<Route path="/configs" element={<ConfigBuilder />} />`

**Step 4: Run the dev server and verify**

```bash
cd webui/frontend && npm run dev
```

Open `http://localhost:5174/configs`. Verify:
- Config list sidebar renders (empty or with any imported configs)
- Form sections render with correct fields and accent colors
- Typing in any field updates the preview pane in real time
- "New" button clears the form
- "Save" button creates a config (check network tab for 201)
- "Export" link triggers a download

**Step 5: Commit**

```bash
cd ../..
git add webui/frontend/src/pages/ConfigBuilder.tsx webui/frontend/src/App.tsx webui/frontend/index.html
git commit -m "feat(webui): add Config Builder page with live preview"
```

---

## Task 7: Training tab integration — "Train with this config"

**Files:**
- Modify: `webui/frontend/src/pages/Training.tsx`
- Modify: `webui/frontend/src/api.ts` (add `exportConfigToFile` helper)

**Goal:** In the "New Run" dialog, add a dropdown to pick a saved config. When selected, the config ID is sent to the backend, which exports `config.py` to a temp path and returns that path for `train.py`.

**Step 1: Add backend endpoint to write config to disk**

In `webui/backend/api/configs.py`, add:

```python
@router.post("/{config_id}/write")
def write_config_to_disk(config_id: str, db: Session = Depends(get_db)):
    """Write exported config.py to a temp file, return its path for train.py."""
    import tempfile
    r = db.get(ConfigRecord, config_id)
    if not r:
        raise HTTPException(status_code=404, detail="Config not found")
    cfg = FullConfig(**r.data)
    py_str = export_to_py(cfg)
    # Write to a stable path under runs/configs/ so it persists for the job
    out_dir = Path("runs/configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{r.id}.py"
    out_path.write_text(py_str)
    return {"path": str(out_path)}
```

Add `from pathlib import Path` at top of `configs.py` if not already imported.

**Step 2: Add API client function**

In `webui/frontend/src/api.ts`, add:

```typescript
export async function writeConfigToDisk(id: string): Promise<{ path: string }> {
  const res = await api.post(`/api/configs/${id}/write`)
  return res.data
}
```

**Step 3: Update `NewRunDialog` in `Training.tsx`**

Add to the imports at top of `Training.tsx`:
```typescript
import { listConfigs, writeConfigToDisk, type ConfigRecord } from '../api'
```

Inside `NewRunDialog`, add a config selector. After the existing recipe selector `<div>`, add:

```tsx
<div>
  <label className="text-xs text-gray-400 block mb-1">From config builder (optional)</label>
  <select
    defaultValue=""
    onChange={async (e) => {
      if (!e.target.value) return
      try {
        const { path } = await writeConfigToDisk(e.target.value)
        setPyConfig(path)
      } catch {
        setError('Failed to export config')
      }
    }}
    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
  >
    <option value="">— use path above —</option>
    {savedConfigs.map((c: ConfigRecord) => (
      <option key={c.id} value={c.id}>{c.name}</option>
    ))}
  </select>
</div>
```

Add the query for saved configs inside `NewRunDialog`:
```typescript
const { data: savedConfigs = [] } = useQuery({ queryKey: ['configs'], queryFn: listConfigs })
```

**Step 4: Verify integration**

1. Create a config in the Config Builder tab
2. Go to Training → New Run
3. Select the config from the dropdown — the config file path should auto-fill
4. Check that a file was written to `runs/configs/<id>.py`

**Step 5: Commit**

```bash
git add webui/backend/api/configs.py webui/frontend/src/pages/Training.tsx webui/frontend/src/api.ts
git commit -m "feat(webui): integrate Config Builder into Training tab new run dialog"
```

---

## Done

After all tasks, the flow is:

1. User opens **Config Builder** tab
2. Creates or imports a config, tweaks params, sees live `config.py` preview
3. Clicks **Save** → stored in DB
4. Goes to **Training** tab → **New Run** → selects config from dropdown
5. Backend writes `config.py` to disk → `train.py` runs with it
