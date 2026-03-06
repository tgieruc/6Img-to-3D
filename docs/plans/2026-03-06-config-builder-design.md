# Config Builder — Design Document

**Date**: 2026-03-06
**Status**: Approved
**Scope**: New "Config" tab in the existing 6Img-to-3D webui

---

## Problem

The current configuration system has several pain points:

1. **Hardcoded absolute paths** — `pif_transforms = "/app/data/Town02/..."` is baked into `config.py`, tied to a specific Docker container layout.
2. **Two incompatible config styles** — the encoder uses MMDetection-style nested `dict(type=...)` registries; the decoder uses a flat dict consumed directly by `TriplaneDecoder(cfg)`. No unified contract.
3. **No validation** — typos in any config key silently become `None` at runtime.
4. **PIF setup is 20+ lines of imperative code in `train.py`** driven by a boolean from config — the boundary between config and runtime is blurred.
5. **`render_rays()` takes the entire `cfg` object** — unclear what it actually needs.
6. **Magic numbers in training loop** — image shape `(48, 64)`, `img_index == 19`, etc.

---

## Goals

- Structured, validated config editing via a web UI
- Import existing `config.py` files (backward compatible)
- Export valid `config.py` files that `train.py` accepts unchanged
- Configs stored persistently in the webui database
- Live preview of the generated `config.py` while editing

---

## Non-Goals

- Changing `train.py` or `eval.py` (the webui generates a config.py, not a new entrypoint)
- Replacing the MMDetection registry pattern in the encoder
- Supporting configs from other projects or datasets

---

## Architecture

```
Frontend (new "Config" tab)
  ├── Config list sidebar (saved configs, clone, delete)
  ├── Config editor (sectioned form with live config.py preview)
  └── Import / Export controls

Backend (/api/configs)
  ├── ConfigRecord (SQLite) — stores FullConfig as JSON
  ├── Pydantic schema — validates all fields
  └── config_io.py
        ├── load_from_py(path) → FullConfig   [import]
        └── export_to_py(FullConfig) → str    [export]
```

**Data flow:**

| Action | Flow |
|--------|------|
| Create | Form → `POST /api/configs` → stored as JSON |
| Import | Upload `config.py` → `mmengine.Config.fromfile()` → map to `FullConfig` → stored |
| Export | `GET /api/configs/{id}/export` → `export_to_py()` → downloadable `.py` |
| Use in training | Training tab picks a config → backend writes exported `config.py` to disk → runs `train.py --py-config <path>` |

`train.py` is not modified. The webui produces a `config.py` indistinguishable from a hand-written one.

---

## Backend: Pydantic Schema

```python
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
    scene_contraction_factor: list[float] = [0.5, 0.1, 0.1]
    offset: list[float] = [-4.0, 0.0, 0.0]   # [z, h, w]
    scale: list[float] = [0.25, 0.25, 0.25]   # [z, h, w]
    # Attention
    num_points_in_pillar: list[int] = [4, 32, 32]
    num_points: list[int] = [8, 64, 64]
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
    towns: list[str] = ["Town01", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
    weather: list[str] = ["ClearNoon"]
    vehicle: list[str] = ["vehicle.tesla.invisible"]
    factor: float = 0.08
    num_imgs: int = 3
    depth: bool = True
    min_cams_train: int = 1
    max_cams_train: int = 6

class ValLoaderConfig(BaseModel):
    batch_size: int = 1
    num_workers: int = 12
    towns: list[str] = ["Town02"]
    weather: list[str] = ["ClearNoon"]
    vehicle: list[str] = ["vehicle.tesla.invisible"]
    spawn_point: list[int] = [3, 7, 12, 48, 98, 66]
    factor: float = 0.25
    depth: bool = True

class DatasetConfig(BaseModel):
    data_path: str = "/app/data/"
    train: TrainLoaderConfig = TrainLoaderConfig()
    val: ValLoaderConfig = ValLoaderConfig()

class FullConfig(BaseModel):
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    dataset: DatasetConfig = DatasetConfig()
    pif: PIFConfig = PIFConfig()
```

---

## Backend: API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/configs` | Create config from `FullConfig` JSON |
| `GET` | `/api/configs` | List all saved configs (id, name, created_at) |
| `GET` | `/api/configs/{id}` | Get full config JSON |
| `PUT` | `/api/configs/{id}` | Update config |
| `DELETE` | `/api/configs/{id}` | Delete config |
| `POST` | `/api/configs/import` | Upload `config.py`, returns parsed `FullConfig` |
| `GET` | `/api/configs/{id}/export` | Returns generated `config.py` as text |
| `POST` | `/api/configs/{id}/clone` | Duplicate a config with a new name |

**Import implementation:**
```python
from mmengine.config import Config as MMConfig

def load_from_py(path: str) -> FullConfig:
    raw = MMConfig.fromfile(path)
    return FullConfig(
        encoder=EncoderConfig(
            dim=raw.get("_dim_", 128),
            tpv_h=raw.get("N_h_", 200),
            # ... etc
        ),
        pif=PIFConfig(
            enabled=raw.get("pif", False),
            transforms_path=raw.get("pif_transforms", ""),
        ),
        # ...
    )
```

**Export implementation:**
`export_to_py()` renders a config.py string from the Pydantic model, reconstructing the MMDetection `dict(type=...)` structure for the encoder using known templates. The output is a valid Python file that `mmengine.Config.fromfile()` can load.

---

## Database

New table `config_records` alongside existing `recipes` and `jobs`:

```python
class ConfigRecord(Base):
    __tablename__ = "config_records"

    id: Mapped[str]           # uuid
    name: Mapped[str]
    data: Mapped[dict]        # JSON — FullConfig as dict
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
```

---

## Frontend: Config Builder Tab

**Overall aesthetic: "Neural Blueprint"**

- Consistent with the existing `bg-gray-950` dark shell
- Typography: `JetBrains Mono` for config values and preview pane, `DM Sans` for section headers
- Color-coded sections via left-border accents

**Layout: two-panel split**

```
┌─── Config List ─────┬─── Editor ────────────────┬─── config.py Preview ───┐
│                     │                            │                         │
│ [+ New Config]      │ [Dataset]  cyan border     │  # Generated config.py  │
│                     │   data_path                │  _dim_ = 128            │
│ my-baseline    ···  │   train towns              │  N_h_ = 200             │
│ pif-experiment ···  │   val spawn points         │  ...                    │
│ high-res-test  ···  │                            │  (syntax highlighted,   │
│                     │ [Encoder]  violet border   │   live updates)         │
│                     │   dim, heads, levels       │                         │
│                     │   resolution (h/w/z)       │                         │
│                     │   scene contraction        │                         │
│                     │                            │                         │
│                     │ [Decoder]  emerald border  │                         │
│                     │   hidden dim/layers        │                         │
│                     │   ray sampling params      │                         │
│                     │                            │                         │
│                     │ [Optimizer] amber border   │                         │
│                     │   lr, epochs, loss weights │                         │
│                     │                            │                         │
│                     │ [PIF]  rose border         │                         │
│                     │   toggle + path picker     │                         │
│                     │   factor slider            │                         │
└─────────────────────┴────────────────────────────┴─────────────────────────┘
```

**Form field types:**
- **Numbers** (lr, loss weights): input + subtle range slider
- **Integers** (dims, layers, bins): number input with ± stepper
- **Booleans**: toggle switch
- **Paths** (data_path, pif_transforms_path): text input with "Browse" button (opens filesystem path dialog via backend)
- **Lists of strings** (towns, vehicles, weathers): tag-input with add/remove
- **Lists of ints** (spawn_points): tag-input

**Import/Export UX:**
- Import: drag-drop zone in the list panel header; accepts `.py` files; shows parsed diff before saving
- Export: "Download config.py" button; also "Copy to clipboard"

**Integration with Training tab:**
- Each config in the list has a "Train with this config" button
- Clicking navigates to the Training tab with the config pre-selected

---

## Config.py Export Format

The generated `config.py` will be structurally identical to the current hand-written one, preserving:
- `_base_` inheritance list (dataset, optimizer, triplane_decoder base files are still generated as separate files or inlined)
- The `self_cross_layer` / `self_layer` dict pattern for transformer layer composition
- All existing variable names (`_dim_`, `N_h_`, `pif`, `pif_transforms`, etc.) so existing training scripts continue to work

Option: offer a "flat" export mode that inlines all base configs into a single file (easier to share, no dependency on `_base_/` directory).

---

## Implementation Order

1. **Backend schema + config_io.py** — Pydantic models, `load_from_py`, `export_to_py`
2. **DB migration** — add `config_records` table
3. **API routes** — `/api/configs` CRUD + import + export
4. **Frontend: Config list + basic form** — new tab, list sidebar, create/edit
5. **Frontend: Live preview pane** — real-time config.py rendering
6. **Frontend: Import flow** — drag-drop + diff preview
7. **Frontend: Training tab integration** — "Train with this config" button
