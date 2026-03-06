# 6Img-to-3D Web UI & Data Pipeline Design

Date: 2026-03-06

## Overview

A full-pipeline web application for 6Img-to-3D that covers:
1. **Split Manager** — build dataset split recipes, export train/val/test JSONL manifests
2. **Training Dashboard** — launch and monitor training runs via MLflow
3. **Evaluation Viewer** — run eval, browse rendered vs GT images, compare experiments

Data originates from `seed4d/` (CARLA simulator), is split/configured via this webapp,
then fed into `6Img-to-3D/` training and evaluation.

---

## Architecture

### Repositories

```
seed4d/data/
    Town/Weather/Vehicle/spawn_point_N/step_N/ego_vehicle/{nuscenes,sphere,...}

6Img-to-3D/webui/          <- NEW
    backend/               (FastAPI + SQLite + MLflow client)
    frontend/              (React + Vite + Tailwind, same stack as seed4d webui)

6Img-to-3D/dataloader/     <- FIXES
    manifest_dataset.py    (wire up RaysDataset from target path)
    generate_manifest.py   (handle ego_vehicle/ path level)
```

### Data Flow

```
Split Manager
  -> recipe.yaml + train.jsonl / val.jsonl / test.jsonl
      -> Training page (passes --manifest to train.py)
          -> MLflow run (metrics, params, checkpoints as artifacts)
              -> Evaluation page (renders novel views, shows metrics)
```

---

## Component 1: Split Manager

### Purpose

Interactively build a split recipe: select which data to use, assign sensor roles
(input/target), and partition scenes into mutually exclusive train/val/test splits.
Export reproducible JSONL manifests + a recipe YAML.

### UI Layout

Three-column layout:

- **Left — Global Filters**: vehicle, weather, input sensor, target sensor.
  All dropdowns populated by scanning the configured `data_dir`.
- **Center — Split Rules**: one panel per split (train/val/test). Each panel lets
  you assign towns, spawn points, and steps. Already-assigned towns are greyed out
  in other splits (mutual exclusivity enforced in UI + backend validation).
- **Right — Preview**: live scene counts per split, sample thumbnails, overlap
  warning if any conflict detected.
- **Bottom**: Export button → generates manifests + recipe, registers in DB.

### Split Recipe Format

```yaml
data_dir: /home/bdw/Documents/seed4d/data
output_dir: /home/bdw/Documents/seed4d
global:
  vehicles: ["vehicle.mini.cooper_s"]
  weathers: ["ClearNoon"]
  input_sensor: nuscenes
  target_sensor: sphere
splits:
  train:
    towns: ["Town01", "Town03", "Town04"]
    spawn_points: all
    steps: all
  val:
    towns: ["Town02"]
    spawn_points: [3, 7, 12]
    steps: all
  test:
    towns: ["Town05"]
    spawn_points: all
    steps: all
```

### Manifest Format (unchanged)

Each line in `train.jsonl` / `val.jsonl` / `test.jsonl`:

```json
{
  "input": "/abs/path/.../ego_vehicle/nuscenes/transforms/transforms_ego.json",
  "target": "/abs/path/.../ego_vehicle/sphere/transforms/transforms_ego.json",
  "town": "Town01", "weather": "ClearNoon",
  "vehicle": "vehicle.mini.cooper_s", "spawn_point": 1, "step": 0
}
```

### Dataloader Fixes Required

1. `utils/generate_manifest.py` — already handles `ego_vehicle/` path level correctly.
   Fix: ensure `CarlaDataset` path logic is consistent with manifest paths.
2. `dataloader/manifest_dataset.py` — currently returns `entry["target"]` as a raw
   string. Fix: load a `RaysDataset` from the target `transforms_ego.json` path and
   return it alongside the input images, matching the `(input_rgb, img_meta, sphere_dataloader)`
   tuple expected by `train.py`.
3. `builder/data_builder.py` — add a manifest-based dataloader branch alongside the
   existing `CarlaDataset` / `PickledCarlaDataset` branches.
4. `train.py` / `eval.py` — add `--manifest` flag as an alternative to `--py-config`
   for dataset specification.

---

## Component 2: Training Dashboard

### Purpose

Launch `train.py` runs, stream logs, visualise live loss/metric curves, compare
experiments. Uses MLflow as the experiment tracking backend.

### MLflow Integration

- `train.py` switches from `tensorboardX.SummaryWriter` to `mlflow.log_metric()` /
  `mlflow.log_params()` / `mlflow.log_artifact()`.
- MLflow stores runs locally in `mlruns/` (file-based, no server required).
- FastAPI backend queries MLflow via Python client:
  `mlflow.search_runs()`, `mlflow.MlflowClient().get_metric_history()`.
- Checkpoints saved as MLflow artifacts (`best_psnr.pth`, `best_lpips.pth`).

### UI Layout

Two-panel layout:

- **Left — Experiment list**: all runs with status badge (running/done/failed),
  best PSNR, manifest used.
- **Right — Run detail**:
  - Config summary (manifest, lr, batch size, etc.)
  - Live loss chart: MSE / LPIPS / TV / Total (tabs), rendered with Recharts,
    polled every 5s from MLflow metrics API
  - Live metrics: PSNR, LPIPS, SSIM (val, updated per checkpoint)
  - Log tail (stdout streaming via SSE)
  - Actions: Stop, go to Eval

- **Compare mode**: select 2+ runs → overlay loss/PSNR curves on one chart.

### New Run Dialog

- Select manifest (dropdown from saved recipes, shows scene counts)
- Select config `.py` (browse or upload)
- Set experiment/run name
- Launch

---

## Component 3: Evaluation Viewer

### Purpose

Run `eval.py` on a finished training run, browse rendered vs GT images per scene,
compare two runs side by side, export a metrics report.

### UI Layout

Two-panel layout:

- **Left — Run list**: completed training runs.
- **Right — Eval detail**:
  - Checkpoint selector (`best_psnr.pth` / `best_lpips.pth` / custom)
  - Manifest selector (val / test)
  - Metrics summary: PSNR / LPIPS / SSIM
  - Scene browser: navigate scenes, show Ground Truth | Rendered | Diff heatmap
  - Compare mode: two runs, same scene, GT / Run-A / Run-B
  - Export report: markdown/HTML with metrics table + sample images

### Eval Job

- "Run Eval" launches `eval.py --resume-from checkpoint.pth --manifest val.jsonl`
  as a subprocess (same job_runner pattern as seed4d).
- Rendered images saved as MLflow artifacts under the run.
- Scene browser loads images from MLflow artifact store.

---

## Backend Design

### Stack

- FastAPI (same as seed4d webui)
- SQLite via SQLAlchemy (recipes, job records)
- MLflow Python client (training metrics + artifacts)
- Subprocess job runner (reuse seed4d pattern)

### API Routes

```
GET  /api/data/scan          -> scan data_dir, return hierarchy
GET  /api/data/options       -> available towns/weathers/vehicles/sensors

POST /api/recipes            -> create/save a split recipe
GET  /api/recipes            -> list saved recipes
GET  /api/recipes/{id}       -> get recipe
POST /api/recipes/{id}/export -> generate JSONL manifests

GET  /api/runs               -> list MLflow runs
GET  /api/runs/{id}/metrics  -> metric history
GET  /api/runs/{id}/artifacts -> list artifacts
POST /api/jobs/train         -> launch train.py
POST /api/jobs/eval          -> launch eval.py
GET  /api/jobs/{id}          -> job status + log
GET  /api/jobs/{id}/stream   -> SSE log stream
DELETE /api/jobs/{id}        -> stop job
```

### Database Models

```
RecipeRecord   id, name, yaml_content, output_dir, created_at
JobRecord      id, type (train|eval), status, mlflow_run_id, pid, log, ...
```

---

## Frontend Design

### Stack

- React + Vite + TypeScript (same as seed4d webui)
- Tailwind CSS
- TanStack Query (data fetching)
- Recharts (loss/metric charts)
- React Router (3 pages)

### Pages

```
/splits   -> Split Manager
/training -> Training Dashboard
/eval     -> Evaluation Viewer
```

---

## Implementation Phases

1. **Phase 1 — Dataloader fixes**: fix `ManifestDataset`, `generate_manifest.py`,
   `data_builder.py`, add `--manifest` flag to `train.py`/`eval.py`.
2. **Phase 2 — Backend scaffold**: FastAPI app, DB models, data scan API, recipe
   CRUD + export.
3. **Phase 3 — Split Manager UI**: global filters, split rules panels, preview,
   export.
4. **Phase 4 — MLflow integration**: switch `train.py` logging, training job
   runner, metrics API.
5. **Phase 5 — Training Dashboard UI**: experiment list, run detail, live charts,
   compare mode.
6. **Phase 6 — Evaluation**: eval job runner, scene browser UI, compare mode,
   report export.
