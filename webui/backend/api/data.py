from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/data", tags=["data"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT.parent / "seed4d" / "data"


def _iter_scenes(data_dir: Path):
    """Yield one dict per ego_vehicle directory."""
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
    result = scan()
    return {k: result[k] for k in ("towns", "weathers", "vehicles", "sensors")}
