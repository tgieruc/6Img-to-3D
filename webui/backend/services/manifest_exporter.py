import json
from pathlib import Path


def _discover_scenes(data_dir: Path, global_filters: dict) -> list[dict]:
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

        entries.append(
            {
                "input": str(input_tf),
                "target": str(target_tf),
                "town": town,
                "weather": weather,
                "vehicle": vehicle,
                "spawn_point": int(sp_part.split("_")[-1]),
                "step": int(step_part.split("_")[-1]),
            }
        )
    return entries


def _matches_rule(entry: dict, rule: dict) -> bool:
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
    """Walk data_dir, filter by recipe, write JSONL manifests. Returns paths + counts."""
    data_dir = Path(recipe["data_dir"])
    output_dir = Path(recipe["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries = _discover_scenes(data_dir, recipe["global"])
    result: dict = {"scene_counts": {}}

    for split_name, rule in recipe.get("splits", {}).items():
        matched = [e for e in all_entries if _matches_rule(e, rule)]
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for entry in matched:
                f.write(json.dumps(entry) + "\n")
        result[split_name] = str(out_path)
        result["scene_counts"][split_name] = len(matched)

    return result
