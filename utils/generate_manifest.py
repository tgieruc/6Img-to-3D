"""Generate train.jsonl / val.jsonl manifests from a seed4d data directory.

Each line is one scene (one step), e.g.:
  {"input": "/abs/path/.../nuscenes/transforms/transforms_ego.json",
   "target": "/abs/path/.../sphere/transforms/transforms_ego.json",
   "town": "Town02", "weather": "ClearNoon",
   "vehicle": "vehicle.mini.cooper_s", "spawn_point": 1, "step": 0}

Usage:
  python utils/generate_manifest.py \\
      --data-dir /home/bdw/Documents/seed4d/data \\
      --output-dir /home/bdw/Documents/seed4d \\
      --input-sensor nuscenes \\
      --target-sensor sphere \\
      --val-towns Town02
"""

import argparse
import json
from pathlib import Path


def discover_entries(data_dir: Path, input_sensor: str, target_sensor: str) -> list[dict]:
    """Walk data_dir and return one entry per ego_vehicle directory."""
    entries = []
    for ego_dir in sorted(data_dir.rglob("ego_vehicle")):
        input_tf = ego_dir / input_sensor / "transforms" / "transforms_ego.json"
        target_tf = ego_dir / target_sensor / "transforms" / "transforms_ego.json"
        if not input_tf.exists():
            print(f"  skip (missing {input_sensor}): {ego_dir}")
            continue
        if not target_tf.exists():
            print(f"  skip (missing {target_sensor}): {ego_dir}")
            continue

        # Parse metadata from path: .../Town/Weather/Vehicle/spawn_point_N/step_N/ego_vehicle
        parts = ego_dir.parts
        step_part = parts[-2]  # step_N
        sp_part = parts[-3]  # spawn_point_N
        vehicle = parts[-4]
        weather = parts[-5]
        town = parts[-6]

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


def generate_manifest(
    data_dir: Path,
    output_dir: Path,
    input_sensor: str,
    target_sensor: str,
    val_towns: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_dir} ...")
    entries = discover_entries(data_dir, input_sensor, target_sensor)
    print(f"Found {len(entries)} scenes total")

    train = [e for e in entries if e["town"] not in val_towns]
    val = [e for e in entries if e["town"] in val_towns]
    print(f"  train: {len(train)}  val: {len(val)}")

    for split, data in [("train", train), ("val", val)]:
        path = output_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate seed4d JSONL manifests")
    parser.add_argument("--data-dir", required=True, type=Path, help="Root of seed4d data")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to write train/val .jsonl")
    parser.add_argument("--input-sensor", default="nuscenes", help="Sensor dir for input images (default: nuscenes)")
    parser.add_argument("--target-sensor", default="sphere", help="Sensor dir for target images (default: sphere)")
    parser.add_argument("--val-towns", default="", help="Comma-separated towns to use as val set (default: none)")
    args = parser.parse_args()

    val_towns = [t.strip() for t in args.val_towns.split(",") if t.strip()]
    generate_manifest(args.data_dir, args.output_dir, args.input_sensor, args.target_sensor, val_towns)


if __name__ == "__main__":
    main()
