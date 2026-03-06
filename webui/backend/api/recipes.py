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
    splits: dict[str, SplitRule]


def _validate_no_overlap(splits: dict[str, SplitRule]) -> None:
    assigned: dict[str, str] = {}
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
    record = RecipeRecord(name=recipe.name, yaml_content=_to_yaml(recipe))
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "name": record.name, "created_at": str(record.created_at)}


@router.get("")
def list_recipes(db: Session = Depends(get_db)):
    return [
        {
            "id": r.id,
            "name": r.name,
            "created_at": str(r.created_at),
            "train_manifest": r.train_manifest,
            "val_manifest": r.val_manifest,
            "scene_counts": r.scene_counts,
        }
        for r in db.query(RecipeRecord).order_by(RecipeRecord.created_at.desc()).all()
    ]


@router.get("/{recipe_id}")
def get_recipe(recipe_id: str, db: Session = Depends(get_db)):
    r = db.get(RecipeRecord, recipe_id)
    if not r:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return {
        "id": r.id,
        "name": r.name,
        "yaml_content": r.yaml_content,
        "train_manifest": r.train_manifest,
        "val_manifest": r.val_manifest,
        "test_manifest": r.test_manifest,
        "scene_counts": r.scene_counts,
    }
