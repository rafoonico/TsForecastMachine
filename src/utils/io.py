from pathlib import Path
import json
import yaml


def read_yaml(path: str):
    """Load a YAML file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_json(obj, path: str):
    """Write a JSON serializable object to disk."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
