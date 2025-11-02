
import yaml
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(__file__).parent / "config.yml"


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


cfg = load_config()