from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config used by every script in the project."""
    config_path = Path(path or os.environ.get("SEG_CONFIG", "configs/default.yaml"))
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
