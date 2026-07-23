"""Small YAML configuration helpers."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def merged_config(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    for dotted_key, value in overrides.items():
        target = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result

