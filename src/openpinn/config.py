"""YAML/JSON configuration loading for reproducible experiments."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_config(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load a YAML/JSON path or defensively copy an existing mapping."""

    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            values = json.load(handle)
        else:
            values = yaml.safe_load(handle)
    if not isinstance(values, Mapping):
        raise ValueError("Experiment configuration must be a mapping.")
    return dict(values)

