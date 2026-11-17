"""Sequential causal time-window variant."""

import math
from collections.abc import Mapping
from typing import Any

from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = dict(values)
    window_ends = selected.pop("window_ends", [0.25, 0.5, 0.75, 1.0])
    if selected:
        raise ValueError(
            "time_window options support only window_ends; got: "
            + ", ".join(sorted(selected))
        )
    return {
        "sampling": {
            "name": "time_window",
            "options": {"window_ends": list(window_ends)},
        }
    }


def _finalize(config: dict, steps: int) -> dict:
    windows = config["sampling"]["options"]["window_ends"]
    config.setdefault("training", {})["resample_every"] = max(
        1, int(math.ceil(steps / len(windows)))
    )
    return config


VARIANT = MethodVariantSpec(
    name="causal_window",
    title="顺序因果时间窗",
    component_file="training.py",
    variant_file="methods/time_window/causal_window.py",
    description="动态问题从早期时间窗开始，逐步扩展到完整时间区间。",
    aliases=("causal", "causal_training"),
    compatible_with=("dynamic",),
    default_patch={
        "sampling": {
            "name": "time_window",
            "options": {"window_ends": [0.25, 0.5, 0.75, 1.0]},
        }
    },
    option_patch=_options,
    finalize_config=_finalize,
)
