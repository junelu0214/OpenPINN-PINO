"""Adam followed by full-batch L-BFGS."""

import math
from collections.abc import Mapping
from typing import Any

from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"optimizer": {"method_options": dict(values)}}


def _finalize(config: dict, steps: int) -> dict:
    if steps < 2:
        raise ValueError("adam_lbfgs requires at least 2 outer training steps.")
    options = dict(config.get("optimizer", {}).get("method_options", {}))
    adam_lr = float(options.get("adam_lr", 1e-3))
    lbfgs_lr = float(options.get("lbfgs_lr", 1.0))
    lbfgs_fraction = float(options.get("lbfgs_fraction", 0.02))
    lbfgs_max_steps = int(options.get("lbfgs_max_steps", 20))
    lbfgs_max_iter = int(options.get("lbfgs_max_iter", 10))
    history_size = int(options.get("history_size", 50))
    line_search_fn = str(options.get("line_search_fn", "strong_wolfe"))
    lbfgs_steps = max(
        1,
        min(
            steps - 1,
            lbfgs_max_steps,
            int(math.ceil(steps * lbfgs_fraction)),
        ),
    )
    config["optimizer"] = {
        "name": "adam_lbfgs",
        "stages": [
            {"name": "adam", "steps": steps - lbfgs_steps, "lr": adam_lr},
            {
                "name": "lbfgs",
                "steps": lbfgs_steps,
                "lr": lbfgs_lr,
                "max_iter": lbfgs_max_iter,
                "history_size": history_size,
                "line_search_fn": line_search_fn,
            },
        ],
    }
    return config


VARIANT = MethodVariantSpec(
    name="adam_then_lbfgs",
    title="Adam 后接 L-BFGS",
    component_file="optimizers.py",
    variant_file="methods/adam_lbfgs/adam_then_lbfgs.py",
    description="先用 Adam，再用全批量 L-BFGS 精修。",
    aliases=("staged_optimizer",),
    option_patch=_options,
    finalize_config=_finalize,
)
