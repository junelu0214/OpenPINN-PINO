"""WAM: solution-gradient-driven movement of collocation points."""

from collections.abc import Mapping
from typing import Any

from ...sampling import AdaptiveMovementSampler
from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = dict(values)
    resample_every = selected.pop("resample_every", None)
    patch: dict[str, Any] = {
        "sampling": {
            "name": "adaptive_movement",
            "options": {"indicator": "solution_gradient", **selected},
        }
    }
    if resample_every is not None:
        patch["training"] = {"resample_every": int(resample_every)}
    return patch


VARIANT = MethodVariantSpec(
    name="gradient_movement",
    title="WAM 梯度配点移动",
    component_file="sampling.py",
    variant_file="methods/adaptive_sampling/gradient_movement.py",
    description="固定一部分内部点，按解梯度指标移动其余配点；训练点总数不变。",
    aliases=("wam",),
    default_patch={
        "sampling": {
            "name": "adaptive_movement",
            "options": {
                "indicator": "solution_gradient",
                "candidate_pool": 512,
                "movable_fraction": 0.5,
                "power": 1.0,
                "epsilon": 1e-12,
            },
        },
        "training": {"resample_every": 50},
    },
    option_patch=_options,
)

__all__ = ["AdaptiveMovementSampler", "VARIANT"]
