"""Current self-adaptive loss-weight variant."""

from collections.abc import Mapping
from typing import Any

from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"loss": {"weighting": {"name": "self_adaptive", **dict(values)}}}


VARIANT = MethodVariantSpec(
    name="self_adaptive",
    title="可训练损失权重",
    component_file="losses.py",
    variant_file="methods/adaptive_weights/self_adaptive.py",
    description="训练中更新 PDE、边界和初值损失的权重。",
    aliases=("self_adaptive_weights",),
    default_patch={
        "loss": {
            "weighting": {
                "name": "self_adaptive",
                "lr": 0.01,
                "min_weight": 1e-6,
                "max_weight": 100.0,
            }
        }
    },
    option_patch=_options,
)
