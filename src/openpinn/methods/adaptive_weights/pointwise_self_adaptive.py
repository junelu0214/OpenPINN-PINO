"""Full pointwise self-adaptive PINN loss masks."""

from collections.abc import Mapping
from typing import Any

from ...losses import PointwiseAdaptiveLossComposer
from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "loss": {"weighting": {"name": "pointwise_self_adaptive", **dict(values)}}
    }


VARIANT = MethodVariantSpec(
    name="pointwise_self_adaptive",
    title="逐点自适应 SA-PINN",
    component_file="losses.py",
    variant_file="methods/adaptive_weights/pointwise_self_adaptive.py",
    description="为每个 PDE、边界和初值训练点学习一个正掩码并做梯度上升。",
    aliases=("sa_pinn", "pointwise_sa_pinn", "soft_attention"),
    compatible_with=("static", "dynamic"),
    default_patch={
        "loss": {
            "weighting": {
                "name": "pointwise_self_adaptive",
                "lr": 0.01,
                "min_weight": 1e-6,
                "max_weight": 100.0,
            }
        }
    },
    option_patch=_options,
)

__all__ = ["PointwiseAdaptiveLossComposer", "VARIANT"]
