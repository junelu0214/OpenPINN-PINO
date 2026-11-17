"""Homoscedastic uncertainty weighting for named PINN loss terms."""

from collections.abc import Mapping
from typing import Any

from ...losses import UncertaintyLossComposer
from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"loss": {"weighting": {"name": "uncertainty", **dict(values)}}}


VARIANT = MethodVariantSpec(
    name="uncertainty_weighting",
    title="不确定性损失权重",
    component_file="losses.py",
    variant_file="methods/adaptive_weights/uncertainty_weighting.py",
    description="学习每个损失项的对数方差，用 exp(-s) 自动缩放不同任务。",
    aliases=("uncertainty", "homoscedastic"),
    default_patch={
        "loss": {
            "weighting": {
                "name": "uncertainty",
                "lr": 0.001,
                "min_log_variance": -10.0,
                "max_log_variance": 10.0,
            }
        }
    },
    option_patch=_options,
)

__all__ = ["UncertaintyLossComposer", "VARIANT"]
