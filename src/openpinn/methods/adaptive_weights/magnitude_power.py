"""Fixed multi-magnitude power-loss balancing from MMPINN."""

from collections.abc import Mapping
from typing import Any

from ...losses import MagnitudePowerLossComposer
from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"loss": {"weighting": {"name": "magnitude_power", **dict(values)}}}


VARIANT = MethodVariantSpec(
    name="magnitude_power",
    title="MMPINN 多量级幂损失",
    component_file="losses.py",
    variant_file="methods/adaptive_weights/magnitude_power.py",
    description="用固定分数幂压缩不同量级损失；不是随训练更新的权重。",
    aliases=("mmpinn_power", "power_loss"),
    default_patch={
        "loss": {
            "weighting": {
                "name": "magnitude_power",
                "roots": {"pde": 3.0, "boundary": 2.0, "initial": 2.0},
                "default_root": 1.0,
                "epsilon": 1e-12,
            }
        }
    },
    option_patch=_options,
)

__all__ = ["MagnitudePowerLossComposer", "VARIANT"]
