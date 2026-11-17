"""ReLoBRaLo relative loss balancing variant."""

from collections.abc import Mapping
from typing import Any

from ...losses import ReLoBRaLoLossComposer
from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"loss": {"weighting": {"name": "relobralo", **dict(values)}}}


VARIANT = MethodVariantSpec(
    name="relobralo",
    title="ReLoBRaLo 相对损失平衡",
    component_file="losses.py",
    variant_file="methods/adaptive_weights/relobralo.py",
    description="根据相对初始/上一步损失和随机回看来平衡多个目标。",
    aliases=("relative_balancing",),
    default_patch={
        "loss": {
            "weighting": {
                "name": "relobralo",
                "alpha": 0.999,
                "temperature": 0.1,
                "expected_lookback": 0.999,
                "epsilon": 1e-12,
                "seed": 0,
            }
        }
    },
    option_patch=_options,
)

__all__ = ["ReLoBRaLoLossComposer", "VARIANT"]
