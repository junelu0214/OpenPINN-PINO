"""Trainable-slope tanh activation variant."""

from collections.abc import Mapping
from typing import Any

from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = dict(values)
    unknown = sorted(set(selected).difference({"initial_slope"}))
    if unknown:
        raise ValueError(
            "adaptive_tanh options support only initial_slope; got: "
            + ", ".join(unknown)
        )
    return {"model": {"activation_options": selected}} if selected else {}

VARIANT = MethodVariantSpec(
    name="adaptive_tanh",
    title="可训练斜率 tanh",
    component_file="models.py",
    variant_file="methods/adaptive_activation/adaptive_tanh.py",
    description="把固定 tanh 换成带可训练斜率的 adaptive_tanh。",
    default_patch={
        "model": {
            "activation": "adaptive_tanh",
            "activation_options": {"initial_slope": 1.0},
        }
    },
    option_patch=_options,
)
