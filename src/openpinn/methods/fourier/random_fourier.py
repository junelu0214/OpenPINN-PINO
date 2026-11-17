"""Fixed random Fourier feature variant."""

from collections.abc import Mapping
from typing import Any

from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"model": {"feature": {"name": "fourier", **dict(values)}}}


VARIANT = MethodVariantSpec(
    name="random_fourier",
    title="固定随机 Fourier 特征",
    component_file="features.py",
    variant_file="methods/fourier/random_fourier.py",
    description="在 MLP 前加入固定随机 Fourier 坐标特征。",
    aliases=("fourier_features", "rff"),
    default_patch={
        "model": {
            "feature": {
                "name": "fourier",
                "num_frequencies": 16,
                "scale": 1.0,
                "seed": 29,
                "include_input": True,
                "trainable": False,
            }
        }
    },
    option_patch=_options,
)
