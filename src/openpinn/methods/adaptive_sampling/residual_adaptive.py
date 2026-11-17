"""Current residual-driven adaptive resampling variant."""

from collections.abc import Mapping
from typing import Any

from ...sampling import ResidualAdaptiveSampler
from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = dict(values)
    resample_every = selected.pop("resample_every", None)
    patch: dict[str, Any] = {
        "sampling": {"name": "residual_adaptive", "options": selected}
    }
    if resample_every is not None:
        patch["training"] = {"resample_every": int(resample_every)}
    return patch


VARIANT = MethodVariantSpec(
    name="residual_adaptive",
    title="残差驱动自适应重采样",
    component_file="sampling.py",
    variant_file="methods/adaptive_sampling/residual_adaptive.py",
    description="根据当前 PDE 残差从候选点中重新选择内部训练点。",
    aliases=("rar",),
    default_patch={
        "sampling": {
            "name": "residual_adaptive",
            "options": {
                "candidate_pool": 512,
                "power": 1.0,
                "uniform_mix": 0.1,
                "epsilon": 1e-12,
            },
        },
        "training": {"resample_every": 50},
    },
    option_patch=_options,
)

__all__ = ["ResidualAdaptiveSampler", "VARIANT"]
