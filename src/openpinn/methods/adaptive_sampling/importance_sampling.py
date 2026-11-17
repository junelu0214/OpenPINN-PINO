"""Loss-proportional importance sampling with inverse-probability correction."""

from collections.abc import Mapping
from typing import Any

from ...sampling import ImportanceSamplingSampler
from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = dict(values)
    resample_every = selected.pop("resample_every", None)
    patch: dict[str, Any] = {
        "sampling": {"name": "importance_sampling", "options": selected}
    }
    if resample_every is not None:
        patch["training"] = {"resample_every": int(resample_every)}
    return patch


VARIANT = MethodVariantSpec(
    name="importance_sampling",
    title="损失重要性采样",
    component_file="sampling.py",
    variant_file="methods/adaptive_sampling/importance_sampling.py",
    description="按逐点 PDE 损失抽样，并用逆概率权重修正训练目标。",
    aliases=("importance", "is_pinn"),
    default_patch={
        "sampling": {
            "name": "importance_sampling",
            "options": {"candidate_pool": 512, "epsilon": 1e-12},
        },
        "training": {"resample_every": 50},
    },
    option_patch=_options,
)

__all__ = ["ImportanceSamplingSampler", "VARIANT"]
