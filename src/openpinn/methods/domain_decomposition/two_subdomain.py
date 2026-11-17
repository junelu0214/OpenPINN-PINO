"""Two-subnetwork one-dimensional interface variant."""

from collections.abc import Mapping
from typing import Any

from .._types import MethodVariantSpec


def _options(values: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = dict(values)
    unknown = sorted(set(selected).difference({"interface"}))
    if unknown:
        raise ValueError(
            "domain_decomposition options support only interface; got: "
            + ", ".join(unknown)
        )
    return {"model": {"options": selected}} if selected else {}


def _finalize(config: dict, steps: int) -> dict:
    del steps
    allowed = {
        "name",
        "input_dim",
        "output_dim",
        "hidden_layers",
        "layers",
        "depth",
        "activation",
        "activation_options",
        "initialization",
        "bias",
        "options",
    }
    config["model"] = {
        key: value for key, value in config["model"].items() if key in allowed
    }
    return config


VARIANT = MethodVariantSpec(
    name="two_subdomain",
    title="两子域多网络分解",
    component_file="models.py, problems.py, losses.py",
    variant_file="methods/domain_decomposition/two_subdomain.py",
    description="在两个材料子区使用独立网络，并加入解与通量界面损失。",
    aliases=("xpinn", "multi_network"),
    compatible_with=("heterogeneous_poisson",),
    default_patch={
        "model": {"name": "domain_decomposition"},
        "loss": {
            "formulation": "interface_strong",
            "weights": {
                "pde": 1.0,
                "boundary": 1.0,
                "interface_solution": 1.0,
                "interface_flux": 1.0,
            },
        },
    },
    option_patch=_options,
    finalize_config=_finalize,
)
