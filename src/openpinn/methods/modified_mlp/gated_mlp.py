"""Current gated ModifiedMLP variant."""

from .._types import MethodVariantSpec

VARIANT = MethodVariantSpec(
    name="gated_mlp",
    title="门控 Modified MLP",
    component_file="models.py",
    variant_file="methods/modified_mlp/gated_mlp.py",
    description="用门控 ModifiedMLP 替换普通 MLP。",
    aliases=("modified_mlp_default",),
    default_patch={"model": {"name": "modified_mlp"}},
)
