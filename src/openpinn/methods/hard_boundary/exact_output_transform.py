"""Exact output-transform boundary variant."""

from .._types import MethodVariantSpec

VARIANT = MethodVariantSpec(
    name="exact_output_transform",
    title="解析输出变换",
    component_file="constraints.py",
    variant_file="methods/hard_boundary/exact_output_transform.py",
    description="把已支持问题的边界/初值条件直接写进网络输出。",
    aliases=("hard_constraint", "exact_bc"),
    compatible_with=("poisson", "burgers"),
    default_patch={"model": {"output_transform": {"name": "hard"}}},
)
