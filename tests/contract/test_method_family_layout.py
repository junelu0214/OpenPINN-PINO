from __future__ import annotations

from pathlib import Path

import openpinn as op


def test_each_method_family_has_a_variant_source_and_config() -> None:
    project_root = Path(__file__).resolve().parents[2]
    for family in op.list_methods():
        variants = op.list_method_variants(family.name)
        assert variants
        assert family.default_variant in {variant.name for variant in variants}
        for variant in variants:
            assert (project_root / "src" / "openpinn" / variant.variant_file).is_file()
            assert (
                project_root
                / "configs"
                / "methods"
                / family.name
                / f"{variant.name}.yaml"
            ).is_file()
