from __future__ import annotations

from pathlib import Path

import yaml

import openpinn as op


def _catalog_entries(project_root: Path) -> list[dict[str, str]]:
    catalog_path = project_root / "docs" / "method_catalog.yaml"
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    return [
        {"method": family["method"], **variant}
        for family in payload["families"]
        for variant in family["variants"]
    ]


def test_method_catalog_matches_runtime_registry_and_detail_pages() -> None:
    project_root = Path(__file__).resolve().parents[2]
    entries = _catalog_entries(project_root)

    catalog_keys = {(item["method"], item["variant"]) for item in entries}
    runtime_keys = {
        (family.name, variant.name)
        for family in op.list_methods()
        for variant in op.list_method_variants(family.name)
    }
    assert catalog_keys == runtime_keys
    assert len(entries) == len(catalog_keys)

    for item in entries:
        detail_page = project_root / "docs" / f"{item['doc']}.rst"
        assert detail_page.is_file()
        source = detail_page.read_text(encoding="utf-8")
        directive = (
            f".. openpinn-method-summary:: {item['method']} {item['variant']}"
        )
        assert directive in source
        assert item["paper_url"].startswith("https://")
        assert item["summary"]
        assert item["suitable_for"]
        assert item["code_license"]
