from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = ROOT / "examples" / "tutorials"
EXPECTED = {
    "01_intro_install_and_configurator.ipynb",
    "02_intro_read_artifacts.ipynb",
    "03_pinn_poisson.ipynb",
    "04_pinn_burgers.ipynb",
    "05_pinn_plate_with_hole.ipynb",
    "05_pinn_poisson_2d.ipynb",
    "06_operator_deeponet_data_only.ipynb",
    "07_operator_fno_physics_only.ipynb",
    "08_operator_hybrid.ipynb",
    "09_extension_custom_pde_and_mode.ipynb",
}
PRIMARY = {
    "04_pinn_burgers.ipynb",
    "05_pinn_poisson_2d.ipynb",
    "07_operator_fno_physics_only.ipynb",
}
DEMOTED = EXPECTED - PRIMARY


def _has_visual_output(output: dict[str, object]) -> bool:
    data = output.get("data", {})
    if not isinstance(data, dict):
        return False
    if "image/png" in data or "image/svg+xml" in data:
        return True
    html = data.get("text/html", "")
    return "<svg" in ("".join(html) if isinstance(html, list) else str(html))


def test_tutorial_notebooks_are_valid_and_have_committed_outputs() -> None:
    notebooks = {path.name: path for path in TUTORIALS.glob("*.ipynb")}
    assert set(notebooks) == EXPECTED
    for name, path in notebooks.items():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4, name
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells, name
        assert any(cell.get("outputs") for cell in code_cells), name
        assert any("openpinn" in "".join(cell.get("source", [])) for cell in code_cells), name
        text = path.read_text(encoding="utf-8").lower()
        assert "method_configurator" in text, name
        assert "smoke" in text, name

    for name in PRIMARY:
        text = notebooks[name].read_text(encoding="utf-8")
        notebook = json.loads(text)
        outputs = [
            output
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            for output in cell.get("outputs", [])
        ]
        visuals = [
            output
            for output in outputs
            if _has_visual_output(output)
        ]
        assert len(visuals) >= 2, name
        assert not any(output.get("output_type") == "error" for output in outputs), name
        assert "configurator_tutorial" in text.lower(), name

    for name in DEMOTED:
        notebook = json.loads(notebooks[name].read_text(encoding="utf-8"))
        assert notebook["metadata"]["nbsphinx"]["orphan"] is True, name


def test_m19_demo_artifacts_are_slim_and_referenced() -> None:
    root = TUTORIALS / "artifacts" / "m19"
    expected = {
        "burgers": {"loss.png", "solution_xt.png", "metrics.json"},
        "poisson_2d_pinn": {
            "loss.png",
            "solution_vs_reference.png",
            "metrics.json",
            "REPRO.md",
            "run_config.json",
        },
        "fno_physics_only": {"loss.png", "solution_vs_reference.png", "metrics.json"},
    }
    found = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    assert found == {f"{case}/{name}" for case, names in expected.items() for name in names}
    assert not any(path.suffix == ".pt" for path in root.rglob("*"))
    burgers = json.loads((root / "burgers" / "metrics.json").read_text(encoding="utf-8"))
    poisson = json.loads((root / "poisson_2d_pinn" / "metrics.json").read_text(encoding="utf-8"))
    fno = json.loads((root / "fno_physics_only" / "metrics.json").read_text(encoding="utf-8"))
    assert burgers["relative_l2_error"] == 0.04131326079368591
    assert burgers["stop_steps"] == 3000 and burgers["passed"] is True
    assert poisson["relative_l2_error"] == 0.033738549798727036
    assert poisson["stop_steps"] == 5000 and poisson["passed"] is True
    assert fno["post_train_relative_l2"] == 0.03891376405954361
    assert fno["stop_steps"] == 10000 and fno["passed"] is False
    sidecar = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.suffix in {".json", ".md"}
    )
    assert "tongji" not in sidecar.lower()
    assert "\\" not in sidecar
    references = {
        "04_pinn_burgers.ipynb": "artifacts/m19/burgers",
        "05_pinn_poisson_2d.ipynb": "artifacts/m19/poisson_2d_pinn",
        "07_operator_fno_physics_only.ipynb": "artifacts/m19/fno_physics_only",
    }
    for name, needle in references.items():
        text = (TUTORIALS / name).read_text(encoding="utf-8")
        assert needle in text
        notebook = json.loads(text)
        pngs = [
            output
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            for output in cell.get("outputs", [])
            if isinstance(output.get("data"), dict) and "image/png" in output["data"]
        ]
        assert len(pngs) >= 2, name


def test_sphinx_renders_notebooks_without_execution() -> None:
    config = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    assert '"nbsphinx"' in config
    assert 'nbsphinx_execute = "never"' in config
    index = (ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
    assert "   configurator_tutorial" in index
    assert "   code_tutorials" in index
    assert "   method_configurator" in index
    assert ".. openpinn-method-configurator::" in index
    assert "五类 Agent" in index
    start_section = index.split(":caption: 开始", 1)[1].split(".. toctree::", 1)[0]
    assert start_section.index("   method_configurator") < start_section.index("   getting_started")
    assert "   install" not in start_section
    assert "   quickstart" not in start_section
    tutorial_section = index.split(":caption: Tutorial 算例", 1)[1].split(".. toctree::", 1)[0]
    assert "   configurator_tutorial" in tutorial_section
    assert "   code_tutorials" in tutorial_section
    assert "   poisson" not in tutorial_section
    assert "   inverse_seepage_consolidation" not in tutorial_section
    method_section = index.split(":caption: 方法生态", 1)[1].split(".. toctree::", 1)[0]
    assert "   method_catalog" not in method_section
    reference_section = index.split(":caption: 使用指南与参考", 1)[1].split(".. toctree::", 1)[0]
    assert "   user_guide" in reference_section
    assert "   api" in reference_section
    assert "   quality_reference" in reference_section
    for former_entry in ("validation", "reproducibility", "faq", "changelog"):
        assert f"   {former_entry}" not in reference_section
    code_hub = (ROOT / "docs" / "code_tutorials.rst").read_text(encoding="utf-8")
    tutorial_toctree = code_hub.split(".. toctree::", 1)[1]
    for name in EXPECTED:
        assert f"tutorials/{name.removesuffix('.ipynb')}" in tutorial_toctree
    assert "按物理问题" in code_hub
    assert "预印本结果" in code_hub
    assert "tutorials/burgers_method_comparison" in tutorial_toctree
    assert "tutorials/periodic_poisson_operator" in tutorial_toctree
    assert "tutorials/plate_hole_agent_workflow" in tutorial_toctree
    assert "inverse_seepage_consolidation" in tutorial_toctree
    assert "稳态 Darcy 渗流的渗透率识别" in code_hub
    assert "Terzaghi 固结问题族的固结系数识别" in code_hub
    getting_started = (ROOT / "docs" / "getting_started.rst").read_text(encoding="utf-8")
    assert "   install" in getting_started and "   quickstart" in getting_started
    quality_reference = (ROOT / "docs" / "quality_reference.rst").read_text(encoding="utf-8")
    for nested_entry in ("validation", "reproducibility", "faq", "changelog"):
        assert f"   {nested_entry}" in quality_reference
    method_ecosystem = (ROOT / "docs" / "method_ecosystem.rst").read_text(encoding="utf-8")
    assert "   method_catalog" in method_ecosystem
    assert (ROOT / "docs" / "locales" / "en" / "LC_MESSAGES" / "index.po").is_file()
    layout = (ROOT / "docs" / "_templates" / "layout.html").read_text(encoding="utf-8")
    assert "doc-language-switch" in layout
    assert "jupyter 全路径" not in index.lower()
    assert "sudo apt-get install -y libgomp1 pandoc" in (
        ROOT / ".github" / "workflows" / "pages.yml"
    ).read_text(encoding="utf-8")


def test_document_pages_use_available_width_without_workbench_overflow() -> None:
    site_css = (ROOT / "docs" / "_static" / "document_site.css").read_text(encoding="utf-8")
    assert ".wy-nav-content" in site_css
    assert "max-width: none" in site_css
    assert "width: auto" in site_css
    assert ".rst-content .wy-table-responsive" in site_css
    for manual_stylesheet in ("manuals.css", "manual_diagrams.css"):
        manual_css = (ROOT / "docs" / "_static" / manual_stylesheet).read_text(encoding="utf-8")
        assert "max-width: 1200px" not in manual_css

    workbench_css = (ROOT / "docs" / "_static" / "method_configurator.css").read_text(
        encoding="utf-8"
    )
    configurator_rule = workbench_css.split(".openpinn-configurator {", 1)[1].split("}", 1)[0]
    assert "width: 100%" in configurator_rule
    assert "max-width: 100%" in configurator_rule
    assert "min-width: 0" in configurator_rule
    assert "@media (max-width: 620px)" in workbench_css
    assert "[data-role=\"parameter-list\"] { grid-template-columns: minmax(0, 1fr); }" in workbench_css
