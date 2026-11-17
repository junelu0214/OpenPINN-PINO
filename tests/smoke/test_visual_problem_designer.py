from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_visual_problem_designer_sources_and_local_licenses_exist() -> None:
    index = (PROJECT_ROOT / "docs/index.rst").read_text(encoding="utf-8")
    page = (PROJECT_ROOT / "docs/visual_problem_designer.rst").read_text(
        encoding="utf-8"
    )
    extension = (
        PROJECT_ROOT / "docs/_ext/openpinn_method_catalog.py"
    ).read_text(encoding="utf-8")
    javascript = (
        PROJECT_ROOT / "docs/_static/visual_problem_designer.js"
    ).read_text(encoding="utf-8")

    assert "   visual_problem_designer" not in index
    assert page.startswith(":orphan:")
    assert ".. openpinn-visual-problem-designer::" in page
    assert "VisualProblemDesignerDirective" in extension
    assert 'type: "pde_residual"' in javascript
    assert 'type: "pde_derivative"' in javascript
    assert "op.hessian(u, x" in javascript
    assert "op.PDEProblem(" in javascript
    assert "op.solve(" in javascript
    assert 'data-role="load-nonlinear-heat"' in extension
    assert "function nonlinearHeatSource" in javascript
    assert 'nameInput.value = "nonlinear_heat_1d"' in javascript
    assert 'message.textContent = "已加载新模型：u_t - 0.1u_xx + u^3 = 0。"' in javascript

    required_assets = (
        "docs/_static/vendor/blockly/blockly_compressed.js",
        "docs/_static/vendor/blockly/zh-hans.js",
        "docs/_static/vendor/blockly/LICENSE",
        "docs/_static/vendor/katex/katex.min.js",
        "docs/_static/vendor/katex/katex.min.css",
        "docs/_static/vendor/katex/LICENSE",
        "THIRD_PARTY_NOTICES.md",
    )
    for relative in required_assets:
        assert (PROJECT_ROOT / relative).is_file(), relative


def test_visual_problem_designer_uses_pinned_permissive_dependencies() -> None:
    notices = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    blockly_license = (
        PROJECT_ROOT / "docs/_static/vendor/blockly/LICENSE"
    ).read_text(encoding="utf-8")
    katex_license = (
        PROJECT_ROOT / "docs/_static/vendor/katex/LICENSE"
    ).read_text(encoding="utf-8")

    assert "Blockly 13.3.0" in notices
    assert "KaTeX 0.18.7" in notices
    assert "Apache License" in blockly_license
    assert "MIT License" in katex_license or "The MIT License" in katex_license
