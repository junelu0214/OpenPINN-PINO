from __future__ import annotations

import math
import json
import shutil
import subprocess
from pathlib import Path

import openpinn as op
import pytest
import torch


def test_configurator_external_reference_short_run(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = op.solve(
        "poisson", methods=[op.method("baseline", "plain_pinn")],
        steps=1, seed=0, model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 4},
        output_root=tmp_path, plot=False,
    )
    metrics = op.evaluate_reference_csv(
        result, project_root / "examples/references/poisson_reference.csv",
        coordinate_columns=("x",), value_column="u",
        problem=op.make_poisson_problem(),
    )
    assert math.isfinite(metrics["rmse"])
    assert (result.output_dir / "reference_comparison.svg").is_file()
    saved = json.loads((result.output_dir / "metrics.json").read_text())
    assert saved["external_reference_rmse"] == pytest.approx(metrics["rmse"])
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert "reference_metrics.json" in manifest["files"]
    assert "reference_comparison.svg" in manifest["files"]


def test_configurator_custom_equation_template_runs(tmp_path: Path) -> None:
    domain = op.Interval(-1.0, 1.0)

    def equation(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        u_xx = op.hessian(u, x, i=0, j=0)
        source = math.pi**2 * torch.sin(math.pi * x[:, :1])
        return -u_xx - source

    def exact_solution(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x[:, :1])

    problem = op.PDEProblem(
        domain,
        equation,
        [op.DirichletBC(0.0, name="boundary")],
        exact_solution=exact_solution,
        name="my_new_equation",
        metadata={
            "kind": "custom_static",
            "equation": "-u_xx = pi^2 sin(pi*x)",
        },
    )
    result = op.solve(
        problem,
        methods=[op.method("baseline", "plain_pinn")],
        steps=1,
        seed=0,
        training={"log_every": 1},
        output_root=tmp_path,
        plot=False,
    )
    assert result.metrics["problem"] == "my_new_equation"
    assert result.metrics["method"] == "baseline"
    assert (result.output_dir / "config.yaml").is_file()
    assert (result.output_dir / "metrics.json").is_file()
    assert (result.output_dir / "failure_notes.md").is_file()


def test_custom_equation_automatically_names_weights_for_balancing(
    tmp_path: Path,
) -> None:
    domain = op.Interval(-1.0, 1.0)

    def equation(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -op.hessian(u, x, i=0, j=0)

    problem = op.PDEProblem(
        domain,
        equation,
        [op.DirichletBC(0.0, name="boundary")],
        name="weighted_custom_equation",
        metadata={"kind": "custom_static", "residual_names": ("pde",)},
    )
    result = op.solve(
        problem,
        methods=[op.method("adaptive_weights", "relobralo")],
        steps=1,
        seed=0,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 4},
        overrides={"evaluation": {"points": 9}},
        output_root=tmp_path,
        plot=False,
    )

    assert result.metrics["method"] == "adaptive_weights"
    assert "final_weight_pde" in result.metrics
    assert "final_weight_boundary" in result.metrics
    assert (result.output_dir / "manifest.json").is_file()


def test_generated_style_mixed_boundary_conditions_run(tmp_path: Path) -> None:
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    domain = op.RectangleWithCircularHoles(
        lower=(x_min, y_min), upper=(x_max, y_max), holes=()
    )

    def equation(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -op.hessian(u, x, i=0, j=0) - op.hessian(u, x, i=1, j=1)

    def on_left(points: torch.Tensor) -> torch.Tensor:
        return torch.isclose(points[:, 0], points.new_tensor(x_min))

    def on_right(points: torch.Tensor) -> torch.Tensor:
        return torch.isclose(points[:, 0], points.new_tensor(x_max))

    def on_bottom(points: torch.Tensor) -> torch.Tensor:
        return torch.isclose(points[:, 1], points.new_tensor(y_min))

    def on_top(points: torch.Tensor) -> torch.Tensor:
        return torch.isclose(points[:, 1], points.new_tensor(y_max))

    problem = op.PDEProblem(
        domain,
        equation,
        [
            op.DirichletBC(0.0, boundary=on_left, name="boundary_left"),
            op.NeumannBC(
                0.0,
                normal=(1.0, 0.0),
                boundary=on_right,
                name="boundary_right",
            ),
            op.RobinBC(
                0.0,
                normal=(0.0, -1.0),
                boundary=on_bottom,
                alpha=1.0,
                beta=0.5,
                name="boundary_bottom",
            ),
            op.DirichletBC(0.0, boundary=on_top, name="boundary_top"),
        ],
        name="mixed_boundary_2d",
        metadata={"kind": "custom_static", "residual_names": ("pde",)},
    )
    result = op.solve(
        problem,
        methods=[op.method("adaptive_weights", "relobralo")],
        steps=1,
        seed=0,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 16, "boundary": 128},
        overrides={"evaluation": {"points": {"x": 5, "y": 5}}},
        output_root=tmp_path,
        plot=False,
    )

    assert result.metrics["method"] == "adaptive_weights"
    assert "boundary_left_rmse" in result.metrics
    assert "boundary_right_rmse" in result.metrics
    assert "boundary_bottom_rmse" in result.metrics
    assert "boundary_top_rmse" in result.metrics


def test_configurator_sources_include_custom_equation_controls() -> None:
    project_root = Path(__file__).resolve().parents[2]
    extension = (project_root / "docs/_ext/openpinn_method_catalog.py").read_text(
        encoding="utf-8"
    )
    javascript = (project_root / "docs/_static/method_configurator.js").read_text(
        encoding="utf-8"
    )
    assert '"name": "custom_equation_template"' in extension
    assert '"template": "custom_visual"' in extension
    assert 'data-role="custom-problem-panel"' in extension
    assert 'data-role="custom-preview"' in extension
    assert 'data-role="custom-dimension"' in extension
    assert 'data-role="custom-equation-mode"' in extension
    assert 'data-role="custom-term-builder"' in extension
    assert 'data-role="custom-add-term"' in extension
    assert extension.count('data-role="custom-boundary-side"') == 5
    assert 'data-side="left"' in extension
    assert 'data-side="right"' in extension
    assert 'data-side="bottom"' in extension
    assert 'data-side="top"' in extension
    assert 'data-side="curve"' in extension
    assert 'data-role="custom-polygon-boundaries"' in extension
    assert 'data-role="custom-geometry-type"' in extension
    assert '<option value="circle">' in extension
    assert '<option value="ellipse">' in extension
    assert '<option value="rounded_rectangle">' in extension
    assert '<option value="spline">' in extension
    assert '<option value="polygon">导入多边形文件（识别图片）</option>' in extension
    assert '<option value="dxf">导入 DXF 图纸（COMSOL/CAD）</option>' in extension
    assert '<option value="comsol_dxf">' not in extension
    assert '<option value="cad_dxf">' not in extension
    assert 'data-role="custom-dxf-import"' in extension
    assert 'data-role="custom-dxf-file"' in extension
    assert 'app.add_js_file("dxf_geometry_import.js", defer="defer")' in extension
    assert 'data-role="custom-circle-radius"' in extension
    assert 'data-role="custom-ellipse-axis-a"' in extension
    assert 'data-role="custom-ellipse-axis-b"' in extension
    assert 'data-role="custom-ellipse-rotation"' in extension
    assert 'data-role="custom-corner-radius"' in extension
    assert 'data-role="custom-polygon-file"' in extension
    assert 'data-role="custom-image-options"' in extension
    assert 'data-role="custom-image-threshold-mode"' in extension
    assert 'data-role="custom-image-threshold"' in extension
    assert 'data-role="custom-image-resolution"' in extension
    assert 'data-role="custom-image-shape-mode"' in extension
    assert 'data-role="custom-image-simplify" type="number" value="2"' in extension
    assert 'data-role="custom-polygon-editor"' in extension
    assert 'data-role="custom-polygon-vertices"' in extension
    assert 'data-role="custom-polygon-transform-apply"' in extension
    assert ".png,.jpg,.jpeg" in extension
    assert '<option value="neumann">' in extension
    assert '<option value="robin">' in extension
    assert 'data-role="custom-initial-preset"' in extension
    assert "function renderCustomPreview()" in javascript
    assert "function readCustomTerms(settings)" in javascript
    assert "function resetCustomTerms()" in javascript
    assert "function readCustomBoundaries(dimension, geometry)" in javascript
    assert "function polygonCoordinatesFromJson(data)" in javascript
    assert "function polygonCoordinatesFromCsv(text)" in javascript
    assert "function loadPolygonImage(file)" in javascript
    assert "OpenPINNImageTrace.imageDataToPolygon" in javascript
    assert "traced.analyticCandidates" in javascript
    assert "traced.curveShape" in javascript
    assert 'customGeometryTypeSelect.value = "spline"' in javascript
    assert 'customGeometryTypeSelect.value = "circle"' in javascript
    assert 'customGeometryTypeSelect.value = "rounded_rectangle"' in javascript
    assert "OpenPINNImageTrace.transformRings" in javascript
    assert "OpenPINNImageTrace.setEdgeLength" in javascript
    assert "OpenPINNImageTrace.setInteriorAngle" in javascript
    assert 'customPreview.addEventListener("pointermove"' in javascript
    assert "OpenPINNImageTrace.clientPointToViewBox" in javascript
    assert "OpenPINNImageTrace.edgeMeasurements" in javascript
    assert 'data-polygon-edge-label="true"' in javascript
    assert 'data-polygon-vertex-number="true"' in javascript
    assert "function validatePolygon(rawOuter, rawHoles, sourceName)" in javascript
    assert "function boundaryLabel(boundary)" in javascript
    assert "op.NeumannBC(" in javascript
    assert "op.RobinBC(" in javascript
    assert "var xOrigin" in javascript
    assert "var xZero" in javascript
    assert "var yZero" in javascript
    assert "var fieldGrid" in javascript
    assert 'entry.condition.type !== "dirichlet"' in javascript
    assert '"x∈[" + settings.xMin' in javascript
    assert "var domainAspect" in javascript
    assert '["helmholtz", "Helmholtz 制造解"]' in javascript
    assert '["allen_cahn", "Allen–Cahn"]' in javascript
    assert '["burgers", "Burgers"]' in javascript
    assert "op.RectangleWithCircularHoles(" in javascript
    assert "op.Disk(" in javascript
    assert "op.Ellipse(" in javascript
    assert "op.RoundedRectangle(" in javascript
    assert "op.ClosedSpline(" in javascript
    assert "def curved_boundary_normal(points):" in javascript
    assert "op.Polygon(POLYGON_VERTICES, holes=POLYGON_HOLES)" in javascript
    assert "def polygon_boundary_normal(points):" in javascript
    assert "function polygonBoundaryDefinitions()" in javascript
    assert "function renderPolygonBoundaries()" in javascript
    assert 'lines.push("def polygon_edge_selector(ring_index, edge_index):")' in javascript
    assert 'lines.push("def polygon_ring_selector(ring_index):")' in javascript
    assert 'lines.push("def make_circle_hole(center_x, center_y, radius, count, start_angle):")' in javascript
    assert 'data-role="custom-dxf-circle-editor"' in extension
    assert 'data-role="custom-dxf-circle-radius"' in extension
    assert 'boundary=\' + selector' in javascript
    assert 'name="boundary_\' + boundary.side' in javascript
    assert "customPolygonBoundaryState" in javascript
    assert 'function isPolygonGeometry(geometry)' in javascript
    assert 'window.OpenPINNDxfImport.parseDetailed(String(reader.result), { unitScale: unitScale })' in javascript
    assert 'data-role="custom-dxf-unit-scale"' in extension
    assert 'data-role="{prefix}-enable"' in extension
    assert 'prefix="observation"' in extension
    assert 'prefix="reference"' in extension
    assert "_observation_panel_html()" in extension
    assert "_reference_panel_html()" in extension
    assert 'op.evaluate_reference_csv(' in javascript
    assert "condition.type = typeSelect.value" in javascript
    assert "condition[input.dataset.boundaryKey] = Number(input.value)" in javascript
    assert "renderCustomPreview();\n          updateCode();" in javascript
    assert "customPolygonData.outer = rings[0]" in javascript
    assert "op.SpaceTimeDomain(SPACE, T_MIN, T_MAX)" in javascript
    assert "op.InitialCondition(example_field" in javascript
    assert "def equation(x, u):" in javascript
    assert "PROBLEM = op.PDEProblem(" in javascript
    assert "op.DirichletBC" in javascript
    assert '"residual_names": ("pde",)' in javascript
    assert 'data-role="evaluation-grid"' not in extension
    assert "EVALUATION_GRID" not in javascript
    assert "周期边界需要成对约束" in extension


def test_browser_image_contour_module() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser contour-module test.")
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [node, str(project_root / "tests/js/polygon_image_trace.test.cjs")],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "POLYGON_IMAGE_TRACE_OK" in completed.stdout


def test_browser_dxf_geometry_import_module() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser DXF-module test.")
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [node, str(project_root / "tests/js/dxf_geometry_import.test.cjs")],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DXF_GEOMETRY_IMPORT_OK" in completed.stdout


def test_problem_designer_sources_include_report_only_workflow() -> None:
    project_root = Path(__file__).resolve().parents[2]
    page = (project_root / "docs/problem_designer.rst").read_text(encoding="utf-8")
    extension = (project_root / "docs/_ext/openpinn_method_catalog.py").read_text(
        encoding="utf-8"
    )
    javascript = (project_root / "docs/_static/problem_designer.js").read_text(
        encoding="utf-8"
    )

    assert ".. openpinn-problem-designer::" in page
    assert "不会把方程注册进配置器" in page
    assert "ProblemDesignerDirective" in extension
    assert "op.check_method_compatibility(" in javascript
    assert "print(REPORT.to_text())" in javascript
    assert 'value="dynamic"' in extension
    assert 'value="heterogeneous"' in extension
