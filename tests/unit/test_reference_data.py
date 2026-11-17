from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

import openpinn as op


def _result(tmp_path: Path, input_dim: int, weights: list[float]) -> op.ExperimentResult:
    model = torch.nn.Linear(input_dim, 1, bias=False).double()
    with torch.no_grad():
        model.weight[:] = torch.tensor([weights], dtype=torch.float64)
    return op.ExperimentResult(
        metrics={}, history=[],
        predictions={"coordinates": np.zeros((2, input_dim)), "prediction": np.zeros((2, 1))},
        output_dir=tmp_path, model=model, compute_ledger={},
    )


def _csv(path: Path, headers: list[str], rows: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def test_reference_csv_exact_and_saved_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "external.csv"
    _csv(path, ["x (mm)", "pressure (kPa)"], [[0, 0], [500, 1], [1000, 2], [1000, 2]])
    result = _result(tmp_path / "run", 1, [2])
    problem = op.PDEProblem(op.Interval(0, 1), lambda x, u: u * 0, name="test")
    metrics = op.evaluate_reference_csv(
        result, path, problem=problem,
        coordinate_columns=("x (mm)",), coordinate_scales=(0.001,),
        value_column="pressure (kPa)",
    )
    assert metrics["rmse"] == pytest.approx(0)
    assert metrics["r2"] == pytest.approx(1)
    assert (result.output_dir / "reference_comparison.svg").is_file()
    assert (result.output_dir / "reference_predictions.csv").is_file()
    metadata = json.loads((result.output_dir / "reference_metrics.json").read_text())
    assert metadata["sample_count"] == 3
    assert metadata["identical_duplicate_rows_dropped"] == 1
    assert metadata["evaluation_only"] is True
    assert result.metrics["external_reference_rmse"] == pytest.approx(0)
    with (result.output_dir / "reference_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))
    assert float(saved_rows[1]["x (mm)"]) == 500


def test_reference_csv_rejects_bad_units_and_conflicting_rows(tmp_path: Path) -> None:
    path = tmp_path / "outside.csv"
    _csv(path, ["x", "u"], [[0, 0], [2, 4]])
    result = _result(tmp_path / "run", 1, [2])
    problem = op.PDEProblem(op.Interval(0, 1), lambda x, u: u * 0)
    with pytest.raises(ValueError, match="outside"):
        op.evaluate_reference_csv(result, path, problem=problem)
    with pytest.raises(ValueError, match="coordinate_scales"):
        op.evaluate_reference_csv(result, path, coordinate_scales=(0,))
    _csv(path, ["x", "u"], [[0, 0], [0, 1]])
    with pytest.raises(ValueError, match="different value"):
        op.evaluate_reference_csv(result, path)


def test_reference_csv_two_dimensional_time_slice(tmp_path: Path) -> None:
    path = tmp_path / "field.csv"
    rows = [[x, y, t, x + y + t] for t in (0, 1) for y in (0, 1) for x in (0, 1)]
    _csv(path, ["x", "y", "t", "p"], rows)
    result = _result(tmp_path / "run", 3, [1, 1, 1])
    domain = op.SpaceTimeDomain(
        op.RectangleWithCircularHoles(lower=(0, 0), upper=(1, 1), holes=()),
        op.Interval(0, 1),
    )
    problem = op.PDEProblem(domain, lambda x, u: u * 0)
    metrics = op.evaluate_reference_csv(
        result, path, problem=problem,
        coordinate_columns=("x", "y", "t"), value_column="p",
    )
    assert metrics["rmse"] == pytest.approx(0)
    svg = (result.output_dir / "reference_comparison.svg").read_text()
    assert "t=1" in svg
    assert "metrics use all rows" in svg
    ET.fromstring(svg)


def test_reference_csv_scales_model_and_reference_independently(tmp_path: Path) -> None:
    path = tmp_path / "physical.csv"
    _csv(path, ["x_m", "p_Pa"], [[0, 0], [40, 20000]])
    result = _result(tmp_path / "run", 1, [1])
    metrics = op.evaluate_reference_csv(
        result, path, coordinate_columns=("x_m",), coordinate_scales=(1 / 40,),
        value_column="p_Pa", value_scale=0.001, prediction_scale=20,
    )
    assert metrics["rmse"] == pytest.approx(0)
    saved = json.loads((result.output_dir / "reference_metrics.json").read_text())
    assert saved["prediction_scale"] == 20


def test_reference_csv_rejects_points_inside_hole(tmp_path: Path) -> None:
    path = tmp_path / "inside_hole.csv"
    _csv(path, ["x", "y", "u"], [[0.2, 0.2, 0], [0.5, 0.5, 0]])
    result = _result(tmp_path / "run", 2, [0, 0])
    domain = op.RectangleWithCircularHoles(
        lower=(0, 0), upper=(1, 1), holes=((0.5, 0.5, 0.1),)
    )
    problem = op.PDEProblem(domain, lambda x, u: u * 0)
    with pytest.raises(ValueError, match="outside"):
        op.evaluate_reference_csv(
            result, path, problem=problem,
            coordinate_columns=("x", "y"), value_column="u",
        )


def test_reference_csv_regular_grid_renders_cells(tmp_path: Path) -> None:
    path = tmp_path / "grid.csv"
    rows = [[x / 7, y / 7, (x + y) / 7] for y in range(8) for x in range(8)]
    _csv(path, ["x", "y", "u"], rows)
    result = _result(tmp_path / "run", 2, [1, 1])
    op.evaluate_reference_csv(result, path, coordinate_columns=("x", "y"), value_column="u")
    svg = (result.output_dir / "reference_comparison.svg").read_text()
    assert svg.count('<rect x="') > 100
    ET.fromstring(svg)


def test_comsol_style_headerless_spreadsheet(tmp_path: Path) -> None:
    sample = Path(__file__).resolve().parents[2] / "examples/references/comsol_spreadsheet_example.txt"
    result = _result(tmp_path / "run", 2, [1, 1])
    metrics = op.evaluate_reference_csv(
        result, sample, coordinate_columns=("#1", "#2"), value_column="#3",
        time_axis=False,
    )
    assert metrics["rmse"] == pytest.approx(0)
    assert metrics["relative_l2"] == pytest.approx(0)


def test_single_exported_time_slice_is_aligned_to_model_time(tmp_path: Path) -> None:
    path = tmp_path / "slice.csv"
    _csv(path, ["x_m", "y_m", "p_kPa"], [[0, 0, 10], [40, 0, 11], [0, 40, 11], [40, 40, 12]])
    result = _result(tmp_path / "run", 3, [1, 1, 1])
    metrics = op.evaluate_reference_csv(
        result, path, coordinate_columns=("x_m", "y_m"), value_column="p_kPa",
        coordinate_scales=(1 / 40, 1 / 40, 1 / 40), fixed_time=20,
        prediction_scale=1, time_axis=True,
    )
    # Model x/40 + y/40 + t/40; reference is intentionally different but aligned.
    assert metrics["rmse"] > 0
    with (result.output_dir / "reference_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(float(row["t"]) == 20 for row in rows)
    assert float(rows[1]["prediction"]) == pytest.approx(1.5)
    svg = (result.output_dir / "reference_comparison.svg").read_text()
    assert "t=20" in svg


def test_operator_observation_loader_reuses_columns_scales_and_fixed_time(tmp_path: Path) -> None:
    path = tmp_path / "operator.csv"
    _csv(path, ["x_mm", "u_kpa"], [[0, 2], [500, 3], [1000, 4]])
    observation = op.load_operator_observation_csv(
        path,
        coordinate_columns=("x_mm",),
        coordinate_scales=(0.001, 2.0),
        value_column="u_kpa",
        value_scale=1000.0,
        fixed_time=0.25,
    )
    assert observation.coordinates.shape == (3, 2)
    assert observation.coordinates[:, 0].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert observation.coordinates[:, 1].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert observation.values[:, 0].tolist() == pytest.approx([2000, 3000, 4000])
