"""Evaluate a saved PINN model against an independent tabular reference field.

Reference rows are never supplied to the optimizer.  Coordinates are evaluated
directly at the exported sample locations; no interpolation or training-grid
reuse is hidden in the reported errors.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from itertools import chain
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from .artifacts import ArtifactWriter
from .metrics import max_absolute_error, mean_absolute_error, r2_score, relative_l2_error, rmse

if TYPE_CHECKING:
    from .experiment import ExperimentResult
    from .problems import PDEProblem


def _read_csv(
    path: Path,
    coordinate_columns: Sequence[str | int],
    value_column: str | int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Reference CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        meaningful = (
            (number, line) for number, line in enumerate(handle, start=1)
            if line.strip() and not line.lstrip().startswith(("%", "#"))
        )
        first = next(meaningful, None)
        if first is None:
            raise ValueError("Reference table has no data rows.")
        delimiter = max((",", ";", "\t"), key=first[1].count)
        if first[1].count(delimiter) == 0:
            delimiter = None

        def cells(line: str) -> list[str]:
            if delimiter is None:
                return line.split()
            return [item.strip() for item in next(csv.reader([line], delimiter=delimiter))]

        first_cells = cells(first[1])
        try:
            [float(item) for item in first_cells]
            headerless = True
        except ValueError:
            headerless = False
        headers = (
            [f"#{index + 1}" for index in range(len(first_cells))]
            if headerless else [item.strip() for item in first_cells]
        )
        if len(set(headers)) != len(headers):
            raise ValueError("Reference table has repeated column headers.")

        def column_index(selector: str | int) -> int:
            if isinstance(selector, int) and not isinstance(selector, bool):
                index = selector
            elif str(selector).strip().startswith("#"):
                try:
                    index = int(str(selector).strip()[1:]) - 1
                except ValueError as exc:
                    raise ValueError(f"Invalid column selector {selector!r}.") from exc
            else:
                name = str(selector).strip()
                if name not in headers:
                    raise ValueError(f"Reference table missing column {name!r}; available: {headers}")
                index = headers.index(name)
            if not 0 <= index < len(headers):
                raise ValueError(f"Column {selector!r} is outside the table with {len(headers)} columns.")
            return index

        required = [column_index(name) for name in [*coordinate_columns, value_column]]
        if len(set(required)) != len(required):
            raise ValueError("Coordinate and value columns must be distinct.")
        rows: dict[tuple[float, ...], float] = {}
        duplicate_count = 0
        data_rows = chain([first], meaningful) if headerless else meaningful
        for line_number, line in data_rows:
            row = cells(line)
            if len(row) != len(headers):
                raise ValueError(f"Reference table line {line_number} has {len(row)} cells; expected {len(headers)}.")
            try:
                point = tuple(float(row[index]) for index in required[:-1])
                value = float(row[required[-1]])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Reference table line {line_number} has an invalid number.") from exc
            if not all(math.isfinite(number) for number in (*point, value)):
                raise ValueError(f"Reference table line {line_number} has NaN or infinity.")
            if point in rows:
                if not math.isclose(rows[point], value, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Reference table line {line_number} repeats a coordinate with a different value.")
                duplicate_count += 1
                continue
            rows[point] = value
    if len(rows) < 2:
        raise ValueError("Reference table needs at least two distinct sample points.")
    return np.asarray(list(rows), dtype=np.float64), np.asarray(list(rows.values()), dtype=np.float64), duplicate_count


@dataclass(frozen=True)
class OperatorObservation:
    """One scalar operator-training field loaded from a local CSV file."""

    coordinates: torch.Tensor
    values: torch.Tensor
    source: Path
    duplicate_count: int = 0


def load_operator_observation_csv(
    path: str | Path,
    *,
    coordinate_columns: Sequence[str | int] = ("x",),
    value_column: str | int = "u",
    coordinate_scales: Sequence[float] | None = None,
    value_scale: float = 1.0,
    fixed_time: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> OperatorObservation:
    """Load local CSV observations for designed-problem operator training.

    The parser and column conventions are identical to
    :func:`evaluate_reference_csv`.  Loading is explicit and local; this API
    performs no upload, interpolation, or optimizer step.
    """

    source = Path(path).expanduser().resolve()
    coordinates, values, duplicates = _read_csv(
        source, tuple(coordinate_columns), value_column
    )
    if fixed_time is not None:
        if not math.isfinite(fixed_time):
            raise ValueError("fixed_time must be finite.")
        coordinates = np.column_stack(
            (coordinates, np.full(len(coordinates), fixed_time))
        )
    scales = (
        np.ones(coordinates.shape[1], dtype=np.float64)
        if coordinate_scales is None
        else np.asarray(coordinate_scales, dtype=np.float64)
    )
    if scales.shape != (coordinates.shape[1],) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError(
            "coordinate_scales must contain one positive finite scale per model coordinate."
        )
    if not math.isfinite(value_scale) or value_scale == 0:
        raise ValueError("value_scale must be finite and non-zero.")
    return OperatorObservation(
        coordinates=torch.as_tensor(coordinates * scales, dtype=dtype),
        values=torch.as_tensor(values * value_scale, dtype=dtype).reshape(-1, 1),
        source=source,
        duplicate_count=duplicates,
    )


def _color(value: float, low: float, high: float) -> str:
    fraction = 0.5 if high <= low else max(0.0, min(1.0, (value - low) / (high - low)))
    red = round(28 + 213 * fraction)
    green = round(94 + 95 * (1 - abs(2 * fraction - 1)))
    blue = round(224 - 194 * fraction)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _comparison_svg(
    coordinates: np.ndarray, reference: np.ndarray, prediction: np.ndarray, *, has_time: bool
) -> str:
    """Render measured points without implying a smoother reference than supplied."""

    dimension = coordinates.shape[1]
    spatial_dimension = dimension - 1 if has_time else dimension
    selected = np.arange(len(coordinates))
    slice_note = ""
    if dimension > spatial_dimension:
        latest = float(np.max(coordinates[:, -1]))
        selected = np.flatnonzero(np.isclose(coordinates[:, -1], latest, rtol=1e-9, atol=1e-12))
        slice_note = f" (t={latest:.6g}; metrics use all rows)"
    if len(selected) > 6000:
        selected = selected[np.linspace(0, len(selected) - 1, 6000, dtype=int)]
        slice_note += " (plot displays up to 6000 points)"
    positions = coordinates[selected, :spatial_dimension]
    values = [reference[selected], prediction[selected], np.abs(prediction[selected] - reference[selected])]
    color_low = float(min(np.min(values[0]), np.min(values[1])))
    color_high = float(max(np.max(values[0]), np.max(values[1])))
    error_high = float(np.max(values[2]))
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="510" viewBox="0 0 1260 510">',
        '<rect width="1260" height="510" fill="white"/>',
        '<text x="35" y="38" font-family="sans-serif" font-size="23" fill="#173248">Independent reference comparison</text>',
        f'<text x="35" y="64" font-family="sans-serif" font-size="13" fill="#526b7d">Values at exported sample coordinates{html.escape(slice_note)}</text>',
    ]
    labels = ("Reference", "Prediction", "Absolute error")
    for panel, field in enumerate(values):
        left, top, width, height = 35 + panel * 410, 115, 350, 320
        pieces.append(f'<text x="{left}" y="97" font-family="sans-serif" font-weight="bold" font-size="18" fill="#173248">{labels[panel]}</text>')
        pieces.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f5f8fa" stroke="#b6c7d5"/>')
        low, high = (0.0, error_high) if panel == 2 else (color_low, color_high)
        x = positions[:, 0]
        x_min, x_max = float(np.min(x)), float(np.max(x))
        x_span = max(x_max - x_min, 1e-12)
        px = left + 8 + (x - x_min) / x_span * (width - 16)
        if spatial_dimension == 1:
            y_min, y_max = float(np.min(field)), float(np.max(field))
            y_span = max(y_max - y_min, 1e-12)
            py = top + height - 8 - (field - y_min) / y_span * (height - 16)
            order = np.argsort(x)
            points = " ".join(f"{px[index]:.2f},{py[index]:.2f}" for index in order)
            pieces.append(f'<polyline points="{points}" fill="none" stroke="#246a9b" stroke-width="2"/>')
        else:
            y = positions[:, 1]
            y_min, y_max = float(np.min(y)), float(np.max(y))
            y_span = max(y_max - y_min, 1e-12)
            py = top + height - 8 - (y - y_min) / y_span * (height - 16)
            unique_x, unique_y = np.unique(x), np.unique(y)
            regular_grid = (
                len(unique_x) >= 8 and len(unique_y) >= 8
                and len(selected) >= 0.5 * len(unique_x) * len(unique_y)
                and np.allclose(np.diff(unique_x), np.diff(unique_x)[0], rtol=1e-4)
                and np.allclose(np.diff(unique_y), np.diff(unique_y)[0], rtol=1e-4)
            )
            cell_width = (width - 16) / max(len(unique_x) - 1, 1)
            cell_height = (height - 16) / max(len(unique_y) - 1, 1)
            # Draw only measured grid cells; missing cells (for example holes) stay blank.
            for point_x, point_y, value in zip(px, py, field, strict=True):
                color = _color(float(value), low, high)
                if regular_grid:
                    pieces.append(f'<rect x="{point_x - cell_width / 2:.2f}" y="{point_y - cell_height / 2:.2f}" width="{cell_width + 0.3:.2f}" height="{cell_height + 0.3:.2f}" fill="{color}"/>')
                else:
                    pieces.append(f'<circle cx="{point_x:.2f}" cy="{point_y:.2f}" r="2.7" fill="{color}"/>')
        pieces.append(f'<text x="{left}" y="{top + height + 24}" font-family="sans-serif" font-size="12" fill="#526b7d">min {low:.5g}   max {high:.5g}</text>')
    pieces.append('<text x="35" y="490" font-family="sans-serif" font-size="12" fill="#526b7d">Regular-grid cells and irregular-grid points show supplied values only; no reference values are interpolated.</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def evaluate_reference_csv(
    result: ExperimentResult,
    path: str | Path,
    *,
    coordinate_columns: Sequence[str | int] = ("x",),
    value_column: str | int = "u",
    problem: PDEProblem | None = None,
    field_index: int = 0,
    coordinate_scales: Sequence[float] | None = None,
    value_scale: float = 1.0,
    prediction_scale: float = 1.0,
    time_axis: bool | None = None,
    fixed_time: float | None = None,
    batch_size: int = 4096,
) -> dict[str, float]:
    """Compare an independent CSV/COMSOL Spreadsheet field and save artifacts.

    ``coordinate_columns`` must follow model input order (for example x,y,t).
    Named headers and 1-based selectors such as ``"#1"`` are supported.
    A COMSOL Spreadsheet table may have ``%`` comment lines and no header;
    comma, semicolon, tab and whitespace separators are recognized.
    For one exported time slice without a time column, pass spatial columns
    and ``fixed_time`` in the exported time unit; its scale is still the last
    entry of ``coordinate_scales``.
    Coordinate scales convert exported coordinates into model input units.
    ``value_scale`` and ``prediction_scale`` independently convert the
    exported and model values into the same *reported* physical unit.
    All supplied coordinates must belong to ``problem.domain`` when supplied;
    out-of-domain samples fail loudly instead of biasing metrics by filtering.
    """

    source = Path(path).expanduser().resolve()
    selectors = tuple(coordinate_columns)
    names = tuple(str(name) for name in selectors)
    input_dim = int(np.asarray(result.predictions["coordinates"]).shape[1])
    if fixed_time is not None and (input_dim < 2 or time_axis is False):
        raise ValueError("fixed_time requires a time-dependent model.")
    if fixed_time is not None and not math.isfinite(fixed_time):
        raise ValueError("fixed_time must be finite.")
    required_columns = input_dim - (fixed_time is not None)
    if len(names) != required_columns:
        raise ValueError(f"Expected {required_columns} coordinate columns in model input order, got {len(names)}.")
    output_dim = int(np.asarray(result.predictions["prediction"]).shape[1])
    if not 0 <= field_index < output_dim:
        raise ValueError(f"field_index must be between 0 and {output_dim - 1}.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    scales = np.ones(input_dim, dtype=np.float64) if coordinate_scales is None else np.asarray(coordinate_scales, dtype=np.float64)
    if scales.shape != (input_dim,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("coordinate_scales must contain one positive finite scale per coordinate.")
    if not math.isfinite(value_scale) or value_scale == 0:
        raise ValueError("value_scale must be finite and non-zero.")
    if not math.isfinite(prediction_scale) or prediction_scale == 0:
        raise ValueError("prediction_scale must be finite and non-zero.")
    exported_coordinates, reference, duplicate_count = _read_csv(source, selectors, value_column)
    if fixed_time is not None:
        exported_coordinates = np.column_stack(
            (exported_coordinates, np.full(len(exported_coordinates), fixed_time))
        )
        names = (*names, "t")
    model_coordinates = exported_coordinates * scales
    reference *= value_scale
    if problem is not None:
        in_domain = problem.domain.contains(torch.as_tensor(model_coordinates, dtype=torch.float64), atol=1e-7)
        outside = int((~in_domain).sum().item())
        if outside:
            raise ValueError(f"Reference has {outside} coordinates outside the problem domain; check geometry, time and units.")
    parameter = next(result.model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32
    predictions = []
    was_training = result.model.training
    result.model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(model_coordinates), batch_size):
                tensor = torch.as_tensor(model_coordinates[start:start + batch_size], device=device, dtype=dtype)
                output = result.model(tensor)
                if output.ndim != 2 or output.shape[1] != output_dim:
                    raise ValueError("Model output shape changed since training.")
                predictions.append(output[:, field_index].detach().cpu().numpy().astype(np.float64))
    finally:
        result.model.train(was_training)
    prediction = np.concatenate(predictions) * prediction_scale
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Model prediction on reference coordinates contains NaN or infinity.")
    metrics = {
        "relative_l2": relative_l2_error(prediction, reference),
        "rmse": rmse(prediction, reference),
        "mae": mean_absolute_error(prediction, reference),
        "max_absolute_error": max_absolute_error(prediction, reference),
        "r2": r2_score(prediction, reference),
    }
    writer = ArtifactWriter(result.output_dir)
    writer.write_json("reference_metrics.json", {
        **metrics,
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_count": len(exported_coordinates),
        "identical_duplicate_rows_dropped": duplicate_count,
        "coordinate_columns": names,
        "coordinate_scales": scales.tolist(),
        "value_column": value_column,
        "value_scale": value_scale,
        "prediction_scale": prediction_scale,
        "field_index": field_index,
        "time_axis": time_axis,
        "fixed_time": fixed_time,
        "evaluation_only": True,
        "interpolation": False,
        "metric_weighting": "uniform_per_distinct_sample_point",
    })
    with (writer.output_dir / "reference_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow([*names, "reference", "prediction", "absolute_error"])
        for point, actual, predicted in zip(exported_coordinates, reference, prediction, strict=True):
            csv_writer.writerow([*point, actual, predicted, abs(predicted - actual)])
    (writer.output_dir / "reference_comparison.svg").write_text(
        _comparison_svg(
            exported_coordinates, reference, prediction,
            has_time=(
                bool(time_axis) if time_axis is not None
                else fixed_time is not None
                or bool(problem is not None and hasattr(problem.domain, "time_index"))
                or re.match(r"^(?:t|time)\b", names[-1].strip(), flags=re.IGNORECASE) is not None
                or input_dim == 3
            ),
        ), encoding="utf-8"
    )
    result.metrics.update({f"external_reference_{name}": value for name, value in metrics.items()})
    writer.write_metrics(result.metrics)
    manifest_path = writer.output_dir / "manifest.json"
    manifest_metadata = {}
    if manifest_path.is_file():
        manifest_metadata = json.loads(manifest_path.read_text(encoding="utf-8")).get("metadata", {})
    writer.write_manifest(metadata=manifest_metadata)
    return metrics
