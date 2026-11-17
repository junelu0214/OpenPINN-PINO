"""Small shared helpers for transparent two-variant method comparisons."""

from __future__ import annotations

import csv
import html
import json
import math
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


COMPARISON_FIELDS = (
    "method",
    "status",
    "relative_l2_error",
    "max_absolute_error",
    "rmse",
    "pde_residual_rmse",
    "boundary_rmse",
    "initial_rmse",
    "interface_solution_jump",
    "interface_flux_jump",
    "elapsed_seconds",
    "parameter_count",
    "trainable_parameter_count",
    "seed",
    "device",
    "optimizer",
    "closure_calls",
    "failure_reason",
)


def result_summary(method: str, result: Any) -> dict[str, Any]:
    """Return the common metrics used by every method comparison."""

    metrics = result.metrics
    ledger = result.compute_ledger
    initial_keys = [key for key in metrics if key.endswith("initial_rmse")]
    boundary_keys = [key for key in metrics if key.endswith("boundary_rmse")]
    return {
        "method": method,
        "status": "completed",
        "relative_l2_error": metrics.get("relative_l2_error"),
        "max_absolute_error": metrics.get("max_absolute_error"),
        "rmse": metrics.get("rmse"),
        "pde_residual_rmse": metrics.get("pde_residual_rmse"),
        "boundary_rmse": metrics.get(
            "boundary_rmse", metrics.get(boundary_keys[0]) if boundary_keys else None
        ),
        "initial_rmse": metrics.get(
            "initial_rmse", metrics.get(initial_keys[0]) if initial_keys else None
        ),
        "interface_solution_jump": metrics.get("interface_solution_jump"),
        "interface_flux_jump": metrics.get("interface_flux_jump"),
        "elapsed_seconds": ledger.get("elapsed_seconds"),
        "parameter_count": ledger.get("parameter_count"),
        "trainable_parameter_count": ledger.get("trainable_parameter_count"),
        "seed": ledger.get("seed"),
        "device": ledger.get("device"),
        "optimizer": ledger.get("optimizer"),
        "closure_calls": ledger.get("closure_calls"),
        "failure_reason": None,
        "output_dir": str(result.output_dir),
    }


def failure_summary(method: str, output_dir: Path, error: BaseException) -> dict[str, Any]:
    """Persist an exception before returning a failed comparison row."""

    output_dir.mkdir(parents=True, exist_ok=True)
    details = {
        "method": method,
        "status": "failed",
        "failure_reason": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "failure.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return details


def write_comparison_files(
    root: Path,
    comparison: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    source_note: str,
    limitations: Sequence[str] = (),
) -> None:
    """Write JSON, CSV and an explicit failure/limitation note."""

    (root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (root / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=COMPARISON_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row.get("status") != "completed"]
    lines = ["# Failure and limitation notes", "", "## Training failures", ""]
    if failures:
        lines.extend(
            f"- {row['method']}: {row.get('failure_reason', 'unknown failure')}"
            for row in failures
        )
    else:
        lines.append("- None. All comparison runs completed with finite saved outputs.")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is one fixed-seed comparison, not a statistical superiority claim.",
            "- Only the named method component changes between the compared runs.",
            f"- {source_note}",
        ]
    )
    lines.extend(f"- {item}" for item in limitations)
    (root / "failure_notes.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _finite_range(values: Sequence[np.ndarray], *, include_zero: bool = False) -> tuple[float, float]:
    merged = np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in values])
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return (0.0, 1.0)
    lower, upper = float(np.min(merged)), float(np.max(merged))
    if include_zero:
        lower, upper = min(0.0, lower), max(0.0, upper)
    if math.isclose(lower, upper):
        padding = max(1.0, abs(lower)) * 0.05
    else:
        padding = (upper - lower) * 0.06
    return lower - padding, upper + padding


def _polyline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    panel: tuple[float, float, float, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    color: str,
    dash: str | None = None,
) -> str:
    px, py, width, height = panel
    x0, x1 = x_range
    y0, y1 = y_range
    points: list[str] = []
    for x_value, y_value in zip(x, y, strict=True):
        if not (math.isfinite(float(x_value)) and math.isfinite(float(y_value))):
            continue
        sx = px + (float(x_value) - x0) / max(x1 - x0, 1e-12) * width
        sy = py + height - (float(y_value) - y0) / max(y1 - y0, 1e-12) * height
        points.append(f"{sx:.2f},{sy:.2f}")
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linejoin="round"{dash_attr}/>'
    )


def _panel(
    *,
    title: str,
    x: np.ndarray,
    curves: Sequence[tuple[str, np.ndarray, str, str | None]],
    panel: tuple[float, float, float, float],
    y_range: tuple[float, float] | None = None,
) -> str:
    px, py, width, height = panel
    x_range = _finite_range([x])
    selected_y = y_range or _finite_range(
        [values for _, values, _, _ in curves], include_zero=True
    )
    parts = [
        f'<text x="{px}" y="{py - 18}" class="panel-title">{html.escape(title)}</text>',
        f'<rect x="{px}" y="{py}" width="{width}" height="{height}" class="plot"/>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        sx, sy = px + fraction * width, py + fraction * height
        parts.append(f'<line x1="{sx}" y1="{py}" x2="{sx}" y2="{py + height}" class="grid"/>')
        parts.append(f'<line x1="{px}" y1="{sy}" x2="{px + width}" y2="{sy}" class="grid"/>')
    for index, (label, values, color, dash) in enumerate(curves):
        parts.append(
            _polyline(
                x,
                values,
                panel=panel,
                x_range=x_range,
                y_range=selected_y,
                color=color,
                dash=dash,
            )
        )
        legend_y = py + 18 + index * 20
        parts.append(f'<line x1="{px + 10}" y1="{legend_y}" x2="{px + 34}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{px + 41}" y="{legend_y + 4}" class="legend">{html.escape(label)}</text>')
    return "\n".join(parts)


def write_static_1d_figure(
    path: Path,
    baseline: Any,
    improved: Any,
    *,
    title: str,
    improved_label: str,
) -> None:
    """Save solution, absolute error and loss curves for a static 1D pair."""

    coordinates = np.asarray(baseline.predictions["coordinates"])[:, 0]
    reference = np.asarray(baseline.predictions["reference"])[:, 0]
    plain_prediction = np.asarray(baseline.predictions["prediction"])[:, 0]
    improved_prediction = np.asarray(improved.predictions["prediction"])[:, 0]
    plain_error = np.abs(plain_prediction - reference)
    improved_error = np.abs(improved_prediction - reference)
    plain_steps = np.asarray([record["step"] for record in baseline.history], dtype=float)
    improved_steps = np.asarray([record["step"] for record in improved.history], dtype=float)
    if not np.array_equal(plain_steps, improved_steps):
        raise ValueError("Compared histories must record identical training steps.")
    plain_loss = np.log10(
        np.maximum([record["total"] for record in baseline.history], 1e-30)
    )
    improved_loss = np.log10(
        np.maximum([record["total"] for record in improved.history], 1e-30)
    )
    panels = [
        _panel(
            title="Solution",
            x=coordinates,
            curves=(
                ("Exact", reference, "#111827", "7 5"),
                ("Plain PINN", plain_prediction, "#2563eb", None),
                (improved_label, improved_prediction, "#dc2626", None),
            ),
            panel=(70, 135, 330, 330),
        ),
        _panel(
            title="Absolute error",
            x=coordinates,
            curves=(
                ("Plain PINN", plain_error, "#2563eb", None),
                (improved_label, improved_error, "#dc2626", None),
            ),
            panel=(475, 135, 330, 330),
            y_range=(0.0, max(float(plain_error.max()), float(improved_error.max()), 1e-12) * 1.06),
        ),
        _panel(
            title="Training loss (log10)",
            x=plain_steps,
            curves=(
                ("Plain PINN", plain_loss, "#2563eb", None),
                (improved_label, improved_loss, "#dc2626", None),
            ),
            panel=(880, 135, 330, 330),
        ),
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="540" viewBox="0 0 1280 540">
<style>
text {{ font-family: Arial, sans-serif; fill: #111827; }}
.title {{ font-size: 25px; font-weight: 700; }}
.subtitle {{ font-size: 13px; fill: #475569; }}
.panel-title {{ font-size: 17px; font-weight: 700; }}
.plot {{ fill: #f8fafc; stroke: #94a3b8; }}
.grid {{ stroke: #e2e8f0; stroke-width: 1; }}
.legend {{ font-size: 11px; fill: #334155; }}
</style>
<rect width="1280" height="540" fill="#ffffff"/>
<text x="50" y="45" class="title">{html.escape(title)}</text>
<text x="50" y="72" class="subtitle">One fixed-seed run; every configured setting matches except the named method component.</text>
{"".join(panels)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_static_reference_figure(path: Path, result: Any, *, title: str) -> None:
    """Save exact solution, absolute error and loss for one static 1D run."""

    coordinates = np.asarray(result.predictions["coordinates"])[:, 0]
    reference = np.asarray(result.predictions["reference"])[:, 0]
    prediction = np.asarray(result.predictions["prediction"])[:, 0]
    error = np.abs(prediction - reference)
    steps = np.asarray([record["step"] for record in result.history], dtype=float)
    loss = np.log10(np.maximum([record["total"] for record in result.history], 1e-30))
    panels = [
        _panel(
            title="Solution",
            x=coordinates,
            curves=(
                ("Exact", reference, "#111827", "7 5"),
                ("Plain PINN", prediction, "#2563eb", None),
            ),
            panel=(70, 135, 330, 330),
        ),
        _panel(
            title="Absolute error",
            x=coordinates,
            curves=(("Plain PINN", error, "#dc2626", None),),
            panel=(475, 135, 330, 330),
            y_range=(0.0, max(float(error.max()), 1e-12) * 1.06),
        ),
        _panel(
            title="Training loss (log10)",
            x=steps,
            curves=(("Plain PINN", loss, "#2563eb", None),),
            panel=(880, 135, 330, 330),
        ),
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="540" viewBox="0 0 1280 540">
<style>
text {{ font-family: Arial, sans-serif; fill: #111827; }}
.title {{ font-size: 25px; font-weight: 700; }}
.subtitle {{ font-size: 13px; fill: #475569; }}
.panel-title {{ font-size: 17px; font-weight: 700; }}
.plot {{ fill: #f8fafc; stroke: #94a3b8; }}
.grid {{ stroke: #e2e8f0; stroke-width: 1; }}
.legend {{ font-size: 11px; fill: #334155; }}
</style>
<rect width="1280" height="540" fill="#ffffff"/>
<text x="50" y="45" class="title">{html.escape(title)}</text>
<text x="50" y="72" class="subtitle">One fixed-seed baseline; the exact solution is evaluation-only.</text>
{"".join(panels)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_dynamic_1d_figure(
    path: Path,
    baseline: Any,
    improved: Any,
    *,
    title: str,
    improved_label: str,
) -> None:
    """Save four reference slices, time-slice errors and training losses."""

    baseline_coordinates = np.asarray(baseline.predictions["coordinates"])
    improved_coordinates = np.asarray(improved.predictions["coordinates"])
    if not np.allclose(baseline_coordinates, improved_coordinates):
        raise ValueError("Compared predictions must use identical coordinates.")
    times = np.unique(baseline_coordinates[:, 1])
    if times.size != 4:
        raise ValueError("Dynamic comparison figures require exactly four time slices.")
    baseline_prediction = np.asarray(baseline.predictions["prediction"])[:, 0]
    improved_prediction = np.asarray(improved.predictions["prediction"])[:, 0]
    if "reference" not in baseline.predictions or "reference" not in improved.predictions:
        raise ValueError("Dynamic comparison figures require a saved reference field.")
    baseline_reference = np.asarray(baseline.predictions["reference"])[:, 0]
    improved_reference = np.asarray(improved.predictions["reference"])[:, 0]
    if not np.allclose(baseline_reference, improved_reference):
        raise ValueError("Compared runs must use the same reference field.")

    panels: list[str] = []
    locations = ((65, 125), (650, 125), (65, 425), (650, 425))
    baseline_slice_errors: list[float] = []
    improved_slice_errors: list[float] = []
    for time_value, (panel_x, panel_y) in zip(times, locations, strict=True):
        selected = np.isclose(baseline_coordinates[:, 1], time_value)
        order = np.argsort(baseline_coordinates[selected, 0])
        x_values = baseline_coordinates[selected, 0][order]
        reference_values = baseline_reference[selected][order]
        baseline_values = baseline_prediction[selected][order]
        improved_values = improved_prediction[selected][order]
        baseline_slice_errors.append(
            float(np.max(np.abs(baseline_values - reference_values)))
        )
        improved_slice_errors.append(
            float(np.max(np.abs(improved_values - reference_values)))
        )
        panels.append(
            _panel(
                title=f"Solution at t={float(time_value):.2f}",
                x=x_values,
                curves=(
                    ("Reference", reference_values, "#111827", "7 5"),
                    ("Plain PINN", baseline_values, "#2563eb", None),
                    (improved_label, improved_values, "#dc2626", None),
                ),
                panel=(panel_x, panel_y, 515, 220),
            )
        )

    baseline_steps = np.asarray(
        [record["step"] for record in baseline.history], dtype=float
    )
    improved_steps = np.asarray(
        [record["step"] for record in improved.history], dtype=float
    )
    if not np.array_equal(baseline_steps, improved_steps):
        raise ValueError("Compared histories must record identical training steps.")
    baseline_loss = np.log10(
        np.maximum([record["total"] for record in baseline.history], 1e-30)
    )
    improved_loss = np.log10(
        np.maximum([record["total"] for record in improved.history], 1e-30)
    )
    maximum_slice_error = max(
        max(baseline_slice_errors), max(improved_slice_errors), 1e-12
    )
    panels.append(
        _panel(
            title="Maximum absolute error by time slice",
            x=np.asarray(times, dtype=float),
            curves=(
                (
                    "Plain PINN",
                    np.asarray(baseline_slice_errors),
                    "#2563eb",
                    None,
                ),
                (
                    improved_label,
                    np.asarray(improved_slice_errors),
                    "#dc2626",
                    None,
                ),
            ),
            panel=(65, 725, 515, 225),
            y_range=(0.0, maximum_slice_error * 1.06),
        )
    )
    panels.append(
        _panel(
            title="Training loss (log10)",
            x=baseline_steps,
            curves=(
                ("Plain PINN", baseline_loss, "#2563eb", None),
                (improved_label, improved_loss, "#dc2626", None),
            ),
            panel=(650, 725, 515, 225),
        )
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1230" height="1010" viewBox="0 0 1230 1010">
<style>
text {{ font-family: Arial, sans-serif; fill: #111827; }}
.title {{ font-size: 25px; font-weight: 700; }}
.subtitle {{ font-size: 13px; fill: #475569; }}
.panel-title {{ font-size: 17px; font-weight: 700; }}
.plot {{ fill: #f8fafc; stroke: #94a3b8; }}
.grid {{ stroke: #e2e8f0; stroke-width: 1; }}
.legend {{ font-size: 11px; fill: #334155; }}
</style>
<rect width="1230" height="1010" fill="#ffffff"/>
<text x="50" y="45" class="title">{html.escape(title)}</text>
<text x="50" y="72" class="subtitle">One fixed-seed run; the independent reference is evaluation-only and every setting matches except the named method component.</text>
{"".join(panels)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_dynamic_reference_figure(
    path: Path,
    result: Any,
    *,
    title: str,
) -> None:
    """Save four exact-versus-PINN slices, absolute errors and loss."""

    coordinates = np.asarray(result.predictions["coordinates"])
    prediction = np.asarray(result.predictions["prediction"])[:, 0]
    reference = np.asarray(result.predictions["reference"])[:, 0]
    times = np.unique(coordinates[:, 1])
    if times.size != 4:
        raise ValueError("Dynamic reference figures require exactly four time slices.")

    panels: list[str] = []
    locations = ((65, 125), (650, 125), (65, 425), (650, 425))
    error_curves: list[tuple[str, np.ndarray, str, str | None]] = []
    error_x: np.ndarray | None = None
    colors = ("#2563eb", "#059669", "#dc2626", "#7c3aed")
    for time_value, (panel_x, panel_y), color in zip(
        times, locations, colors, strict=True
    ):
        selected = np.isclose(coordinates[:, 1], time_value)
        order = np.argsort(coordinates[selected, 0])
        x_values = coordinates[selected, 0][order]
        predicted_values = prediction[selected][order]
        reference_values = reference[selected][order]
        panels.append(
            _panel(
                title=f"Solution at t={float(time_value):.2f}",
                x=x_values,
                curves=(
                    ("Reference", reference_values, "#111827", "7 5"),
                    ("Plain PINN", predicted_values, "#2563eb", None),
                ),
                panel=(panel_x, panel_y, 515, 220),
            )
        )
        error_x = x_values
        error_curves.append(
            (
                f"t={float(time_value):.2f}",
                np.abs(predicted_values - reference_values),
                color,
                None,
            )
        )

    if error_x is None:
        raise ValueError("Dynamic reference figure received no evaluation points.")
    maximum_error = max(float(values.max()) for _, values, _, _ in error_curves)
    panels.append(
        _panel(
            title="Absolute error by time slice",
            x=error_x,
            curves=error_curves,
            panel=(65, 725, 515, 220),
            y_range=(0.0, max(maximum_error, 1e-12) * 1.06),
        )
    )
    steps = np.asarray([record["step"] for record in result.history], dtype=float)
    loss = np.log10(
        np.maximum([record["total"] for record in result.history], 1e-30)
    )
    panels.append(
        _panel(
            title="Training loss (log10)",
            x=steps,
            curves=(("Plain PINN", loss, "#2563eb", None),),
            panel=(650, 725, 515, 220),
        )
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1230" height="1010" viewBox="0 0 1230 1010">
<style>
text {{ font-family: Arial, sans-serif; fill: #111827; }}
.title {{ font-size: 25px; font-weight: 700; }}
.subtitle {{ font-size: 13px; fill: #475569; }}
.panel-title {{ font-size: 17px; font-weight: 700; }}
.plot {{ fill: #f8fafc; stroke: #94a3b8; }}
.grid {{ stroke: #e2e8f0; stroke-width: 1; }}
.legend {{ font-size: 11px; fill: #334155; }}
</style>
<rect width="1230" height="1010" fill="#ffffff"/>
<text x="50" y="45" class="title">{html.escape(title)}</text>
<text x="50" y="72" class="subtitle">The independent Cole–Hopf reference is used for evaluation only, never for training.</text>
{"".join(panels)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_static_1d_three_way_figure(
    path: Path,
    adam: Any,
    lbfgs: Any,
    staged: Any,
    *,
    title: str,
) -> None:
    """Save solution, error and normalized-progress losses for three optimizers."""

    coordinates = np.asarray(adam.predictions["coordinates"])[:, 0]
    reference = np.asarray(adam.predictions["reference"])[:, 0]
    variants = (
        ("Adam", adam, "#2563eb"),
        ("L-BFGS", lbfgs, "#059669"),
        ("Adam + L-BFGS", staged, "#dc2626"),
    )
    for _, result, _ in variants[1:]:
        if not np.allclose(result.predictions["coordinates"][:, 0], coordinates):
            raise ValueError("Compared predictions must use identical coordinates.")
    solution_curves = [("Exact", reference, "#111827", "7 5")]
    error_curves = []
    progress = np.linspace(0.0, 100.0, 101)
    loss_curves = []
    for label, result, color in variants:
        prediction = np.asarray(result.predictions["prediction"])[:, 0]
        solution_curves.append((label, prediction, color, None))
        error_curves.append((label, np.abs(prediction - reference), color, None))
        raw_loss = np.log10(
            np.maximum([record["total"] for record in result.history], 1e-30)
        )
        raw_progress = np.linspace(0.0, 100.0, raw_loss.size)
        loss_curves.append(
            (label, np.interp(progress, raw_progress, raw_loss), color, None)
        )
    max_error = max(float(values.max()) for _, values, _, _ in error_curves)
    panels = [
        _panel(
            title="Solution",
            x=coordinates,
            curves=solution_curves,
            panel=(70, 135, 330, 330),
        ),
        _panel(
            title="Absolute error",
            x=coordinates,
            curves=error_curves,
            panel=(475, 135, 330, 330),
            y_range=(0.0, max(max_error, 1e-12) * 1.06),
        ),
        _panel(
            title="Training loss vs progress (log10)",
            x=progress,
            curves=loss_curves,
            panel=(880, 135, 330, 330),
        ),
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="540" viewBox="0 0 1280 540">
<style>
text {{ font-family: Arial, sans-serif; fill: #111827; }}
.title {{ font-size: 25px; font-weight: 700; }}
.subtitle {{ font-size: 13px; fill: #475569; }}
.panel-title {{ font-size: 17px; font-weight: 700; }}
.plot {{ fill: #f8fafc; stroke: #94a3b8; }}
.grid {{ stroke: #e2e8f0; stroke-width: 1; }}
.legend {{ font-size: 11px; fill: #334155; }}
</style>
<rect width="1280" height="540" fill="#ffffff"/>
<text x="50" y="45" class="title">{html.escape(title)}</text>
<text x="50" y="72" class="subtitle">Loss curves use normalized outer-step progress; compare closure calls and wall time in the table.</text>
{"".join(panels)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")
