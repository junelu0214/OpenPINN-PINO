"""Fair one-setting comparison of plain and Fourier-feature Poisson PINNs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import traceback
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import openpinn as op


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reference_runs" / "poisson_fourier.yaml"
COMPARISON_FIELDS = (
    "method",
    "status",
    "relative_l2_error",
    "max_absolute_error",
    "rmse",
    "pde_residual_rmse",
    "boundary_rmse",
    "elapsed_seconds",
    "parameter_count",
    "seed",
    "device",
    "failure_reason",
)


def _variant_config(config: Mapping[str, Any], feature: Mapping[str, Any]) -> dict[str, Any]:
    selected = deepcopy(dict(config))
    selected.setdefault("model", {})["feature"] = dict(feature)
    return selected


def _summary(method: str, result: Any) -> dict[str, Any]:
    metrics = result.metrics
    ledger = result.compute_ledger
    return {
        "method": method,
        "status": "completed",
        "relative_l2_error": metrics.get("relative_l2_error"),
        "max_absolute_error": metrics.get("max_absolute_error"),
        "rmse": metrics.get("rmse"),
        "pde_residual_rmse": metrics.get("pde_residual_rmse"),
        "boundary_rmse": metrics.get("boundary_rmse"),
        "elapsed_seconds": ledger.get("elapsed_seconds"),
        "parameter_count": ledger.get("parameter_count"),
        "seed": ledger.get("seed"),
        "device": ledger.get("device"),
        "failure_reason": None,
        "output_dir": str(result.output_dir),
    }


def _failure(method: str, output_dir: Path, error: BaseException) -> dict[str, Any]:
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


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _range(values: Sequence[np.ndarray], *, include_zero: bool = False) -> tuple[float, float]:
    merged = _finite(np.concatenate([np.asarray(value).reshape(-1) for value in values]))
    if merged.size == 0:
        return (0.0, 1.0)
    lower = float(np.min(merged))
    upper = float(np.max(merged))
    if include_zero:
        lower = min(0.0, lower)
        upper = max(0.0, upper)
    if math.isclose(lower, upper):
        padding = max(1.0, abs(lower)) * 0.05
        return (lower - padding, upper + padding)
    padding = (upper - lower) * 0.06
    return (lower - padding, upper + padding)


def _polyline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    panel: tuple[float, float, float, float],
    color: str,
    width: float = 2.0,
    dash: str | None = None,
) -> str:
    left, right, bottom, top = bounds
    px, py, pw, ph = panel
    usable_x = max(right - left, 1e-12)
    usable_y = max(top - bottom, 1e-12)
    points = []
    for x_value, y_value in zip(x, y):
        if not (math.isfinite(float(x_value)) and math.isfinite(float(y_value))):
            continue
        sx = px + (float(x_value) - left) / usable_x * pw
        sy = py + ph - (float(y_value) - bottom) / usable_y * ph
        points.append(f"{sx:.2f},{sy:.2f}")
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"'
        f'{dash_attribute}/>'
    )


def _panel(
    *,
    x: np.ndarray,
    curves: Sequence[tuple[str, np.ndarray, str, str | None]],
    title: str,
    x_label: str,
    y_label: str,
    panel: tuple[float, float, float, float],
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> str:
    px, py, pw, ph = panel
    selected_x_bounds = x_bounds or _range([x])
    selected_y_bounds = y_bounds or _range([curve[1] for curve in curves], include_zero=True)
    left, right = selected_x_bounds
    bottom, top = selected_y_bounds
    parts = [
        f'<text x="{px:.1f}" y="{py - 22:.1f}" class="panel-title">{html.escape(title)}</text>',
        f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="{ph:.1f}" class="plot-bg"/>',
    ]
    for index in range(5):
        fraction = index / 4
        sx = px + fraction * pw
        sy = py + fraction * ph
        x_value = left + fraction * (right - left)
        y_value = top - fraction * (top - bottom)
        parts.extend(
            [
                f'<line x1="{sx:.1f}" y1="{py:.1f}" x2="{sx:.1f}" y2="{py + ph:.1f}" class="grid"/>',
                f'<line x1="{px:.1f}" y1="{sy:.1f}" x2="{px + pw:.1f}" y2="{sy:.1f}" class="grid"/>',
                f'<text x="{sx:.1f}" y="{py + ph + 20:.1f}" text-anchor="middle" class="tick">{x_value:.2g}</text>',
                f'<text x="{px - 8:.1f}" y="{sy + 4:.1f}" text-anchor="end" class="tick">{y_value:.2g}</text>',
            ]
        )
    parts.extend(
        [
            f'<line x1="{px:.1f}" y1="{py + ph:.1f}" x2="{px + pw:.1f}" y2="{py + ph:.1f}" class="axis"/>',
            f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{py + ph:.1f}" class="axis"/>',
            f'<text x="{px + pw / 2:.1f}" y="{py + ph + 47:.1f}" text-anchor="middle" class="axis-label">{html.escape(x_label)}</text>',
            f'<text x="{px - 48:.1f}" y="{py + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 {px - 48:.1f} {py + ph / 2:.1f})" class="axis-label">{html.escape(y_label)}</text>',
        ]
    )
    bounds = (left, right, bottom, top)
    for _, values, color, dash in curves:
        parts.append(
            _polyline(
                x,
                values,
                bounds=bounds,
                panel=panel,
                color=color,
                dash=dash,
            )
        )
    legend_x = px + 10
    legend_y = py + 18
    for index, (label, _, color, dash) in enumerate(curves):
        y_position = legend_y + index * 20
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{legend_x:.1f}" y1="{y_position:.1f}" x2="{legend_x + 24:.1f}" y2="{y_position:.1f}" stroke="{color}" stroke-width="2.5"{dash_attribute}/>'
        )
        parts.append(
            f'<text x="{legend_x + 31:.1f}" y="{y_position + 4:.1f}" class="legend">{html.escape(label)}</text>'
        )
    return "\n".join(parts)


def _write_figure(path: Path, baseline: Any, fourier: Any) -> None:
    coordinates = np.asarray(baseline.predictions["coordinates"])[:, 0]
    reference = np.asarray(baseline.predictions["reference"])[:, 0]
    baseline_prediction = np.asarray(baseline.predictions["prediction"])[:, 0]
    fourier_prediction = np.asarray(fourier.predictions["prediction"])[:, 0]
    baseline_error = np.abs(baseline_prediction - reference)
    fourier_error = np.abs(fourier_prediction - reference)
    error_upper = max(
        float(np.max(baseline_error)), float(np.max(fourier_error)), 1e-12
    ) * 1.06

    baseline_steps = np.asarray([record["step"] for record in baseline.history])
    fourier_steps = np.asarray([record["step"] for record in fourier.history])
    if not np.array_equal(baseline_steps, fourier_steps):
        raise ValueError("Baseline and Fourier histories do not share the same steps.")
    baseline_loss = np.log10(
        np.maximum(np.asarray([record["total"] for record in baseline.history]), 1e-30)
    )
    fourier_loss = np.log10(
        np.maximum(np.asarray([record["total"] for record in fourier.history]), 1e-30)
    )

    panels = [
        _panel(
            x=coordinates,
            curves=(
                ("Exact", reference, "#111827", "7 5"),
                ("Plain PINN", baseline_prediction, "#2563eb", None),
                ("Fourier PINN", fourier_prediction, "#dc2626", None),
            ),
            title="Solution",
            x_label="x",
            y_label="u(x)",
            panel=(85, 155, 320, 330),
        ),
        _panel(
            x=coordinates,
            curves=(
                ("Plain PINN", baseline_error, "#2563eb", None),
                ("Fourier PINN", fourier_error, "#dc2626", None),
            ),
            title="Absolute error",
            x_label="x",
            y_label="|prediction - exact|",
            panel=(495, 155, 320, 330),
            y_bounds=(0.0, error_upper),
        ),
        _panel(
            x=baseline_steps,
            curves=(
                ("Plain PINN", baseline_loss, "#2563eb", None),
                ("Fourier PINN", fourier_loss, "#dc2626", None),
            ),
            title="Training loss",
            x_label="step",
            y_label="log10(total loss)",
            panel=(905, 155, 320, 330),
            x_bounds=(float(baseline_steps[0]), float(baseline_steps[-1])),
        ),
    ]
    baseline_error_value = float(baseline.metrics["relative_l2_error"])
    fourier_error_value = float(fourier.metrics["relative_l2_error"])
    baseline_time = float(baseline.compute_ledger["elapsed_seconds"])
    fourier_time = float(fourier.compute_ledger["elapsed_seconds"])
    baseline_parameters = int(baseline.compute_ledger["parameter_count"])
    fourier_parameters = int(fourier.compute_ledger["parameter_count"])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="650" viewBox="0 0 1280 650">
<style>
  text {{ font-family: Arial, Helvetica, sans-serif; fill: #111827; }}
  .title {{ font-size: 26px; font-weight: 700; }}
  .subtitle {{ font-size: 14px; fill: #4b5563; }}
  .panel-title {{ font-size: 18px; font-weight: 700; }}
  .plot-bg {{ fill: #f8fafc; stroke: #cbd5e1; }}
  .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
  .axis {{ stroke: #475569; stroke-width: 1.2; }}
  .tick {{ font-size: 11px; fill: #64748b; }}
  .axis-label {{ font-size: 12px; fill: #334155; }}
  .legend {{ font-size: 11px; fill: #334155; }}
  .card {{ fill: #f1f5f9; stroke: #cbd5e1; }}
  .card-title {{ font-size: 12px; font-weight: 700; fill: #475569; }}
  .card-value {{ font-size: 18px; font-weight: 700; }}
</style>
<rect width="1280" height="650" fill="#ffffff"/>
<text x="50" y="48" class="title">Poisson: plain PINN vs fixed Fourier features</text>
<text x="50" y="75" class="subtitle">One deterministic CPU run; all settings match except model.feature.</text>
{"".join(panels)}
<rect x="85" y="550" width="255" height="70" rx="8" class="card"/>
<text x="103" y="574" class="card-title">PLAIN PINN RELATIVE L2</text>
<text x="103" y="603" class="card-value">{baseline_error_value:.5g}</text>
<rect x="365" y="550" width="255" height="70" rx="8" class="card"/>
<text x="383" y="574" class="card-title">FOURIER PINN RELATIVE L2</text>
<text x="383" y="603" class="card-value">{fourier_error_value:.5g}</text>
<rect x="645" y="550" width="255" height="70" rx="8" class="card"/>
<text x="663" y="574" class="card-title">TRAINING TIME (PLAIN / FOURIER)</text>
<text x="663" y="603" class="card-value">{baseline_time:.3f}s / {fourier_time:.3f}s</text>
<rect x="925" y="550" width="255" height="70" rx="8" class="card"/>
<text x="943" y="574" class="card-title">PARAMETERS (PLAIN / FOURIER)</text>
<text x="943" y="603" class="card-value">{baseline_parameters} / {fourier_parameters}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_failure_notes(
    path: Path, rows: Sequence[Mapping[str, Any]], *, parameter_note: str | None
) -> None:
    failures = [row for row in rows if row.get("status") != "completed"]
    lines = ["# Failure and limitation notes", ""]
    if failures:
        lines.append("## Training failures")
        lines.append("")
        for failure in failures:
            lines.append(
                f"- {failure['method']}: {failure.get('failure_reason', 'unknown failure')}"
            )
    else:
        lines.extend(
            [
                "## Training failures",
                "",
                "- None. Both runs completed and all saved numeric outputs were finite.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is one fixed-seed CPU comparison, not a statistical superiority claim.",
            "- The two runs use the same PDE, points, widths, optimizer, steps and experiment seed.",
            "- Fourier features change the network input width, so trainable parameter counts differ.",
            "- No external repository code was copied or adapted for this comparison.",
        ]
    )
    if parameter_note:
        lines.append(f"- {parameter_note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(
    config_source: str | Path | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run both variants and preserve results even when one variant fails."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    method_config = op.load_config(config_source)
    feature_config = dict(method_config.get("model", {}).get("feature", {}))
    if str(feature_config.get("name", "")).lower() != "fourier":
        raise ValueError("The method config must select model.feature.name: fourier.")
    variants = {
        "plain_pinn": _variant_config(method_config, {"name": "identity"}),
        "fourier_pinn": _variant_config(method_config, feature_config),
    }

    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for method, config in variants.items():
        variant_dir = root / method
        try:
            result = op.Experiment.from_config(config).run(output_dir=variant_dir)
        except Exception as error:  # Preserve a useful failure artifact before exiting.
            rows.append(_failure(method, variant_dir, error))
        else:
            results[method] = result
            rows.append(_summary(method, result))

    completed = len(results) == 2
    differences: dict[str, Any] = {}
    parameter_note = None
    if completed:
        plain = results["plain_pinn"]
        fourier = results["fourier_pinn"]
        for key in (
            "relative_l2_error",
            "max_absolute_error",
            "rmse",
            "pde_residual_rmse",
        ):
            differences[f"fourier_minus_plain_{key}"] = float(fourier.metrics[key]) - float(
                plain.metrics[key]
            )
        differences["fourier_minus_plain_elapsed_seconds"] = float(
            fourier.compute_ledger["elapsed_seconds"]
        ) - float(plain.compute_ledger["elapsed_seconds"])
        differences["fourier_minus_plain_parameter_count"] = int(
            fourier.compute_ledger["parameter_count"]
        ) - int(plain.compute_ledger["parameter_count"])
        parameter_note = (
            "Observed trainable parameters: "
            f"plain={plain.compute_ledger['parameter_count']}, "
            f"Fourier={fourier.compute_ledger['parameter_count']}."
        )
        _write_figure(root / "comparison.svg", plain, fourier)

    comparison = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "accuracy_claim": False,
        "changed_component": "model.feature: identity -> fourier",
        "fairness": {
            "same_problem": True,
            "same_sampling": True,
            "same_hidden_layers": True,
            "same_optimizer": True,
            "same_steps": True,
            "same_experiment_seed": True,
            "parameter_count_is_expected_to_differ": True,
        },
        "variants": rows,
        "differences": differences,
    }
    (root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_comparison_csv(root / "comparison.csv", rows)
    _write_failure_notes(root / "failure_notes.md", rows, parameter_note=parameter_note)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare plain and Fourier-feature PINNs on one Poisson setting."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "poisson_fourier_comparison",
    )
    args = parser.parse_args()
    comparison = run_comparison(args.config, args.output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, default=str))
    if comparison["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
