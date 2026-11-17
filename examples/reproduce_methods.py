"""Run the ten completed method comparisons with a fixed multi-seed protocol."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import openpinn as op

from .burgers_adaptive_activation import run_comparison as run_adaptive_activation
from .burgers_adaptive_sampling import run_comparison as run_adaptive_sampling
from .burgers_adaptive_weights import run_comparison as run_adaptive_weights
from .burgers_time_window import run_comparison as run_time_window
from .burgers_transfer import run_comparison as run_transfer
from .heterogeneous_poisson_domain_decomposition import (
    run_comparison as run_domain_decomposition,
)
from .poisson_adam_lbfgs import run_comparison as run_adam_lbfgs
from .poisson_fourier import run_comparison as run_fourier
from .poisson_hard_boundary import run_comparison as run_hard_boundary
from .poisson_modified_mlp import run_comparison as run_modified_mlp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (20260827, 20260828, 20260829)
NUMERIC_FIELDS = (
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
    "closure_calls",
)


@dataclass(frozen=True)
class MethodSpec:
    """The immutable comparison contract for one completed method."""

    slug: str
    label: str
    case: str
    config_name: str
    baseline_variant: str
    method_variant: str
    primary_metric: str
    runner: Callable[[str | Path | Mapping[str, Any], str | Path], dict[str, Any]]


METHODS = (
    MethodSpec(
        "fourier",
        "Fourier features",
        "high-frequency Poisson",
        "poisson_fourier.yaml",
        "plain_pinn",
        "fourier_pinn",
        "relative_l2_error",
        run_fourier,
    ),
    MethodSpec(
        "modified_mlp",
        "Modified MLP",
        "high-frequency Poisson",
        "poisson_modified_mlp.yaml",
        "plain_pinn",
        "modified_mlp",
        "relative_l2_error",
        run_modified_mlp,
    ),
    MethodSpec(
        "adaptive_activation",
        "Adaptive activation",
        "Burgers",
        "burgers_adaptive_activation.yaml",
        "plain_pinn",
        "adaptive_tanh",
        "relative_l2_error",
        run_adaptive_activation,
    ),
    MethodSpec(
        "hard_boundary",
        "Hard boundary",
        "high-frequency Poisson",
        "poisson_hard_boundary.yaml",
        "plain_pinn",
        "hard_boundary",
        "relative_l2_error",
        run_hard_boundary,
    ),
    MethodSpec(
        "adaptive_weights",
        "Adaptive weights",
        "Burgers",
        "burgers_adaptive_weights.yaml",
        "plain_pinn",
        "adaptive_weights",
        "relative_l2_error",
        run_adaptive_weights,
    ),
    MethodSpec(
        "adaptive_sampling",
        "Adaptive sampling",
        "Burgers",
        "burgers_adaptive_sampling.yaml",
        "plain_pinn",
        "adaptive_sampling",
        "relative_l2_error",
        run_adaptive_sampling,
    ),
    MethodSpec(
        "adam_lbfgs",
        "Adam + L-BFGS",
        "low-frequency Poisson",
        "poisson_adam_lbfgs.yaml",
        "adam_only",
        "adam_lbfgs",
        "relative_l2_error",
        run_adam_lbfgs,
    ),
    MethodSpec(
        "time_window",
        "Time window",
        "Burgers",
        "burgers_time_window.yaml",
        "plain_pinn",
        "time_window",
        "relative_l2_error",
        run_time_window,
    ),
    MethodSpec(
        "transfer",
        "Transfer learning",
        "Burgers",
        "burgers_transfer.yaml",
        "scratch_target",
        "full_transfer_target",
        "relative_l2_error",
        run_transfer,
    ),
    MethodSpec(
        "domain_decomposition",
        "Domain decomposition",
        "heterogeneous Poisson",
        "heterogeneous_poisson_domain_decomposition.yaml",
        "plain_pinn",
        "domain_decomposition",
        "relative_l2_error",
        run_domain_decomposition,
    ),
)


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return None
    return selected if math.isfinite(selected) else None


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return mean and sample standard deviation by method variant and metric."""

    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        for metric in NUMERIC_FIELDS:
            value = _finite_number(row.get(metric))
            if value is not None:
                key = (str(row["method_slug"]), str(row["variant"]), metric)
                groups.setdefault(key, []).append(value)

    summary: list[dict[str, Any]] = []
    for (method_slug, variant, metric), values in sorted(groups.items()):
        summary.append(
            {
                "method_slug": method_slug,
                "variant": variant,
                "metric": metric,
                "count": len(values),
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "minimum": min(values),
                "maximum": max(values),
            }
        )
    return summary


def primary_pairs(
    rows: Sequence[Mapping[str, Any]], specs: Sequence[MethodSpec]
) -> list[dict[str, Any]]:
    """Build paired per-seed primary-metric and timing ratios."""

    by_key = {
        (str(row["method_slug"]), int(row["seed"]), str(row["variant"])): row
        for row in rows
        if row.get("status") == "completed" and row.get("seed") is not None
    }
    pairs: list[dict[str, Any]] = []
    seeds = sorted({int(row["seed"]) for row in rows if row.get("seed") is not None})
    for spec in specs:
        for seed in seeds:
            baseline = by_key.get((spec.slug, seed, spec.baseline_variant))
            method = by_key.get((spec.slug, seed, spec.method_variant))
            if baseline is None or method is None:
                continue
            baseline_value = _finite_number(baseline.get(spec.primary_metric))
            method_value = _finite_number(method.get(spec.primary_metric))
            baseline_time = _finite_number(baseline.get("elapsed_seconds"))
            method_time = _finite_number(method.get("elapsed_seconds"))
            if baseline_value is None or method_value is None:
                continue
            metric_ratio = method_value / baseline_value if baseline_value > 0.0 else None
            time_ratio = (
                method_time / baseline_time
                if baseline_time is not None
                and method_time is not None
                and baseline_time > 0.0
                else None
            )
            pairs.append(
                {
                    "method_slug": spec.slug,
                    "method_label": spec.label,
                    "case": spec.case,
                    "seed": seed,
                    "primary_metric": spec.primary_metric,
                    "baseline_variant": spec.baseline_variant,
                    "method_variant": spec.method_variant,
                    "baseline_value": baseline_value,
                    "method_value": method_value,
                    "method_minus_baseline": method_value - baseline_value,
                    "method_over_baseline": metric_ratio,
                    "log10_method_over_baseline": (
                        math.log10(metric_ratio)
                        if metric_ratio is not None and metric_ratio > 0.0
                        else None
                    ),
                    "baseline_seconds": baseline_time,
                    "method_seconds": method_time,
                    "method_over_baseline_time": time_ratio,
                }
            )
    return pairs


def summarize_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the paired primary metric and timing ratios for every method."""

    summary: list[dict[str, Any]] = []
    method_slugs = list(dict.fromkeys(str(pair["method_slug"]) for pair in pairs))
    for slug in method_slugs:
        selected = [pair for pair in pairs if pair["method_slug"] == slug]
        ratios = [
            float(pair["method_over_baseline"])
            for pair in selected
            if _finite_number(pair.get("method_over_baseline")) is not None
        ]
        log_ratios = [math.log10(value) for value in ratios if value > 0.0]
        time_ratios = [
            float(pair["method_over_baseline_time"])
            for pair in selected
            if _finite_number(pair.get("method_over_baseline_time")) is not None
        ]
        summary.append(
            {
                "method_slug": slug,
                "method_label": selected[0]["method_label"],
                "case": selected[0]["case"],
                "primary_metric": selected[0]["primary_metric"],
                "count": len(ratios),
                "mean_method_over_baseline": statistics.fmean(ratios),
                "sample_std_method_over_baseline": (
                    statistics.stdev(ratios) if len(ratios) > 1 else 0.0
                ),
                "mean_log10_method_over_baseline": statistics.fmean(log_ratios),
                "sample_std_log10_method_over_baseline": (
                    statistics.stdev(log_ratios) if len(log_ratios) > 1 else 0.0
                ),
                "mean_method_over_baseline_time": statistics.fmean(time_ratios),
                "sample_std_method_over_baseline_time": (
                    statistics.stdev(time_ratios) if len(time_ratios) > 1 else 0.0
                ),
                "improved_seed_count": sum(value < 1.0 for value in ratios),
            }
        )
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_ratio_figure(path: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    """Write an SVG of paired primary-metric ratios on a log10 scale."""

    width, height = 1320, 650
    left, right, top, bottom = 95.0, 35.0, 90.0, 145.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(row["mean_log10_method_over_baseline"]) for row in summary]
    errors = [float(row["sample_std_log10_method_over_baseline"]) for row in summary]
    low = min([-1.0] + [value - error for value, error in zip(values, errors, strict=True)])
    high = max([1.0] + [value + error for value, error in zip(values, errors, strict=True)])
    padding = max((high - low) * 0.08, 0.1)
    low, high = low - padding, high + padding

    def y_position(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:24px;font-weight:700}.sub{font-size:14px;fill:#4b5563}.tick{font-size:12px}.label{font-size:12px}.axis{stroke:#374151;stroke-width:1.3}.grid{stroke:#e5e7eb;stroke-width:1}</style>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" class="title">Three-seed paired primary-metric ratios</text>',
        f'<text x="{width / 2}" y="59" text-anchor="middle" class="sub">log10(method / baseline); below zero means a smaller error or residual</text>',
    ]
    for index in range(6):
        value = low + index / 5 * (high - low)
        y = y_position(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{value:.2f}</text>')
    zero_y = y_position(0.0)
    parts.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_width}" y2="{zero_y:.2f}" stroke="#111827" stroke-width="2"/>')
    step = plot_width / max(len(summary), 1)
    for index, row in enumerate(summary):
        x = left + (index + 0.5) * step
        value = float(row["mean_log10_method_over_baseline"])
        error = float(row["sample_std_log10_method_over_baseline"])
        y, y_low, y_high = y_position(value), y_position(value - error), y_position(value + error)
        color = "#15803d" if value < 0.0 else "#b91c1c"
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" y2="{y_high:.2f}" stroke="{color}" stroke-width="2"/>',
                f'<line x1="{x - 7:.2f}" y1="{y_low:.2f}" x2="{x + 7:.2f}" y2="{y_low:.2f}" stroke="{color}" stroke-width="2"/>',
                f'<line x1="{x - 7:.2f}" y1="{y_high:.2f}" x2="{x + 7:.2f}" y2="{y_high:.2f}" stroke="{color}" stroke-width="2"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 18:.2f}" text-anchor="end" transform="rotate(-42 {x:.2f} {top + plot_height + 18:.2f})" class="label">{html.escape(str(row["method_label"]))}</text>',
            ]
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>',
            f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2})" class="sub">log10(primary metric ratio), mean ± sample SD</text>',
            f'<text x="{left}" y="{height - 18}" class="sub">Primary metric: relative L2 for Poisson; PDE residual RMSE for Burgers.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def run_protocol(
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    specs: Sequence[MethodSpec] = METHODS,
) -> dict[str, Any]:
    """Run every selected method sequentially and preserve failures."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Reproducibility directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    selected_seeds = tuple(int(seed) for seed in seeds)
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Seeds must be a non-empty sequence of distinct integers.")

    protocol = {
        "schema_version": 1,
        "seeds": list(selected_seeds),
        "execution": "sequential",
        "configuration_policy": "Only experiment.seed changes from each committed method config.",
        "statistics": "Arithmetic mean and sample standard deviation; paired by seed.",
        "accuracy_claim": False,
        "methods": [
            {
                "order": index,
                "slug": spec.slug,
                "label": spec.label,
                "case": spec.case,
                "config": str((PROJECT_ROOT / "configs" / "reference_runs" / spec.config_name).resolve()),
                "baseline_variant": spec.baseline_variant,
                "method_variant": spec.method_variant,
                "primary_metric": spec.primary_metric,
            }
            for index, spec in enumerate(specs, start=1)
        ],
    }
    (root / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        base_config = op.load_config(PROJECT_ROOT / "configs" / "reference_runs" / spec.config_name)
        method_root = root / f"{index:02d}_{spec.slug}"
        for seed in selected_seeds:
            config = op.load_config(base_config)
            config.setdefault("experiment", {})["seed"] = seed
            run_dir = method_root / f"seed_{seed}"
            try:
                comparison = spec.runner(config, run_dir)
            except Exception as error:
                run_dir.mkdir(parents=True, exist_ok=True)
                failure = {
                    "method_slug": spec.slug,
                    "seed": seed,
                    "status": "failed",
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                (run_dir / "protocol_failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
                run_records.append(failure)
                continue

            run_records.append(
                {
                    "method_slug": spec.slug,
                    "seed": seed,
                    "status": comparison.get("status", "unknown"),
                    "output_dir": str(run_dir),
                }
            )
            for variant in comparison.get("variants", []):
                row = {
                    "method_slug": spec.slug,
                    "method_label": spec.label,
                    "case": spec.case,
                    "seed": seed,
                    "primary_metric": spec.primary_metric,
                    "variant": variant.get("method"),
                    **{
                        key: value
                        for key, value in variant.items()
                        if key not in {"method", "seed"}
                    },
                }
                rows.append(row)

    aggregate = summarize_rows(rows)
    pairs = primary_pairs(rows, specs)
    paired_summary = summarize_pairs(pairs)
    failures = [record for record in run_records if record.get("status") != "completed"]
    status = "completed" if not failures else "completed_with_failures"
    result = {
        "schema_version": 1,
        "status": status,
        "seeds": list(selected_seeds),
        "method_count": len(specs),
        "comparison_run_count": len(run_records),
        "completed_comparison_count": len(run_records) - len(failures),
        "failed_comparison_count": len(failures),
        "paired_summary": paired_summary,
        "failures": failures,
    }
    (root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(root / "run_status.csv", run_records)
    _write_csv(root / "all_variants.csv", rows)
    _write_csv(root / "aggregate_metrics.csv", aggregate)
    _write_csv(root / "paired_primary_metrics.csv", pairs)
    _write_csv(root / "paired_summary.csv", paired_summary)
    if paired_summary:
        write_ratio_figure(root / "primary_metric_ratios.svg", paired_summary)

    notes = [
        "# Reproducibility failure and limitation notes",
        "",
        "## Training failures",
        "",
    ]
    if failures:
        notes.extend(
            f'- {record["method_slug"]}, seed {record["seed"]}: {record.get("failure_reason", "comparison failed")}'
            for record in failures
        )
    else:
        notes.append("- None. All method comparisons completed for all three seeds.")
    notes.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Three seeds measure basic repeatability but do not establish universal superiority.",
            "- No method hyperparameters or training-step budgets were changed for this rerun.",
            "- Primary metrics are compared only within the same benchmark and method pair, never ranked across different PDEs.",
            "- Burgers relative L2 errors use the bundled independent Cole--Hopf reference for evaluation only; training never sees that reference.",
            "- Individual seed directories retain error metrics, time, predictions, figures, checkpoints and their method-specific limitation notes.",
        ]
    )
    (root / "failure_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "reproducibility_three_seed",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[spec.slug for spec in METHODS],
        default=[spec.slug for spec in METHODS],
    )
    args = parser.parse_args()
    requested = set(args.methods)
    specs = tuple(spec for spec in METHODS if spec.slug in requested)
    result = run_protocol(args.output_dir, seeds=args.seeds, specs=specs)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
