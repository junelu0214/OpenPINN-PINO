"""Compare one registered method variant with plain PINN and save all evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import openpinn as op

from ._comparison import (
    result_summary,
    write_comparison_files,
    write_dynamic_1d_figure,
    write_static_1d_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_comparison(
    *,
    problem: str,
    family: str,
    variant: str,
    output_dir: str | Path,
    steps: int = 300,
    seed: int = 20260829,
    method_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a fixed-seed, one-component comparison with standard artifacts."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    dynamic = problem.lower().replace("_", "") in {"burgers", "burgers1d"}
    model = {"hidden_layers": [24, 24]}
    sampling = (
        {"interior": 48, "boundary": 12, "initial": 12}
        if dynamic
        else {"interior": 48, "boundary": 12}
    )
    evaluation = (
        {"points": {"space": 33, "time": 4}, "time_slices": [0.0, 0.33, 0.67, 1.0]}
        if dynamic
        else {"points": 65}
    )
    shared = {
        "steps": int(steps),
        "seed": int(seed),
        "model": model,
        "sampling": sampling,
        "training": {"resample_every": 20, "log_every": 0},
        "overrides": {"evaluation": evaluation},
        "plot": True,
    }
    plain = op.solve(
        problem,
        method="baseline",
        variant="plain_pinn",
        output_dir=root / "plain_pinn",
        **shared,
    )
    improved = op.solve(
        problem,
        method=family,
        variant=variant,
        method_options=dict(method_options or {}),
        output_dir=root / variant,
        **shared,
    )

    rows = [result_summary("plain_pinn", plain), result_summary(variant, improved)]
    comparable = (
        "relative_l2_error",
        "max_absolute_error",
        "rmse",
        "pde_residual_rmse",
    )
    differences = {
        f"{variant}_minus_plain_{name}": float(improved.metrics[name])
        - float(plain.metrics[name])
        for name in comparable
        if name in plain.metrics and name in improved.metrics
    }
    differences[f"{variant}_minus_plain_elapsed_seconds"] = float(
        improved.compute_ledger["elapsed_seconds"]
    ) - float(plain.compute_ledger["elapsed_seconds"])
    differences[f"{variant}_minus_plain_parameter_count"] = int(
        improved.compute_ledger["parameter_count"]
    ) - int(plain.compute_ledger["parameter_count"])

    if dynamic:
        write_dynamic_1d_figure(
            root / "comparison.svg",
            plain,
            improved,
            title=f"Burgers: plain PINN vs {variant}",
            improved_label=variant,
        )
    else:
        write_static_1d_figure(
            root / "comparison.svg",
            plain,
            improved,
            title=f"Poisson: plain PINN vs {variant}",
            improved_label=variant,
        )

    comparison = {
        "schema_version": 1,
        "status": "completed",
        "accuracy_claim": False,
        "validation_scope": "300-step reference short run unless overridden; not a convergence claim",
        "problem": problem,
        "family": family,
        "variant": variant,
        "steps": int(steps),
        "seed": int(seed),
        "changed_component": (
            "sampling variant only"
            if family == "adaptive_sampling"
            else "loss-balancing variant only"
        ),
        "fairness": {
            "same_problem": True,
            "same_network": True,
            "same_training_point_counts": True,
            "same_optimizer": True,
            "same_steps": True,
            "same_seed": True,
        },
        "variants": rows,
        "differences": differences,
        "method_diagnostics": {
            "sampler": improved.compute_ledger.get("sampler_diagnostics", {}),
            "adaptive_weight_parameter_count": improved.compute_ledger.get(
                "adaptive_weight_parameter_count", 0
            ),
            "final_weights": {
                key.removeprefix("final_weight_"): value
                for key, value in improved.metrics.items()
                if key.startswith("final_weight_")
            },
        },
    }
    write_comparison_files(
        root,
        comparison,
        rows,
        source_note=(
            "The method was independently implemented from its paper equations; "
            "no source from the license-unclear reference repositories was copied."
        ),
        limitations=(
            "This reference short run checks integration and saved outputs, not final convergence.",
            "One seed and one small network cannot establish statistical superiority.",
            "A positive, negative, or mixed error difference is recorded without hiding it.",
        ),
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("poisson", "burgers"), required=True)
    parser.add_argument("--family", choices=("adaptive_sampling", "adaptive_weights"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method-options-json",
        default="{}",
        help="JSON mapping passed as method_options.",
    )
    args = parser.parse_args()
    options = json.loads(args.method_options_json)
    if not isinstance(options, dict):
        raise TypeError("--method-options-json must decode to an object.")
    result = run_comparison(
        problem=args.problem,
        family=args.family,
        variant=args.variant,
        output_dir=args.output_dir,
        steps=args.steps,
        seed=args.seed,
        method_options=options,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
