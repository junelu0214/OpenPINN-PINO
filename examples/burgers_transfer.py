"""Burgers comparison of scratch, full and last-layer transfer learning."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import openpinn as op

from ._comparison import (
    failure_summary,
    result_summary,
    write_comparison_files,
    write_dynamic_1d_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reference_runs" / "burgers_transfer.yaml"


def _source_config(target: Mapping[str, Any]) -> dict[str, Any]:
    source = deepcopy(dict(target))
    source["experiment"]["name"] = "burgers_transfer_source"
    source["experiment"]["description"] = "Pretraining source on t in [0, 0.5]."
    source["experiment"]["seed"] = int(target["experiment"]["seed"]) - 1
    source["problem"]["domain"]["t"] = [0.0, 0.5]
    source["training"]["initial_checkpoint"] = None
    source["model"]["trainable_last_linear_layers"] = None
    source["evaluation"]["time_slices"] = [0.0, 0.17, 0.33, 0.5]
    return source


def run_comparison(
    config_source: str | Path | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Pretrain a short domain, then compare full and partial target transfer."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    target_config = op.load_config(config_source)
    if "initial_checkpoint" not in target_config.get("training", {}):
        raise ValueError("Transfer config must expose training.initial_checkpoint.")

    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    source_dir = root / "source_pretraining"
    try:
        source_result = op.Experiment.from_config(_source_config(target_config)).run(
            output_dir=source_dir
        )
    except Exception as error:
        rows.append(failure_summary("source_pretraining", source_dir, error))
    else:
        results["source_pretraining"] = source_result
        rows.append(result_summary("source_pretraining", source_result))

    scratch_config = deepcopy(target_config)
    scratch_config["training"]["initial_checkpoint"] = None
    scratch_config["model"]["trainable_last_linear_layers"] = None
    scratch_dir = root / "scratch_target"
    try:
        scratch_result = op.Experiment.from_config(scratch_config).run(
            output_dir=scratch_dir
        )
    except Exception as error:
        rows.append(failure_summary("scratch_target", scratch_dir, error))
    else:
        results["scratch_target"] = scratch_result
        rows.append(result_summary("scratch_target", scratch_result))

    full_transfer_dir = root / "full_transfer_target"
    if "source_pretraining" in results:
        full_transfer_config = deepcopy(target_config)
        full_transfer_config["training"]["initial_checkpoint"] = str(
            source_dir / "checkpoint.pt"
        )
        full_transfer_config["model"]["trainable_last_linear_layers"] = None
        try:
            full_transfer_result = op.Experiment.from_config(full_transfer_config).run(
                output_dir=full_transfer_dir
            )
        except Exception as error:
            rows.append(failure_summary("full_transfer_target", full_transfer_dir, error))
        else:
            results["full_transfer_target"] = full_transfer_result
            rows.append(result_summary("full_transfer_target", full_transfer_result))
    else:
        try:
            raise RuntimeError("Full transfer skipped because source pretraining failed.")
        except RuntimeError as error:
            rows.append(failure_summary("full_transfer_target", full_transfer_dir, error))

    partial_transfer_dir = root / "partial_transfer_target"
    if "source_pretraining" in results:
        partial_transfer_config = deepcopy(target_config)
        partial_transfer_config["training"]["initial_checkpoint"] = str(
            source_dir / "checkpoint.pt"
        )
        partial_transfer_config["model"]["trainable_last_linear_layers"] = 1
        try:
            partial_transfer_result = op.Experiment.from_config(
                partial_transfer_config
            ).run(output_dir=partial_transfer_dir)
        except Exception as error:
            rows.append(
                failure_summary("partial_transfer_target", partial_transfer_dir, error)
            )
        else:
            results["partial_transfer_target"] = partial_transfer_result
            rows.append(result_summary("partial_transfer_target", partial_transfer_result))
    else:
        try:
            raise RuntimeError("Partial transfer skipped because source pretraining failed.")
        except RuntimeError as error:
            rows.append(
                failure_summary("partial_transfer_target", partial_transfer_dir, error)
            )

    completed = len(results) == 4
    differences: dict[str, float] = {}
    timing: dict[str, float] = {}
    if completed:
        source = results["source_pretraining"]
        scratch = results["scratch_target"]
        full_transfer = results["full_transfer_target"]
        partial_transfer = results["partial_transfer_target"]
        for name, result in (
            ("full_transfer", full_transfer),
            ("partial_transfer", partial_transfer),
        ):
            for key in (
                "relative_l2_error",
                "max_absolute_error",
                "rmse",
                "pde_residual_rmse",
                "boundary_rmse",
                "initial_rmse",
            ):
                differences[f"{name}_minus_scratch_{key}"] = float(
                    result.metrics[key]
                ) - float(scratch.metrics[key])
            differences[f"{name}_minus_scratch_target_seconds"] = float(
                result.compute_ledger["elapsed_seconds"]
            ) - float(scratch.compute_ledger["elapsed_seconds"])
        source_seconds = float(source.compute_ledger["elapsed_seconds"])
        scratch_seconds = float(scratch.compute_ledger["elapsed_seconds"])
        full_transfer_seconds = float(full_transfer.compute_ledger["elapsed_seconds"])
        partial_transfer_seconds = float(
            partial_transfer.compute_ledger["elapsed_seconds"]
        )
        timing = {
            "source_pretraining_seconds": source_seconds,
            "scratch_target_seconds": scratch_seconds,
            "full_transfer_target_seconds": full_transfer_seconds,
            "partial_transfer_target_seconds": partial_transfer_seconds,
            "full_transfer_pipeline_seconds": source_seconds + full_transfer_seconds,
            "partial_transfer_pipeline_seconds": (
                source_seconds + partial_transfer_seconds
            ),
        }
        write_dynamic_1d_figure(
            root / "comparison.svg",
            scratch,
            full_transfer,
            title="Burgers target: scratch vs full-network transfer",
            improved_label="Full transfer",
        )
        write_dynamic_1d_figure(
            root / "partial_transfer_comparison.svg",
            scratch,
            partial_transfer,
            title="Burgers target: scratch vs last-layer transfer",
            improved_label="Last-layer transfer",
        )

    comparison = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "accuracy_claim": False,
        "changed_component": (
            "target initialization: random -> source model checkpoint; "
            "partial variant also freezes all but the last linear layer"
        ),
        "method_scope": (
            "model-only checkpoint warm start with new optimizer state; compares "
            "full-network fine-tuning against last-linear-layer fine-tuning"
        ),
        "fairness": {
            "same_target_problem": True,
            "same_target_sampling": True,
            "same_target_network": True,
            "same_target_loss": True,
            "same_target_optimizer": True,
            "same_target_steps": True,
            "same_target_seed": True,
            "same_total_parameter_count": True,
            "partial_variant_has_fewer_trainable_parameters": True,
            "pretraining_cost_reported_separately": True,
        },
        "variants": rows,
        "timing": timing,
        "differences": differences,
    }
    write_comparison_files(
        root,
        comparison,
        rows,
        source_note=(
            "Model-only checkpoint loading was independently implemented; no code from "
            "the license-unclear PINN-extrapolation repository was copied or adapted."
        ),
        limitations=(
            "Full transfer updates every network weight. Partial transfer updates only the final linear layer, so its lower trainable-parameter count is intentional and recorded.",
            "The source pretraining cost must be included unless one source model is reused across many targets.",
            "Burgers solution errors use the bundled Cole--Hopf reference for evaluation only; it is never supplied to training.",
        ),
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "burgers_transfer_comparison",
    )
    args = parser.parse_args()
    comparison = run_comparison(args.config, args.output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
