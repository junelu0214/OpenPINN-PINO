"""Short-run every code option exposed by the documentation configurator.

Validation outputs are intentionally kept under ``results/`` rather than in
the method implementation folders.  The script uses the same public
``op.method(...)`` calls and parameter defaults that the web page renders.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import openpinn as op


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = (
    PROJECT_ROOT / "results" / "configurator_validation" / "20260904_all_variants"
)


CASES = (
    ("baseline", "plain_pinn", "poisson", 1),
    ("modified_mlp", "gated_mlp", "poisson", 1),
    ("adaptive_activation", "adaptive_tanh", "poisson", 1),
    ("fourier", "random_fourier", "poisson", 1),
    ("hard_boundary", "exact_output_transform", "poisson", 1),
    ("adaptive_weights", "self_adaptive", "poisson", 1),
    ("adaptive_weights", "uncertainty_weighting", "poisson", 1),
    ("adaptive_weights", "relobralo", "poisson", 1),
    ("adaptive_weights", "magnitude_power", "poisson", 1),
    ("adaptive_weights", "pointwise_self_adaptive", "poisson", 1),
    ("adaptive_sampling", "residual_adaptive", "poisson", 1),
    ("adaptive_sampling", "residual_movement", "poisson", 1),
    ("adaptive_sampling", "gradient_movement", "poisson", 1),
    ("adaptive_sampling", "importance_sampling", "poisson", 1),
    ("adam_lbfgs", "adam_then_lbfgs", "poisson", 2),
    ("time_window", "causal_window", "burgers", 1),
    ("domain_decomposition", "two_subdomain", "heterogeneous_poisson", 1),
)


def default_options(method: str, variant: str) -> dict[str, Any]:
    """Return every non-empty default shown in the configurator."""

    schema = op.method_parameter_schema(method, variant)
    values = {
        name: item["default"]
        for name, item in schema.items()
        if item["default"] is not None
    }
    if method == "adaptive_sampling":
        values["resample_every"] = 1
    return values


def run_case(
    method: str,
    variant: str,
    problem: str,
    steps: int,
    *,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_options = default_options(method, variant)
    selected_options.update(options or {})
    started = time.perf_counter()
    result = op.solve(
        problem,
        methods=[op.method(method, variant, **selected_options)],
        steps=steps,
        seed=0,
        output_root=VALIDATION_ROOT / "runs" / f"{method}__{variant}",
    )
    return {
        "status": "passed",
        "problem": problem,
        "method": method,
        "variant": variant,
        "steps": steps,
        "options": selected_options,
        "elapsed_seconds": time.perf_counter() - started,
        "relative_l2_error": result.metrics.get("relative_l2_error"),
        "output_dir": str(result.output_dir),
    }


def main() -> int:
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for method, variant, problem, steps in CASES:
        try:
            record = run_case(method, variant, problem, steps)
            records.append(record)
            print(f"PASS {method}/{variant} -> {record['output_dir']}")
        except Exception as error:  # pragma: no cover - validation reporting path
            record = {
                "status": "failed",
                "problem": problem,
                "method": method,
                "variant": variant,
                "steps": steps,
                "error": f"{type(error).__name__}: {error}",
            }
            records.append(record)
            failures.append(record)
            print(f"FAIL {method}/{variant}: {record['error']}")

    source = op.solve(
        "poisson",
        methods=[op.method("baseline", "plain_pinn")],
        steps=1,
        seed=0,
        output_root=VALIDATION_ROOT / "transfer_source",
    )
    checkpoint = source.output_dir / "checkpoint.pt"
    try:
        transfer_record = run_case(
            "transfer",
            "checkpoint_transfer",
            "poisson",
            1,
            options={"checkpoint": checkpoint},
        )
        records.append(transfer_record)
        print(
            "PASS transfer/checkpoint_transfer -> "
            f"{transfer_record['output_dir']}"
        )
    except Exception as error:  # pragma: no cover - validation reporting path
        transfer_record = {
            "status": "failed",
            "problem": "poisson",
            "method": "transfer",
            "variant": "checkpoint_transfer",
            "steps": 1,
            "error": f"{type(error).__name__}: {error}",
        }
        records.append(transfer_record)
        failures.append(transfer_record)
        print(f"FAIL transfer/checkpoint_transfer: {transfer_record['error']}")

    summary = {
        "purpose": "Short execution of every concrete method shown by the web configurator.",
        "accuracy_claim": False,
        "passed": sum(item["status"] == "passed" for item in records),
        "failed": len(failures),
        "records": records,
    }
    (VALIDATION_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    rows = [
        "# 配置生成器逐方法短跑记录",
        "",
        "本验证只检查网页所列方法及参数能否进入真实训练并保存产物，不用于比较精度。",
        "",
        f"- 通过：{summary['passed']}",
        f"- 失败：{summary['failed']}",
        "",
        "| method | variant | problem | steps | status | output |",
        "|---|---|---|---:|---|---|",
    ]
    rows.extend(
        "| {method} | {variant} | {problem} | {steps} | {status} | {output} |".format(
            method=item["method"],
            variant=item["variant"],
            problem=item["problem"],
            steps=item["steps"],
            status=item["status"],
            output=item.get("output_dir", item.get("error", "")),
        )
        for item in records
    )
    (VALIDATION_ROOT / "README.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
