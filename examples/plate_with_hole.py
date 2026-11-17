"""Plate-with-hole 2D linear-elastic demo from ProblemSpec + MethodSpec (M3).

Loads the M2 validated YAML specs, builds the problem via codegen, trains
a PINN for a configurable number of steps, and reports metrics including
an optional Kirsch analytical comparison.

Usage
-----
Smoke test (few steps, CI-safe):
    python examples/plate_with_hole.py --smoke

Longer training:
    python examples/plate_with_hole.py --steps 5000

The ``--smoke`` flag trains for 10 steps (a sanity check, not an accuracy run).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import openpinn as op
from openpinn.codegen import from_specs
from openpinn.kirsch import (
    compare_displacement_l2,
    kirsch_hoop_stress_at_hole,
    kirsch_scf,
)


SPEC_DIR = Path(__file__).resolve().parent / "specs"
PROBLEM_YAML = SPEC_DIR / "plate_with_hole_problem.yaml"
METHOD_YAML = SPEC_DIR / "pinn_method.yaml"


def run(
    *,
    steps: int = 500,
    smoke: bool = False,
    output_dir: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build problem from specs, train, evaluate, compare to Kirsch."""

    effective_steps = 10 if smoke else steps
    problem, kwargs = from_specs(
        str(PROBLEM_YAML),
        str(METHOD_YAML),
        steps=effective_steps,
        device=device,
    )

    if smoke:
        kwargs["sampling"] = {"name": "uniform", "interior": 64, "boundary": 32}
        kwargs["model"] = {
            "name": "mlp",
            "hidden_layers": [32, 32],
            "activation": "tanh",
        }

    result = op.solve(
        problem,
        **kwargs,
        run_name=f"plate_with_hole_{'smoke' if smoke else f'{effective_steps}steps'}",
        output_dir=output_dir,
        plot=False,
    )

    summary: dict[str, Any] = {
        "example": "plate_with_hole",
        "steps": effective_steps,
        "smoke": smoke,
        "output_dir": str(result.output_dir),
        "final_total_loss": result.metrics.get("final_total_loss"),
        "pde_residual_rmse": result.metrics.get("pde_residual_rmse"),
        "status": (
            "completed_with_finite_outputs"
            if math.isfinite(result.metrics.get("final_total_loss", float("nan")))
            else "completed_non_finite"
        ),
    }

    # Kirsch analytical comparison
    prob_spec = op.load_problem_spec(str(PROBLEM_YAML))
    E = prob_spec.constitutive.E
    nu = prob_spec.constitutive.nu
    hole = prob_spec.geometry.holes[0]
    a = hole.radius
    center = tuple(hole.center)

    # Find uniaxial far-field tension from the right-edge traction BC
    sigma_inf = 0.0
    for bc in prob_spec.boundaries:
        if bc.region == "right" and bc.type.value == "neumann":
            vals = bc.value if isinstance(bc.value, list) else [bc.value]
            for dof, val in zip(bc.dofs, vals):
                if dof == "ux":
                    sigma_inf = float(val)
                    break

    if sigma_inf != 0.0 and "prediction" in result.predictions:
        coords = result.predictions["coordinates"]
        pred = result.predictions["prediction"]
        pred_ux = pred[:, 0]
        pred_uy = pred[:, 1]

        kirsch_metrics = compare_displacement_l2(
            pred_ux, pred_uy, coords,
            sigma_inf=sigma_inf,
            a=a,
            E=E,
            nu=nu,
            center=center,
        )
        summary["kirsch_comparison"] = kirsch_metrics
        summary["kirsch_scf"] = kirsch_scf()
        summary["kirsch_note"] = (
            "Kirsch solution assumes an infinite plate; the finite plate "
            f"({prob_spec.geometry.outer.x_max - prob_spec.geometry.outer.x_min}m x "
            f"{prob_spec.geometry.outer.y_max - prob_spec.geometry.outer.y_min}m) "
            f"with hole radius {a}m introduces boundary-proximity error that is "
            "NOT a PINN defect. Best comparison is near the hole (r < ~4a)."
        )
    else:
        summary["kirsch_comparison"] = None
        summary["kirsch_note"] = "Kirsch comparison skipped (no traction or predictions)."

    # Hoop stress reference at hole boundary
    theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    hoop = kirsch_hoop_stress_at_hole(theta, sigma_inf)
    summary["kirsch_max_hoop_stress"] = float(np.max(hoop))
    summary["kirsch_scf_value"] = float(np.max(hoop) / sigma_inf) if sigma_inf != 0 else None

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plate-with-hole 2D linear-elastic PINN demo. "
            "Loads ProblemSpec + MethodSpec YAMLs, trains, and compares to Kirsch."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="10-step CPU smoke test (sanity check, not accuracy).",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Training steps (ignored with --smoke).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (cpu or cuda).",
    )
    args = parser.parse_args()

    summary = run(
        steps=args.steps,
        smoke=args.smoke,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
