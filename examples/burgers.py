"""Dynamic one-dimensional viscous Burgers baseline and smoke example."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import openpinn as op
import torch

try:
    from ._comparison import write_dynamic_reference_figure
    from ._runner import build_parser, print_summary, run_config, select_config
except ImportError:  # Allows ``python examples/burgers.py`` after installation.
    from _comparison import write_dynamic_reference_figure
    from _runner import build_parser, print_summary, run_config, select_config


def _reference_validation(result: Any) -> dict[str, Any]:
    """Reproduce the reference-oracle checks inside each saved run."""

    viscosity = float(result.metrics["reference_viscosity"])
    coarse = op.BurgersColeHopfReference(viscosity, quadrature_points=512)
    fine = op.BurgersColeHopfReference(viscosity, quadrature_points=1024)
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    times = torch.tensor([0.01, 0.33, 0.67, 1.0], dtype=torch.float64)
    convergence_coordinates = torch.cartesian_prod(x, times)
    convergence_error = float(
        (coarse(convergence_coordinates) - fine(convergence_coordinates))
        .abs()
        .max()
    )

    initial_coordinates = torch.column_stack(
        (x, torch.zeros_like(x))
    )
    initial_error = float(
        (fine(initial_coordinates) + torch.sin(math.pi * x)[:, None]).abs().max()
    )
    boundary_coordinates = torch.tensor(
        [[-1.0, 0.01], [-1.0, 0.5], [1.0, 0.01], [1.0, 1.0]],
        dtype=torch.float64,
    )
    boundary_error = float(fine(boundary_coordinates).abs().max())

    pde_coordinates = torch.tensor(
        [[-0.8, 0.10], [-0.3, 0.25], [0.2, 0.50], [0.7, 0.90]],
        dtype=torch.float64,
    )
    spatial_step = 5e-4
    time_step = 1e-4
    x_shift = torch.tensor([spatial_step, 0.0], dtype=torch.float64)
    t_shift = torch.tensor([0.0, time_step], dtype=torch.float64)
    value = fine(pde_coordinates)
    u_x = (
        fine(pde_coordinates + x_shift) - fine(pde_coordinates - x_shift)
    ) / (2.0 * spatial_step)
    u_xx = (
        fine(pde_coordinates + x_shift)
        - 2.0 * value
        + fine(pde_coordinates - x_shift)
    ) / (spatial_step**2)
    u_t = (
        fine(pde_coordinates + t_shift) - fine(pde_coordinates - t_shift)
    ) / (2.0 * time_step)
    pde_residual = u_t + value * u_x - viscosity * u_xx
    pde_error = float(pde_residual.abs().max())
    tolerances = {
        "quadrature_convergence_max_abs": 2e-10,
        "initial_condition_max_abs": 1e-13,
        "boundary_condition_max_abs": 1e-13,
        "finite_difference_pde_max_abs": 1e-4,
    }
    observed = {
        "quadrature_512_vs_1024_max_abs": convergence_error,
        "initial_condition_max_abs": initial_error,
        "boundary_condition_max_abs": boundary_error,
        "finite_difference_pde_max_abs": pde_error,
    }
    passed = (
        convergence_error <= tolerances["quadrature_convergence_max_abs"]
        and initial_error <= tolerances["initial_condition_max_abs"]
        and boundary_error <= tolerances["boundary_condition_max_abs"]
        and pde_error <= tolerances["finite_difference_pde_max_abs"]
    )
    if not passed:
        raise RuntimeError(f"Burgers reference validation failed: {observed}")
    return {
        "status": "passed",
        "observed": observed,
        "tolerances": tolerances,
        "notes": {
            "quadrature": "Maximum difference after doubling Gauss-Legendre points.",
            "pde": "Centered finite-difference check at four interior space-time points.",
            "training_use": "Reference is evaluation-only.",
        },
    }


def _write_reference_artifacts(result: Any) -> None:
    """Add the Burgers reference figure and an explicit limitation record."""

    output_dir = Path(result.output_dir)
    write_dynamic_reference_figure(
        output_dir / "reference_comparison.svg",
        result,
        title="Viscous Burgers: plain PINN against Cole–Hopf reference",
    )
    writer = op.ArtifactWriter(output_dir)
    writer.write_json("reference_validation.json", _reference_validation(result))
    coordinates = np.asarray(result.predictions["coordinates"])
    prediction = np.asarray(result.predictions["prediction"])[:, 0]
    reference = np.asarray(result.predictions["reference"])[:, 0]
    pointwise_error = np.abs(prediction - reference)
    worst_index = int(np.argmax(pointwise_error))
    worst_x, worst_time = coordinates[worst_index]
    worst_error = float(pointwise_error[worst_index])
    (output_dir / "failure_notes.md").write_text(
        "\n".join(
            (
                "# Failure and limitation notes",
                "",
                "## Training failures",
                "",
                "- None. Training and saved evaluation outputs are finite.",
                "",
                "## Reference and interpretation limits",
                "",
                "- The independent Cole–Hopf reference is used only after training; it does not enter the loss or sampling.",
                "- The evaluator is an original OpenPINN implementation based on the Cole (1951) and Hopf (1950) transformations; no external source code was copied.",
                "- The reported result is one fixed-seed configuration, not a statistical or paper-accuracy claim.",
                f"- Largest saved pointwise error: {worst_error:.8g} at x={float(worst_x):.8g}, t={float(worst_time):.8g}; inspect the shock-transition panels before interpreting the aggregate error.",
                "- Positive reference times below t=0.002 are outside the validated quadrature range and are rejected explicitly.",
                "- When running directly from an uninstalled source checkout, install the package or set PYTHONPATH=src.",
                "",
            )
        ),
        encoding="utf-8",
    )
    writer.write_manifest(
        metadata={
            "problem": "burgers_1d",
            "seed": result.compute_ledger["seed"],
            "schema": "openpinn-starter-v1",
            "reference_artifacts": True,
        }
    )


def run(
    *,
    smoke: bool = False,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> Any:
    """Run dynamic Burgers through the public ``openpinn`` API."""

    selected = select_config("burgers", smoke=smoke, custom=config_path)
    result = run_config(selected, output_dir=output_dir)
    _write_reference_artifacts(result)
    return result


def main() -> None:
    parser = build_parser("burgers")
    args = parser.parse_args()
    selected = select_config("burgers", smoke=args.smoke, custom=args.config)
    result = run_config(selected, output_dir=args.output_dir)
    _write_reference_artifacts(result)
    print_summary("burgers_dynamic", selected, result)


if __name__ == "__main__":
    main()
