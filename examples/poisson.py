"""Static one-dimensional Poisson baseline and smoke example."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import openpinn as op

try:
    from ._comparison import write_static_reference_figure
    from ._runner import build_parser, print_summary, run_config, select_config
except ImportError:  # Allows ``python examples/poisson.py`` after installation.
    from _comparison import write_static_reference_figure
    from _runner import build_parser, print_summary, run_config, select_config


def _write_reference_artifacts(result: Any) -> None:
    """Add a baseline figure and explicit interpretation limits."""

    output_dir = Path(result.output_dir)
    write_static_reference_figure(
        output_dir / "reference_comparison.svg",
        result,
        title="Poisson: plain PINN against the analytic solution",
    )
    coordinates = np.asarray(result.predictions["coordinates"])[:, 0]
    prediction = np.asarray(result.predictions["prediction"])[:, 0]
    reference = np.asarray(result.predictions["reference"])[:, 0]
    error = np.abs(prediction - reference)
    worst_index = int(np.argmax(error))
    (output_dir / "failure_notes.md").write_text(
        "\n".join(
            (
                "# Failure and limitation notes",
                "",
                "## Training failures",
                "",
                "- None. Training and saved evaluation outputs are finite.",
                "",
                "## Interpretation limits",
                "",
                "- The analytic solution is used only after training; it does not enter the loss or sampling.",
                "- This is one fixed-seed baseline, not a statistical or paper-accuracy claim.",
                f"- Largest saved pointwise error: {float(error[worst_index]):.8g} at x={float(coordinates[worst_index]):.8g}.",
                "",
            )
        ),
        encoding="utf-8",
    )
    op.ArtifactWriter(output_dir).write_manifest(
        metadata={
            "problem": "poisson_1d",
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
    """Run static Poisson through the public ``openpinn`` API."""

    selected = select_config("poisson", smoke=smoke, custom=config_path)
    result = run_config(selected, output_dir=output_dir)
    _write_reference_artifacts(result)
    return result


def main() -> None:
    parser = build_parser("poisson")
    args = parser.parse_args()
    selected = select_config("poisson", smoke=args.smoke, custom=args.config)
    result = run_config(selected, output_dir=args.output_dir)
    _write_reference_artifacts(result)
    print_summary("poisson_static", selected, result)


if __name__ == "__main__":
    main()
