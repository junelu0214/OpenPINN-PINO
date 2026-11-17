"""Two-material one-dimensional diffusion baseline and smoke example."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from ._runner import build_parser, print_summary, run_config, select_config
except ImportError:  # Allows ``python examples/heterogeneous_poisson.py``.
    from _runner import build_parser, print_summary, run_config, select_config


def run(
    *,
    smoke: bool = False,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> Any:
    """Run heterogeneous Poisson through the public ``openpinn`` API."""

    selected = select_config(
        "heterogeneous_poisson", smoke=smoke, custom=config_path
    )
    return run_config(selected, output_dir=output_dir)


def main() -> None:
    parser = build_parser("heterogeneous Poisson")
    args = parser.parse_args()
    selected = select_config(
        "heterogeneous_poisson", smoke=args.smoke, custom=args.config
    )
    result = run_config(selected, output_dir=args.output_dir)
    print_summary("heterogeneous_poisson_static", selected, result)


if __name__ == "__main__":
    main()
