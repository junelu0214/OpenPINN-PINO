"""Two-field manufactured coupling and extensibility smoke example."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from ._runner import build_parser, print_summary, run_config, select_config
except ImportError:  # Allows ``python examples/coupled_manufactured.py``.
    from _runner import build_parser, print_summary, run_config, select_config


def run(
    *,
    smoke: bool = False,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> Any:
    """Run the coupled contract case through the public OpenPINN API."""

    selected = select_config(
        "coupled_manufactured", smoke=smoke, custom=config_path
    )
    return run_config(selected, output_dir=output_dir)


def main() -> None:
    parser = build_parser("coupled_manufactured")
    args = parser.parse_args()
    selected = select_config(
        "coupled_manufactured", smoke=args.smoke, custom=args.config
    )
    result = run_config(selected, output_dir=args.output_dir)
    print_summary("coupled_manufactured_contract", selected, result)


if __name__ == "__main__":
    main()
