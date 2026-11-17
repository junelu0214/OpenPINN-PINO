"""Command-line entry point for the compact OpenPINN API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .api import build_config, result_summary, solve, solve_config
from .method_registry import list_method_variants, list_methods


def _value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _options(items: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Method option {item!r} must use key=value syntax.")
        key, value = item.split("=", 1)
        values[key] = _value(value)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openpinn",
        description="Define/select a PDE, choose one method name, train and save outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a compact YAML or command-line experiment.")
    run.add_argument("--config", type=Path, help="Compact YAML file.")
    run.add_argument("--problem", default="poisson", help="Built-in problem shorthand.")
    run.add_argument("--method", default="baseline", help="Method shorthand.")
    run.add_argument("--variant", help="Concrete variant inside the method family.")
    run.add_argument(
        "--steps",
        type=int,
        help="Training steps. Overrides training.steps when --config is used.",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="cpu")
    run.add_argument("--output-dir", type=Path)
    run.add_argument(
        "--method-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeat for method-specific options; JSON values are accepted.",
    )

    methods = subparsers.add_parser("list-methods", help="List method shorthands.")
    methods.add_argument(
        "--family",
        help="Show the concrete variants registered under one method family.",
    )
    methods.set_defaults(command="list-methods")

    show = subparsers.add_parser("show-config", help="Print the resolved low-level config.")
    show.add_argument("--problem", default="poisson")
    show.add_argument("--method", default="baseline")
    show.add_argument("--variant")
    show.add_argument("--steps", type=int, default=1000)

    harness = subparsers.add_parser(
        "harness",
        add_help=False,
        help="Use the deterministic JSON interface for external general-purpose agents.",
    )
    harness.add_argument("harness_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv and effective_argv[0] == "harness":
        from .harness.cli import main as harness_main

        return harness_main(effective_argv[1:])
    args = build_parser().parse_args(effective_argv)
    if args.command == "list-methods":
        if args.family:
            for variant in list_method_variants(args.family):
                print(
                    f"{args.family:22} {variant.name:24} "
                    f"{variant.variant_file:48} {variant.title}"
                )
            return 0
        for spec in list_methods():
            print(
                f"{spec.name:22} {spec.default_variant:24} "
                f"{spec.component_file:34} {spec.title}"
            )
        return 0
    if args.command == "show-config":
        config, _ = build_config(
            args.problem,
            method=args.method,
            variant=args.variant,
            steps=args.steps,
        )
        print(json.dumps(config, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.config is not None:
        result = solve_config(
            args.config,
            output_dir=args.output_dir,
            steps=args.steps,
        )
    else:
        result = solve(
            args.problem,
            method=args.method,
            variant=args.variant,
            steps=1000 if args.steps is None else args.steps,
            seed=args.seed,
            device=args.device,
            method_options=_options(args.method_option),
            output_dir=args.output_dir,
        )
    print(result_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
