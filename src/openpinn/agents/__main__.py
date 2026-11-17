"""CLI entry point: ``python -m openpinn.agents.validator``.

Validates one ProblemSpec YAML and optionally a MethodSpec YAML,
printing a structured report and exiting non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import sys

from .validator import validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m openpinn.agents.validator",
        description=(
            "Specification Validation Agent: validate ProblemSpec/MethodSpec YAML and check "
            "codegen readiness before training or code generation."
        ),
    )
    parser.add_argument(
        "problem",
        help="Path to a ProblemSpec YAML file.",
    )
    parser.add_argument(
        "method",
        nargs="?",
        default=None,
        help="Path to a MethodSpec YAML file (optional).",
    )
    parser.add_argument(
        "--code",
        default=None,
        help="Path to a generated Python file to check for compliance.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output the report as JSON instead of human-readable text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    code_text: str | None = None
    if args.code is not None:
        with open(args.code, "r", encoding="utf-8") as fh:
            code_text = fh.read()

    report = validate(
        problem=args.problem,
        method=args.method,
        code=code_text,
    )

    if args.json_output:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        if report.ok:
            print("PASS: all validation checks passed.")
        else:
            print("FAIL: validation errors detected.\n")

        if report.errors:
            print(f"Errors ({len(report.errors)}):")
            for err in report.errors:
                print(f"  [{err.code}] {err.path}: {err.message}")

        if report.warnings:
            print(f"\nWarnings ({len(report.warnings)}):")
            for w in report.warnings:
                print(f"  [{w.code}] {w.path}: {w.message}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
