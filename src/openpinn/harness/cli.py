"""JSON command line interface for the OpenPINN workflow harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core import check_task, initialize_task, inspect_run, run_task, workflow_reference


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openpinn harness",
        description="Run agent-authored scientific code with an OpenPINN-inspired workflow and evidence contract.",
    )
    commands = parser.add_subparsers(dest="harness_command", required=True)
    commands.add_parser("workflow", help="Print the workflow, planning axes, and task contract.")
    init = commands.add_parser("init", help="Create an unrestricted agent workspace.")
    init.add_argument("task_dir", type=Path)
    check = commands.add_parser("check", help="Check workspace structure and Python syntax, not physics capability.")
    check.add_argument("task_dir", type=Path)
    run = commands.add_parser("run", help="Execute the agent-authored Python entrypoint with a wall-time bound.")
    run.add_argument("task_dir", type=Path)
    inspect = commands.add_parser("inspect", help="Inspect a task directory, run directory, or Harness artifact.")
    inspect.add_argument("source", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.harness_command == "workflow":
        result = workflow_reference()
    elif args.harness_command == "init":
        result = initialize_task(args.task_dir)
    elif args.harness_command == "check":
        result = check_task(args.task_dir)
    elif args.harness_command == "run":
        result = run_task(args.task_dir)
    else:
        result = inspect_run(args.source)
    _print(result)
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
