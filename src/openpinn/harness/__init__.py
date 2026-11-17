"""Agent-facing OpenPINN workflow harness.

The harness deliberately contains no language-model loop or physics capability
gate. General-purpose agents own the scientific solution; this package provides
workflow scaffolding, bounded execution, and machine-readable evidence.
"""

from .core import (
    check_task,
    initialize_task,
    inspect_run,
    load_task,
    run_task,
    task_template,
    workflow_reference,
)
from .contracts import HarnessTask

__all__ = [
    "HarnessTask",
    "check_task",
    "initialize_task",
    "inspect_run",
    "load_task",
    "run_task",
    "task_template",
    "workflow_reference",
]
