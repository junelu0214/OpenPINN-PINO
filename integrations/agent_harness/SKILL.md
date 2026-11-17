---
name: openpinn-workflow
description: Solve scientific problems by authoring code in OpenPINN and using its workflow harness for bounded execution and evidence.
---

# OpenPINN Workflow Harness

You own the scientific solution. The Harness has no physics capability gate and
does not decide whether a problem is supported. Use OpenPINN components where
helpful; extend them or write task-local PyTorch code when needed.

## Required workflow

1. Run `openpinn harness workflow` to read the task and evidence contract.
2. Run `openpinn harness init openpinn_tasks/TASK_ID`.
3. Translate the user's problem into `problem.md`: fields, equations, domain,
   initial/boundary conditions, parameters, units, requested outputs, and open
   scientific choices. Ask the user only when a missing choice materially changes
   the physical problem; do not invent such a choice.
4. Write `plan.md`: representation, sampling, losses, optimizer, differentiation,
   nondimensionalization, smoke budget, reference evidence, metrics, and risks.
5. Implement `solution/run.py`. Prefer existing OpenPINN public APIs and extension
   points, but do not stop because the problem is absent from a registry. Add
   task-local components, operators, tests, or reference calculations as needed.
6. Read runtime paths and budgets from `OPENPINN_*` environment variables. Write
   `run_result.json` and declared artifacts under `OPENPINN_OUTPUT_DIR`.
7. Run `openpinn harness check TASK_DIR`; resolve blocking execution errors and
   consider every non-blocking advisory.
8. Run a small smoke using `openpinn harness run TASK_DIR`, then inspect with
   `openpinn harness inspect TASK_DIR`. Use a new output directory for each run.
9. Diagnose the evidence, revise code or method, update the task budget/output
   directory, and repeat. Expand compute only after the smoke is finite and useful.
10. Report Agent claims separately from Harness observations. Do not convert
    successful execution, finite output, or decreasing loss into an unsupported
    convergence or accuracy claim.

## Execution boundary

- The Harness executes your workspace Python as a subprocess without a shell.
- It is not an OS security sandbox; only use it in a trusted workspace.
- Keep all writes inside the declared output directory.
- Do not hard-code credentials. The Harness never needs a model API key.
- Treat timeout, nonzero exit, missing/invalid result JSON, and non-finite values
  as unsuccessful evidence that requires diagnosis.
