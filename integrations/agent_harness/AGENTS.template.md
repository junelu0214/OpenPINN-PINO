# Solving scientific problems with OpenPINN

The general-purpose agent owns problem understanding, method selection, code,
verification, and diagnosis. Do not stop because a problem is not registered in
OpenPINN. Prefer public OpenPINN components and extension points, then implement
task-local code when necessary.

Use `openpinn harness workflow → init → check → run → inspect`. Maintain
`problem.md` and `plan.md`, start with a small smoke, write `run_result.json`, and
keep Agent claims separate from Harness-observed evidence. Never infer convergence
or accuracy from process success, finite numbers, or loss reduction alone. Read
the full workflow in `integrations/agent_harness/SKILL.md`.
