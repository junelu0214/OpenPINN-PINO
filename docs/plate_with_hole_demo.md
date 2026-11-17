# Tutorial: 2D plate with a hole

This tutorial describes the M3 codegen and plate-with-hole demo that builds
on the M2 `ProblemSpec`/`MethodSpec` YAML schema. It is intentionally not a
solution-path option in the Sphinx configurator. The runnable example,
codegen, Kirsch reference, and tests remain supported here.

## M2 to M3 flow

1. **M2** defined `ProblemSpec` and `MethodSpec` pydantic models in
   `src/openpinn/spec.py`, plus validated example YAMLs under
   `examples/specs/`.
2. **M3** adds `src/openpinn/codegen.py` which consumes validated specs
   and produces a runnable `PDEProblem` + solver kwargs that feed
   directly into the existing `openpinn.solve()` API.  No parallel
   stack is introduced.

## Quick start

```bash
# Smoke test (10 steps, ~3 seconds, CPU, CI-safe)
python examples/plate_with_hole.py --smoke

# Longer training (500 steps, default)
python examples/plate_with_hole.py

# Custom step count
python examples/plate_with_hole.py --steps 5000
```

The script loads two YAML files from `examples/specs/`:

- `plate_with_hole_problem.yaml` — 2D rectangle (2 m x 2 m) with a
  0.25 m radius central hole, steel (E = 210 GPa, nu = 0.3), left
  edge fixed, right edge uniaxial traction 1 MPa, other edges
  traction-free.
- `pinn_method.yaml` — baseline PINN, 4-layer MLP, Adam optimizer.

## Codegen API

```python
from openpinn.codegen import from_specs
import openpinn as op

problem, kwargs = from_specs(
    "examples/specs/plate_with_hole_problem.yaml",
    "examples/specs/pinn_method.yaml",
    steps=500,
)
result = op.solve(problem, **kwargs)
```

`from_specs` validates both YAMLs through the M2 pydantic models,
builds a `PDEProblem` with 2D isotropic plane-stress equilibrium
residuals via autodiff, maps boundary conditions (Dirichlet and
stress-based traction Neumann), and returns kwargs compatible with the
existing `openpinn.solve()` path.

## Kirsch analytical comparison

The module `src/openpinn/kirsch.py` provides the classical Kirsch
closed-form solution for an infinite plate with a circular hole under
uniaxial tension:

- `kirsch_stress_polar` / `kirsch_stress_cartesian` — stress field.
- `kirsch_displacement` — displacement field (plane stress or strain).
- `kirsch_hoop_stress_at_hole` — hoop stress at the hole boundary.
- `kirsch_scf()` — returns 3.0 (the infinite-plate stress concentration
  factor).
- `compare_displacement_l2` — relative L2 displacement error between
  PINN predictions and the Kirsch reference.

### Honest limits of the Kirsch comparison

The Kirsch solution is exact only for an **infinite** plate.  The
example geometry is a **finite** 2 m x 2 m plate with a 0.25 m radius
hole (plate-to-hole ratio = 8).  This means:

- **Near the hole** (r < ~4a = 1 m): the Kirsch solution is a good
  approximation and the comparison is meaningful.
- **Near the outer boundaries**: the Kirsch solution diverges from the
  finite-plate solution because the traction-free and fixed boundary
  conditions on the outer edges are not present in the infinite plate.
  A large Kirsch mismatch at these locations does NOT indicate a PINN
  defect.
- **Short training**: a 10-step smoke run cannot converge to a
  physically meaningful displacement field.  The Kirsch comparison is
  scaffolded so the path is exercised, but numeric thresholds require
  longer training and are not enforced in CI.

## Tests

```bash
# M3 unit tests (codegen + Kirsch, ~2 seconds)
python -m pytest tests/unit/test_codegen.py tests/unit/test_kirsch.py -v

# M3 smoke tests (short training, ~3 seconds)
python -m pytest tests/smoke/test_plate_with_hole_smoke.py -v

# All tests (M2 + M3 + existing)
python -m pytest tests/ -q
```

## Files added

| File | Purpose |
|------|---------|
| `src/openpinn/codegen.py` | ProblemSpec+MethodSpec -> PDEProblem + solve kwargs |
| `src/openpinn/kirsch.py` | Kirsch analytical reference (stress, displacement, SCF) |
| `examples/plate_with_hole.py` | Runnable demo entrypoint |
| `tests/unit/test_codegen.py` | Unit tests: codegen positive + negative cases |
| `tests/unit/test_kirsch.py` | Unit tests: Kirsch analytical formulae |
| `tests/smoke/test_plate_with_hole_smoke.py` | Smoke: short training + Kirsch comparison path |
| `docs/plate_with_hole_demo.md` | This document |
