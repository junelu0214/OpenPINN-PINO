# Specification Validation Agent: final ProblemSpec/MethodSpec gate

The Specification Validation Agent, formerly called Agent A, is a structured,
pre-codegen validation gate that checks
`ProblemSpec` and `MethodSpec` inputs before training or code generation
proceeds.  It is purely library/CLI-based and does not depend on an LLM,
web UI, or interactive chat interface.

---

## What it does

1. **Composes M2 schema validation** — delegates to
   `openpinn.spec.load_problem_spec` / `load_method_spec` for pydantic
   schema enforcement; does *not* fork or rewrite the schema.
2. **Adds agent-level checks** — boundary ID uniqueness, codegen-readiness
   whitelists (physics, dimensions, geometry, constitutive model), and
   optional static code compliance (import whitelist, no PDE residuals
   inside `forward()`).
3. **Returns a structured report** — `ValidationReport` with `ok: bool`,
   `errors` (with code/path/message), and optional `warnings`.
4. **Blocks codegen on failure** — when used as a pre-gate via
   `from_specs(pre_validate=True)`, invalid specs raise a `ValueError`
   with the full error report before any codegen or training work starts.

---

## Relation to M2 and M3

| Layer | Purpose |
|-------|---------|
| **M2** (`openpinn.spec`) | Pydantic schema: field types, value ranges, cross-field rules. Source of truth for YAML validation. |
| **Specification Validation Agent** (`openpinn.agents.validation` (legacy `openpinn.agents.validator` remains supported)) | Composes M2 validation + codegen readiness + code compliance. Does not mutate YAML. |
| **M3** (`openpinn.codegen`) | Converts validated specs to runnable PDEProblem + solve kwargs. `from_specs` runs the Validation Agent by default on the same normalized pair it sends to codegen. |

---

## Quick start

### Python API

```python
from openpinn.agents import validate_specs

report = validate_specs(
    "examples/specs/plate_with_hole_problem.yaml",
    "examples/specs/pinn_method.yaml",
)

if report.ok:
    print("Ready for codegen")
else:
    for err in report.errors:
        print(f"[{err.code}] {err.path}: {err.message}")
```

### Pre-codegen gate in M3

```python
from openpinn.codegen import from_specs

# Raises ValueError if Specification Validation Agent finds errors
problem, kwargs = from_specs(
    "problem.yaml",
    "method.yaml",
    pre_validate=True,
)
```

### CLI

```bash
# Exits 0 on success, non-zero on failure
python -m openpinn.agents.validator problem.yaml method.yaml

# JSON output
python -m openpinn.agents.validator problem.yaml method.yaml --json

# With code compliance check
python -m openpinn.agents.validator problem.yaml method.yaml --code generated.py
```

---

## `validate_specs()` signature

```python
def validate_specs(
    problem: ProblemSpec | str | dict,
    method: MethodSpec | str | dict,
    *,
    code: str | None = None,
) -> ValidationReport:
```

**Parameters:**

- `problem` — A `ProblemSpec` instance, path to a YAML file, or raw dict.
- `method` — The final `MethodSpec` instance, path, or dict. Final-combination validation requires it.
- `code` — Optional Python source string for static compliance checking.

**Returns** a `ValidationReport`:

```python
@dataclass
class ValidationReport:
    ok: bool                           # True when no errors
    errors: list[ValidationError]      # blocking issues
    warnings: list[ValidationError]    # non-blocking advisories
    problem_spec: ProblemSpec | None
    method_spec: MethodSpec | None
```

Each `ValidationError` has:

| Field | Type | Description |
|-------|------|-------------|
| `code` | `str` | Stable error code, e.g. `schema.missing`, `codegen.unsupported_physics` |
| `path` | `str` | Dot-separated field path, e.g. `problem.constitutive.E` |
| `message` | `str` | Human-readable explanation |
| `severity` | `str` | `error` or `warning` |

---

## Checks performed

### M2 schema (delegated)

All pydantic-enforced rules from `ProblemSpec` and `MethodSpec`:
field types, value ranges, extra-key rejection, cross-field constraints
(bounds ordering, BC completeness, dof consistency, etc.).

### Agent-level checks

| Code | Description |
|------|-------------|
| `codegen.unsupported_dim` | dim not in M3 supported set (currently {2}) |
| `codegen.unsupported_physics` | physics not supported for codegen |
| `codegen.unsupported_geometry` | outer geometry type not supported |
| `codegen.unsupported_constitutive` | constitutive model not supported |
| `codegen.unsupported_field_kind` | field kind not supported |
| `codegen.grid_differentiator_pinn` | *(warning)* grid diff with PINN path |
| `boundary.duplicate_id` | duplicate boundary condition ids |
| `boundary.traction_free_explicit` | *(warning)* explicit zero-traction BC |

### Code compliance (when `code` argument provided)

| Code | Description |
|------|-------------|
| `code.syntax_error` | generated code cannot be parsed |
| `code.forbidden_import` | import not in allowed whitelist |
| `code.pde_in_forward` | PDE residual computation inside `forward()` |

---

## Architecture constraints

- **Core must not import agents**: `openpinn.agents` is an optional package.
  Core modules (`spec.py`, `codegen.py`, `training.py`, etc.) never import
  it at module level.  The `from_specs(pre_validate=True)` path uses a
  lazy import inside the function body.
- **No schema duplication**: Specification Validation Agent reuses M2 pydantic models as-is.
- **No YAML mutation**: Specification Validation Agent is read-only; it never rewrites or
  silently fixes input specs.
- **No LLM dependency**: validation is purely deterministic.
- **Browser boundary**: the static documentation can run only a shape or
  exported-contract precheck. It never labels that result as full Python
  Validation. Planning imports keep copy/download locked until the final pair is
  checked in Python; exported plate scripts call `validate_specs` again before
  `from_specs`.
- **M35 generic browser layer**: a version 1 `GenericProblemDescription`
  retains free-form physics labels, multiple fields, coordinates, parameters,
  equation forms, geometry, boundary and initial conditions, couplings, and
  opaque material. Its `LayeredValidationReport` uses `pass`, `fail`,
  `unknown`, `unsupported`, and `skipped`; it has no aggregate `ok` that could
  be mistaken for trainability. The well-posedness and physical-correctness
  layer is always `unknown`, and the restricted expression checker never
  evaluates user input.
- **Generic export boundary**: the browser can download the generic problem,
  method draft, and report only as an explicitly labelled non-training
  artifact. Runnable training export stays locked because the static browser
  has no same-source Python target gate. The Executable adapter preserves all
  supported fields or rejects with a path and reason. A successful mapping
  still requires M34 Python Validation and cannot bypass it.
- **M36 Hybrid semantic track**: deterministic structure results remain
  separate from a browser-direct OpenAI-compatible semantic review. Semantic
  status is bounded to `no_concern_identified`, `concerns`, or
  `insufficient_information`; each concern must cite an existing input path
  and an excerpt actually present at that path. These are model observations,
  never proof. Transport, CORS, authentication, timeout, cancellation, stale
  response, or schema failure leaves semantic review incomplete without a
  rules fallback. Validation API settings are independent unless the user
  explicitly chooses to reuse Planning settings. Keys never enter reports or
  downloads. Neither semantic output nor an unregistered method can override
  deterministic failures, M34, or the training-export lock.
- **M37 Code Building Agent**: the browser accepts only a current
  `HybridValidationReport` whose deterministic track has no failure and whose
  semantic track is complete. It deterministically emits a generic Python
  assembly skeleton plus a `BuildReport`; the product path does not require
  the legacy elasticity adapter. Missing template fields, a stale fingerprint,
  or incomplete semantics hard-block source output. Semantic concerns may
  proceed only when they remain visible in both outputs. Every successful
  result declares `backend_runnable=false`, leaves backend assembly explicitly
  unimplemented, and starts no training. No LLM writes or explains code.
- **M38 Diagnostic Evaluation Agent**: generated generic skeletons write
  `openpinn_run/run_artifact.json` when invoked, including the pair fingerprint,
  backend capability, run status, metric slots, and failure-stack slots. The
  current generic skeleton records `not_runnable`. The browser reads JSON or a
  ZIP containing that artifact locally, rejects stale or invalid fingerprints,
  and keeps deterministic findings separate from a configurable real LLM
  diagnosis. Transport or schema failures remain incomplete without a rules
  fallback. `DiagnosticReport` repair hints are non-binding;
  `repair_performed=false`, and no code, spec, or hyperparameter is mutated.
- **Stage A trainable-problem mapping contract**: the existing generic problem
  now carries first-class coordinate bounds/units, field identity/units,
  canonical parameters, equations, geometry, boundaries, and initial
  conditions. `trainable_contract.mapping_complete` is a deterministic mapping
  signal for later stages, not proof or current backend capability. An
  equivalent unit-less `diffusivity` is explicitly normalized to the
  unit-bearing `alpha` referenced by the equation; conflicts remain visible and
  incomplete. Generic CodeBuilder remains non-runnable with its explicit
  `NotImplementedError`; shared engines and physics adapters are later stages.
- **Scope**: the current final-spec engine supports the implemented 2D
  linear-elastic, plane-stress plate path. Unsupported physics and plane strain
  fail or receive an explicit warning. Passing Validation does not prove that a
  boundary-value problem is well posed, that training converges, or that the
  physical model is correct. Kirsch's infinite-plate solution is not an exact
  acceptance oracle for a finite fixed-and-loaded plate.
