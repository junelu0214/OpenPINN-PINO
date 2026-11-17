# ProblemSpec and MethodSpec Guide (M2)

OpenPINN uses two machine-readable YAML specifications to describe a physics
problem and a solver configuration **before** any training or code generation
runs.  The validation layer catches errors (missing fields, bad types, physics
violations) up front so that a later M3 codegen step can trust its inputs.

---

## Quick start

```python
from openpinn.spec import load_problem_spec, load_method_spec

problem = load_problem_spec("plate_with_hole_problem.yaml")
method  = load_method_spec("pinn_method.yaml")
# Both return pydantic model instances; invalid specs raise ValidationError.
```

---

## ProblemSpec schema

A ProblemSpec YAML must contain these top-level keys:

| Key            | Type    | Required | Description |
|----------------|---------|----------|-------------|
| `dim`          | `2`     | yes      | Spatial dimension (currently only 2D). |
| `physics`      | enum    | yes      | Physics type. Allowed: `linear_elasticity`. |
| `geometry`     | object  | yes      | Domain definition (see below). |
| `field`        | object  | yes      | Physical field (see below). |
| `constitutive` | object  | yes      | Material / constitutive law (see below). |
| `boundaries`   | list    | yes      | Boundary conditions (see below). |
| `units`        | object  | no       | Unit labels (defaults: m, N, Pa). |

Unknown keys at any level are rejected.

### `geometry`

```yaml
geometry:
  outer:
    type: rectangle
    x_min: 0.0
    x_max: 2.0
    y_min: 0.0
    y_max: 2.0
  holes:
    - type: circle
      center: [1.0, 1.0]
      radius: 0.25
```

- `outer` must be a `rectangle` with `x_min < x_max` and `y_min < y_max`.
- `holes` is an optional list of `circle` entries.  Each hole must lie
  **strictly** inside the outer rectangle.  Holes must not overlap.

### `field`

```yaml
field:
  kind: displacement
  components: [ux, uy]
```

- `kind`: currently `displacement`.
- `components`: unique non-empty list of component names.

### `constitutive`

```yaml
constitutive:
  model: isotropic_linear
  E: 210.0e9
  nu: 0.3
  stress_voigt: true
```

- `model`: currently `isotropic_linear`.
- `E`: Young's modulus, must be **strictly positive**.
- `nu`: Poisson's ratio, must satisfy `-1.0 <= nu < 0.5` (2D isotropic constraint).
- `stress_voigt`: boolean, defaults to `true`.

### `boundaries`

```yaml
boundaries:
  - id: fix_left
    region: left
    type: dirichlet
    dofs: [ux, uy]
    value: [0.0, 0.0]
```

- `id`: unique human-readable label.
- `region`: edge name (`left`, `right`, `top`, `bottom`, `hole_0`, `hole_1`, ...).
- `type`: `dirichlet` or `neumann`.
- `dofs`: list of field component names this BC constrains.
- `value`: scalar (applied to all listed dofs) or list matching `len(dofs)`.

**Validation rules:**

1. Every outer edge (`left`, `right`, `top`, `bottom`) and every hole
   boundary (`hole_0`, ...) must have at least one BC.
2. The same degree of freedom on the same region cannot have both a
   Dirichlet and a Neumann condition (conflict).
3. `dofs` entries must be among the declared `field.components`.

### `units`

```yaml
units:
  length: m
  force: N
  stress: Pa
```

Informational.  If omitted, defaults to SI (m / N / Pa).

---

## MethodSpec schema

| Key              | Type    | Required | Description |
|------------------|---------|----------|-------------|
| `method`         | string  | yes      | Registered method or operator name. |
| `variant`        | string  | no       | Variant within the method family. |
| `path`           | enum    | no       | `pinn` (default) or `operator`. |
| `sampling`       | object  | no       | Sampling config (see below). |
| `loss`           | object  | no       | Loss config (see below). |
| `optimizer`      | object  | no       | Optimizer config (see below). |
| `network`        | object  | no       | Network architecture. |
| `differentiator` | object  | no       | Derivative backend selected from the component registry. |
| `training`       | dict    | no       | Extra training params (e.g. `steps`). |

### Method name validation

- **PINN path** (`path: pinn`): `method` and `variant` must exist in the
  OpenPINN method registry.  Available families include `baseline`, `fourier`,
  `adaptive_weights`, `adaptive_sampling`, `modified_mlp`, `hard_boundary`,
  `adam_lbfgs`, `adaptive_activation`, `time_window`, `transfer`, and
  `domain_decomposition`.
- **Operator path** (`path: operator`): `method` must be a selectable operator
  model hook (`deeponet` or `fno`). PINO is not a model choice. `variant` is
  not supported on this path.

### `sampling`

```yaml
sampling:
  name: uniform
  interior: 512
  boundary: 128
```

- `name` must be a registered **sampler** (`uniform`, `residual_adaptive`, ...).
- `interior` and `boundary` must be >= 1.

### `loss`

```yaml
loss:
  formulation: strong
  weighting: fixed
  weights:
    pde: 1.0
    boundary: 10.0
```

- `formulation` must be a registered **formulation** (`strong`, `interface_strong`).
- `weighting` must be a registered **loss** (`fixed`, `self_adaptive`, `relobralo`, ...).
  Defaults to `fixed`.  Unknown names are rejected; there is no silent fallback.

### `optimizer`

```yaml
optimizer:
  name: adam
  lr: 0.001
```

- `name` must be a registered **optimizer** (`adam`, `adamw`, `lbfgs`, `adam_lbfgs`).
- `lr`: must be > 0.

### `network`

- `name` must be a registered **model** (`mlp`, `modified_mlp`, `domain_decomposition`).

### `differentiator`

- `mode` must be a registered **derivative** (`autodiff`, `grid`, `finite_difference`).

---

## Running validation

```bash
# From Python
python -c "
from openpinn.spec import load_problem_spec, load_method_spec
load_problem_spec('examples/specs/plate_with_hole_problem.yaml')
load_method_spec('examples/specs/pinn_method.yaml')
print('Both specs are valid.')
"
```

Invalid specs raise `pydantic.ValidationError` with per-field messages:

```
pydantic.ValidationError: 1 validation error for ProblemSpec
constitutive.E
  Input should be greater than 0 [type=greater_than, ...]
```

### Programmatic validation without exceptions

```python
from openpinn.spec import validate_problem_spec, validate_method_spec

errors = validate_problem_spec("problem.yaml")
if errors:
    print("Invalid:", errors)
```

---

## Example files

See `examples/specs/` for ready-to-use examples:

- `plate_with_hole_problem.yaml` — 2D plate with a central hole under uniaxial tension.
- `pinn_method.yaml` — Plain PINN baseline solver configuration.
- `deeponet_method.yaml` — DeepONet operator-learning path.

---

## Design notes

- Schemas use **pydantic v2** with `extra="forbid"` so unknown keys are
  always rejected rather than silently ignored.
- The spec layer is deliberately separated from training and codegen: M3
  will consume validated `ProblemSpec` / `MethodSpec` objects to generate
  runnable training scripts and Kirsch analytical comparisons.
- `MethodSpec` validates names against the **live** method, operator, and
  component registries (`model`, `sampler`, `loss`, `optimizer`,
  `derivative`), so newly registered methods and components are accepted
  and unknown names fail with an explicit error.  See `docs/extension.md`.
