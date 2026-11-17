# Second development: registries, custom PDEs, and custom BCs (M4)

Built-in components are enough to train a PINN. Second development is
registering a new factory, or writing a `PDEProblem` with your residual and
boundary conditions. Names in MethodSpec / experiment YAML resolve through
`ComponentRegistry`. Unknown names raise `ValueError`; there is no silent
fallback to a baseline.

---

## Public registries

Five kinds are first-class and support `register`, `available`, and `create`:

| Kind | Built-in names (canonical) | Object contract |
|------|----------------------------|-----------------|
| `model` | `mlp`, `modified_mlp`, `domain_decomposition` | `torch.nn.Module` with `output_dim` |
| `sampler` | `uniform`, `residual_adaptive`, `adaptive_movement`, `importance_sampling`, `time_window` | `SamplerProtocol` |
| `loss` | `fixed`, `self_adaptive`, `pointwise_self_adaptive`, `uncertainty`, `relobralo`, `magnitude_power` | object with `compose(terms)` |
| `optimizer` | `adam`, `adamw`, `lbfgs`, `adam_lbfgs` | `OptimizerAdapter` |
| `derivative` | `autodiff`, `grid`, `finite_difference` | `grad`/`hessian` (autodiff) or `partial` (grid) |

The same registry also keeps `problem`, `feature`, `constraint`, and
`formulation`. Prefer extending this object over rewriting `Experiment`.

```python
import openpinn as op

registry = op.default_component_registry()
print(registry.available("model"))
print(registry.available("loss"))
model = registry.create(
    "model", "mlp",
    config={"name": "mlp", "hidden_layers": [8, 8]},
    problem=op.make_poisson_problem(),
    registry=registry,
)
derivative = registry.create("derivative", "autodiff")
u_x = derivative.grad(model(x), x)  # x.requires_grad must be True
```

Register a project-specific factory on a **copy** (each call to
`default_component_registry()` is isolated):

```python
registry = op.default_component_registry()
registry.register("loss", "my_loss", my_loss_factory)
# YAML name "my_loss" now resolves on this registry only.
op.Experiment.from_config(config, registry=registry)
```

Unknown names fail immediately:

```python
registry.create("sampler", "does_not_exist")
# ValueError: Component sampler:'does_not_exist' is not registered; available: ...
```

MethodSpec YAML uses the same rule. `sampling.name`, `network.name`,
`loss.weighting`, `optimizer.name`, and `differentiator.mode` must exist in
the live component registry.

---

## Custom PDE

A PDE is a callable residual plus geometry and conditions, not a network
class. Put the equation on `PDEProblem`, then optionally register it as a
`problem` factory.

```python
import math
import openpinn as op
import torch
from torch import Tensor


def build_problem() -> op.PDEProblem:
    domain = op.Interval(-1.0, 1.0)

    def exact(x: Tensor) -> Tensor:
        return torch.sin(math.pi * x[:, :1])

    def residual(x: Tensor, u: Tensor) -> Tensor:
        # -u_xx - u = (pi^2 - 1) sin(pi x)
        return (
            -op.hessian(u, x, i=0, j=0)
            - u
            - (math.pi**2 - 1.0) * torch.sin(math.pi * x[:, :1])
        )

    return op.PDEProblem(
        domain,
        residual,
        [op.DirichletBC(0.0, name="boundary")],
        exact_solution=exact,
        name="helmholtz_1d",
        metadata={"kind": "custom_static", "residual_names": ("pde",)},
    )


registry = op.default_component_registry()
registry.register("problem", "helmholtz_1d", build_problem)
problem = registry.create("problem", "helmholtz_1d")
result = op.solve(problem, method="baseline", steps=5)
```

A runnable copy lives in `examples/custom_pde.py`.

You can also pass the `PDEProblem` object to `op.solve` without registering
it. Registration is needed when a YAML `problem.name` should resolve to your
equation.

---

## Custom boundary condition

Subclass `openpinn.conditions.Condition`, implement `error(model, points)`,
and pass the instance into `PDEProblem`. Built-in `DirichletBC`, `NeumannBC`,
`RobinBC`, and `InitialCondition` follow the same contract.

```python
from openpinn.conditions import Condition, evaluate_value


class EndpointFluxBC(Condition):
    """Prescribe du/dx at selected 1D boundary points."""

    def __init__(self, value, boundary=None, *, name="flux", weight=1.0):
        super().__init__(value=value, name=name, weight=weight, selector=boundary)
        self.sample_kind = "boundary"

    def error(self, model, points):
        selected = self.select(points)
        if not selected.requires_grad:
            selected = selected.detach().requires_grad_(True)
        prediction = model(selected)
        flux = op.grad(prediction, selected)[:, :1]
        target = evaluate_value(self.value, selected, like=flux)
        return flux - target
```

A runnable mixed Dirichlet / flux example lives in `examples/custom_bc.py`.

There is no separate `condition` registry kind. Custom BCs are objects on a
`PDEProblem`; register the **problem** if YAML should select it by name.

---

## Do not write PDE residuals inside `network.forward`

`forward` returns the physical field (or a hard-constraint transform of it).
The residual belongs on `PDEProblem.residual` / `evaluate_residual`.
Formulations and trainers call that residual; the network does not.

This is the same rule M3 codegen follows: it builds a `PDEProblem` and does
not hand-write residuals into the network.

```python
# Wrong — residual inside forward
class BadNet(torch.nn.Module):
    def forward(self, x):
        u = self.net(x)
        residual = -op.hessian(u, x, i=0, j=0) - f(x)
        return residual

# Right — network predicts u; PDEProblem owns the residual
class MLP(torch.nn.Module):
    def forward(self, x):
        return self.net(x)
```

The Specification Validation Agent (`openpinn.agents.validation`, with the
legacy `openpinn.agents.validator` import still supported) also rejects generated
`forward` methods that call `grad` / `hessian`. The core helper below is independent
of `openpinn.agents` and can be used on library or project source:

```python
import inspect
import openpinn as op

op.assert_no_pde_residual_in_forward(inspect.getsource(op.PINNModel))
op.assert_no_pde_residual_in_forward(path_to_my_model_file)
```

The checker flags `evaluate_residual` / `make_physics_residual` / `.residual()`
inside `forward`, and assignments named `residual` that call `grad` or
`hessian`. It inspects source with `ast`; it does not execute the network.

---

## What this milestone does not change

- No rewrite of `Experiment`. Loss construction now goes through
  `registry.create("loss", ...)`, matching model / sampler / optimizer.
- No new physics families, 3D, contact, or web UI.
- Core training and codegen do not import `openpinn.agents`.
