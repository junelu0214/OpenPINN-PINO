"""Generate runnable OpenPINN configuration from ProblemSpec + MethodSpec (M3).

This module bridges the M2 validated YAML schemas to the existing
``openpinn.solve`` / ``Experiment`` runtime.  It does NOT hand-write PDE
residuals into the network; instead it builds a ``PDEProblem`` with the
correct physics via autodiff, then maps MethodSpec fields onto the existing
configurator / build_config path.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .autodiff import grad, hessian
from .conditions import DirichletBC, Condition
from .geometry import RectangleWithCircularHoles
from .problems import PDEProblem
from .spec import ProblemSpec, MethodSpec, load_problem_spec, load_method_spec


def _plane_stress_constitutive(E: float, nu: float) -> tuple[float, float, float]:
    """Return (C11, C12, C66) for isotropic plane stress."""
    factor = E / (1.0 - nu**2)
    C11 = factor
    C12 = factor * nu
    C66 = factor * (1.0 - nu) / 2.0
    return C11, C12, C66


def _compute_stress(
    coords: Tensor, pred: Tensor, C11: float, C12: float, C66: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute Cauchy stress from displacement via autodiff."""
    ux_grad = grad(pred, coords, component=0)
    uy_grad = grad(pred, coords, component=1)
    ux_x = ux_grad[:, 0:1]
    ux_y = ux_grad[:, 1:2]
    uy_x = uy_grad[:, 0:1]
    uy_y = uy_grad[:, 1:2]

    eps_xx = ux_x
    eps_yy = uy_y
    gamma_xy = ux_y + uy_x

    sig_xx = C11 * eps_xx + C12 * eps_yy
    sig_yy = C12 * eps_xx + C11 * eps_yy
    sig_xy = C66 * gamma_xy
    return sig_xx, sig_yy, sig_xy


def _build_elasticity_residual(C11: float, C12: float, C66: float):
    """Return a residual callable for 2D isotropic linear elasticity (plane stress).

    PDE (strong form, equilibrium):
        d(sigma_xx)/dx + d(sigma_xy)/dy = 0
        d(sigma_xy)/dx + d(sigma_yy)/dy = 0
    """

    def residual(coords: Tensor, pred: Tensor) -> dict[str, Tensor]:
        sig_xx, sig_yy, sig_xy = _compute_stress(coords, pred, C11, C12, C66)

        sig_xx_x = grad(sig_xx, coords)[:, 0:1]
        sig_xy_y = grad(sig_xy, coords)[:, 1:2]
        sig_xy_x = grad(sig_xy, coords)[:, 0:1]
        sig_yy_y = grad(sig_yy, coords)[:, 1:2]

        return {
            "momentum_x": sig_xx_x + sig_xy_y,
            "momentum_y": sig_xy_x + sig_yy_y,
        }

    return residual


class TractionBC(Condition):
    """Traction (stress-based Neumann) BC for 2D linear elasticity.

    Prescribes sigma . n = t on a boundary, where sigma is computed from
    the constitutive law applied to the displacement field.
    """

    def __init__(
        self,
        traction_x: float,
        traction_y: float,
        normal_fn,
        C11: float,
        C12: float,
        C66: float,
        boundary=None,
        *,
        name: str = "traction",
    ) -> None:
        super().__init__(value=0.0, name=name, selector=boundary)
        self.traction_x = traction_x
        self.traction_y = traction_y
        self.normal_fn = normal_fn
        self.C11 = C11
        self.C12 = C12
        self.C66 = C66
        self.sample_kind = "boundary"

    def error(self, model, points: Tensor) -> Tensor:
        selected = self.select(points)
        if not selected.requires_grad:
            selected = selected.detach().requires_grad_(True)
        prediction = model(selected)

        sig_xx, sig_yy, sig_xy = _compute_stress(
            selected, prediction, self.C11, self.C12, self.C66
        )

        n = self.normal_fn(selected)
        nx = n[:, 0:1]
        ny = n[:, 1:2]

        tx_pred = sig_xx * nx + sig_xy * ny
        ty_pred = sig_xy * nx + sig_yy * ny

        tx_target = selected.new_tensor(self.traction_x)
        ty_target = selected.new_tensor(self.traction_y)

        err_x = tx_pred - tx_target
        err_y = ty_pred - ty_target
        return torch.cat([err_x, err_y], dim=1)


def _edge_selector(region: str, geom: RectangleWithCircularHoles):
    """Return a callable that selects points on a named edge/hole boundary."""
    o_lower = geom.lower
    o_upper = geom.upper

    if region == "left":
        def sel(pts: Tensor) -> Tensor:
            return torch.isclose(pts[:, 0], pts.new_tensor(o_lower[0]), atol=1e-7, rtol=0.0)
        return sel
    elif region == "right":
        def sel(pts: Tensor) -> Tensor:
            return torch.isclose(pts[:, 0], pts.new_tensor(o_upper[0]), atol=1e-7, rtol=0.0)
        return sel
    elif region == "bottom":
        def sel(pts: Tensor) -> Tensor:
            return torch.isclose(pts[:, 1], pts.new_tensor(o_lower[1]), atol=1e-7, rtol=0.0)
        return sel
    elif region == "top":
        def sel(pts: Tensor) -> Tensor:
            return torch.isclose(pts[:, 1], pts.new_tensor(o_upper[1]), atol=1e-7, rtol=0.0)
        return sel
    elif region.startswith("hole_"):
        idx = int(region.split("_", 1)[1])
        cx, cy, r = geom.holes[idx]

        def sel(pts: Tensor) -> Tensor:
            center = pts.new_tensor((cx, cy))
            dist = torch.linalg.vector_norm(pts - center, dim=1)
            return torch.isclose(dist, dist.new_tensor(r), atol=1e-5, rtol=0.0)
        return sel
    else:
        raise ValueError(f"Unknown boundary region: {region!r}")


def _edge_normal(region: str, geom: RectangleWithCircularHoles):
    """Return a callable outward normal for a named boundary region."""
    if region == "left":
        return lambda pts: pts.new_tensor([[-1.0, 0.0]]).expand(pts.shape[0], -1)
    elif region == "right":
        return lambda pts: pts.new_tensor([[1.0, 0.0]]).expand(pts.shape[0], -1)
    elif region == "bottom":
        return lambda pts: pts.new_tensor([[0.0, -1.0]]).expand(pts.shape[0], -1)
    elif region == "top":
        return lambda pts: pts.new_tensor([[0.0, 1.0]]).expand(pts.shape[0], -1)
    elif region.startswith("hole_"):
        idx = int(region.split("_", 1)[1])
        cx, cy, r = geom.holes[idx]

        def normal_fn(pts: Tensor) -> Tensor:
            center = pts.new_tensor((cx, cy))
            direction = center - pts
            return direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        return normal_fn
    else:
        raise ValueError(f"Unknown boundary region for normal: {region!r}")


def build_problem(spec: ProblemSpec) -> PDEProblem:
    """Convert a validated ProblemSpec into a runnable PDEProblem.

    Currently supports ``linear_elasticity`` with ``isotropic_linear``
    constitutive law on a 2D rectangle-with-holes domain.
    """
    if spec.physics.value != "linear_elasticity":
        raise ValueError(f"Unsupported physics type: {spec.physics.value}")
    if spec.constitutive.model.value != "isotropic_linear":
        raise ValueError(f"Unsupported constitutive model: {spec.constitutive.model.value}")
    if spec.constitutive.assumption.value != "plane_stress":
        raise ValueError(
            "The linear-elasticity code generator currently supports only "
            f"assumption='plane_stress', got {spec.constitutive.assumption.value!r}"
        )

    outer = spec.geometry.outer
    holes_tuples = tuple(
        (h.center[0], h.center[1], h.radius) for h in spec.geometry.holes
    )
    geom = RectangleWithCircularHoles(
        lower=(outer.x_min, outer.y_min),
        upper=(outer.x_max, outer.y_max),
        holes=holes_tuples,
        stratified_boundary=True,
    )

    C11, C12, C66 = _plane_stress_constitutive(spec.constitutive.E, spec.constitutive.nu)
    residual_fn = _build_elasticity_residual(C11, C12, C66)

    conditions: list[Condition] = []
    components = spec.field.components

    # Group BCs by region to build composite conditions
    region_bcs: dict[str, list] = {}
    for bc in spec.boundaries:
        region_bcs.setdefault(bc.region, []).append(bc)

    for region, bcs in region_bcs.items():
        selector = _edge_selector(region, geom)

        # Check if this region has any Dirichlet BCs
        dirichlet_bcs = [b for b in bcs if b.type.value == "dirichlet"]
        neumann_bcs = [b for b in bcs if b.type.value == "neumann"]

        if dirichlet_bcs:
            # Collect all Dirichlet dofs for this region
            dof_values: dict[str, float] = {}
            for db in dirichlet_bcs:
                vals = db.value if isinstance(db.value, list) else [db.value] * len(db.dofs)
                for dof_name, val in zip(db.dofs, vals):
                    dof_values[dof_name] = float(val)

            comp_vals = {components.index(d): v for d, v in dof_values.items()}

            def _make_dirichlet_fn(vmap, n_comp):
                def val_fn(pts: Tensor) -> Tensor:
                    out = torch.zeros(
                        pts.shape[0], n_comp, device=pts.device, dtype=pts.dtype
                    )
                    for ci, cv in vmap.items():
                        out[:, ci] = cv
                    return out
                return val_fn

            conditions.append(
                DirichletBC(
                    _make_dirichlet_fn(comp_vals, len(components)),
                    boundary=selector,
                    name=f"dirichlet_{region}",
                )
            )

        if neumann_bcs:
            # Collect all traction components for this region
            tx, ty = 0.0, 0.0
            for nb in neumann_bcs:
                vals = nb.value if isinstance(nb.value, list) else [nb.value] * len(nb.dofs)
                for dof_name, val in zip(nb.dofs, vals):
                    if dof_name == components[0]:
                        tx = float(val)
                    elif dof_name == components[1]:
                        ty = float(val)

            normal_fn = _edge_normal(region, geom)
            conditions.append(
                TractionBC(
                    tx, ty, normal_fn, C11, C12, C66,
                    boundary=selector,
                    name=f"traction_{region}",
                )
            )

    return PDEProblem(
        geom,
        residual_fn,
        conditions,
        name="plate_with_hole",
        field_names=tuple(components),
        output_dim=len(components),
        metadata={
            "kind": "static",
            "physics": spec.physics.value,
            "constitutive": {
                "E": spec.constitutive.E,
                "nu": spec.constitutive.nu,
                "assumption": spec.constitutive.assumption.value,
            },
            "residual_names": ("momentum_x", "momentum_y"),
        },
    )


def build_solve_kwargs(
    problem_spec: ProblemSpec,
    method_spec: MethodSpec,
    *,
    steps: int | None = None,
    device: str = "cpu",
) -> tuple[dict[str, Any], PDEProblem]:
    """Map ProblemSpec + MethodSpec into (kwargs, problem).

    The returned problem can be passed to ``op.solve(problem, **kwargs)``.
    """
    problem = build_problem(problem_spec)
    effective_steps = steps if steps is not None else method_spec.training.get("steps", 1000)

    kwargs: dict[str, Any] = {
        "method": method_spec.method,
        "steps": effective_steps,
        "device": device,
        "model": {
            "name": method_spec.network.name,
            "hidden_layers": method_spec.network.hidden_layers,
            "activation": method_spec.network.activation,
        },
        "sampling": {
            "name": method_spec.sampling.name,
            "interior": method_spec.sampling.interior,
            "boundary": method_spec.sampling.boundary,
        },
        "overrides": {
            "optimizer": {
                "name": method_spec.optimizer.name,
                "lr": method_spec.optimizer.lr,
            },
            "loss": {
                "formulation": method_spec.loss.formulation,
                "weighting": method_spec.loss.weighting,
            },
        },
    }

    if method_spec.variant is not None:
        kwargs["variant"] = method_spec.variant

    # MethodSpec loss weights use generic names (pde, boundary) that must
    # be mapped to the actual residual/condition names produced by the problem.
    # If the MethodSpec weights don't match, generate appropriate weights.
    residual_names = list(problem.metadata.get("residual_names", ("pde",)))
    condition_names = [c.name for c in problem.conditions]

    if method_spec.loss.weights is not None:
        spec_weights = method_spec.loss.weights
        pde_weight = spec_weights.get("pde", 1.0)
        bc_weight = spec_weights.get("boundary", 1.0)
        mapped_weights = {}
        for rn in residual_names:
            mapped_weights[rn] = pde_weight
        for cn in condition_names:
            mapped_weights[cn] = bc_weight
        kwargs["overrides"]["loss"]["weights"] = mapped_weights

    return kwargs, problem


def from_specs(
    problem_source: str | dict[str, Any],
    method_source: str | dict[str, Any],
    *,
    steps: int | None = None,
    device: str = "cpu",
    pre_validate: bool = True,
) -> tuple[PDEProblem, dict[str, Any]]:
    """Load and validate specs, then return (problem, solve_kwargs).

    This is the main codegen entry point.  The returned problem can be
    passed to ``op.solve(problem, **solve_kwargs)`` directly.

    Parameters
    ----------
    pre_validate:
        Defaults to ``True`` and runs the Specification Validation Agent
        before codegen.  If validation fails, a ``ValueError`` is raised with
        the structured error report.  The ``openpinn.agents`` package is
        imported lazily so core modules never depend on it at module level.
    """
    prob_spec = load_problem_spec(problem_source)
    meth_spec = load_method_spec(method_source)

    if pre_validate:
        from .agents.validation import validate_specs

        report = validate_specs(prob_spec, meth_spec)
        if not report.ok:
            msgs = "; ".join(
                f"[{e.code}] {e.path}: {e.message}" for e in report.errors
            )
            raise ValueError(
                f"Specification Validation Agent validation failed with "
                f"{len(report.errors)} error(s): {msgs}"
            )

    kwargs, problem = build_solve_kwargs(prob_spec, meth_spec, steps=steps, device=device)
    return problem, kwargs
