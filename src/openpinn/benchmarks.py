"""Reference problem constructors for starter acceptance tests."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from .autodiff import grad, hessian
from .conditions import DirichletBC, InitialCondition
from .geometry import Interval, RectangleWithCircularHoles, SpaceTimeDomain
from .problems import PDEProblem


class BurgersColeHopfReference:
    """Independent Cole–Hopf reference evaluator for the 1D Burgers case.

    The transformed heat equation is evaluated with Gauss–Legendre
    quadrature and reflected Gaussian images.  Computation is deliberately
    performed in float64 and in log space so the small default viscosity does
    not underflow.  This object is an evaluation oracle only: it is never
    called by the trainer or loss function.
    """

    def __init__(
        self,
        viscosity: float,
        *,
        lower: float = -1.0,
        upper: float = 1.0,
        initial_time: float = 0.0,
        quadrature_points: int = 1024,
        image_radius: int = 2,
        minimum_positive_time: float = 0.002,
        batch_size: int = 64,
    ) -> None:
        self.viscosity = float(viscosity)
        self.lower = float(lower)
        self.upper = float(upper)
        self.initial_time = float(initial_time)
        self.quadrature_points = int(quadrature_points)
        self.image_radius = int(image_radius)
        self.minimum_positive_time = float(minimum_positive_time)
        self.batch_size = int(batch_size)
        if not math.isfinite(self.viscosity) or self.viscosity <= 0.0:
            raise ValueError("Burgers reference viscosity must be positive.")
        if not self.lower < self.upper:
            raise ValueError("Burgers reference interval must have positive length.")
        if self.quadrature_points < 16:
            raise ValueError("quadrature_points must be at least 16.")
        if self.image_radius < 0:
            raise ValueError("image_radius must be non-negative.")
        if self.minimum_positive_time <= 0.0:
            raise ValueError("minimum_positive_time must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        raw_nodes, raw_weights = np.polynomial.legendre.leggauss(
            self.quadrature_points
        )
        midpoint = 0.5 * (self.lower + self.upper)
        half_length = 0.5 * (self.upper - self.lower)
        nodes = midpoint + half_length * raw_nodes
        weights = half_length * raw_weights
        beta = 1.0 / (2.0 * math.pi * self.viscosity)
        self._nodes = torch.from_numpy(nodes)
        # Subtracting one constant from the transformed initial field is
        # harmless because u = -2*nu*d_x(log(phi)); it keeps the exponent <= 0.
        self._log_initial_weight = torch.from_numpy(
            np.log(weights) - beta * (np.cos(math.pi * nodes) + 1.0)
        )

    @property
    def metadata(self) -> dict[str, object]:
        """Return sufficient provenance to audit the evaluation oracle."""

        return {
            "method": "Cole-Hopf reflected heat-kernel quadrature",
            "viscosity": self.viscosity,
            "quadrature_points": self.quadrature_points,
            "image_radius": self.image_radius,
            "minimum_positive_time": self.minimum_positive_time,
            "internal_dtype": "float64",
            "cole_doi": "10.1090/qam/42889",
            "hopf_doi": "10.1002/cpa.3160030302",
            "external_code_used": False,
        }

    def __call__(self, coordinates: Tensor) -> Tensor:
        """Evaluate ``u(x,t)`` at an ``(N, 2)`` coordinate tensor."""

        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Burgers reference coordinates must have shape (N, 2).")
        if not torch.is_floating_point(coordinates):
            raise TypeError("Burgers reference coordinates must be floating point.")
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("Burgers reference coordinates must be finite.")

        original_dtype = coordinates.dtype
        work = coordinates.detach().to(dtype=torch.float64)
        x = work[:, 0]
        elapsed = work[:, 1] - self.initial_time
        if bool((elapsed < 0.0).any()):
            raise ValueError("Burgers reference received a time before initial_time.")
        too_early = (elapsed > 0.0) & (elapsed < self.minimum_positive_time)
        if bool(too_early.any()):
            raise ValueError(
                "Positive evaluation times fall below the validated quadrature range "
                f"t >= {self.minimum_positive_time:g}."
            )

        result = torch.empty_like(x)
        initial_mask = elapsed == 0.0
        result[initial_mask] = -torch.sin(math.pi * x[initial_mask])

        nodes = self._nodes.to(device=work.device)
        log_initial_weight = self._log_initial_weight.to(device=work.device)
        interval_length = self.upper - self.lower
        shifts = (
            torch.arange(
                -self.image_radius,
                self.image_radius + 1,
                dtype=torch.float64,
                device=work.device,
            )
            * (2.0 * interval_length)
        )
        for time_value in torch.unique(elapsed[~initial_mask]):
            selected = torch.nonzero(elapsed == time_value, as_tuple=False).flatten()
            time_scalar = float(time_value.cpu())
            denominator = 4.0 * self.viscosity * time_scalar
            for start in range(0, selected.numel(), self.batch_size):
                batch_indices = selected[start : start + self.batch_size]
                x_batch = x[batch_indices, None, None]
                y = nodes[None, :, None]
                image_shifts = shifts[None, None, :]
                direct = x_batch - y + image_shifts
                reflected = x_batch + y - 2.0 * self.lower + image_shifts
                distances = torch.cat((direct, reflected), dim=2)
                log_terms = (
                    log_initial_weight[None, :, None]
                    - distances.square() / denominator
                )
                maximum = log_terms.amax(dim=(1, 2), keepdim=True)
                normalized = torch.exp(log_terms - maximum)
                weighted_distance = (normalized * distances).sum(dim=(1, 2))
                result[batch_indices] = (
                    weighted_distance / normalized.sum(dim=(1, 2)) / time_scalar
                )

        boundary = torch.isclose(
            x, x.new_tensor(self.lower), rtol=0.0, atol=1e-12
        ) | torch.isclose(x, x.new_tensor(self.upper), rtol=0.0, atol=1e-12)
        result[boundary] = 0.0
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("Burgers reference evaluation produced non-finite values.")
        return result[:, None].to(dtype=original_dtype)


def make_poisson_problem(
    domain: Interval | None = None,
    *,
    wave_number: float = 1.0,
    amplitude: float = 1.0,
    k: float | None = None,
) -> PDEProblem:
    """Build static 1D Poisson with an analytic sine solution.

    The default is ``-u_xx = pi^2 sin(pi x)`` on ``[-1, 1]`` with zero
    Dirichlet data and exact solution ``sin(pi x)``.
    """

    selected_domain = domain or Interval(-1.0, 1.0)
    if k is not None:
        wave_number = k
    frequency = float(wave_number) * math.pi
    scale = float(amplitude)

    def exact(coordinates: Tensor) -> Tensor:
        return scale * torch.sin(frequency * coordinates[:, :1])

    def residual(coordinates: Tensor, prediction: Tensor) -> Tensor:
        forcing = (frequency**2) * exact(coordinates)
        return -hessian(prediction, coordinates, i=0, j=0) - forcing

    return PDEProblem(
        selected_domain,
        residual,
        [DirichletBC(exact, name="boundary")],
        exact_solution=exact,
        name="poisson_1d",
        metadata={
            "kind": "static",
            "equation": "-u_xx = (k*pi)^2 A sin(k*pi*x)",
            "wave_number": float(wave_number),
            "amplitude": scale,
        },
    )


def make_piecewise_poisson_problem(
    domain: Interval | None = None,
    *,
    interface: float = 0.0,
    left_diffusivity: float = 1.0,
    right_diffusivity: float = 4.0,
) -> PDEProblem:
    """Build a two-material 1D diffusion benchmark with an exact interface solution.

    The problem solves ``-(a(x) u_x)_x = 0`` with ``u(-1)=0`` and
    ``u(1)=1``.  The piecewise-constant coefficient changes at ``interface``.
    The analytic field is continuous and has continuous normal flux, while its
    slope changes across the material interface.  It is a compact
    heterogeneous smoke benchmark, not a domain-decomposition implementation.
    """

    selected_domain = domain or Interval(-1.0, 1.0)
    lower, upper = selected_domain.bounds
    interface = float(interface)
    left_diffusivity = float(left_diffusivity)
    right_diffusivity = float(right_diffusivity)
    if not lower < interface < upper:
        raise ValueError("interface must lie strictly inside the interval.")
    if left_diffusivity <= 0.0 or right_diffusivity <= 0.0:
        raise ValueError("Diffusivities must be positive.")

    resistance = (interface - lower) / left_diffusivity + (
        upper - interface
    ) / right_diffusivity
    flux_value = 1.0 / resistance
    interface_value = flux_value * (interface - lower) / left_diffusivity

    def coefficient(coordinates: Tensor) -> Tensor:
        x = coordinates[:, :1]
        return torch.where(
            x <= interface,
            x.new_full(x.shape, left_diffusivity),
            x.new_full(x.shape, right_diffusivity),
        )

    def exact(coordinates: Tensor) -> Tensor:
        x = coordinates[:, :1]
        left = flux_value * (x - lower) / left_diffusivity
        right = interface_value + flux_value * (x - interface) / right_diffusivity
        return torch.where(x <= interface, left, right)

    def residual(coordinates: Tensor, prediction: Tensor) -> Tensor:
        u_x = grad(prediction, coordinates)[:, :1]
        physical_flux = coefficient(coordinates) * u_x
        return -grad(physical_flux, coordinates)[:, :1]

    return PDEProblem(
        selected_domain,
        residual,
        [DirichletBC(exact, name="boundary")],
        exact_solution=exact,
        name="piecewise_poisson_1d",
        metadata={
            "kind": "heterogeneous_static",
            "equation": "-(a(x) u_x)_x = 0",
            "interface": interface,
            "left_diffusivity": left_diffusivity,
            "right_diffusivity": right_diffusivity,
            "exact_flux": flux_value,
        },
    )


def make_coupled_manufactured_problem(
    domain: Interval | None = None,
    *,
    coupling_strength: float = 1.0,
) -> PDEProblem:
    """Build a two-field elliptic contract case with an explicit coupling term.

    The exact fields are ``u=x^2`` and ``v=x^3``.  Two coupled balance
    equations and the compatible relation ``v_x-3u=0`` are returned as three
    separately named residuals.  This deliberately overdetermined
    manufactured system proves the multi-field extension path; it is not an
    engineering-accuracy benchmark.
    """

    selected_domain = domain or Interval(-1.0, 1.0)
    kappa = float(coupling_strength)
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("coupling_strength must be finite and positive.")

    def exact(coordinates: Tensor) -> Tensor:
        x = coordinates[:, :1]
        return torch.cat((x.square(), x.pow(3)), dim=1)

    def boundary_value(coordinates: Tensor) -> Tensor:
        x = coordinates[:, :1]
        return torch.cat((torch.ones_like(x), x), dim=1)

    def residual(
        coordinates: Tensor, prediction: Tensor
    ) -> dict[str, Tensor]:
        x = coordinates[:, :1]
        u = prediction[:, 0:1]
        v = prediction[:, 1:2]
        u_xx = hessian(prediction, coordinates, component=0, i=0, j=0)
        v_x = grad(prediction, coordinates, component=1)[:, 0:1]
        v_xx = hessian(prediction, coordinates, component=1, i=0, j=0)
        forcing_u = -2.0 + kappa * (x.square() - x.pow(3))
        forcing_v = -6.0 * x + kappa * (x.pow(3) - x.square())
        return {
            "equation_u": -u_xx + kappa * (u - v) - forcing_u,
            "equation_v": -v_xx + kappa * (v - u) - forcing_v,
            "coupling": v_x - 3.0 * u,
        }

    return PDEProblem(
        selected_domain,
        residual,
        [DirichletBC(boundary_value, name="boundary")],
        exact_solution=exact,
        name="coupled_elliptic_manufactured_1d",
        field_names=("u", "v"),
        metadata={
            "kind": "coupled_static_contract",
            "equations": (
                "-u_xx + kappa*(u-v) = f_u",
                "-v_xx + kappa*(v-u) = f_v",
                "v_x - 3*u = 0",
            ),
            "residual_names": ("equation_u", "equation_v", "coupling"),
            "coupling_strength": kappa,
            "accuracy_claim": False,
        },
    )


def make_burgers_problem(
    domain: SpaceTimeDomain | None = None,
    *,
    viscosity: float = 0.01 / math.pi,
    nu: float | None = None,
    reference_quadrature_points: int = 1024,
) -> PDEProblem:
    """Build the dynamic 1D viscous Burgers benchmark.

    The equation is ``u_t + u*u_x - nu*u_xx = 0`` on
    ``x in [-1,1], t in [0,1]`` with zero spatial boundary values and initial
    state ``-sin(pi*x)``.  A Cole–Hopf heat-kernel quadrature supplies an
    independent reference field for evaluation; it is not used in training.
    """

    selected_domain = domain or SpaceTimeDomain(
        Interval(-1.0, 1.0), Interval(0.0, 1.0)
    )
    if nu is not None:
        viscosity = nu
    viscosity = float(viscosity)
    if viscosity <= 0.0:
        raise ValueError("Burgers viscosity must be positive.")
    if any(
        abs(math.sin(math.pi * endpoint)) > 1e-12
        for endpoint in selected_domain.space.bounds
    ):
        raise ValueError(
            "Burgers spatial endpoints must be zeros of sin(pi*x) so the initial "
            "and zero Dirichlet data are compatible."
        )

    reference = BurgersColeHopfReference(
        viscosity,
        lower=selected_domain.space.lower,
        upper=selected_domain.space.upper,
        initial_time=selected_domain.time.lower,
        quadrature_points=reference_quadrature_points,
    )

    def initial(coordinates: Tensor) -> Tensor:
        return -torch.sin(math.pi * coordinates[:, :1])

    def residual(coordinates: Tensor, prediction: Tensor) -> Tensor:
        first = grad(prediction, coordinates)
        u_x = first[:, 0:1]
        u_t = first[:, 1:2]
        u_xx = hessian(prediction, coordinates, i=0, j=0)
        return u_t + prediction * u_x - viscosity * u_xx

    return PDEProblem(
        selected_domain,
        residual,
        [
            DirichletBC(0.0, name="boundary"),
            InitialCondition(initial, name="initial"),
        ],
        exact_solution=reference,
        name="burgers_1d",
        metadata={
            "kind": "dynamic",
            "equation": "u_t + u*u_x - nu*u_xx = 0",
            "viscosity": viscosity,
            "reference": reference.metadata,
        },
    )


def make_tunnel_consolidation_model1_problem(
    *,
    width_m: float = 40.0,
    height_m: float = 40.0,
    duration_years: float = 40.0,
    tunnel_radius_m: float = 4.0,
    tunnel_centers_m: tuple[tuple[float, float], ...] = ((10.0, 20.0), (30.0, 20.0)),
    initial_pressure_kpa: float = 20.0,
    consolidation_x_m2_per_year: float = 0.30,
    consolidation_y_m2_per_year: float = 0.60,
) -> PDEProblem:
    """Build the paper's Model 1 twin-tunnel consolidation PDE.

    The network works with the paper's nondimensional variables
    ``X=x/B``, ``Y=y/H``, ``tau=t/T`` and ``U=u/p0``.  It solves
    ``U_tau - Cx_hat*U_XX - Cy_hat*U_YY = 0`` with ``U=0`` on the outer
    rectangle and both tunnel walls, and ``U=1`` initially in the soil.

    The Word document fixes the 40 m square, two 4 m-radius tunnels and
    40-year horizon.  It does not explicitly state Model 1's ``p0``, ``Cx``
    or ``Cy`` in that subsection, so those values remain visible parameters;
    the defaults are documented assumptions rather than paper ground truth.
    """

    physical_values = {
        "width_m": float(width_m),
        "height_m": float(height_m),
        "duration_years": float(duration_years),
        "tunnel_radius_m": float(tunnel_radius_m),
        "initial_pressure_kpa": float(initial_pressure_kpa),
        "consolidation_x_m2_per_year": float(consolidation_x_m2_per_year),
        "consolidation_y_m2_per_year": float(consolidation_y_m2_per_year),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in physical_values.values()):
        raise ValueError("All Model 1 dimensions, duration, pressure and coefficients must be positive.")
    centers = tuple((float(x), float(y)) for x, y in tunnel_centers_m)
    if len(centers) != 2:
        raise ValueError("Model 1 requires exactly two tunnel centers.")
    normalized_holes = tuple(
        (x / width_m, y / height_m, tunnel_radius_m / width_m)
        for x, y in centers
    )
    if not math.isclose(width_m, height_m):
        # Circular physical tunnels become ellipses after unequal x/y scaling;
        # this minimal geometry intentionally implements the square Model 1.
        raise ValueError("Model 1 currently requires width_m == height_m.")
    space = RectangleWithCircularHoles(
        (0.0, 0.0),
        (1.0, 1.0),
        normalized_holes,
    )
    domain = SpaceTimeDomain(space, Interval(0.0, 1.0))
    coefficient_x = consolidation_x_m2_per_year * duration_years / width_m**2
    coefficient_y = consolidation_y_m2_per_year * duration_years / height_m**2

    def residual(coordinates: Tensor, prediction: Tensor) -> Tensor:
        first = grad(prediction, coordinates)
        u_tau = first[:, 2:3]
        u_xx = hessian(prediction, coordinates, i=0, j=0)
        u_yy = hessian(prediction, coordinates, i=1, j=1)
        return u_tau - coefficient_x * u_xx - coefficient_y * u_yy

    return PDEProblem(
        domain,
        residual,
        [
            DirichletBC(0.0, name="boundary"),
            InitialCondition(1.0, name="initial"),
        ],
        name="tunnel_consolidation_model1",
        field_names=("U",),
        metadata={
            "kind": "dynamic",
            "equation": "U_tau - Cx_hat*U_XX - Cy_hat*U_YY = 0",
            "coordinate_order": ("X", "Y", "tau"),
            "nondimensional": True,
            "physical_parameters": {**physical_values, "tunnel_centers_m": centers},
            "dimensionless_coefficients": {
                "consolidation_x": coefficient_x,
                "consolidation_y": coefficient_y,
            },
            "source": {
                "document": "8.28 9点.docx",
                "section": "Model 1 / 模型一：均质各向异性土体中的双隧道固结",
                "external_code_used": False,
            },
            "parameter_note": (
                "The Model 1 subsection fixes geometry and duration but does not list "
                "p0/Cx/Cy explicitly. Defaults are editable assumptions and must not be "
                "reported as extracted Model 1 ground truth."
            ),
            "reference_solution": None,
        },
    )
