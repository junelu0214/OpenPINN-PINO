from __future__ import annotations

import torch

import openpinn as op


REQUIRED_PUBLIC_API = {
    "Adam",
    "AdamW",
    "ArtifactWriter",
    "BurgersColeHopfReference",
    "AutodiffDerivative",
    "ComponentRegistry",
    "assert_no_pde_residual_in_forward",
    "DirichletBC",
    "Experiment",
    "ExperimentResult",
    "FourierFeatures",
    "FormulationProtocol",
    "HardConstraint",
    "IdentityFeature",
    "IdentityOutput",
    "InitialCondition",
    "Interval",
    "LBFGS",
    "LossComposer",
    "MLP",
    "ModifiedMLP",
    "NeumannBC",
    "PDEProblem",
    "PINNModel",
    "Polygon",
    "RobinBC",
    "SamplerProtocol",
    "SpaceTimeDomain",
    "StrongFormulation",
    "Trainer",
    "UniformSampler",
    "grad",
    "hessian",
    "default_component_registry",
    "load_config",
    "make_burgers_problem",
    "make_coupled_manufactured_problem",
    "make_piecewise_poisson_problem",
    "make_optimizer",
    "make_poisson_problem",
}


def test_expected_library_first_api_is_exported_from_one_namespace() -> None:
    missing = sorted(REQUIRED_PUBLIC_API.difference(dir(op)))
    assert not missing, f"openpinn public API is missing: {missing}"


def test_benchmark_factories_return_named_problem_objects() -> None:
    poisson = op.make_poisson_problem()
    burgers = op.make_burgers_problem()
    heterogeneous = op.make_piecewise_poisson_problem()

    assert isinstance(poisson, op.PDEProblem)
    assert isinstance(burgers, op.PDEProblem)
    assert isinstance(heterogeneous, op.PDEProblem)
    assert poisson.domain.dim == 1
    assert burgers.domain.dim == 2
    assert poisson.name.lower().startswith("poisson")
    assert burgers.name.lower().startswith("burgers")
    assert heterogeneous.metadata["kind"] == "heterogeneous_static"


def test_poisson_factory_keeps_reference_and_boundary_data_consistent() -> None:
    problem = op.make_poisson_problem(wave_number=0.5)
    boundary = problem.domain.sample_boundary(2).requires_grad_(True)

    error = problem.conditions[0].error(
        lambda points: problem.exact(points), boundary
    )

    torch.testing.assert_close(error, torch.zeros_like(error), atol=0.0, rtol=0.0)
