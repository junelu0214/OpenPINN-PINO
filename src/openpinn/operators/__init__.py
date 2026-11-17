"""Operator-learning models that sit beside the PINN method catalog.

DeepONet is a selectable coordinate operator. FNO is a selectable grid
operator with discrete :class:`~openpinn.GridDerivative` support. PINO is an
advanced FNO-backed structured-grid training helper, not a third selectable
model.
"""

from .deeponet import (
    DEEPONET_PARAMETER_SCHEMA,
    DeepONet,
    DeepONetConfig,
    antiderivative_batch,
    build_deeponet,
)
from .designed_problem import (
    CoordinateProblemOnGrid,
    evaluate_operator_reference_csv,
    train_operator_problem,
)
from .fno import (
    FNO,
    FNOConfig,
    FNO_PARAMETER_SCHEMA,
    SpectralConv,
    build_fno,
    fourier_derivative_batch,
)
from .grid_derivative import (
    FiniteDifferenceGridDerivative,
    SpectralGridDerivative,
    make_grid_derivative,
)
from .pde_problem import GridPDEProblem, PeriodicPoisson1D
from .pino import (
    PINO,
    PINOConfig,
    PINO_PARAMETER_SCHEMA,
    build_pino,
    poisson_operator_batch,
    train_pino,
)
from .selection import (
    OperatorTrainResult,
    TRAINING_MODES,
    build_operator_config,
    train_operator,
)
from .uniform_grid import UniformGrid

__all__ = [
    "DEEPONET_PARAMETER_SCHEMA",
    "DeepONet",
    "DeepONetConfig",
    "CoordinateProblemOnGrid",
    "FNO",
    "FNOConfig",
    "FNO_PARAMETER_SCHEMA",
    "GridPDEProblem",
    "PINO",
    "PINOConfig",
    "PINO_PARAMETER_SCHEMA",
    "PeriodicPoisson1D",
    "FiniteDifferenceGridDerivative",
    "OperatorTrainResult",
    "TRAINING_MODES",
    "SpectralConv",
    "SpectralGridDerivative",
    "UniformGrid",
    "antiderivative_batch",
    "build_deeponet",
    "build_fno",
    "build_pino",
    "build_operator_config",
    "evaluate_operator_reference_csv",
    "fourier_derivative_batch",
    "make_grid_derivative",
    "poisson_operator_batch",
    "train_pino",
    "train_operator",
    "train_operator_problem",
]
