"""Small tensor/array error metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def _tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach()
    return torch.as_tensor(np.asarray(value))


def rmse(prediction: Any, target: Any = 0.0) -> float:
    """Root-mean-square error as a Python float."""

    pred = _tensor(prediction)
    ref = _tensor(target).to(device=pred.device, dtype=pred.dtype)
    return float(torch.sqrt(torch.mean((pred - ref).square())).cpu())


def mean_absolute_error(prediction: Any, target: Any = 0.0) -> float:
    """Mean absolute error as a Python float."""

    pred = _tensor(prediction)
    ref = _tensor(target).to(device=pred.device, dtype=pred.dtype)
    return float(torch.mean(torch.abs(pred - ref)).cpu())


def relative_l2_error(prediction: Any, target: Any, *, eps: float = 1e-12) -> float:
    """Relative Euclidean error ``||prediction-target|| / ||target||``."""

    pred = _tensor(prediction)
    ref = _tensor(target).to(device=pred.device, dtype=pred.dtype)
    numerator = torch.linalg.vector_norm(pred - ref)
    denominator = torch.linalg.vector_norm(ref).clamp_min(eps)
    return float((numerator / denominator).cpu())


def max_absolute_error(prediction: Any, target: Any = 0.0) -> float:
    """Maximum absolute error as a Python float."""

    pred = _tensor(prediction)
    ref = _tensor(target).to(device=pred.device, dtype=pred.dtype)
    return float(torch.max(torch.abs(pred - ref)).cpu())


def r2_score(prediction: Any, target: Any, *, eps: float = 1e-12) -> float:
    """Coefficient of determination with a finite constant-target convention.

    A constant target has no variance.  In that case an exact prediction is
    assigned 1.0 and every non-exact prediction 0.0, matching the finite-force
    convention commonly used by metric libraries.
    """

    pred = _tensor(prediction)
    ref = _tensor(target).to(device=pred.device, dtype=pred.dtype)
    residual_sum = torch.sum((pred - ref).square())
    total_sum = torch.sum((ref - torch.mean(ref)).square())
    if float(total_sum.detach().cpu()) <= eps:
        return 1.0 if float(residual_sum.detach().cpu()) <= eps else 0.0
    return float((1.0 - residual_sum / total_sum).cpu())
