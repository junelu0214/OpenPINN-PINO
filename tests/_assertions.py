"""Small numerical assertions shared by contract and smoke tests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any

import numpy as np
import torch


def numeric_leaves(value: Any) -> list[np.ndarray]:
    if isinstance(value, torch.Tensor):
        return [value.detach().cpu().numpy()]
    if isinstance(value, np.ndarray):
        return [value] if np.issubdtype(value.dtype, np.number) else []
    if isinstance(value, Number) and not isinstance(value, bool):
        return [np.asarray(value)]
    if isinstance(value, Mapping):
        leaves: list[np.ndarray] = []
        for item in value.values():
            leaves.extend(numeric_leaves(item))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        leaves = []
        for item in value:
            leaves.extend(numeric_leaves(item))
        return leaves
    return []


def assert_finite_tree(value: Any) -> None:
    leaves = numeric_leaves(value)
    assert leaves, "expected at least one numeric value"
    for leaf in leaves:
        assert np.isfinite(leaf).all()


def assert_nested_allclose(left: Any, right: Any, *, atol: float = 1e-7) -> None:
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert set(left) == set(right)
        for key in left:
            assert_nested_allclose(left[key], right[key], atol=atol)
        return
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, torch.as_tensor(right), rtol=0.0, atol=atol)
        return
    if isinstance(left, np.ndarray):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=atol)
        return
    if isinstance(left, Number) and not isinstance(left, bool):
        assert math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        assert isinstance(right, Sequence)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_nested_allclose(left_item, right_item, atol=atol)
        return
    assert left == right

