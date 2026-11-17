"""Tests for M3 Kirsch analytical reference helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openpinn.kirsch import (
    compare_displacement_l2,
    kirsch_displacement,
    kirsch_hoop_stress_at_hole,
    kirsch_scf,
    kirsch_stress_cartesian,
    kirsch_stress_polar,
)


class TestKirschStress:

    def test_scf_is_three(self):
        assert kirsch_scf() == 3.0

    def test_hoop_stress_at_hole_max(self):
        theta = np.linspace(0, 2 * np.pi, 360)
        hoop = kirsch_hoop_stress_at_hole(theta, sigma_inf=1.0)
        assert np.isclose(np.max(hoop), 3.0, atol=0.01)

    def test_hoop_stress_at_hole_min(self):
        theta = np.linspace(0, 2 * np.pi, 360)
        hoop = kirsch_hoop_stress_at_hole(theta, sigma_inf=1.0)
        assert np.isclose(np.min(hoop), -1.0, atol=0.01)

    def test_stress_polar_far_field(self):
        r = np.array([1000.0])
        theta = np.array([0.0])
        result = kirsch_stress_polar(r, theta, sigma_inf=1.0, a=1.0)
        assert np.isclose(result["sigma_rr"][0], 1.0, atol=0.01)

    def test_stress_polar_traction_free_hole(self):
        theta = np.linspace(0, 2 * np.pi, 36)
        r = np.ones_like(theta) * 1.0
        result = kirsch_stress_polar(r, theta, sigma_inf=1.0, a=1.0)
        np.testing.assert_allclose(result["sigma_rr"], 0.0, atol=1e-12)

    def test_stress_cartesian_symmetric(self):
        x = np.array([0.0, 0.0, 2.0, -2.0])
        y = np.array([2.0, -2.0, 0.0, 0.0])
        result = kirsch_stress_cartesian(x, y, sigma_inf=1.0, a=1.0)
        np.testing.assert_allclose(
            result["sigma_xx"][0], result["sigma_xx"][1], atol=1e-10
        )

    def test_stress_scf_at_hole_boundary(self):
        theta = np.array([np.pi / 2])
        r = np.array([1.0])
        result = kirsch_stress_polar(r, theta, sigma_inf=1.0, a=1.0)
        assert np.isclose(result["sigma_tt"][0], 3.0, atol=1e-10)


class TestKirschDisplacement:

    def test_displacement_finite(self):
        x = np.array([2.0, 3.0, 0.0, 1.5])
        y = np.array([0.0, 0.0, 2.0, 1.5])
        result = kirsch_displacement(x, y, sigma_inf=1.0, a=1.0, E=1.0, nu=0.3)
        assert np.all(np.isfinite(result["ux"]))
        assert np.all(np.isfinite(result["uy"]))

    def test_displacement_far_field_ux_positive(self):
        x = np.array([100.0])
        y = np.array([0.0])
        result = kirsch_displacement(x, y, sigma_inf=1.0, a=1.0, E=1.0, nu=0.3)
        assert result["ux"][0] > 0

    def test_displacement_with_center(self):
        x = np.array([3.0])
        y = np.array([1.0])
        r1 = kirsch_displacement(x, y, sigma_inf=1.0, a=1.0, E=1.0, nu=0.3, center=(0, 0))
        r2 = kirsch_displacement(x + 5, y + 5, sigma_inf=1.0, a=1.0, E=1.0, nu=0.3, center=(5, 5))
        np.testing.assert_allclose(r1["ux"], r2["ux"], atol=1e-10)


class TestCompareDisplacementL2:

    def test_perfect_match_zero_error(self):
        x = np.array([2.0, 3.0, 0.0, 1.5])
        y = np.array([0.0, 0.0, 2.0, 1.5])
        coords = np.stack([x, y], axis=-1)
        ref = kirsch_displacement(x, y, sigma_inf=1.0, a=1.0, E=1.0, nu=0.3)
        result = compare_displacement_l2(
            ref["ux"], ref["uy"], coords,
            sigma_inf=1.0, a=1.0, E=1.0, nu=0.3,
        )
        assert result["relative_l2_total"] < 1e-10

    def test_zero_prediction_nonzero_error(self):
        x = np.array([2.0, 3.0, 4.0])
        y = np.array([0.0, 0.0, 0.0])
        coords = np.stack([x, y], axis=-1)
        result = compare_displacement_l2(
            np.zeros(3), np.zeros(3), coords,
            sigma_inf=1.0, a=1.0, E=1.0, nu=0.3,
        )
        assert result["relative_l2_total"] > 0.5

    def test_returns_n_points(self):
        x = np.array([2.0, 3.0])
        y = np.array([0.0, 0.0])
        coords = np.stack([x, y], axis=-1)
        result = compare_displacement_l2(
            np.zeros(2), np.zeros(2), coords,
            sigma_inf=1.0, a=1.0, E=1.0, nu=0.3,
        )
        assert result["n_comparison_points"] == 2
