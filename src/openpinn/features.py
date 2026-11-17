"""Input feature transforms."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class IdentityFeature(nn.Module):
    """Pass coordinates to the network unchanged."""

    def __init__(self, input_dim: int | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.out_features = input_dim

    def forward(self, coordinates: Tensor) -> Tensor:
        """Return coordinates unchanged."""

        return coordinates

    def output_dimension(self, input_dim: int | None = None) -> int:
        """Return the feature width for a coordinate width."""

        dimension = self.input_dim if input_dim is None else input_dim
        if dimension is None:
            raise ValueError("input_dim is required to determine the output dimension.")
        return int(dimension)


class FourierFeatures(nn.Module):
    """Fixed random Fourier feature mapping.

    The mapping is ``[sin(2*pi*xB), cos(2*pi*xB)]`` with an optional copy of
    the original coordinates.  Frequencies are deterministic for a fixed
    seed.  This is a feature transform only; the starter does not claim that a
    chosen scale is optimal for any PDE.
    """

    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 16,
        scale: float = 1.0,
        seed: int = 0,
        *,
        include_input: bool = False,
        trainable: bool = False,
        frequencies: Tensor | None = None,
        mapping_size: int | None = None,
    ) -> None:
        super().__init__()
        if mapping_size is not None:
            num_frequencies = mapping_size
        if input_dim <= 0 or num_frequencies <= 0:
            raise ValueError("input_dim and num_frequencies must be positive.")
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.scale = float(scale)
        self.seed = int(seed)
        self.include_input = bool(include_input)

        if frequencies is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            matrix = torch.randn(
                self.input_dim, self.num_frequencies, generator=generator
            ) * self.scale
        else:
            matrix = torch.as_tensor(frequencies, dtype=torch.get_default_dtype())
            expected = (self.input_dim, self.num_frequencies)
            if tuple(matrix.shape) != expected:
                raise ValueError(f"frequencies must have shape {expected}.")
        if trainable:
            self.frequencies = nn.Parameter(matrix)
        else:
            self.register_buffer("frequencies", matrix)
        self.out_features = 2 * self.num_frequencies + (
            self.input_dim if self.include_input else 0
        )

    def output_dimension(self, input_dim: int | None = None) -> int:
        """Return the fixed output feature width."""

        if input_dim is not None and int(input_dim) != self.input_dim:
            raise ValueError("FourierFeatures received a different input dimension.")
        return self.out_features

    def forward(self, coordinates: Tensor) -> Tensor:
        """Map ``(N, input_dim)`` coordinates to Fourier features."""

        if coordinates.ndim != 2 or coordinates.shape[1] != self.input_dim:
            raise ValueError(
                f"coordinates must have shape (N, {self.input_dim}), got {tuple(coordinates.shape)}."
            )
        projected = 2.0 * math.pi * (coordinates @ self.frequencies)
        encoded = torch.cat((torch.sin(projected), torch.cos(projected)), dim=1)
        return torch.cat((coordinates, encoded), dim=1) if self.include_input else encoded

