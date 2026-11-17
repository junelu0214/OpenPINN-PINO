"""Output transforms for identity and analytic hard constraints."""

from __future__ import annotations

from typing import Callable, Union

from torch import Tensor, nn

from .conditions import ValueLike, evaluate_value

DistanceLike = Union[Tensor, float, Callable[[Tensor], Tensor]]


class IdentityOutput(nn.Module):
    """Leave the raw network output unchanged."""

    def forward(self, coordinates: Tensor, raw_output: Tensor) -> Tensor:
        """Return ``raw_output``."""

        del coordinates
        return raw_output


class HardConstraint(nn.Module):
    """Apply the analytic transform ``g(x) + d(x) * network(x)``.

    The caller supplies a distance-like function ``d`` that vanishes wherever
    the constraint must be exact and an extension ``g`` of the prescribed
    value.  OpenPINN does not infer a distance function for arbitrary geometry.
    Users should verify smoothness and boundary error for their own domain.
    """

    def __init__(
        self,
        distance: DistanceLike,
        boundary_value: ValueLike = 0.0,
    ) -> None:
        super().__init__()
        self.distance = distance
        self.boundary_value = boundary_value

    def _distance(self, coordinates: Tensor, raw_output: Tensor) -> Tensor:
        values = self.distance(coordinates) if callable(self.distance) else self.distance
        if not isinstance(values, Tensor):
            values = raw_output.new_tensor(float(values))
        else:
            values = values.to(device=raw_output.device, dtype=raw_output.dtype)
        if values.ndim == 0:
            values = values.expand(raw_output.shape[0], 1)
        elif values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[0] not in (1, raw_output.shape[0]):
            raise ValueError("Hard-constraint distance has an incompatible batch size.")
        return values

    def forward(self, coordinates: Tensor, raw_output: Tensor) -> Tensor:
        """Return the exactly transformed model output."""

        distance = self._distance(coordinates, raw_output)
        extension = evaluate_value(self.boundary_value, coordinates, like=raw_output)
        try:
            return extension + distance * raw_output
        except RuntimeError as exc:
            raise ValueError(
                "Hard-constraint distance/value cannot broadcast to the model output."
            ) from exc
