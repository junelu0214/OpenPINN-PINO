"""Baseline multilayer perceptron and composed PINN model."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .constraints import IdentityOutput
from .features import IdentityFeature


class AdaptiveTanh(nn.Module):
    """Tanh with one trainable positive slope per hidden layer.

    This minimal layer-wise variant follows the trainable-activation idea in
    Jagtap, Kawaguchi and Karniadakis (2020), DOI
    ``10.1016/j.jcp.2019.109136``.  It is an independent PyTorch
    implementation and does not copy the MIT-licensed TensorFlow reference
    implementation.
    """

    def __init__(self, initial_slope: float = 1.0) -> None:
        super().__init__()
        if initial_slope <= 0.0:
            raise ValueError("AdaptiveTanh initial_slope must be positive.")
        self.slope = nn.Parameter(torch.tensor(float(initial_slope)))

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply tanh after scaling by the learned slope."""

        return torch.tanh(self.slope * inputs)


def _activation_factory(
    activation: str | nn.Module | Callable[[], nn.Module],
    options: Mapping[str, object] | None = None,
) -> nn.Module:
    selected_options = dict(options or {})
    if isinstance(activation, nn.Module):
        if selected_options:
            raise ValueError("activation_options cannot be used with an activation module.")
        return copy.deepcopy(activation)
    if callable(activation) and not isinstance(activation, str):
        if selected_options:
            raise ValueError("activation_options cannot be used with an activation factory.")
        result = activation()
        if not isinstance(result, nn.Module):
            raise TypeError("Activation factory must return torch.nn.Module.")
        return result
    name = str(activation).lower().replace("_", "")
    activations: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "sigmoid": nn.Sigmoid,
        "adaptivetanh": AdaptiveTanh,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation {activation!r}.")
    if name != "adaptivetanh" and selected_options:
        raise ValueError("activation_options are currently supported only by adaptive_tanh.")
    return activations[name](**selected_options)


class MLP(nn.Module):
    """A fully connected baseline network with Xavier initialization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_layers: Sequence[int] = (64, 64, 64),
        activation: str | nn.Module | Callable[[], nn.Module] = "tanh",
        *,
        bias: bool = True,
        activation_options: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        widths = [int(input_dim), *(int(width) for width in hidden_layers), int(output_dim)]
        if any(width <= 0 for width in widths):
            raise ValueError("All MLP layer widths must be positive.")
        self.input_dim = widths[0]
        self.output_dim = widths[-1]
        self.hidden_layers = tuple(widths[1:-1])
        self.activation_name = activation if isinstance(activation, str) else type(activation).__name__

        modules: list[nn.Module] = []
        self.linear_layers = nn.ModuleList()
        for index, (in_width, out_width) in enumerate(zip(widths[:-1], widths[1:])):
            layer = nn.Linear(in_width, out_width, bias=bias)
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            self.linear_layers.append(layer)
            modules.append(layer)
            if index < len(widths) - 2:
                modules.append(_activation_factory(activation, activation_options))
        self.network = nn.Sequential(*modules)

    def forward(self, features: Tensor) -> Tensor:
        """Evaluate the MLP."""

        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"MLP expects shape (N, {self.input_dim}), got {tuple(features.shape)}."
            )
        return self.network(features)


class ModifiedMLP(nn.Module):
    """Gated MLP proposed for mitigating PINN gradient pathologies.

    This is an independent PyTorch implementation of the architecture in
    Wang, Teng and Perdikaris (2021), DOI ``10.1137/20M1318043``.  It does not
    copy code from the paper's reference repository, whose license was not
    explicit when the OpenPINN handover was prepared.

    All hidden layers use one width because the two input encoders are blended
    elementwise with every hidden state.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_layers: Sequence[int] = (64, 64, 64),
        activation: str | nn.Module | Callable[[], nn.Module] = "tanh",
        *,
        bias: bool = True,
        activation_options: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        widths = tuple(int(width) for width in hidden_layers)
        if input_dim <= 0 or output_dim <= 0 or not widths:
            raise ValueError("input_dim, output_dim and hidden_layers must be positive.")
        if any(width <= 0 for width in widths):
            raise ValueError("All ModifiedMLP hidden widths must be positive.")
        if len(set(widths)) != 1:
            raise ValueError(
                "ModifiedMLP requires equal hidden widths for elementwise gating."
            )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = widths
        self.activation_name = (
            activation if isinstance(activation, str) else type(activation).__name__
        )
        hidden_width = widths[0]

        self.encoder_u = nn.Linear(self.input_dim, hidden_width, bias=bias)
        self.encoder_v = nn.Linear(self.input_dim, hidden_width, bias=bias)
        self.gate_layers = nn.ModuleList(
            [
                nn.Linear(
                    self.input_dim if index == 0 else hidden_width,
                    hidden_width,
                    bias=bias,
                )
                for index in range(len(widths))
            ]
        )
        self.output_layer = nn.Linear(hidden_width, self.output_dim, bias=bias)
        self.encoder_u_activation = _activation_factory(activation, activation_options)
        self.encoder_v_activation = _activation_factory(activation, activation_options)
        self.gate_activations = nn.ModuleList(
            [_activation_factory(activation, activation_options) for _ in widths]
        )

        self.linear_layers = nn.ModuleList(
            [self.encoder_u, self.encoder_v, *self.gate_layers, self.output_layer]
        )
        for layer in self.linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Evaluate the two encoders and gated hidden-state blends."""

        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"ModifiedMLP expects shape (N, {self.input_dim}), "
                f"got {tuple(features.shape)}."
            )
        encoded_u = self.encoder_u_activation(self.encoder_u(features))
        encoded_v = self.encoder_v_activation(self.encoder_v(features))
        hidden = self.gate_activations[0](self.gate_layers[0](features))
        for layer, activation in zip(
            self.gate_layers[1:], self.gate_activations[1:], strict=True
        ):
            gate = activation(layer(hidden))
            hidden = (1.0 - gate) * encoded_u + gate * encoded_v
        return self.output_layer(hidden)


class DomainDecompositionPINN(nn.Module):
    """Two independent MLPs joined at one 1D spatial interface.

    The accompanying interface formulation supplies solution and flux
    continuity losses. This is an independent minimal two-subdomain design
    inspired by XPINNs, DOI ``10.4208/cicp.OA-2020-0164``; no reference code
    is copied.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_layers: Sequence[int] = (32, 32, 32),
        activation: str | nn.Module | Callable[[], nn.Module] = "tanh",
        *,
        interface: float = 0.0,
        bias: bool = True,
        activation_options: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if int(input_dim) != 1:
            raise ValueError("The starter domain decomposition supports only 1D input.")
        if int(output_dim) != 1:
            raise ValueError("The starter domain decomposition supports one output field.")
        self.input_dim = 1
        self.output_dim = 1
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        self.interface = float(interface)
        self.left_network = MLP(
            1,
            1,
            hidden_layers=self.hidden_layers,
            activation=activation,
            bias=bias,
            activation_options=activation_options,
        )
        self.right_network = MLP(
            1,
            1,
            hidden_layers=self.hidden_layers,
            activation=activation,
            bias=bias,
            activation_options=activation_options,
        )

    def branch_values(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate both subnetworks at common coordinates."""

        if coordinates.ndim != 2 or coordinates.shape[1] != 1:
            raise ValueError(
                f"DomainDecompositionPINN expects shape (N, 1), got {tuple(coordinates.shape)}."
            )
        return self.left_network(coordinates), self.right_network(coordinates)

    def forward(self, coordinates: Tensor) -> Tensor:
        """Dispatch each coordinate to its subdomain network."""

        left, right = self.branch_values(coordinates)
        return torch.where(coordinates[:, :1] <= self.interface, left, right)


class PINNModel(nn.Module):
    """Compose input features, a neural network and an output transform."""

    def __init__(
        self,
        network: nn.Module,
        feature_transform: nn.Module | None = None,
        output_transform: nn.Module | None = None,
        *,
        feature: nn.Module | None = None,
        output: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if feature_transform is not None and feature is not None:
            raise ValueError("Pass feature_transform or feature, not both.")
        if output_transform is not None and output is not None:
            raise ValueError("Pass output_transform or output, not both.")
        self.feature_transform = feature_transform or feature or IdentityFeature()
        self.network = network
        self.output_transform = output_transform or output or IdentityOutput()

    @property
    def net(self) -> nn.Module:
        """Familiar short alias for the underlying network."""

        return self.network

    @property
    def output_dim(self) -> int:
        """Declared number of physical fields produced by the network."""

        declared = getattr(self.network, "output_dim", None)
        if declared is None:
            declared = getattr(self.network, "out_features", None)
        if declared is None:
            raise AttributeError(
                "The underlying network must expose output_dim or out_features."
            )
        return int(declared)

    def raw(self, coordinates: Tensor) -> Tensor:
        """Return the unconstrained network output."""

        return self.network(self.feature_transform(coordinates))

    def forward(self, coordinates: Tensor) -> Tensor:
        """Evaluate the complete feature/network/output pipeline."""

        raw_output = self.raw(coordinates)
        return self.output_transform(coordinates, raw_output)


def freeze_all_but_last_linear_layers(module: nn.Module, last_n: int = 1) -> None:
    """Freeze a module except for its last ``last_n`` linear layers.

    Checkpoint loading changes tensor values but not ``requires_grad`` flags, so
    this helper can be applied when a model is built and remain effective after
    a strict model-only warm start.
    """

    count = int(last_n)
    if count <= 0:
        raise ValueError("last_n must be a positive integer.")
    linear_layers = [
        child for child in module.modules() if isinstance(child, nn.Linear)
    ]
    if count > len(linear_layers):
        raise ValueError(
            f"Cannot keep {count} linear layers trainable; model has "
            f"{len(linear_layers)}."
        )
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for layer in linear_layers[-count:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
