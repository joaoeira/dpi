"""Periodic angle embedding for neural networks."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn

from .value_network import ValueNetwork
from .policy_network import PolicyNetwork, StateBoundedPolicy

Tensor = torch.Tensor


class PeriodicAngleEmbedding(nn.Module):
    """Embed angles as (sin(theta), cos(theta)) for periodicity.

    Transforms input from (batch, n_angles + n_passthrough) to
    (batch, 2*n_angles + n_passthrough) via the mapping:
        theta_i -> (sin(theta_i), cos(theta_i))

    This ensures that theta and theta + 2*pi*k produce identical embeddings.
    """

    def __init__(self, n_angles: int, n_passthrough: int = 0) -> None:
        """Initialize angle embedding.

        Args:
            n_angles: Number of angle dimensions to embed
            n_passthrough: Number of trailing dimensions to pass through unchanged
                          (e.g., for non-angular state components like frequencies)
        """
        super().__init__()
        self.n_angles = n_angles
        self.n_passthrough = n_passthrough

    def forward(self, x: Tensor) -> Tensor:
        """Embed angles.

        Args:
            x: Input tensor, shape (batch, n_angles + n_passthrough)

        Returns:
            Embedded tensor, shape (batch, 2*n_angles + n_passthrough)
        """
        angles = x[:, :self.n_angles]
        passthrough = x[:, self.n_angles:] if self.n_passthrough > 0 else None

        sin_theta = torch.sin(angles)
        cos_theta = torch.cos(angles)

        embedded = torch.cat([sin_theta, cos_theta], dim=1)

        if passthrough is not None:
            embedded = torch.cat([embedded, passthrough], dim=1)

        return embedded

    @property
    def output_dim(self) -> int:
        return 2 * self.n_angles + self.n_passthrough


class PeriodicValueNetwork(nn.Module):
    """Value network with periodic angle embedding."""

    def __init__(
        self,
        n_angles: int,
        n_passthrough: int = 0,
        hidden_dims: Sequence[int] = (256, 128, 64),
        activation: str = "silu",
        output_transform: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        """Initialize periodic value network.

        Args:
            n_angles: Number of angle dimensions to embed
            n_passthrough: Number of trailing dimensions to pass through
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            output_transform: Optional output transform function(s, v) -> v'
        """
        super().__init__()
        self.embedding = PeriodicAngleEmbedding(n_angles, n_passthrough)
        self.value_net = ValueNetwork(
            input_dim=self.embedding.output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            output_transform=None,  # Handle transform ourselves
        )
        self.output_transform = output_transform

    def forward(self, s: Tensor) -> Tensor:
        embedded = self.embedding(s)
        v = self.value_net(embedded)
        if self.output_transform is not None:
            return self.output_transform(s, v)
        return v


class PeriodicPolicyNetwork(nn.Module):
    """Policy network with periodic angle embedding."""

    def __init__(
        self,
        n_angles: int,
        output_dim: int,
        n_passthrough: int = 0,
        hidden_dims: Sequence[int] = (256, 128, 64),
        activation: str = "relu",
        output_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Initialize periodic policy network.

        Args:
            n_angles: Number of angle dimensions to embed
            output_dim: Control output dimension
            n_passthrough: Number of trailing dimensions to pass through
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            output_bounds: Optional (low, high) bounds for output
        """
        super().__init__()
        self.embedding = PeriodicAngleEmbedding(n_angles, n_passthrough)
        self.policy_net = PolicyNetwork(
            input_dim=self.embedding.output_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            output_bounds=output_bounds,
        )

    def forward(self, s: Tensor) -> Tensor:
        embedded = self.embedding(s)
        return self.policy_net(embedded)


class PeriodicStateBoundedPolicy(nn.Module):
    """State-bounded policy with periodic angle embedding.

    The bounds_fn receives the original state (with raw angles),
    while the internal network receives the embedded state.
    """

    def __init__(
        self,
        n_angles: int,
        output_dim: int,
        bounds_fn: Callable[[Tensor], Tuple[Tensor, Tensor]],
        n_passthrough: int = 0,
        hidden_dims: Sequence[int] = (256, 128, 64),
        activation: str = "relu",
    ) -> None:
        """Initialize periodic state-bounded policy.

        Args:
            n_angles: Number of angle dimensions to embed
            output_dim: Control output dimension
            bounds_fn: Function (s) -> (c_min, c_max) for state-dependent bounds
            n_passthrough: Number of trailing dimensions to pass through
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
        """
        super().__init__()
        self.embedding = PeriodicAngleEmbedding(n_angles, n_passthrough)
        self.base_net = PolicyNetwork(
            input_dim=self.embedding.output_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            output_bounds=None,
        )
        self.bounds_fn = bounds_fn

    def forward(self, s: Tensor) -> Tensor:
        # Embed angles for network input
        embedded = self.embedding(s)
        raw = self.base_net(embedded)

        # Get bounds from original state
        c_min, c_max = self.bounds_fn(s)
        span = torch.clamp(c_max - c_min, min=1e-8)
        return c_min + span * torch.sigmoid(raw)
