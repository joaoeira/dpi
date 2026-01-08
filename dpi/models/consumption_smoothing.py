"""Consumption smoothing model with multi-factor income risk."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
import numpy as np

from .base import BaseModel

Tensor = torch.Tensor


class ConsumptionSmoothing(BaseModel):
    """Multi-factor consumption smoothing model.

    State: s = [w, x_1, ..., x_K]
        - w: wealth
        - x_k: income factor k (K factors total)

    Control: c (consumption), scalar

    Dynamics:
        - dw = (r * w + Y(x) - c) dt  (no diffusion in wealth)
        - dx_k = -theta_k * x_k dt + sigma_k dZ_k  (OU processes)

    Income:
        - Y(x) = Ybar * exp(sum_k beta_k * x_k)

    Utility:
        - CRRA: u(c) = c^(1-gamma) / (1-gamma)
        - Log utility when gamma = 1
    """

    def __init__(
        self,
        K: int = 1,
        rho: float = 0.04,
        r: float = 0.02,
        gamma: float = 2.0,
        Ybar: float = 1.0,
        theta: Optional[Union[float, Sequence[float]]] = None,
        sigma: Optional[Union[float, Sequence[float]]] = None,
        beta: Optional[Union[float, Sequence[float]]] = None,
        correlation: Optional[Tensor] = None,
        w_min: float = 0.1,
        w_max: float = 10.0,
        x_std_range: float = 3.0,
        *,
        device=None,
        dtype=None,
    ) -> None:
        """Initialize the consumption smoothing model.

        Args:
            K: Number of income factors.
            rho: Discount rate.
            r: Risk-free interest rate.
            gamma: Risk aversion coefficient (CRRA parameter).
            Ybar: Mean income level.
            theta: Mean reversion speeds for factors. Scalar or length-K sequence.
                   Default: 0.5 for all factors.
            sigma: Volatilities for factors. Scalar or length-K sequence.
                   Default: 0.2 for all factors.
            beta: Income loadings on factors. Scalar or length-K sequence.
                  Default: 1.0 for all factors.
            correlation: (K, K) correlation matrix for factor shocks.
                        Default: identity (uncorrelated).
            w_min: Minimum wealth for sampling.
            w_max: Maximum wealth for sampling.
            x_std_range: Sample factors within +/- this many standard deviations.
            device: PyTorch device.
            dtype: PyTorch dtype.
        """
        self.K = K
        self._rho = float(rho)
        self.r = float(r)
        self.gamma = float(gamma)
        self.Ybar = float(Ybar)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.x_std_range = float(x_std_range)
        self.device = device
        self.dtype = dtype

        # Process theta (mean reversion)
        if theta is None:
            self._theta = torch.full((K,), 0.5, device=device, dtype=dtype)
        elif isinstance(theta, (int, float)):
            self._theta = torch.full((K,), float(theta), device=device, dtype=dtype)
        else:
            self._theta = torch.tensor(theta, device=device, dtype=dtype)

        # Process sigma (volatility)
        if sigma is None:
            self._sigma = torch.full((K,), 0.2, device=device, dtype=dtype)
        elif isinstance(sigma, (int, float)):
            self._sigma = torch.full((K,), float(sigma), device=device, dtype=dtype)
        else:
            self._sigma = torch.tensor(sigma, device=device, dtype=dtype)

        # Process beta (income loadings)
        if beta is None:
            self._beta = torch.full((K,), 1.0, device=device, dtype=dtype)
        elif isinstance(beta, (int, float)):
            self._beta = torch.full((K,), float(beta), device=device, dtype=dtype)
        else:
            self._beta = torch.tensor(beta, device=device, dtype=dtype)

        # Process correlation matrix
        if correlation is None:
            self._corr = torch.eye(K, device=device, dtype=dtype)
        else:
            self._corr = correlation.to(device=device, dtype=dtype)

        # Compute Cholesky factor for correlated shocks
        # diffusion matrix = diag(sigma) @ L where L = cholesky(correlation)
        self._cholesky = torch.linalg.cholesky(self._corr)

        # Compute stationary standard deviation for each factor
        # For OU: stationary variance = sigma^2 / (2 * theta)
        self._x_stationary_std = self._sigma / torch.sqrt(2.0 * self._theta)

    @property
    def discount_rate(self) -> float:
        return self._rho

    @property
    def state_dim(self) -> int:
        """State dimension: wealth + K factors."""
        return 1 + self.K

    def income(self, x: Tensor) -> Tensor:
        """Compute income Y(x) = Ybar * exp(sum_k beta_k * x_k).

        Args:
            x: Factor values, shape (batch, K).

        Returns:
            Income, shape (batch, 1).
        """
        # Clip factor sum for numerical stability
        factor_sum = (x * self._beta).sum(dim=1, keepdim=True)
        factor_sum = torch.clamp(factor_sum, min=-10.0, max=10.0)
        return self.Ybar * torch.exp(factor_sum)

    def drift(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Compute drift for the state [w, x_1, ..., x_K].

        Args:
            s: State tensor, shape (batch, K+1). First column is wealth,
               remaining columns are factors.
            c: Consumption control, shape (batch, 1). If None, uses a
               simple rule c = r*w + Y(x) (consume all income).

        Returns:
            Drift tensor, shape (batch, K+1).
        """
        batch_size = s.shape[0]
        w = s[:, 0:1]  # (batch, 1)
        x = s[:, 1:]   # (batch, K)

        # Income
        Y = self.income(x)  # (batch, 1)

        # Default consumption rule if not provided
        if c is None:
            c = self.r * w + Y

        # Wealth drift: dw = (r * w + Y - c) dt
        dw = self.r * w + Y - c  # (batch, 1)

        # Factor drift: dx_k = -theta_k * x_k dt
        dx = -self._theta.unsqueeze(0) * x  # (batch, K)

        return torch.cat([dw, dx], dim=1)  # (batch, K+1)

    def diffusion(self, s: Tensor, c: Optional[Tensor] = None):
        """Compute diffusion columns for each shock.

        The diffusion matrix is structured as:
            [0, 0, ..., 0]        <- wealth has no diffusion
            [g_1,1, g_1,2, ...]   <- factor 1 diffusion across shocks
            [g_2,1, g_2,2, ...]   <- factor 2 diffusion across shocks
            ...

        With correlated shocks: g = diag(sigma) @ L where L = cholesky(corr).

        Args:
            s: State tensor, shape (batch, K+1).
            c: Control tensor (unused for diffusion).

        Returns:
            List of K tensors, each shape (batch, K+1), one per shock.
        """
        batch_size = s.shape[0]
        state_dim = 1 + self.K

        # Diffusion matrix for factors: diag(sigma) @ L
        # Shape: (K, K) where row i is factor i, column j is shock j
        diff_matrix = torch.diag(self._sigma) @ self._cholesky  # (K, K)

        # Build diffusion columns (one per shock)
        result = []
        for j in range(self.K):
            # Column j: zeros for wealth, diff_matrix[:, j] for factors
            col = torch.zeros((batch_size, state_dim), device=s.device, dtype=s.dtype)
            col[:, 1:] = diff_matrix[:, j].unsqueeze(0)  # Broadcast to batch
            result.append(col)

        return result

    def utility(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Compute CRRA utility u(c).

        Args:
            s: State tensor, shape (batch, K+1).
            c: Consumption, shape (batch, 1). If None, uses default rule.

        Returns:
            Utility, shape (batch, 1).
        """
        if c is None:
            # Default consumption: consume income + interest
            w = s[:, 0:1]
            x = s[:, 1:]
            Y = self.income(x)
            c = self.r * w + Y

        # Ensure positive consumption
        c = torch.clamp(c, min=1e-8)

        if abs(self.gamma - 1.0) < 1e-8:
            # Log utility
            return torch.log(c)
        else:
            # CRRA utility
            return (c ** (1.0 - self.gamma)) / (1.0 - self.gamma)

    def sample_states(self, n: int) -> Tensor:
        """Sample n states from the state space.

        Samples wealth uniformly and factors from truncated normal
        (approximating the stationary distribution).

        Args:
            n: Number of states to sample.

        Returns:
            States tensor, shape (n, K+1).
        """
        # Sample wealth uniformly in [w_min, w_max]
        w = torch.rand((n, 1), device=self.device, dtype=self.dtype)
        w = self.w_min + (self.w_max - self.w_min) * w

        # Sample factors from correlated normal (stationary scale)
        # Shocks are correlated via the Cholesky of the correlation matrix.
        z = torch.randn((n, self.K), device=self.device, dtype=self.dtype)
        eps = z @ self._cholesky.T
        x = eps * self._x_stationary_std.unsqueeze(0)

        # Truncate to reasonable range
        x_max = self.x_std_range * self._x_stationary_std.unsqueeze(0)
        x = torch.clamp(x, min=-x_max, max=x_max)

        return torch.cat([w, x], dim=1)

    def consumption_bounds(self, s: Tensor) -> tuple[Tensor, Tensor]:
        """Compute reasonable consumption bounds given state.

        Args:
            s: State tensor, shape (batch, K+1).

        Returns:
            Tuple of (c_min, c_max), each shape (batch, 1).
        """
        w = s[:, 0:1]
        x = s[:, 1:]
        Y = self.income(x)

        # Minimum consumption: small positive value
        c_min = torch.full_like(w, 1e-4)

        # Maximum consumption: keep wealth above w_min in one-unit time step
        c_max = self.r * w + Y + torch.clamp(w - self.w_min, min=0.0)

        return c_min, c_max

    def marginal_propensity_to_consume(self, s: Tensor, c: Tensor) -> Tensor:
        """Compute MPC = dc/dw using Euler equation.

        For diagnostic purposes.

        Args:
            s: State tensor, shape (batch, K+1).
            c: Consumption tensor, shape (batch, 1).

        Returns:
            MPC tensor, shape (batch, 1).
        """
        # This is a simplified approximation; true MPC requires solving the model
        # For CRRA: u'(c) = c^(-gamma), so MPC from Euler is related to r/rho
        return torch.full_like(c, self.r / self._rho)
