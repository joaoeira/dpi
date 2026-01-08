"""Health management model with multi-dimensional health states."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch

from .base import BaseModel

Tensor = torch.Tensor


def _as_tensor(x, K: int, device=None, dtype=None) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, (int, float)):
        return torch.full((K,), float(x), device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


class HealthManagement(BaseModel):
    """Multi-factor health management model.

    State: s = [h_1, ..., h_K, a]
        - h_k: health dimensions
        - a: age

    Controls: u = [e, d, l]
        - e: exercise effort
        - d: diet quality
        - l: leisure

    Consumption: c = w(h) * n - p_d * d^2, with n = Tbar - sleep - e - l.
    """

    def __init__(
        self,
        K: int = 8,
        rho: float = 0.04,
        gamma: float = 2.0,
        alpha: float = 0.6,
        eta: float = 0.1,
        omega: Optional[Sequence[float]] = None,
        theta_age_min: float = 20.0,
        theta_age_max: float = 80.0,
        delta_base: Union[float, Sequence[float]] = 0.05,
        delta_slope: Union[float, Sequence[float]] = 0.01,
        nu0: Union[float, Sequence[float]] = -0.01,
        nu1: Union[float, Sequence[float]] = -0.02,
        phi: Optional[Sequence[Sequence[float]]] = None,
        beta_e: Union[float, Sequence[float]] = 0.05,
        beta_d: Union[float, Sequence[float]] = 0.03,
        sigma: Union[float, Sequence[float]] = 0.08,
        correlation: Optional[Tensor] = None,
        wage_base: float = 1.0,
        diet_cost: float = 0.0,
        kappa_d: float = 0.0,
        mu_e: float = 0.0,
        nu_e: float = 0.0,
        wage_weights: Optional[Sequence[float]] = None,
        tbar: float = 1.0,
        sleep: float = 0.3,
        e_max: float = 0.2,
        l_min: float = 0.05,
        n_min: float = 0.2,
        d_min: float = 0.0,
        d_max: float = 1.0,
        h_std_range: float = 3.0,
        *,
        device=None,
        dtype=None,
    ) -> None:
        self.K = K
        self._rho = float(rho)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.age_min = float(theta_age_min)
        self.age_max = float(theta_age_max)
        self.tbar = float(tbar)
        self.sleep = float(sleep)
        self.e_max = float(e_max)
        self.l_min = float(l_min)
        self.n_min = float(n_min)
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.h_std_range = float(h_std_range)
        self.wage_base = float(wage_base)
        self.device = device
        self.dtype = dtype

        self._omega = _as_tensor(omega or [0.4] * K, K, device, dtype)
        self._delta_base = _as_tensor(delta_base, K, device, dtype)
        self._delta_slope = _as_tensor(delta_slope, K, device, dtype)
        self._nu0 = _as_tensor(nu0, K, device, dtype)
        self._nu1 = _as_tensor(nu1, K, device, dtype)
        self._beta_e = _as_tensor(beta_e, K, device, dtype)
        self._beta_d = _as_tensor(beta_d, K, device, dtype)
        self._sigma = _as_tensor(sigma, K, device, dtype)
        self.diet_cost = float(diet_cost)
        self.kappa_d = float(kappa_d)
        self.mu_e = float(mu_e)
        self.nu_e = float(nu_e)

        if wage_weights is None:
            wage_weights = [0.3] * K
        self._wage_weights = _as_tensor(wage_weights, K, device, dtype)

        if phi is None:
            self._phi = torch.zeros((K, K), device=device, dtype=dtype)
        else:
            self._phi = torch.tensor(phi, device=device, dtype=dtype)

        if correlation is None:
            self._corr = torch.eye(K, device=device, dtype=dtype)
        else:
            self._corr = correlation.to(device=device, dtype=dtype)

        self._cholesky = torch.linalg.cholesky(self._corr)

        # Stationary scale proxy for sampling
        self._h_stationary_std = self._sigma / torch.sqrt(2.0 * torch.clamp(self._delta_base, min=1e-4))

        # Ensure leisure bound is feasible given e_max and n_min
        self.l_max = self.tbar - self.sleep - self.e_max - self.n_min
        if self.l_max < self.l_min:
            raise ValueError("Infeasible time budget: increase tbar or reduce e_max/n_min.")

    @property
    def discount_rate(self) -> float:
        return self._rho

    @property
    def state_dim(self) -> int:
        return self.K + 1

    def _age_norm(self, age: Tensor) -> Tensor:
        return torch.clamp((age - self.age_min) / (self.age_max - self.age_min), 0.0, 1.0)

    def _age_excess(self, age: Tensor, anchor: float) -> Tensor:
        return torch.clamp(age - anchor, min=0.0)

    def wage(self, h: Tensor) -> Tensor:
        """Wage as a function of health."""
        score = (h * self._wage_weights).sum(dim=1, keepdim=True)
        score = torch.clamp(score, min=-5.0, max=5.0)
        return self.wage_base * torch.exp(score)

    def quality(self, h: Tensor) -> Tensor:
        """Health quality index Q(h)."""
        linear = (h * self._omega).sum(dim=1, keepdim=True)
        quad = 0.5 * self.eta * (h ** 2).sum(dim=1, keepdim=True)
        return torch.exp(linear - quad)

    def controls_to_outcomes(self, s: Tensor, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Map controls to (e, d, l, c)."""
        e = u[:, 0:1]
        d = u[:, 1:2]
        l = u[:, 2:3]

        h = s[:, : self.K]
        wage = self.wage(h)

        n = self.tbar - self.sleep - e - l
        n = torch.clamp(n, min=self.n_min)
        c = wage * n - self.diet_cost * (d ** 2)

        return e, d, l, c

    def drift(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        h = s[:, : self.K]
        age = s[:, self.K : self.K + 1]

        if c is None:
            raise ValueError("Controls are required for drift in this model.")

        e, d, _l, _cons = self.controls_to_outcomes(s, c)

        age_norm = self._age_norm(age)
        age_excess = self._age_excess(age, 30.0)
        delta = self._delta_base + self._delta_slope * age_excess
        nu = self._nu0 + self._nu1 * age_norm

        cross = h @ self._phi.T
        dh = -delta * h + cross + self._beta_e * e + self._beta_d * d + nu

        dage = torch.ones_like(age)
        return torch.cat([dh, dage], dim=1)

    def diffusion(self, s: Tensor, c: Optional[Tensor] = None):
        batch = s.shape[0]
        state_dim = self.K + 1

        diff_matrix = torch.diag(self._sigma) @ self._cholesky

        result = []
        for j in range(self.K):
            col = torch.zeros((batch, state_dim), device=s.device, dtype=s.dtype)
            col[:, : self.K] = diff_matrix[:, j].unsqueeze(0)
            result.append(col)

        return result

    def utility(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        if c is None:
            raise ValueError("Controls are required for utility in this model.")

        h = s[:, : self.K]
        e, d, l, cons = self.controls_to_outcomes(s, c)

        cons = torch.clamp(cons, min=1e-8)
        l = torch.clamp(l, min=1e-6)

        q = self.quality(h)
        mood = 1.0 + self.mu_e * e - self.nu_e * (e ** 2)
        mood = torch.clamp(mood, min=1e-4)
        diet_penalty = torch.exp(-self.kappa_d * (d ** 2))
        inside = (cons ** self.alpha) * (l ** (1.0 - self.alpha)) * q * mood * diet_penalty

        if abs(self.gamma - 1.0) < 1e-8:
            return torch.log(inside)
        return (inside ** (1.0 - self.gamma)) / (1.0 - self.gamma)

    def control_bounds(self, s: Tensor) -> tuple[Tensor, Tensor]:
        batch = s.shape[0]
        device = s.device
        dtype = s.dtype

        e_min = torch.zeros((batch, 1), device=device, dtype=dtype)
        e_max = torch.full((batch, 1), self.e_max, device=device, dtype=dtype)

        d_min = torch.full((batch, 1), self.d_min, device=device, dtype=dtype)
        d_max = torch.full((batch, 1), self.d_max, device=device, dtype=dtype)

        l_min = torch.full((batch, 1), self.l_min, device=device, dtype=dtype)
        l_max = torch.full((batch, 1), self.l_max, device=device, dtype=dtype)

        c_min = torch.cat([e_min, d_min, l_min], dim=1)
        c_max = torch.cat([e_max, d_max, l_max], dim=1)

        return c_min, c_max

    def sample_states(self, n: int) -> Tensor:
        # Sample health factors with correlation
        z = torch.randn((n, self.K), device=self.device, dtype=self.dtype)
        eps = z @ self._cholesky.T
        h = eps * self._h_stationary_std.unsqueeze(0)

        h_max = self.h_std_range * self._h_stationary_std.unsqueeze(0)
        h = torch.clamp(h, min=-h_max, max=h_max)

        age = torch.rand((n, 1), device=self.device, dtype=self.dtype)
        age = self.age_min + (self.age_max - self.age_min) * age

        return torch.cat([h, age], dim=1)


def default_phi_matrix() -> list[list[float]]:
    return [
        [0.0, -0.1, 0.05, 0.05, -0.05, 0.1, 0.0, 0.0],
        [0.1, 0.0, 0.1, 0.05, -0.1, 0.05, 0.0, 0.0],
        [0.05, 0.15, 0.0, 0.05, -0.1, 0.1, 0.0, 0.0],
        [0.05, 0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.1, 0.05, 0.05, 0.0, 0.0, 0.2, 0.05, 0.0],
        [0.05, 0.0, 0.05, 0.0, 0.15, 0.0, 0.0, 0.05],
        [0.1, 0.0, 0.05, 0.0, 0.1, 0.15, 0.0, 0.0],
        [0.05, 0.0, 0.1, 0.0, 0.1, 0.15, 0.0, 0.0],
    ]
