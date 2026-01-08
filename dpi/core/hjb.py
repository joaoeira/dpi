"""HJB residual computation."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .ito import compute_drift_hyperdual

Tensor = torch.Tensor


def compute_hjb_residual(
    model,
    s: Tensor,
    c: Optional[Tensor],
    V: Callable[[Tensor], Tensor],
    *,
    method: str = "hyperdual",
    ito_method: str = "finite_diff",
    h: float = 5e-2,
    stencil: str = "three",
) -> Tensor:
    """Compute the HJB residual for a batch of states.

    HJB(s, c, V) = u(c) - rho * V(s) + E[dV]/dt
    """
    if method.lower() != "hyperdual":
        raise ValueError("only 'hyperdual' method is supported for HJB residuals")

    drift = model.drift(s, c)
    diffusion = model.diffusion(s, c)
    utility = model.utility(s, c)

    drift_term = compute_drift_hyperdual(
        V,
        s,
        drift,
        diffusion,
        h=h,
        stencil=stencil,
        method=ito_method,
    )

    v = V(s)
    if utility.shape != v.shape:
        utility = utility.view_as(v)
    if drift_term.shape != v.shape:
        drift_term = drift_term.view_as(v)
    return utility - model.discount_rate * v + drift_term
