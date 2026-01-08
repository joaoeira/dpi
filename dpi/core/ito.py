"""Hyper-dual Ito's lemma utilities."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Union
import math

import torch

Tensor = torch.Tensor


def _normalize_diffusion(diffusion: Union[Tensor, Sequence[Tensor]]) -> List[Tensor]:
    if isinstance(diffusion, torch.Tensor):
        if diffusion.dim() == 1:
            return [diffusion]
        if diffusion.dim() == 2:
            return [diffusion]
        if diffusion.dim() == 3:
            return list(diffusion.unbind(dim=-1))
        raise ValueError("diffusion tensor must have shape (batch, n) or (batch, n, m)")
    if isinstance(diffusion, (list, tuple)):
        if not diffusion:
            raise ValueError("diffusion list must be non-empty")
        return list(diffusion)
    raise TypeError("diffusion must be a Tensor or a list/tuple of Tensors")


def second_derivative_fd(
    F: Callable[[float], Tensor],
    h: float,
    stencil: str = "three",
) -> Tensor:
    """Compute a second derivative with central finite differences.

    F is a function of a scalar epsilon returning a Tensor.
    """
    if h <= 0:
        raise ValueError("h must be positive")

    stencil_key = stencil.lower()
    if stencil_key == "three":
        coeffs = { -1.0: 1.0, 0.0: -2.0, 1.0: 1.0 }
        denom = h * h
    elif stencil_key == "five":
        coeffs = { -2.0: -1.0, -1.0: 16.0, 0.0: -30.0, 1.0: 16.0, 2.0: -1.0 }
        denom = 12.0 * h * h
    elif stencil_key == "seven":
        coeffs = {
            -3.0: 2.0,
            -2.0: -27.0,
            -1.0: 270.0,
            0.0: -490.0,
            1.0: 270.0,
            2.0: -27.0,
            3.0: 2.0,
        }
        denom = 180.0 * h * h
    elif stencil_key == "nine":
        coeffs = {
            -4.0: -9.0,
            -3.0: 128.0,
            -2.0: -1008.0,
            -1.0: 8064.0,
            0.0: -14350.0,
            1.0: 8064.0,
            2.0: -1008.0,
            3.0: 128.0,
            4.0: -9.0,
        }
        denom = 5040.0 * h * h
    else:
        raise ValueError("stencil must be one of: three, five, seven, nine")

    result = None
    for k, coeff in coeffs.items():
        term = F(k * h)
        result = term * coeff if result is None else result + term * coeff
    return result / denom


def _second_derivative_autodiff(F: Callable[[Tensor], Tensor], device: torch.device, dtype: torch.dtype) -> Tensor:
    eps = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

    def F_eps(eps_tensor: Tensor) -> Tensor:
        return F(eps_tensor).reshape(-1)

    def dF(eps_tensor: Tensor) -> Tensor:
        return torch.autograd.functional.jacobian(F_eps, eps_tensor, create_graph=True)

    return torch.autograd.functional.jacobian(dF, eps)


def compute_drift_hyperdual(
    V: Callable[[Tensor], Tensor],
    s: Tensor,
    drift: Tensor,
    diffusion: Union[Tensor, Sequence[Tensor]],
    *,
    h: float = 5e-2,
    stencil: str = "three",
    method: str = "finite_diff",
) -> Tensor:
    """Compute E[dV]/dt using the hyper-dual auxiliary function.

    The result has the same batch shape as V(s).
    """
    diffusions = _normalize_diffusion(diffusion)
    m = len(diffusions)
    if m <= 0:
        raise ValueError("diffusion must include at least one shock column")

    sqrt2 = math.sqrt(2.0)

    def F(eps: Union[float, Tensor]) -> Tensor:
        total = None
        eps2 = eps * eps
        for g_i in diffusions:
            s_pert = s + g_i * (eps / sqrt2) + drift * (eps2 / (2.0 * m))
            v = V(s_pert)
            total = v if total is None else total + v
        return total

    method_key = method.lower()
    if method_key == "finite_diff":
        return second_derivative_fd(F, h, stencil=stencil)
    if method_key == "autodiff":
        return _second_derivative_autodiff(F, s.device, s.dtype)

    raise ValueError("method must be 'finite_diff' or 'autodiff'")
