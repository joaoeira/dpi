"""Hyper-dual Ito's lemma utilities."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union
import math

import torch

Tensor = torch.Tensor


def _normalize_diffusion_list(diffusion: Union[Tensor, Sequence[Tensor]]) -> List[Tensor]:
    """Normalize diffusion to list of column tensors (legacy format)."""
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


def _normalize_diffusion_tensor(diffusion: Union[Tensor, Sequence[Tensor]]) -> Tensor:
    """Normalize diffusion to 3D tensor of shape (batch, n, m).

    Supports multiple input formats:
    - Single tensor (batch, n): m=1, returns (batch, n, 1)
    - Single tensor (batch, n, m): returns as-is
    - List of m tensors (batch, n): stacks to (batch, n, m)
    """
    if isinstance(diffusion, torch.Tensor):
        if diffusion.dim() == 2:
            # (batch, n) -> (batch, n, 1)
            return diffusion.unsqueeze(-1)
        if diffusion.dim() == 3:
            # (batch, n, m) -> as-is
            return diffusion
        raise ValueError("diffusion tensor must have shape (batch, n) or (batch, n, m)")

    if isinstance(diffusion, (list, tuple)):
        if not diffusion:
            raise ValueError("diffusion list must be non-empty")
        # Stack list of (batch, n) to (batch, n, m)
        return torch.stack(list(diffusion), dim=-1)

    raise TypeError("diffusion must be a Tensor or list of Tensors")


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


def _get_stencil_coeffs(stencil: str) -> tuple:
    """Get finite difference stencil coefficients and denominator factor.

    Returns (coeffs_dict, denom_factor) where denom = denom_factor * h^2.
    """
    stencil_key = stencil.lower()
    if stencil_key == "three":
        return {-1.0: 1.0, 0.0: -2.0, 1.0: 1.0}, 1.0
    if stencil_key == "five":
        return {-2.0: -1.0, -1.0: 16.0, 0.0: -30.0, 1.0: 16.0, 2.0: -1.0}, 12.0
    if stencil_key == "seven":
        return {
            -3.0: 2.0, -2.0: -27.0, -1.0: 270.0, 0.0: -490.0,
            1.0: 270.0, 2.0: -27.0, 3.0: 2.0,
        }, 180.0
    if stencil_key == "nine":
        return {
            -4.0: -9.0, -3.0: 128.0, -2.0: -1008.0, -1.0: 8064.0,
            0.0: -14350.0, 1.0: 8064.0, 2.0: -1008.0, 3.0: 128.0, 4.0: -9.0,
        }, 5040.0
    raise ValueError("stencil must be one of: three, five, seven, nine")


def _compute_drift_vectorized(
    V: Callable[[Tensor], Tensor],
    s: Tensor,
    drift: Tensor,
    diffusion: Union[Tensor, Sequence[Tensor]],
    *,
    h: float = 5e-2,
    stencil: str = "three",
    chunk_size: Optional[int] = None,
) -> Tensor:
    """Compute E[dV]/dt using vectorized shock batching.

    This version batches all m shock perturbations into a single V() call
    per stencil point, reducing complexity from O(m * stencil_points) to
    O(stencil_points) V() evaluations.

    Args:
        V: Value function, callable from (B, n) -> (B, 1)
        s: States, shape (B, n)
        drift: Drift vector, shape (B, n)
        diffusion: Diffusion matrix, shape (B, n, m) or list of m tensors (B, n)
        h: Finite difference step size
        stencil: Stencil type ("three", "five", "seven", "nine")
        chunk_size: If set, process shocks in chunks to limit memory.
                   None means process all m shocks in one batched call.

    Returns:
        E[dV]/dt, shape (B, 1)
    """
    # Normalize diffusion to (B, n, m)
    G = _normalize_diffusion_tensor(diffusion)
    B, n, m = G.shape

    sqrt2 = math.sqrt(2.0)
    coeffs, denom_factor = _get_stencil_coeffs(stencil)
    denom = denom_factor * h * h

    def F_vectorized(eps: float) -> Tensor:
        """Compute sum_i V(s + g_i * eps + drift * eps^2 / (2m)) with batched V call."""
        eps2 = eps * eps
        drift_term = drift * (eps2 / (2.0 * m))  # (B, n)

        if chunk_size is not None and m > chunk_size:
            # Process shocks in chunks to limit memory
            total = torch.zeros((B, 1), device=s.device, dtype=s.dtype)
            for start in range(0, m, chunk_size):
                end = min(start + chunk_size, m)
                m_chunk = end - start

                # g_chunk: (B, n, m_chunk)
                g_chunk = G[:, :, start:end]

                # Perturb states for this chunk
                # g_scaled: (B, n, m_chunk) -> (B, m_chunk, n) after permute
                g_scaled = g_chunk * (eps / sqrt2)
                g_scaled = g_scaled.permute(0, 2, 1)  # (B, m_chunk, n)
                s_expanded = s.unsqueeze(1) + drift_term.unsqueeze(1)  # (B, 1, n)
                s_pert = s_expanded + g_scaled  # (B, m_chunk, n)

                # Flatten for batched V call
                s_pert_flat = s_pert.reshape(B * m_chunk, n)  # (B*m_chunk, n)
                v_flat = V(s_pert_flat)  # (B*m_chunk, 1)
                v = v_flat.reshape(B, m_chunk, 1)  # (B, m_chunk, 1)

                total = total + v.sum(dim=1)  # (B, 1)

            return total
        else:
            # Process all shocks in one batched call
            # s_pert[b, i, :] = s[b, :] + G[b, :, i] * eps/sqrt2 + drift[b, :] * eps^2/(2m)

            # G: (B, n, m) -> g_scaled: (B, m, n) after permute
            g_scaled = G * (eps / sqrt2)  # (B, n, m)
            g_scaled = g_scaled.permute(0, 2, 1)  # (B, m, n)

            # s_expanded: (B, 1, n)
            s_expanded = s.unsqueeze(1) + drift_term.unsqueeze(1)  # (B, 1, n)

            # s_pert: (B, m, n)
            s_pert = s_expanded + g_scaled  # Broadcasting: (B, 1, n) + (B, m, n)

            # Flatten batch and shock dimensions for single V call
            s_pert_flat = s_pert.reshape(B * m, n)  # (B*m, n)

            # Single V call for all shocks
            v_flat = V(s_pert_flat)  # (B*m, 1)

            # Reshape and sum over shocks
            v = v_flat.reshape(B, m, 1)  # (B, m, 1)
            total = v.sum(dim=1)  # (B, 1)

            return total

    # Apply finite difference stencil
    result = None
    for k, coeff in coeffs.items():
        term = F_vectorized(k * h)
        result = term * coeff if result is None else result + term * coeff

    return result / denom


def compute_drift_hyperdual(
    V: Callable[[Tensor], Tensor],
    s: Tensor,
    drift: Tensor,
    diffusion: Union[Tensor, Sequence[Tensor]],
    *,
    h: float = 5e-2,
    stencil: str = "three",
    method: str = "finite_diff",
    vectorized: bool = True,
    chunk_size: Optional[int] = None,
) -> Tensor:
    """Compute E[dV]/dt using the hyper-dual auxiliary function.

    Args:
        V: Value function, callable from (B, n) -> (B, 1)
        s: States, shape (B, n)
        drift: Drift vector, shape (B, n)
        diffusion: Diffusion, shape (B, n, m) or list of m tensors (B, n)
        h: Finite difference step size
        stencil: Stencil type ("three", "five", "seven", "nine")
        method: "finite_diff" or "autodiff"
        vectorized: If True, use batched shock evaluation (recommended for m > 10).
                   Reduces O(m) V() calls to O(1) per stencil point.
        chunk_size: For vectorized mode, process shocks in chunks if set.
                   Useful for memory management with large m.

    Returns:
        E[dV]/dt, shape (B, 1)
    """
    method_key = method.lower()

    # Use vectorized implementation for finite_diff when enabled
    if method_key == "finite_diff" and vectorized:
        return _compute_drift_vectorized(
            V, s, drift, diffusion,
            h=h, stencil=stencil, chunk_size=chunk_size
        )

    # Legacy sequential implementation
    diffusions = _normalize_diffusion_list(diffusion)
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

    if method_key == "finite_diff":
        return second_derivative_fd(F, h, stencil=stencil)
    if method_key == "autodiff":
        return _second_derivative_autodiff(F, s.device, s.dtype)

    raise ValueError("method must be 'finite_diff' or 'autodiff'")
