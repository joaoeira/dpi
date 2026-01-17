"""Controlled stochastic Kuramoto oscillator model."""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch

from .base import BaseModel

Tensor = torch.Tensor


def ring_graph(N: int, k: int = 1) -> Tuple[Tensor, Tensor]:
    """Create ring graph edge list with k nearest neighbors on each side.

    Args:
        N: Number of nodes
        k: Number of neighbors on each side (total 2k neighbors per node)

    Returns:
        src: Source node indices, shape (2*N*k,)
        dst: Destination node indices, shape (2*N*k,)
    """
    edges_src = []
    edges_dst = []
    for i in range(N):
        for offset in range(1, k + 1):
            # Forward neighbor
            j_fwd = (i + offset) % N
            edges_src.append(i)
            edges_dst.append(j_fwd)
            # Backward neighbor
            j_bwd = (i - offset) % N
            edges_src.append(i)
            edges_dst.append(j_bwd)
    return torch.tensor(edges_src, dtype=torch.long), torch.tensor(edges_dst, dtype=torch.long)


def watts_strogatz_graph(
    N: int,
    k: int = 2,
    p: float = 0.1,
    seed: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Create small-world graph via Watts-Strogatz rewiring.

    Args:
        N: Number of nodes
        k: Each node connected to k nearest neighbors on each side
        p: Rewiring probability
        seed: Random seed for reproducibility

    Returns:
        src, dst: Edge list tensors
    """
    rng = torch.Generator().manual_seed(seed)

    # Start with ring
    edges = set()
    for i in range(N):
        for offset in range(1, k + 1):
            j = (i + offset) % N
            edges.add((min(i, j), max(i, j)))

    # Rewire with probability p
    edges_list = list(edges)
    for idx, (i, j) in enumerate(edges_list):
        if torch.rand(1, generator=rng).item() < p:
            # Remove edge (i, j), add edge (i, new_j)
            edges.discard((i, j))
            # Pick new target avoiding self-loop and existing edges
            candidates = [
                c for c in range(N)
                if c != i and (min(i, c), max(i, c)) not in edges
            ]
            if candidates:
                new_j = candidates[int(torch.randint(len(candidates), (1,), generator=rng).item())]
                edges.add((min(i, new_j), max(i, new_j)))
            else:
                edges.add((i, j))  # Keep original if no valid rewire

    # Convert to directed edges (both directions)
    src_list, dst_list = [], []
    for (i, j) in edges:
        src_list.extend([i, j])
        dst_list.extend([j, i])

    return torch.tensor(src_list, dtype=torch.long), torch.tensor(dst_list, dtype=torch.long)


class KuramotoModel(BaseModel):
    """Controlled stochastic Kuramoto model on a sparse graph.

    State: theta in R^N (angles, can be unbounded but dynamics are 2pi-periodic)
    Control: u in R^M where M <= N (partial actuation)

    Dynamics:
        dtheta_i = (omega_i + coupling_i + u_i) dt + sigma_i dB_i

    where coupling_i = (K/degree_i) * sum_{j in neighbors(i)} sin(theta_j - theta_i)

    Utility:
        u(theta, u) = -(1 - R(theta)^2) - (lambda/2) * ||u||^2

    where R is the Kuramoto order parameter.
    """

    def __init__(
        self,
        N: int,
        rho: float = 0.01,
        K: float = 1.0,
        sigma: float = 0.1,
        omega_std: float = 0.5,
        control_cost: float = 0.1,
        control_mask: Optional[Tensor] = None,
        edge_src: Optional[Tensor] = None,
        edge_dst: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        graph_type: Literal["ring", "watts_strogatz"] = "ring",
        ring_k: int = 2,
        ws_p: float = 0.1,
        u_max: float = 1.0,
        omega_seed: int = 42,
        graph_seed: int = 0,
        *,
        device=None,
        dtype=None,
    ) -> None:
        """Initialize Kuramoto model.

        Args:
            N: Number of oscillators
            rho: Discount rate
            K: Coupling strength
            sigma: Noise intensity (scalar or per-oscillator)
            omega_std: Standard deviation of natural frequencies
            control_cost: Quadratic control cost coefficient (lambda/2)
            control_mask: Boolean tensor (N,) indicating controlled nodes.
                         If None, all nodes are controlled (M=N).
            edge_src, edge_dst: Edge list for graph. If None, uses graph_type.
            edge_weight: Optional edge weights. If None, all weights are 1.
            graph_type: "ring" or "watts_strogatz" if edge list not provided
            ring_k: Neighbors on each side for ring/WS base
            ws_p: Rewiring probability for Watts-Strogatz
            u_max: Control bound magnitude
            omega_seed: Random seed for natural frequencies
            graph_seed: Random seed for graph construction
            device, dtype: PyTorch device and dtype
        """
        self.N = N
        self._rho = float(rho)
        self.K = float(K)
        self.control_cost = float(control_cost)
        self.u_max = float(u_max)
        self.device = device
        self.dtype = dtype

        # Natural frequencies: omega_i ~ N(0, omega_std^2)
        gen = torch.Generator().manual_seed(omega_seed)
        self._omega = torch.randn(N, generator=gen, device=device, dtype=dtype) * omega_std

        # Noise intensity
        if isinstance(sigma, (int, float)):
            self._sigma = torch.full((N,), float(sigma), device=device, dtype=dtype)
        else:
            self._sigma = torch.as_tensor(sigma, device=device, dtype=dtype)

        # Control mask: which nodes are actuated
        if control_mask is None:
            self._control_mask = torch.ones(N, dtype=torch.bool, device=device)
            self.M = N
        else:
            self._control_mask = control_mask.to(device=device, dtype=torch.bool)
            self.M = int(self._control_mask.sum().item())

        # Build graph
        if edge_src is None or edge_dst is None:
            if graph_type == "ring":
                edge_src, edge_dst = ring_graph(N, k=ring_k)
            else:
                edge_src, edge_dst = watts_strogatz_graph(N, k=ring_k, p=ws_p, seed=graph_seed)

        self._edge_src = edge_src.to(device=device)
        self._edge_dst = edge_dst.to(device=device)

        if edge_weight is None:
            self._edge_weight = torch.ones(self._edge_src.shape[0], device=device, dtype=dtype)
        else:
            self._edge_weight = edge_weight.to(device=device, dtype=dtype)

        # Precompute node degrees for normalization
        self._degree = torch.zeros(N, device=device, dtype=dtype)
        self._degree.scatter_add_(0, self._edge_dst, torch.ones_like(self._edge_weight))
        self._degree = torch.clamp(self._degree, min=1.0)

        # Indices of controlled nodes (for mapping control vector to full state)
        self._controlled_indices = torch.nonzero(self._control_mask, as_tuple=True)[0]

    @property
    def discount_rate(self) -> float:
        return self._rho

    @property
    def state_dim(self) -> int:
        return self.N

    @property
    def control_dim(self) -> int:
        return self.M

    def order_parameter(self, theta: Tensor) -> Tensor:
        """Compute Kuramoto order parameter R(theta).

        R = |1/N * sum_j exp(i * theta_j)|

        Args:
            theta: Angles, shape (batch, N)

        Returns:
            R: Order parameter, shape (batch, 1), in [0, 1]
        """
        # Use real arithmetic: R = sqrt((sum cos)^2 + (sum sin)^2) / N
        cos_mean = torch.cos(theta).mean(dim=1, keepdim=True)
        sin_mean = torch.sin(theta).mean(dim=1, keepdim=True)
        R = torch.sqrt(cos_mean ** 2 + sin_mean ** 2)
        return R

    def _sparse_coupling(self, theta: Tensor) -> Tensor:
        """Compute sparse graph coupling sum_j A_ij sin(theta_j - theta_i).

        Uses scatter_add for O(E) complexity where E is number of edges.

        Args:
            theta: Angles, shape (batch, N)

        Returns:
            coupling: Shape (batch, N)
        """
        batch_size = theta.shape[0]

        # Get source and destination angles
        theta_src = theta[:, self._edge_src]  # (batch, E)
        theta_dst = theta[:, self._edge_dst]  # (batch, E)

        # sin(theta_src - theta_dst) = influence of src on dst
        sin_diff = torch.sin(theta_src - theta_dst) * self._edge_weight  # (batch, E)

        # Aggregate at destination nodes
        coupling = torch.zeros((batch_size, self.N), device=theta.device, dtype=theta.dtype)

        # Expand edge_dst for batched scatter_add
        edge_dst_expanded = self._edge_dst.unsqueeze(0).expand(batch_size, -1)  # (batch, E)
        coupling.scatter_add_(1, edge_dst_expanded, sin_diff)

        # Normalize by degree
        coupling = coupling / self._degree.unsqueeze(0)

        return self.K * coupling

    def _expand_control(self, c: Optional[Tensor], batch_size: int) -> Tensor:
        """Expand control from M-dimensional to N-dimensional.

        Args:
            c: Control, shape (batch, M) or None
            batch_size: Batch size for output

        Returns:
            c_full: Shape (batch, N), zeros at uncontrolled nodes
        """
        device = self.device if self.device is not None else (c.device if c is not None else None)
        dtype = self.dtype if self.dtype is not None else (c.dtype if c is not None else torch.float32)

        if c is None:
            return torch.zeros((batch_size, self.N), device=device, dtype=dtype)

        c_full = torch.zeros((batch_size, self.N), device=c.device, dtype=c.dtype)
        c_full[:, self._controlled_indices] = c
        return c_full

    def drift(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Compute drift for Kuramoto dynamics.

        drift_i = omega_i + coupling_i + u_i

        Args:
            s: State (angles), shape (batch, N)
            c: Control, shape (batch, M) or None

        Returns:
            drift: Shape (batch, N)
        """
        batch_size = s.shape[0]
        coupling = self._sparse_coupling(s)
        u_full = self._expand_control(c, batch_size)

        return self._omega.unsqueeze(0) + coupling + u_full

    def diffusion(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Compute diffusion matrix (diagonal noise).

        Each oscillator has independent noise: g_i = sigma_i * e_i

        Returns tensor of shape (batch, N, N) where [:, :, i] is the i-th column.
        """
        batch_size = s.shape[0]

        # Diagonal diffusion: each shock affects only one oscillator
        # Shape: (batch, N, N) where [:, :, i] = sigma_i * e_i
        diffusion_matrix = torch.diag(self._sigma).unsqueeze(0).expand(batch_size, -1, -1)

        return diffusion_matrix.clone()

    def utility(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Compute utility: -(1 - R^2) - (lambda/2) ||u||^2.

        Args:
            s: State (angles), shape (batch, N)
            c: Control, shape (batch, M) or None

        Returns:
            utility: Shape (batch, 1)
        """
        R = self.order_parameter(s)
        sync_reward = -(1.0 - R ** 2)  # Maximized when R=1 (full sync)

        if c is not None:
            control_penalty = -self.control_cost * (c ** 2).sum(dim=1, keepdim=True)
        else:
            control_penalty = torch.zeros_like(sync_reward)

        return sync_reward + control_penalty

    def sample_states(self, n: int) -> Tensor:
        """Sample initial angles uniformly on [0, 2*pi]^N."""
        return torch.rand((n, self.N), device=self.device, dtype=self.dtype) * 2 * math.pi

    def sample_states_shift_invariant(self, n: int) -> Tensor:
        """Sample with theta_0 = 0 (shift-invariant reference frame)."""
        theta = torch.rand((n, self.N), device=self.device, dtype=self.dtype) * 2 * math.pi
        theta[:, 0] = 0.0
        return theta

    def control_bounds(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        """Return control bounds for StateBoundedPolicy wrapper.

        Returns:
            (c_min, c_max): Each shape (batch, M)
        """
        batch_size = s.shape[0]
        c_min = torch.full((batch_size, self.M), -self.u_max, device=s.device, dtype=s.dtype)
        c_max = torch.full((batch_size, self.M), self.u_max, device=s.device, dtype=s.dtype)
        return c_min, c_max


class SwingModel(KuramotoModel):
    """Second-order swing equation model.

    State: [theta, omega] in R^{2N}

    Dynamics:
        dtheta_i = omega_i dt
        domega_i = (-alpha_i * omega_i + P_i + coupling_i + u_i) dt + sigma_i dB_i
    """

    def __init__(
        self,
        N: int,
        rho: float = 0.01,
        K: float = 1.0,
        alpha: float = 0.1,
        P_mean: float = 0.0,
        P_std: float = 0.1,
        sigma: float = 0.1,
        control_cost: float = 0.1,
        omega_cost: float = 0.1,
        control_mask: Optional[Tensor] = None,
        P_seed: int = 43,
        **kwargs,
    ) -> None:
        """Initialize swing model.

        Args:
            N: Number of oscillators
            alpha: Damping coefficient
            P_mean: Mean power injection
            P_std: Standard deviation of power injections
            omega_cost: Cost coefficient for frequency deviations
            P_seed: Random seed for power injections
            **kwargs: Arguments passed to KuramotoModel
        """
        super().__init__(
            N=N, rho=rho, K=K, sigma=sigma,
            control_cost=control_cost, control_mask=control_mask, **kwargs
        )

        self.alpha = float(alpha)
        self.omega_cost = float(omega_cost)

        # Power injections P_i
        gen = torch.Generator().manual_seed(P_seed)
        self._P = torch.randn(N, generator=gen, device=self.device, dtype=self.dtype) * P_std + P_mean

    @property
    def state_dim(self) -> int:
        return 2 * self.N

    def drift(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Compute drift for swing dynamics."""
        batch_size = s.shape[0]
        theta = s[:, :self.N]
        omega = s[:, self.N:]

        # Coupling (only affects frequency, not angle)
        coupling = self._sparse_coupling(theta)
        u_full = self._expand_control(c, batch_size)

        # dtheta = omega
        dtheta = omega

        # domega = -alpha*omega + P + coupling + u
        domega = -self.alpha * omega + self._P.unsqueeze(0) + coupling + u_full

        return torch.cat([dtheta, domega], dim=1)

    def diffusion(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Diffusion only in omega components."""
        batch_size = s.shape[0]

        # Noise only in frequency states (last N components)
        # Shape: (batch, 2N, N)
        diffusion = torch.zeros(
            (batch_size, 2 * self.N, self.N),
            device=s.device, dtype=s.dtype
        )
        diffusion[:, self.N:, :] = torch.diag(self._sigma).unsqueeze(0)

        return diffusion

    def utility(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Utility based on phase coherence and frequency regulation."""
        theta = s[:, :self.N]
        omega = s[:, self.N:]

        R = self.order_parameter(theta)
        sync_reward = -(1.0 - R ** 2)

        # Penalize frequency deviations
        freq_penalty = -self.omega_cost * (omega ** 2).mean(dim=1, keepdim=True)

        if c is not None:
            control_penalty = -self.control_cost * (c ** 2).sum(dim=1, keepdim=True)
        else:
            control_penalty = torch.zeros_like(sync_reward)

        return sync_reward + freq_penalty + control_penalty

    def sample_states(self, n: int) -> Tensor:
        """Sample [theta, omega] with theta uniform, omega near zero."""
        theta = torch.rand((n, self.N), device=self.device, dtype=self.dtype) * 2 * math.pi
        omega = torch.randn((n, self.N), device=self.device, dtype=self.dtype) * 0.1
        return torch.cat([theta, omega], dim=1)
