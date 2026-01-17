"""Kuramoto oscillator experiment with DPI."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpi.algorithms.dpi import DPITrainer
from dpi.core.hjb import compute_hjb_residual
from dpi.models.kuramoto import KuramotoModel, SwingModel
from dpi.networks.angle_embedding import PeriodicValueNetwork, PeriodicStateBoundedPolicy
from dpi.utils.training import DPIConfig, TrainingHistory

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_model(
    N: int,
    K: float,
    sigma: float,
    control_fraction: float,
    graph_type: str,
    model_type: str,
    device=None,
    dtype=None,
) -> KuramotoModel:
    """Create Kuramoto or Swing model with specified parameters."""

    # Partial actuation: control only a fraction of nodes
    M = max(1, int(N * control_fraction))
    control_mask = torch.zeros(N, dtype=torch.bool, device=device)
    # Actuate evenly spaced nodes
    controlled_indices = torch.linspace(0, N - 1, M).long()
    control_mask[controlled_indices] = True

    kwargs = dict(
        N=N,
        K=K,
        sigma=sigma,
        control_mask=control_mask,
        graph_type=graph_type,
        ring_k=2,
        ws_p=0.1,
        u_max=1.0,
        control_cost=0.1,
        device=device,
        dtype=dtype,
    )

    if model_type == "swing":
        return SwingModel(**kwargs, alpha=0.1)
    return KuramotoModel(**kwargs)


def create_networks(
    model: KuramotoModel,
    hidden_dims: tuple = (256, 128, 64),
):
    """Create periodic value and policy networks."""

    # For Kuramoto: state is theta (N angles)
    # For Swing: state is [theta, omega] (N angles + N frequencies)
    if isinstance(model, SwingModel):
        n_angles = model.N
        n_passthrough = model.N  # omega components
    else:
        n_angles = model.N
        n_passthrough = 0

    value_net = PeriodicValueNetwork(
        n_angles=n_angles,
        n_passthrough=n_passthrough,
        hidden_dims=hidden_dims,
        activation="silu",
    )

    policy_net = PeriodicStateBoundedPolicy(
        n_angles=n_angles,
        output_dim=model.M,
        bounds_fn=model.control_bounds,
        n_passthrough=n_passthrough,
        hidden_dims=hidden_dims,
        activation="relu",
    )

    return value_net, policy_net


def compute_hjb_stats(
    model: KuramotoModel,
    value_net,
    policy_net,
    n_samples: int = 1000,
    device=None,
    dtype=None,
) -> dict:
    """Compute HJB residual statistics."""
    with torch.no_grad():
        s = model.sample_states(n_samples)
        s = s.to(device=device, dtype=dtype)
        u = policy_net(s)

        hjb = compute_hjb_residual(
            model, s, u, value_net,
            ito_method="finite_diff",
            h=1e-2,
            stencil="five",
            vectorized=True,
            chunk_size=64,
        )

        return {
            "mean": hjb.mean().item(),
            "std": hjb.std().item(),
            "max_abs": hjb.abs().max().item(),
        }


def simulate_trajectory(
    model: KuramotoModel,
    policy_net,
    theta_init: torch.Tensor,
    T: float = 10.0,
    dt: float = 0.01,
    use_control: bool = True,
    noise: Optional[torch.Tensor] = None,
    device=None,
    dtype=None,
) -> dict:
    """Simulate controlled trajectory using Euler-Maruyama."""
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)

    if noise is not None:
        if noise.shape != (n_steps, model.N):
            raise ValueError(f"noise must have shape ({n_steps}, {model.N})")

    state = theta_init.clone()
    state_history = [state.cpu().numpy().copy()]

    # Compute initial order parameter
    theta = state[:model.N] if isinstance(model, SwingModel) else state
    R_history = [model.order_parameter(theta.unsqueeze(0)).item()]
    u_history = []

    with torch.no_grad():
        for step in range(n_steps):
            s = state.unsqueeze(0)

            if use_control:
                u = policy_net(s)
            else:
                u = torch.zeros((1, model.M), device=device, dtype=dtype)

            u_history.append(u.squeeze(0).cpu().numpy().copy())

            # Euler-Maruyama step
            drift = model.drift(s, u).squeeze(0)

            # Sample noise
            if isinstance(model, SwingModel):
                # Noise only in omega components
                if noise is None:
                    dW_omega = torch.randn(model.N, device=device, dtype=dtype) * sqrt_dt
                else:
                    dW_omega = noise[step] * sqrt_dt
                diffusion_noise = model._sigma * dW_omega
                diffusion_full = torch.zeros(model.state_dim, device=device, dtype=dtype)
                diffusion_full[model.N:] = diffusion_noise
            else:
                if noise is None:
                    dW = torch.randn(model.N, device=device, dtype=dtype) * sqrt_dt
                else:
                    dW = noise[step] * sqrt_dt
                diffusion_full = model._sigma * dW

            state = state + drift * dt + diffusion_full

            # Record
            state_history.append(state.cpu().numpy().copy())
            theta = state[:model.N] if isinstance(model, SwingModel) else state
            R_history.append(model.order_parameter(theta.unsqueeze(0)).item())

    return {
        "state": np.array(state_history),
        "R": np.array(R_history),
        "u": np.array(u_history),
        "t": np.arange(n_steps + 1) * dt,
    }


def plot_synchrogram(
    traj_controlled: dict,
    traj_uncontrolled: Optional[dict],
    N: int,
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """Plot a synchrogram (oscillator index vs time) colored by phase."""
    if not HAS_MATPLOTLIB:
        return

    def _theta_mod(traj: dict) -> tuple[np.ndarray, np.ndarray]:
        theta = traj["state"][:, :N]
        theta_mod = np.mod(theta, 2 * np.pi).T  # (N, T+1)
        t = traj["t"]
        return theta_mod, t

    theta_c, t_c = _theta_mod(traj_controlled)

    if traj_uncontrolled is not None:
        theta_u, t_u = _theta_mod(traj_uncontrolled)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
        panels = [
            (axes[0], theta_u, t_u, "No control"),
            (axes[1], theta_c, t_c, "With control"),
        ]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        axes = [ax]
        panels = [(ax, theta_c, t_c, "With control")]

    for ax, theta_mod, t, title in panels:
        im = ax.imshow(
            theta_mod,
            aspect="auto",
            origin="lower",
            cmap="hsv",
            vmin=0.0,
            vmax=2 * np.pi,
            extent=(t[0], t[-1], 0, N - 1),
        )
        ax.set_ylabel("Oscillator index")
        ax.set_title(f"Synchrogram ({title})")
        ax.grid(False)

    axes[-1].set_xlabel("Time")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Phase (mod 2π)")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved synchrogram to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_order_parameter(
    traj_controlled: dict,
    traj_uncontrolled: Optional[dict] = None,
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """Plot order parameter R(t) over time."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(traj_controlled["t"], traj_controlled["R"], "b-", linewidth=1.5, label="With control")

    if traj_uncontrolled is not None:
        ax.plot(traj_uncontrolled["t"], traj_uncontrolled["R"], "r--", linewidth=1.5, label="No control")

    ax.axhline(y=1.0, color="g", linestyle=":", alpha=0.5, label="Full sync (R=1)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Order parameter R")
    ax.set_title("Kuramoto Order Parameter Under Learned Control")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved order parameter plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_phase_evolution(
    traj: dict,
    N: int,
    N_show: int = 20,
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """Plot phase evolution for a subset of oscillators."""
    if not HAS_MATPLOTLIB:
        return

    state = traj["state"]
    t = traj["t"]
    theta = state[:, :N]  # Extract theta (first N components)

    # Show subset of oscillators
    indices = np.linspace(0, N - 1, min(N_show, N)).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in indices:
        theta_i = np.mod(theta[:, i], 2 * np.pi)
        ax.plot(t, theta_i, alpha=0.6, linewidth=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Phase (mod 2pi)")
    ax.set_title(f"Phase Evolution ({len(indices)} oscillators shown)")
    ax.set_ylim(0, 2 * np.pi)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved phase evolution plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_control_energy(
    traj: dict,
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """Plot control energy over time."""
    if not HAS_MATPLOTLIB:
        return

    u = traj["u"]
    t = traj["t"][:-1]  # Control has one less sample

    energy = (u ** 2).sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, energy, "b-", linewidth=1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Control energy ||u||^2")
    ax.set_title("Control Energy Over Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved control energy plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_loss_history(
    history: TrainingHistory,
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """Plot training loss history."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.semilogy(history.value_loss, label="Value loss", alpha=0.8)
    if history.policy_loss[0] is not None:
        policy_losses = [p for p in history.policy_loss if p is not None]
        ax.semilogy(policy_losses, label="Policy loss", alpha=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved loss history plot to {save_path}")
    else:
        plt.show()

    plt.close()


def run_experiment(
    N: int,
    n_iterations: int,
    K: float,
    sigma: float,
    control_fraction: float,
    graph_type: str,
    model_type: str,
    save_plots: bool,
    plot_dir: Optional[pathlib.Path],
    verbose: bool = True,
) -> dict:
    """Run full Kuramoto experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    if verbose:
        print("\n" + "=" * 60)
        print(f"Kuramoto Experiment: N={N}, type={model_type}")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Coupling K={K}, Noise sigma={sigma}")
        print(f"Control fraction: {control_fraction:.1%}")
        print(f"Graph: {graph_type}")
        print(f"Iterations: {n_iterations}")

    # Create model and networks
    model = create_model(
        N, K, sigma, control_fraction, graph_type, model_type,
        device=device, dtype=dtype
    )

    # Scale hidden dims with N
    if N <= 50:
        hidden_dims = (128, 64)
    elif N <= 200:
        hidden_dims = (256, 128)
    else:
        hidden_dims = (512, 256, 128)

    value_net, policy_net = create_networks(model, hidden_dims)
    value_net = value_net.to(device=device, dtype=dtype)
    policy_net = policy_net.to(device=device, dtype=dtype)

    if verbose:
        n_value_params = sum(p.numel() for p in value_net.parameters())
        n_policy_params = sum(p.numel() for p in policy_net.parameters())
        print(f"Controlled nodes: {model.M}/{model.N}")
        print(f"Value net params: {n_value_params:,}")
        print(f"Policy net params: {n_policy_params:,}")

    # Configure training
    if N <= 50:
        value_lr, policy_lr = 1e-3, 5e-4
        batch_size = 2048
        chunk_size = None
    elif N <= 200:
        value_lr, policy_lr = 5e-4, 2e-4
        batch_size = 1024
        chunk_size = 64
    else:
        value_lr, policy_lr = 2e-4, 1e-4
        batch_size = 512
        chunk_size = 32

    config = DPIConfig(
        value_lr=value_lr,
        policy_lr=policy_lr,
        batch_size=batch_size,
        eval_method="implicit",
        delta_t=1.0,
        n_eval_steps=5,
        n_improve_steps=1,
        ito_method="finite_diff",
        fd_step_size=1e-2,
        fd_stencil="five",
        lr_decay=0.995,
        lr_decay_steps=500,
        vectorized=True,
        chunk_size=chunk_size,
    )

    trainer = DPITrainer(model, value_net, policy_net, config)

    if verbose:
        print("\nTraining...")

    start_time = time.time()
    history = trainer.train(n_iterations)
    elapsed = time.time() - start_time

    if verbose:
        print(f"Training complete in {elapsed:.2f}s")
        print(f"Final value loss: {history.value_loss[-1]:.6f}")
        if history.policy_loss[-1] is not None:
            print(f"Final policy loss: {history.policy_loss[-1]:.6f}")

    # Evaluate
    hjb_stats = compute_hjb_stats(model, value_net, policy_net, device=device, dtype=dtype)

    if verbose:
        print("\nHJB Residual Stats:")
        print(f"  Mean: {hjb_stats['mean']:.6f}")
        print(f"  Std:  {hjb_stats['std']:.6f}")
        print(f"  Max:  {hjb_stats['max_abs']:.6f}")

    # Simulate trajectories
    state_init = model.sample_states(1).squeeze(0)
    sim_steps = int(20.0 / 0.01)
    noise = torch.randn((sim_steps, model.N), device=device, dtype=dtype)

    if verbose:
        print("\nSimulating trajectories...")

    traj_controlled = simulate_trajectory(
        model, policy_net, state_init, T=20.0, dt=0.01,
        use_control=True, noise=noise, device=device, dtype=dtype
    )

    traj_uncontrolled = simulate_trajectory(
        model, policy_net, state_init.clone(), T=20.0, dt=0.01,
        use_control=False, noise=noise, device=device, dtype=dtype
    )

    if verbose:
        print(f"Controlled:   R(0)={traj_controlled['R'][0]:.3f} -> R(T)={traj_controlled['R'][-1]:.3f}")
        print(f"Uncontrolled: R(0)={traj_uncontrolled['R'][0]:.3f} -> R(T)={traj_uncontrolled['R'][-1]:.3f}")

    # Plotting
    if save_plots and HAS_MATPLOTLIB:
        if plot_dir is None:
            plot_dir = ROOT / "plots"

        suffix = f"N{N}_{model_type}"

        plot_order_parameter(
            traj_controlled,
            traj_uncontrolled,
            save_path=plot_dir / f"kuramoto_R_{suffix}.png"
        )

        plot_phase_evolution(
            traj_controlled,
            model.N,
            save_path=plot_dir / f"kuramoto_phases_{suffix}.png"
        )

        plot_control_energy(
            traj_controlled,
            save_path=plot_dir / f"kuramoto_control_{suffix}.png"
        )

        plot_loss_history(
            history,
            save_path=plot_dir / f"kuramoto_loss_{suffix}.png"
        )

        plot_synchrogram(
            traj_controlled,
            traj_uncontrolled,
            model.N,
            save_path=plot_dir / f"kuramoto_synchrogram_{suffix}.png",
        )

    return {
        "N": N,
        "K": K,
        "sigma": sigma,
        "control_fraction": control_fraction,
        "graph_type": graph_type,
        "model_type": model_type,
        "iterations": n_iterations,
        "elapsed": elapsed,
        "final_value_loss": history.value_loss[-1],
        "final_policy_loss": history.policy_loss[-1],
        "hjb_stats": hjb_stats,
        "R_initial": traj_controlled["R"][0],
        "R_final_controlled": traj_controlled["R"][-1],
        "R_final_uncontrolled": traj_uncontrolled["R"][-1],
        "history": history,
        "traj_controlled": traj_controlled,
        "traj_uncontrolled": traj_uncontrolled,
        "noise": noise.detach().cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kuramoto oscillator experiment")
    parser.add_argument("--N", type=int, default=20, help="Number of oscillators")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--K", type=float, default=1.0, help="Coupling strength")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise intensity")
    parser.add_argument("--control-fraction", type=float, default=0.2,
                        help="Fraction of nodes with control")
    parser.add_argument("--graph", choices=["ring", "watts_strogatz"], default="ring")
    parser.add_argument("--model", choices=["kuramoto", "swing"], default="kuramoto")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(42)

    run_experiment(
        N=args.N,
        n_iterations=args.iterations,
        K=args.K,
        sigma=args.sigma,
        control_fraction=args.control_fraction,
        graph_type=args.graph,
        model_type=args.model,
        save_plots=not args.no_plots,
        plot_dir=ROOT / "plots",
    )


if __name__ == "__main__":
    main()
