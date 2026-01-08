"""Minimal smoke test for DPI training."""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpi.algorithms.dpi import DPITrainer
from dpi.models.two_trees import TwoTrees
from dpi.networks.value_network import ValueNetwork
from dpi.utils.training import DPIConfig


def main() -> None:
    torch.manual_seed(0)

    model = TwoTrees(rho=0.04, sigma=0.2)
    value_net = ValueNetwork(input_dim=1, hidden_dims=(32, 32), activation="silu")

    config = DPIConfig(
        value_lr=1e-3,
        batch_size=128,
        eval_method="explicit",
        delta_t=1.0,
        n_eval_steps=2,
        n_improve_steps=0,
        ito_method="finite_diff",
        fd_step_size=1e-4,
        fd_stencil="five",
    )

    trainer = DPITrainer(model, value_net, policy_network=None, config=config)
    history = trainer.train(n_iterations=5)

    print("Smoke training complete. Last value loss:", history.value_loss[-1])


if __name__ == "__main__":
    main()
