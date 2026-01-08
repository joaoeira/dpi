import unittest
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpi.algorithms.policy_eval import policy_evaluation_step
from dpi.algorithms.policy_improve import policy_improvement_step
from dpi.models.base import BaseModel
from dpi.networks.value_network import ValueNetwork
from dpi.networks.policy_network import PolicyNetwork


class ZeroDynamicsModel(BaseModel):
    def __init__(self, rho=0.1):
        self._rho = float(rho)

    @property
    def discount_rate(self):
        return self._rho

    def drift(self, s, c=None):
        return torch.zeros_like(s)

    def diffusion(self, s, c=None):
        return [torch.zeros_like(s)]

    def utility(self, s, c=None):
        if c is None:
            return torch.zeros((s.shape[0], 1), dtype=s.dtype, device=s.device)
        return -0.5 * (c ** 2).sum(dim=1, keepdim=True)

    def sample_states(self, n):
        return torch.randn(n, 2)


class TestDPISteps(unittest.TestCase):
    def test_policy_improve_and_eval_update_params(self):
        torch.manual_seed(0)
        model = ZeroDynamicsModel(rho=0.1)
        value_net = ValueNetwork(input_dim=2, hidden_dims=(8,), activation="silu")
        policy_net = PolicyNetwork(input_dim=2, output_dim=1, hidden_dims=(8,), activation="relu")
        value_opt = torch.optim.Adam(value_net.parameters(), lr=1e-2)
        policy_opt = torch.optim.Adam(policy_net.parameters(), lr=1e-2)

        s = model.sample_states(32)

        policy_before = [p.detach().clone() for p in policy_net.parameters()]
        loss_pi = policy_improvement_step(
            model,
            s,
            policy_net,
            value_net,
            policy_opt,
            ito_method="finite_diff",
            fd_step_size=1e-4,
            fd_stencil="three",
        )
        policy_after = list(policy_net.parameters())
        self.assertIsNotNone(loss_pi)
        self.assertTrue(any((a - b).abs().sum().item() > 0 for a, b in zip(policy_before, policy_after)))

        value_before = [p.detach().clone() for p in value_net.parameters()]
        loss_v = policy_evaluation_step(
            model,
            s,
            value_net,
            policy_net,
            value_opt,
            eval_method="explicit",
            delta_t=1.0,
            ito_method="finite_diff",
            fd_step_size=1e-4,
            fd_stencil="three",
        )
        value_after = list(value_net.parameters())
        self.assertIsNotNone(loss_v)
        self.assertTrue(any((a - b).abs().sum().item() > 0 for a, b in zip(value_before, value_after)))


if __name__ == "__main__":
    unittest.main()
