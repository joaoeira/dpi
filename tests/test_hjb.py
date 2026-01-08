import unittest
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpi.core.hjb import compute_hjb_residual
from dpi.models.base import BaseModel


class LinearModel(BaseModel):
    def __init__(self, rho=0.1):
        self._rho = float(rho)

    @property
    def discount_rate(self):
        return self._rho

    def drift(self, s, c=None):
        return torch.full_like(s, 0.3)

    def diffusion(self, s, c=None):
        return [torch.full_like(s, 0.5), torch.full_like(s, -0.25)]

    def utility(self, s, c=None):
        return torch.zeros((s.shape[0], 1), dtype=s.dtype, device=s.device)

    def sample_states(self, n):
        return torch.randn(n, 3)


class TestHJB(unittest.TestCase):
    def test_hjb_residual_matches_quadratic(self):
        torch.manual_seed(1)
        s = torch.randn(5, 3, dtype=torch.float64)
        model = LinearModel(rho=0.2)

        def V(x):
            return (x ** 2).sum(dim=1, keepdim=True)

        drift = model.drift(s)
        g1, g2 = model.diffusion(s)

        expected = (2.0 * s * drift).sum(dim=1, keepdim=True)
        expected = expected + (g1 ** 2).sum(dim=1, keepdim=True)
        expected = expected + (g2 ** 2).sum(dim=1, keepdim=True)
        expected = expected - model.discount_rate * V(s)

        hjb = compute_hjb_residual(
            model,
            s,
            c=None,
            V=V,
            ito_method="finite_diff",
            h=1e-4,
            stencil="five",
        )

        self.assertTrue(torch.allclose(hjb, expected, rtol=1e-4, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
