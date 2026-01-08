import unittest
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpi.core.ito import compute_drift_hyperdual, second_derivative_fd


class TestIto(unittest.TestCase):
    def test_second_derivative_fd_quadratic(self):
        a, b, c = 3.5, -1.2, 0.7

        def F(eps):
            return torch.tensor(a * eps * eps + b * eps + c, dtype=torch.float64)

        for stencil in ("three", "five", "seven", "nine"):
            d2 = second_derivative_fd(F, h=1e-4, stencil=stencil)
            self.assertAlmostEqual(d2.item(), 2.0 * a, places=6)

    def test_compute_drift_hyperdual_quadratic(self):
        torch.manual_seed(0)
        s = torch.randn(4, 3, dtype=torch.float64)
        drift = torch.randn_like(s)
        g1 = torch.randn_like(s)
        g2 = torch.randn_like(s)

        def V(x):
            return (x ** 2).sum(dim=1, keepdim=True)

        expected = (2.0 * s * drift).sum(dim=1, keepdim=True)
        expected = expected + (g1 ** 2).sum(dim=1, keepdim=True)
        expected = expected + (g2 ** 2).sum(dim=1, keepdim=True)

        drift_est = compute_drift_hyperdual(
            V,
            s,
            drift,
            [g1, g2],
            h=1e-4,
            stencil="five",
            method="finite_diff",
        )

        self.assertTrue(torch.allclose(drift_est, expected, rtol=1e-4, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
