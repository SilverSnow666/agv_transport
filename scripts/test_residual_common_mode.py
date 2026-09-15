"""No Isaac Sim needed: verify projection bounds, zero sum and isolation."""
import unittest
import torch

from scripts.residual_common_mode import project_zero_mean


class CommonModeTests(unittest.TestCase):
    def test_centering_preserves_differences_when_feasible(self):
        original = torch.tensor([[.2, .4, .6], [0., 0., 0.]], dtype=torch.float64)
        saved = original.clone()
        result, scale = project_zero_mean(original)
        torch.testing.assert_close(result.mean(1), torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(result[:, 1:] - result[:, :1], original[:, 1:] - original[:, :1])
        torch.testing.assert_close(scale, torch.ones_like(scale))
        torch.testing.assert_close(original, saved)

    def test_extreme_projection_is_scaled_not_clipped(self):
        result, scale = project_zero_mean(torch.tensor([[1., 1., -1.]]))
        torch.testing.assert_close(result, torch.tensor([[.5, .5, -1.]]))
        torch.testing.assert_close(scale, torch.tensor([[.75]]))

    def test_random_batch_invariants(self):
        rng = torch.Generator().manual_seed(137)
        a = 2 * torch.rand((10000, 3), generator=rng) - 1
        z, s = project_zero_mean(a)
        self.assertLess(float(z.mean(1).abs().max()), 1e-7)
        self.assertLessEqual(float(z.abs().max()), 1)
        self.assertGreaterEqual(float(s.min()), .75 - 1e-6)
        torch.testing.assert_close(project_zero_mean(z)[0], z)

    def test_invalid_actions_rejected(self):
        for a in (torch.zeros(3), torch.ones(1, 4), torch.full((1, 3), float('nan')), torch.full((1, 3), 2.)):
            with self.assertRaises(ValueError):
                project_zero_mean(a)


if __name__ == "__main__":
    unittest.main()
