"""
GenVision Phase 5.2A: ResDiff-TransUNet Comprehensive Unit Test Suite
====================================================================

Covers the 20 mandatory test criteria specified in Phase 5.2A:
1.  Schedule dimensions and values.
2.  Cumulative alpha monotonicity.
3.  Forward noising shapes.
4.  Forward noising numerical stability.
5.  Residual calculation.
6.  Residual scaling and inverse scaling.
7.  Transformer frozen-weight verification.
8.  CSEB embedding shape.
9.  ResDiff-UNet initialization.
10. Batch size 1 forward pass.
11. Batch size 2 forward pass.
12. Correct output dimensions.
13. Backpropagation through denoiser.
14. No transformer gradients.
15. DDIM timestep ordering.
16. DDIM eta-zero reproducibility with identical initial noise.
17. DDIM final output range.
18. Lambda-zero baseline identity.
19. Lambda boundary validation.
20. No NaN or Inf across the complete pipeline.
"""

from pathlib import Path
import sys
import unittest
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.diffusion.scheduler import GaussianDiffusionScheduler
from models.diffusion.residual import ResidualProcessor, FrozenTransformerConditioner
from models.diffusion.resdiff_unet import ResDiffUNet
from models.diffusion.ddim_sampler import DDIMSampler
from models.diffusion.residual_gating import ResidualGating


class TestResDiffPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scheduler = GaussianDiffusionScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        cls.processor = ResidualProcessor(s_r=4.0)
        cls.denoiser = ResDiffUNet()
        cls.ckpt_path = PROJECT_ROOT / "models" / "transformer_controlled_best.pth"
        cls.conditioner = FrozenTransformerConditioner(checkpoint_path=cls.ckpt_path)
        cls.sampler = DDIMSampler(cls.scheduler, cls.denoiser)
        cls.gating = ResidualGating(lambda_val=1.0)

    # 1. Schedule dimensions and values
    def test_01_schedule_dimensions_and_values(self):
        self.assertEqual(len(self.scheduler.betas), 1000)
        self.assertEqual(len(self.scheduler.alphas), 1000)
        self.assertEqual(len(self.scheduler.alphas_cumprod), 1000)
        self.assertAlmostEqual(self.scheduler.betas[0].item(), 0.0001, places=6)
        self.assertAlmostEqual(self.scheduler.betas[-1].item(), 0.02, places=6)

    # 2. Cumulative alpha monotonicity
    def test_02_cumulative_alpha_monotonicity(self):
        alphas_bar = self.scheduler.alphas_cumprod.cpu().numpy()
        # Strictly decreasing: alphas_bar[t] < alphas_bar[t-1]
        diffs = np.diff(alphas_bar)
        self.assertTrue(np.all(diffs < 0), "alphas_cumprod is not strictly monotonically decreasing")
        self.assertGreater(alphas_bar[0], 0.99)
        self.assertLess(alphas_bar[-1], 0.01)

    # 3. Forward noising shapes
    def test_03_forward_noising_shapes(self):
        x_0 = torch.randn(2, 3, 256, 256)
        t = torch.tensor([50, 500], dtype=torch.long)
        x_t, noise = self.scheduler.q_sample(x_0, t)
        self.assertEqual(x_t.shape, x_0.shape)
        self.assertEqual(noise.shape, x_0.shape)

    # 4. Forward noising numerical stability
    def test_04_forward_noising_numerical_stability(self):
        x_0 = torch.randn(4, 3, 64, 64)
        t = torch.tensor([0, 250, 750, 999], dtype=torch.long)
        x_t, noise = self.scheduler.q_sample(x_0, t)
        self.assertFalse(torch.isnan(x_t).any())
        self.assertFalse(torch.isinf(x_t).any())
        # At t=0, x_t should be almost equal to x_0 (alpha_bar_0 = 1 - 1e-4 = 0.9999)
        t_zero = torch.zeros(1, dtype=torch.long)
        x_zero_t, _ = self.scheduler.q_sample(x_0[:1], t_zero, noise=torch.zeros_like(x_0[:1]))
        self.assertTrue(torch.allclose(x_0[:1], x_zero_t, atol=1e-3))

    # 5. Residual calculation
    def test_05_residual_calculation(self):
        clean = torch.full((1, 3, 32, 32), 0.8)
        pred = torch.full((1, 3, 32, 32), 0.5)
        res = ResidualProcessor.compute_residual(clean, pred)
        self.assertTrue(torch.allclose(res, torch.full((1, 3, 32, 32), 0.3)))

    # 6. Residual scaling and inverse scaling
    def test_06_residual_scaling_and_inverse_scaling(self):
        res = torch.tensor([[-0.25, 0.0, 0.125, 0.3]])
        scaled = self.processor.scale_residual(res)
        expected_scaled = torch.tensor([[-1.0, 0.0, 0.5, 1.2]])
        self.assertTrue(torch.allclose(scaled, expected_scaled))
        unscaled = self.processor.unscale_residual(scaled)
        self.assertTrue(torch.allclose(unscaled, res))

    # 7. Transformer frozen-weight verification
    def test_07_transformer_frozen_weights(self):
        for name, param in self.conditioner.model.named_parameters():
            self.assertFalse(param.requires_grad, f"Parameter {name} is not frozen!")

    # 8. CSEB embedding shape
    def test_08_cseb_embedding_shape(self):
        dummy_cloudy = torch.rand(2, 3, 256, 256)
        x_hat, e_cseb = self.conditioner(dummy_cloudy)
        self.assertEqual(x_hat.shape, (2, 3, 256, 256))
        self.assertEqual(e_cseb.shape, (2, 192))
        self.assertFalse(x_hat.requires_grad)
        self.assertFalse(e_cseb.requires_grad)

    # 9. ResDiff-UNet initialization
    def test_09_resdiff_unet_initialization(self):
        model = ResDiffUNet()
        self.assertIsInstance(model, torch.nn.Module)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 1_000_000)

    # 10. Batch size 1 forward pass
    def test_10_batch_size_1_forward_pass(self):
        x_in = torch.randn(1, 9, 256, 256)
        t = torch.tensor([100], dtype=torch.long)
        e_cseb = torch.randn(1, 192)
        out = self.denoiser(x_in, t, e_cseb)
        self.assertEqual(out.shape, (1, 3, 256, 256))
        self.assertFalse(torch.isnan(out).any())

    # 11. Batch size 2 forward pass
    def test_11_batch_size_2_forward_pass(self):
        x_in = torch.randn(2, 9, 256, 256)
        t = torch.tensor([100, 500], dtype=torch.long)
        e_cseb = torch.randn(2, 192)
        out = self.denoiser(x_in, t, e_cseb)
        self.assertEqual(out.shape, (2, 3, 256, 256))
        self.assertFalse(torch.isnan(out).any())

    # 12. Correct output dimensions
    def test_12_correct_output_dimensions(self):
        b, h, w = 1, 256, 256
        x_in = torch.randn(b, 9, h, w)
        t = torch.tensor([250], dtype=torch.long)
        out = self.denoiser(x_in, t, None)
        self.assertEqual(out.shape, (b, 3, h, w))

    # 13. Backpropagation through denoiser
    def test_13_backprop_through_denoiser(self):
        model = ResDiffUNet()
        x_in = torch.randn(2, 9, 64, 64)
        t = torch.tensor([50, 150], dtype=torch.long)
        e_cseb = torch.randn(2, 192)
        target_noise = torch.randn(2, 3, 64, 64)

        pred_noise = model(x_in, t, e_cseb)
        loss = torch.mean(torch.abs(target_noise - pred_noise))
        loss.backward()

        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                self.assertFalse(torch.isnan(param.grad).any())
        self.assertTrue(has_grad, "No gradients were computed for denoiser parameters")

    # 14. No transformer gradients
    def test_14_no_transformer_gradients(self):
        dummy_cloudy = torch.rand(1, 3, 64, 64)
        x_hat, e_cseb = self.conditioner(dummy_cloudy)
        loss = x_hat.sum() + e_cseb.sum()
        with self.assertRaises(RuntimeError):
            loss.backward()

    # 15. DDIM timestep ordering
    def test_15_ddim_timestep_ordering(self):
        timesteps = DDIMSampler.compute_timesteps(num_steps=10, total_timesteps=1000)
        self.assertEqual(len(timesteps), 10)
        self.assertEqual(timesteps[0], 999)
        self.assertEqual(timesteps[-1], 0)
        for i in range(len(timesteps) - 1):
            self.assertGreater(timesteps[i], timesteps[i + 1])

    # 16. DDIM eta-zero reproducibility
    def test_16_ddim_eta_zero_reproducibility(self):
        cloudy = torch.rand(1, 3, 64, 64)
        x_hat = torch.rand(1, 3, 64, 64)
        e_cseb = torch.rand(1, 192)
        fixed_noise = torch.randn(1, 3, 64, 64)

        # Run 1
        res1 = self.sampler.sample(
            cloudy, x_hat, e_cseb, num_steps=3, eta=0.0, initial_noise=fixed_noise.clone()
        )
        # Run 2
        res2 = self.sampler.sample(
            cloudy, x_hat, e_cseb, num_steps=3, eta=0.0, initial_noise=fixed_noise.clone()
        )
        self.assertTrue(torch.allclose(res1, res2, atol=1e-7))

    # 17. DDIM final output range
    def test_17_ddim_final_output_range(self):
        cloudy = torch.rand(1, 3, 64, 64)
        x_hat = torch.rand(1, 3, 64, 64)
        res = self.sampler.sample(cloudy, x_hat, None, num_steps=3, eta=0.0)
        self.assertFalse(torch.isnan(res).any())
        self.assertFalse(torch.isinf(res).any())
        # Descaled residual should lie in a reasonable bound
        self.assertLess(torch.max(torch.abs(res)).item(), 5.0)

    # 18. Lambda-zero baseline identity
    def test_18_lambda_zero_baseline_identity(self):
        x_hat = torch.rand(2, 3, 64, 64)
        pred_res = torch.randn(2, 3, 64, 64)
        out = self.gating.integrate(x_hat, pred_res, lambda_val=0.0)
        self.assertTrue(torch.allclose(out, x_hat.clamp(0.0, 1.0)))

    # 19. Lambda boundary validation
    def test_19_lambda_boundary_validation(self):
        self.gating.set_lambda(0.0)
        self.gating.set_lambda(1.0)
        self.gating.set_lambda(0.5)
        with self.assertRaises(ValueError):
            self.gating.set_lambda(-0.1)
        with self.assertRaises(ValueError):
            self.gating.set_lambda(1.1)

    # 20. Complete pipeline NaN/Inf check
    def test_20_complete_pipeline_nan_inf(self):
        cloudy = torch.rand(1, 3, 64, 64)
        gt_clean = torch.rand(1, 3, 64, 64)

        # 1. Condition
        x_hat, e_cseb = self.conditioner(cloudy)
        # 2. Residual
        r = ResidualProcessor.compute_residual(gt_clean, x_hat)
        r_tilde = self.processor.scale_residual(r)
        # 3. Noising
        t = torch.tensor([500], dtype=torch.long)
        r_t, noise = self.scheduler.q_sample(r_tilde, t)
        # 4. Denoiser forward
        x_in = ResidualProcessor.prepare_spatial_conditioning(r_t, x_hat, cloudy)
        eps_pred = self.denoiser(x_in, t, e_cseb)
        # 5. Reverse sample
        r_pred = self.sampler.sample(cloudy, x_hat, e_cseb, num_steps=2, eta=0.0)
        # 6. Gated assembly
        x_final = self.gating.integrate(x_hat, r_pred, lambda_val=0.5)

        for tensor_name, tensor in [
            ("x_hat", x_hat),
            ("e_cseb", e_cseb),
            ("r", r),
            ("r_tilde", r_tilde),
            ("r_t", r_t),
            ("eps_pred", eps_pred),
            ("r_pred", r_pred),
            ("x_final", x_final),
        ]:
            self.assertFalse(torch.isnan(tensor).any(), f"NaN detected in {tensor_name}")
            self.assertFalse(torch.isinf(tensor).any(), f"Inf detected in {tensor_name}")


if __name__ == "__main__":
    unittest.main()
