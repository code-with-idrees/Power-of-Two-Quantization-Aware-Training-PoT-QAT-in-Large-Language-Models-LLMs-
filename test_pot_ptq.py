"""
Comprehensive Unit & Integration Test Suite for POT-PTQ Implementation.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pot_ptq.core import (
    qmax_for_bits,
    ste_round_clamp,
    to_groups,
    from_groups,
    pot_quantize_dequantize,
    dequant_naive_fp,
    dequant_bitwise,
)
from pot_ptq.step1 import data_agnostic_scale_init, naive_scale_init
from pot_ptq.layers import PoTLinear


class TestPoTMathAndCore(unittest.TestCase):
    def test_qmax(self):
        self.assertEqual(qmax_for_bits(2), 1)  # 2-bit -> qmax = 2^1 - 1 = 1
        self.assertEqual(qmax_for_bits(3), 3)  # 3-bit -> qmax = 2^2 - 1 = 3
        self.assertEqual(qmax_for_bits(4), 7)  # 4-bit -> qmax = 2^3 - 1 = 7

    def test_grouping_roundtrip(self):
        torch.manual_seed(42)
        W = torch.randn(64, 256)
        group_size = 128
        Wg, pad = to_groups(W, group_size)
        self.assertEqual(Wg.shape, (64, 2, 128))
        self.assertEqual(pad, 0)
        W_recon = from_groups(Wg, pad, 256)
        self.assertTrue(torch.allclose(W, W_recon))

    def test_grouping_with_padding(self):
        torch.manual_seed(42)
        W = torch.randn(32, 200)
        group_size = 128
        Wg, pad = to_groups(W, group_size)
        self.assertEqual(pad, 56)
        self.assertEqual(Wg.shape, (32, 2, 128))
        W_recon = from_groups(Wg, pad, 200)
        self.assertEqual(W_recon.shape, (32, 200))
        self.assertTrue(torch.allclose(W, W_recon))

    def test_ste_gradient(self):
        # STE should pass gradients unchanged through rounding
        x = torch.tensor([0.2, 0.7, 1.4, 2.8], requires_grad=True)
        qmax = 3
        y = ste_round_clamp(x, qmax)
        self.assertTrue(torch.allclose(y.detach(), torch.tensor([0.0, 1.0, 1.0, 3.0])))
        loss = y.sum()
        loss.backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

    def test_bitwise_dequantization_match(self):
        # Section 4: bitwise dequantization must match floating-point formula
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(123)
        n = 5000
        scale = (torch.rand(n, device=device) * 0.05 + 1e-3).to(torch.float16)
        sign = torch.where(torch.rand(n, device=device) > 0.5, 1.0, -1.0).to(torch.float16)
        exponent = torch.randint(0, 4, (n,), device=device)

        w_naive = dequant_naive_fp(scale, sign, exponent)
        w_bitwise = dequant_bitwise(scale, sign, exponent)

        max_err = (w_naive.float() - w_bitwise.float()).abs().max().item()
        self.assertLess(max_err, 1e-4)


class TestStep1ScaleInit(unittest.TestCase):
    def test_step1_grid_search(self):
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        W = torch.randn(128, 256, device=device) * 0.02
        
        # Step 1 with candidate search
        S_opt, P_opt, _ = data_agnostic_scale_init(W, n_bits=3, group_size=128, n_candidates=200, device=device)
        # Naive baseline (b=1)
        S_naive, P_naive, _ = naive_scale_init(W, n_bits=3, group_size=128, device=device)

        # Reconstructed weight MSE comparison
        W_recon_opt = pot_quantize_dequantize(W, S_opt, n_bits=3, group_size=128)
        W_recon_naive = pot_quantize_dequantize(W, S_naive, n_bits=3, group_size=128)

        mse_opt = ((W - W_recon_opt) ** 2).mean().item()
        mse_naive = ((W - W_recon_naive) ** 2).mean().item()

        # Step 1 grid search must achieve lower or equal MSE compared to naive b=1
        self.assertLessEqual(mse_opt, mse_naive)


class TestPoTLinearModule(unittest.TestCase):
    def test_pot_linear_forward_and_freeze(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lin = nn.Linear(256, 128).to(device)
        pot = PoTLinear(lin, n_bits=3, group_size=128, compute_dtype=torch.float16).to(device)

        x = torch.randn(4, 16, 256, device=device, dtype=torch.float16)
        out = pot(x)
        self.assertEqual(out.shape, (4, 16, 128))
        self.assertEqual(out.dtype, torch.float16)

        # Test gradient flows to Gamma
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(pot.Gamma.grad)
        self.assertFalse(torch.isnan(pot.Gamma.grad).any())

        # Test freeze
        pot.freeze()
        self.assertFalse(pot.Gamma.requires_grad)
        self.assertTrue(torch.all(pot.Gamma == 0))


if __name__ == "__main__":
    unittest.main()
