"""
PoTLinear: PyTorch Module replacement for nn.Linear supporting Power-of-Two quantization.
Based on Section 3.1, 3.3, Eq. 15-18 of "POT-PTQ: A Two-step Power-of-Two Post-training for LLMs".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .core import qmax_for_bits, ste_round_clamp, to_groups, from_groups
from .step1 import data_agnostic_scale_init, naive_scale_init


class PoTLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with Power-of-Two (PoT) Post-Training Quantization.
    
    Attributes:
        S (Buffer): Quantization scale per group, initialized by Step 1 (Eq. 9).
        P (Buffer): Quantization sign bit per element {-1, +1} (Eq. 1).
        Gamma (nn.Parameter): Learnable scale residual for Step 2 fine-tuning (Eq. 15).
        W_orig_grouped (Buffer, optional): Original weight matrix for error tracking.
    """

    def __init__(
        self,
        linear: nn.Linear,
        n_bits: int,
        group_size: int = 128,
        compute_dtype: torch.dtype = torch.float16,
        init_method: str = "step1",  # "step1" (Algorithm 1) or "naive" (b=1)
        store_orig: bool = True,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_bits = n_bits
        self.group_size = group_size
        self.qmax = qmax_for_bits(n_bits)
        self.compute_dtype = compute_dtype

        W = linear.weight.data.clone().float()
        
        # Step 1: Initialize scale S
        if init_method == "step1":
            S, P, pad = data_agnostic_scale_init(W, n_bits=n_bits, group_size=group_size)
        elif init_method == "naive":
            S, P, pad = naive_scale_init(W, n_bits=n_bits, group_size=group_size)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        self.pad = pad
        
        # Buffers for fixed quantization state
        self.register_buffer("S", S)  # (out_f, n_groups, 1)
        self.register_buffer("P", P)  # (out_f, n_groups, group_size)
        
        if store_orig:
            Wg_orig, _ = to_groups(W, group_size)
            self.register_buffer("W_orig_grouped", Wg_orig)
        else:
            self.W_orig_grouped = None

        # Step 2 learnable residual Gamma (Eq. 15), initialized to 0
        self.Gamma = nn.Parameter(torch.zeros_like(S))

        # Bias handling
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone().to(compute_dtype))
        else:
            self.register_parameter("bias", None)

    def dequantized_weight(self, use_ste: bool = True) -> torch.Tensor:
        """
        Dequantizes the weight matrix on-the-fly:
          Shat = S * (1 + Gamma)                        (Eq. 15)
          E = clamp(round(log2(|W| / Shat)), 0, qmax)   (Eq. 16 with STE)
          W_hat = Shat * P * 2^E                         (Eq. 17)
        """
        if self.W_orig_grouped is None and hasattr(self, "E"):
            # Fast inference path using frozen integer exponents E
            Wg_hat = self.S * self.P * torch.pow(2.0, self.E.to(self.compute_dtype))
            W_hat = from_groups(Wg_hat, self.pad, self.in_features)
            return W_hat.to(self.compute_dtype)

        # Refined scale with learnable residual (Eq. 15)
        Shat = self.S * (1.0 + self.Gamma)
        Shat_safe = torch.clamp(Shat.abs(), min=1e-12) * torch.sign(Shat + 1e-12)

        # Retrieve magnitude and compute discrete exponent (Eq. 16)
        absW = self.W_orig_grouped.abs()
        log_ratio = torch.log2(torch.clamp(absW / Shat_safe.detach().abs().clamp(min=1e-12), min=1e-12))
        E = torch.clamp(torch.round(log_ratio), 0, self.qmax)

        # Dequantized grouped weight (Eq. 17)
        # Gradient of Shat propagates through scaling the discrete PoT representation (Eq. 18)
        Wg_hat = Shat_safe * self.P * torch.pow(2.0, E.detach())
        W_hat = from_groups(Wg_hat, self.pad, self.in_features)
        return W_hat.to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_hat = self.dequantized_weight(use_ste=(self.training or (self.Gamma is not None and self.Gamma.requires_grad)))
        bias = self.bias if self.bias is not None else None
        return F.linear(x.to(self.compute_dtype), W_hat, bias)

    @torch.no_grad()
    def freeze(self):
        """
        Bakes optimized Gamma into base scale S:
        S <- S * (1 + Gamma), then resets Gamma to 0.
        Saves frozen integer exponents E and frees W_orig_grouped for minimal VRAM usage.
        """
        self.S.mul_(1.0 + self.Gamma)
        self.Gamma.zero_()
        self.Gamma.requires_grad_(False)
        if self.W_orig_grouped is not None:
            absW = self.W_orig_grouped.abs()
            log_ratio = torch.log2(torch.clamp(absW / self.S.abs().clamp(min=1e-12), min=1e-12))
            self.register_buffer("E", torch.clamp(torch.round(log_ratio), 0, self.qmax).to(torch.int8))
            self.W_orig_grouped = None

    @torch.no_grad()
    def weight_only_mse(self) -> float:
        """
        Computes the Mean Squared Error between quantized weights and original unquantized weights.
        """
        if self.W_orig_grouped is None:
            return 0.0
        W_hat = self.dequantized_weight(use_ste=False).float()
        Wg_hat, _ = to_groups(W_hat, self.group_size)
        return ((Wg_hat - self.W_orig_grouped) ** 2).mean().item()
