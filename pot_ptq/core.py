"""
Core mathematical functions and bitwise operations for PoT Quantization.
Based on Section 3.1, 3.2, 4.1, 4.2 of "POT-PTQ: A Two-step Power-of-Two Post-training for LLMs".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def qmax_for_bits(n_bits: int) -> int:
    """
    Computes qmax for n-bit PoT quantization:
    qmax = 2^(n-1) - 1  (Eq. 2 in paper)
    - 2-bit: qmax = 2^1 - 1 = 1 (exponents in {0, 1}, 2 levels * 2 signs = 4 states)
    - 3-bit: qmax = 2^2 - 1 = 3 (exponents in {0, 1, 2, 3}, 4 levels * 2 signs = 8 states)
    - 4-bit: qmax = 2^3 - 1 = 7 (exponents in {0..7}, 8 levels * 2 signs = 16 states)
    """
    assert n_bits >= 2, f"PoT quantization requires at least 2 bits, got {n_bits}"
    return (1 << (n_bits - 1)) - 1


def ste_round_clamp(x: torch.Tensor, qmax: int) -> torch.Tensor:
    """
    Straight-Through Estimator (STE) for discrete exponent rounding and clamping (Eq. 16, 18).
    Forward: returns clamp(round(x), 0, qmax)
    Backward: passes gradients directly through x (dE/dx approx 1).
    """
    rounded = torch.clamp(torch.round(x), 0, qmax)
    return x + (rounded - x).detach()


def to_groups(W: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, int]:
    """
    Partitions a 2D weight matrix (out_features, in_features) along the in_features dimension
    into contiguous groups of size `group_size` (Eq. 5).
    Returns (W_grouped, pad_amount) where W_grouped is (out_features, n_groups, group_size).
    """
    out_f, in_f = W.shape
    pad = (-in_f) % group_size
    if pad > 0:
        W = F.pad(W, (0, pad))
    n_groups = W.shape[1] // group_size
    Wg = W.view(out_f, n_groups, group_size)
    return Wg, pad


def from_groups(Wg: torch.Tensor, pad: int, in_features: int) -> torch.Tensor:
    """
    Reconstructs the 2D weight matrix (out_features, in_features) from grouped tensor (out_features, n_groups, group_size).
    """
    out_f = Wg.shape[0]
    W = Wg.reshape(out_f, -1)
    if pad > 0:
        W = W[:, :in_features]
    return W


def pot_quantize_dequantize(
    W: torch.Tensor,
    S: torch.Tensor,
    n_bits: int,
    group_size: int = 128,
    use_ste: bool = False,
) -> torch.Tensor:
    """
    Quantizes and dequantizes weight tensor W using scale S and n_bits PoT quantization.
    Formula (Eq. 1, 4, 16, 17):
      W_hat = S * sign(W) * 2^E
      where E = clamp(round(log2(|W| / S)), 0, qmax)
    """
    qmax = qmax_for_bits(n_bits)
    Wg, pad = to_groups(W, group_size)
    
    P = torch.sign(Wg)
    P[P == 0] = 1.0
    
    absW = Wg.abs().clamp(min=1e-12)
    S_safe = S.abs().clamp(min=1e-12)
    
    log_ratio = torch.log2(absW / S_safe)
    if use_ste:
        E = ste_round_clamp(log_ratio, qmax)
    else:
        E = torch.clamp(torch.round(log_ratio), 0, qmax)
        
    Wg_hat = S * P * torch.pow(2.0, E)
    return from_groups(Wg_hat, pad, W.shape[1])


# ---------------- Section 4: Hardware-Friendly Bitwise Dequantization ----------------

def dequant_naive_fp(scale: torch.Tensor, sign: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """
    Conventional floating-point dequantization: w = sign * scale * 2^E (Eq. 20-21).
    Requires floating point power and multiplication.
    """
    return sign * scale.to(torch.float16) * torch.pow(2.0, exponent.to(torch.float16))


def dequant_bitwise(scale: torch.Tensor, sign: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """
    Fast GPU-friendly PoT dequantization via bit manipulation and integer addition (Section 4.2, Figure 4).
    
    In IEEE 754 FP16:
    [15]    Sign bit S
    [14:10] Exponent bits (5 bits, bias 15)
    [9:0]   Mantissa bits (10 bits)
    
    Multiplying by 2^E corresponds to adding E to the exponent field:
    Magnitude integer bits + (E << 10).
    Sign bit is determined by XOR of scale sign and PoT sign.
    """
    assert scale.dtype == torch.float16, "Scale must be FP16 for bitwise dequantization"
    scale = scale.contiguous()
    scale_bits = scale.view(torch.int16).to(torch.int32) & 0xFFFF
    
    MANTISSA_BITS = 10
    exp_shift = (exponent.to(torch.int32) & 0x1F) << MANTISSA_BITS
    
    orig_sign_bit = (scale_bits >> 15) & 0x1
    magnitude_bits = scale_bits & 0x7FFF
    new_magnitude = (magnitude_bits + exp_shift) & 0x7FFF  # Integer addition (Sec 4.2 b)
    
    pot_sign_bit = torch.where(sign < 0, torch.ones_like(orig_sign_bit), torch.zeros_like(orig_sign_bit))
    final_sign_bit = orig_sign_bit ^ pot_sign_bit           # Bit manipulation (Sec 4.2 a)
    
    result_bits = (new_magnitude | (final_sign_bit << 15)).to(torch.int16)
    return result_bits.view(torch.float16)
