"""
Algorithm 1: Parallel Data-Agnostic Scale Initialization (Step 1).
Based on Section 3.2, Algorithm 1, Eq. 6-10 of "POT-PTQ: A Two-step Power-of-Two Post-training for LLMs".
"""

import torch
from typing import Tuple
from .core import qmax_for_bits, to_groups


@torch.no_grad()
def data_agnostic_scale_init(
    W: torch.Tensor,
    n_bits: int,
    group_size: int = 128,
    n_candidates: int = 200,
    device: torch.device = None,
    chunk_rows: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Parallel Data-Agnostic Scale Initialization (Algorithm 1).
    Performs grid search over candidate multipliers B = {0.01 * i | i = 1..200}
    to minimize per-group reconstruction error Q1(b) = ||W_group - W_group(b)||_2^2.

    Memory-efficient row-chunked implementation ensures low VRAM footprint
    even for large transformer weight matrices (e.g., 2048 x 8192 or 4096 x 11008).

    Args:
        W: 2D weight tensor (out_features, in_features)
        n_bits: Target PoT precision (e.g., 2 or 3)
        group_size: Quantization group size (e.g., 128)
        n_candidates: Number of grid search multipliers (paper default: 200)
        device: Execution device (GPU recommended)
        chunk_rows: Number of output rows to process per vectorized batch

    Returns:
        S: Optimal scale tensor of shape (out_features, n_groups, 1)
        P: Sign matrix tensor of shape (out_features, n_groups, group_size)
        pad: Number of padded columns
    """
    device = device or W.device
    qmax = qmax_for_bits(n_bits)
    
    # Compute in float32 for high numerical fidelity
    Wf = W.to(device=device, dtype=torch.float32)
    Wg, pad = to_groups(Wf, group_size)  # Shape: (out_f, n_groups, group_size)
    out_f, n_groups, g_size = Wg.shape

    # Sign tensor P in {-1, +1} (Eq. 1)
    P = torch.sign(Wg)
    P[P == 0] = 1.0

    absWg = Wg.abs()
    
    # Base scale s0 estimation (Eq. 10):
    # s0 = max|W_group| / (2^(qmax - 1))
    denom = 2 ** max(qmax - 1, 0)
    s0 = absWg.amax(dim=-1, keepdim=True) / float(denom)  # (out_f, n_groups, 1)
    s0 = torch.clamp(s0, min=1e-12)

    # Grid search candidate multipliers B = {0.01 * i | i = 1, ..., 200} (Algorithm 1, Line 3)
    B = torch.arange(1, n_candidates + 1, device=device, dtype=torch.float32) * 0.01  # (n_candidates,)

    S = torch.empty((out_f, n_groups, 1), device=device, dtype=torch.float32)

    # Process in memory-safe chunks over out_features
    for start in range(0, out_f, chunk_rows):
        end = min(start + chunk_rows, out_f)
        
        Wg_chunk = Wg[start:end]         # (chunk_len, n_groups, group_size)
        s0_chunk = s0[start:end]         # (chunk_len, n_groups, 1)
        P_chunk = P[start:end]           # (chunk_len, n_groups, group_size)
        absW_chunk = absWg[start:end]    # (chunk_len, n_groups, group_size)

        # Candidate scales: s_b = s0 * b  (Line 6)
        # Shape: (chunk_len, n_groups, 1, n_candidates)
        sb = s0_chunk.unsqueeze(-1) * B.view(1, 1, 1, -1)
        sb = torch.clamp(sb, min=1e-12)

        # Quantized exponent E(b) (Line 7, Eq. 6)
        absW_expanded = absW_chunk.unsqueeze(-1)  # (chunk_len, n_groups, group_size, 1)
        log_ratio = torch.log2(torch.clamp(absW_expanded / sb, min=1e-12))
        E = torch.clamp(torch.round(log_ratio), 0, qmax)  # (chunk_len, n_groups, group_size, n_candidates)

        # Reconstructed weight group W_group(b) (Line 8, Eq. 7)
        Wg_hat = sb * P_chunk.unsqueeze(-1) * torch.pow(2.0, E)

        # Reconstruction error Q1(b) = ||W_group - W_group(b)||_2^2 (Line 9, Eq. 8)
        Q1 = ((Wg_chunk.unsqueeze(-1) - Wg_hat) ** 2).sum(dim=2)  # (chunk_len, n_groups, n_candidates)

        # Select optimal multiplier b* (Line 10-12, Eq. 9)
        best_idx = torch.argmin(Q1, dim=-1)  # (chunk_len, n_groups)
        b_star = B[best_idx]                 # (chunk_len, n_groups)

        # Optimal scale s* = s0 * b* (Line 15)
        S[start:end] = (s0_chunk.squeeze(-1) * b_star).unsqueeze(-1)

    return S, P, pad


@torch.no_grad()
def naive_scale_init(
    W: torch.Tensor,
    n_bits: int,
    group_size: int = 128,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Naive baseline scale initialization (b = 1.0, s = s0) used for ablation studies (Table 4).
    """
    return data_agnostic_scale_init(
        W, n_bits, group_size=group_size, n_candidates=1, device=device
    )
