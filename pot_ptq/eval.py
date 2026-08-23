"""
Evaluation and Data Utilities for WikiText-2 Perplexity Benchmark.
Based on Section 5.1, 5.2, Table 1 of "POT-PTQ: A Two-step Power-of-Two Post-training for LLMs".
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from datasets import load_dataset


def get_wikitext2_data(tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads and tokenizes WikiText-2 (wikitext-2-raw-v1) train and test splits.
    """
    wikitext_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wikitext_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    test_text = "\n\n".join(wikitext_test["text"])
    train_text = "\n\n".join(wikitext_train["text"])

    test_ids = tokenizer(test_text, return_tensors="pt").input_ids
    train_ids = tokenizer(train_text, return_tensors="pt").input_ids
    return train_ids, test_ids


def get_calibration_batches(
    train_ids: torch.Tensor,
    n_seqs: int = 128,
    seq_len: int = 2048,
    device: torch.device = None,
    seed: int = 0,
) -> List[torch.Tensor]:
    """
    Samples `n_seqs` random contiguous sequences of length `seq_len` from training tokens.
    (Section 5.1: 128 randomly sampled 2048-token sequences from WikiText-2).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(seed)
    available_tokens = train_ids.shape[1]
    if available_tokens < 2:
        raise ValueError("Training data must contain at least two tokens")
    seq_len = min(seq_len, available_tokens - 1)
    max_start = available_tokens - seq_len
    starts = torch.randint(0, max_start + 1, (n_seqs,), generator=g)
    return [train_ids[:, s : s + seq_len].to(device) for s in starts]


@torch.no_grad()
def evaluate_wikitext2_ppl(
    model: nn.Module,
    test_ids: torch.Tensor,
    seqlen: int = 2048,
    device: torch.device = None,
    max_chunks: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[float, int]:
    """
    Evaluates WikiText-2 perplexity using non-overlapping windows of `seqlen` tokens (Table 1 protocol).

    Computes exponentiated average token negative log-likelihood:
      PPL = exp( sum(chunk_loss * seqlen) / total_tokens )
    """
    device = device or next(model.parameters()).device
    model.eval()

    n_chunks = test_ids.shape[1] // seqlen
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)

    nlls = []
    total_tokens = 0

    for i in range(n_chunks):
        batch = test_ids[:, i * seqlen : (i + 1) * seqlen].to(device)
        # Hugging Face standard: passing labels computes cross-entropy loss over shift targets
        out = model(batch, labels=batch)
        neg_log_lik = out.loss.float() * seqlen
        nlls.append(neg_log_lik.item())
        total_tokens += seqlen

        if device.type == "cuda" and i % 20 == 0:
            torch.cuda.empty_cache()

    if total_tokens == 0:
        return float("nan"), 0

    avg_nll = sum(nlls) / total_tokens
    ppl = math.exp(avg_nll)
    return ppl, n_chunks


def generate_sample(
    model: nn.Module,
    tokenizer,
    prompt: str = "The history of artificial intelligence began",
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = None,
) -> str:
    """
    Generates text continuation to inspect quality of quantized model.
    """
    device = device or next(model.parameters()).device
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)
