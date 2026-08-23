"""
POT-PTQ ~1B LLM Reproduction CLI Runner.
Reproduces "POT-PTQ: A Two-step Power-of-Two Post-training for LLMs"
(Wang et al., arXiv:2507.11959).

Target: meta-llama/Llama-3.2-1B (~1.24B parameters)
Benchmark: WikiText-2 perplexity (Table 1 protocol)

Usage:
    python run_pot_ptq_1b.py
    python run_pot_ptq_1b.py --model_name TinyLlama/TinyLlama_v1.1
    python run_pot_ptq_1b.py --epochs_3bit 10 --epochs_2bit 40 --calib_seqs 128
    python run_pot_ptq_1b.py --run_ablation --run_benchmark
"""

import argparse
import copy
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from pot_ptq import (
        quantize_model,
        evaluate_wikitext2_ppl,
        get_wikitext2_data,
        get_calibration_batches,
        dequant_naive_fp,
        dequant_bitwise,
    )
    from pot_ptq.eval import generate_sample
except ModuleNotFoundError as exc:
    if exc.name == "pot_ptq":
        raise ModuleNotFoundError(
            "The 'pot_ptq' package is missing. Upload the complete 'pot_ptq' "
            "folder next to run_pot_ptq_1b.py in Kaggle."
        ) from exc
    raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="POT-PTQ: Two-Step Power-of-Two Post-Training Quantization for LLMs")

    # Model
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama_v1.1",
                        help="HuggingFace model ID (default: TinyLlama/TinyLlama_v1.1)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated models (or set HF_TOKEN env var)")

    # Quantization hyperparameters (Paper Sec. 5.1 & Table 1)
    parser.add_argument("--group_size", type=int, default=64,
                        help="Quantization group size G (paper uses 64 or 128)")
    parser.add_argument("--calib_seqs", type=int, default=64,
                        help="Number of calibration sequences (paper: 128)")
    parser.add_argument("--calib_seqlen", type=int, default=1024,
                        help="Calibration sequence length (paper: 2048)")
    parser.add_argument("--epochs_3bit", type=int, default=8,
                        help="Fine-tuning epochs for 3-bit (paper: 10)")
    parser.add_argument("--epochs_2bit", type=int, default=20,
                        help="Fine-tuning epochs for 2-bit (paper: 40)")
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="Learning rate for Gamma residual (paper: 1e-3 to 2e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Regularization lambda (paper: 1e-1 to 1e-2)")

    # Evaluation
    parser.add_argument("--eval_seqlen", type=int, default=2048,
                        help="Evaluation sequence length (Table 1 protocol)")
    parser.add_argument("--eval_max_chunks", type=int, default=None,
                        help="Max test chunks to evaluate (None for full set)")

    # Optional experiments
    parser.add_argument("--run_ablation", action="store_true",
                        help="Run 2-bit Step 1 vs Step 2 ablation (Table 4)")
    parser.add_argument("--run_benchmark", action="store_true",
                        help="Run bitwise dequantization microbenchmark (Table 6)")
    parser.add_argument("--prompt", type=str,
                        default="The history of artificial intelligence began",
                        help="Prompt for text generation sanity check")
    parser.add_argument("--skip_bits", type=str, default=None,
                        help="Comma-separated bit-widths to skip, e.g. '2' to skip 2-bit")

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    # Kaggle/Jupyter can append its own arguments (for example, -f kernel.json).
    # Ignore unknown arguments so the CLI can be run from a notebook cell.
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)

    skip_bits = set()
    if args.skip_bits:
        skip_bits = {int(b.strip()) for b in args.skip_bits.split(",")}

    print("=" * 70, flush=True)
    print("  POT-PTQ: Two-Step Power-of-Two Post-Training Quantization for LLMs",
          flush=True)
    print("  Wang et al., arXiv:2507.11959 (July 2025)", flush=True)
    print("=" * 70, flush=True)
    print(f"Model:           {args.model_name}", flush=True)
    if device.type == "cuda":
        print(f"Device:          {device} ({torch.cuda.get_device_name(0)})",
              flush=True)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM:            {vram:.1f} GB", flush=True)
    else:
        print(f"Device:          {device} (CPU)", flush=True)
    print(f"Group size:      {args.group_size}", flush=True)
    print(f"Calibration:     {args.calib_seqs} seqs x {args.calib_seqlen} tokens",
          flush=True)
    print(f"Eval seq length: {args.eval_seqlen}", flush=True)
    print("=" * 70, flush=True)

    # ── 1. Load Model ──
    print(f"\n[1/5] Loading {args.model_name}...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=model_dtype, token=hf_token)
    except Exception as e:
        print(f"Failed to load {args.model_name}: {e}", flush=True)
        fallback = "TinyLlama/TinyLlama_v1.1"
        print(f"Falling back to {fallback}...", flush=True)
        args.model_name = fallback
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=model_dtype)

    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n_params / 1e9:.3f}B parameters.", flush=True)

    # ── 2. Prepare WikiText-2 Data ──
    print("\n[2/5] Preparing WikiText-2 dataset...", flush=True)
    train_ids, test_ids = get_wikitext2_data(tokenizer)
    calib_batches = get_calibration_batches(
        train_ids, args.calib_seqs, args.calib_seqlen, device=device)
    print(f"Train tokens: {train_ids.shape[1]:,}, "
          f"Test tokens: {test_ids.shape[1]:,}", flush=True)
    print(f"Calibration: {len(calib_batches)} sequences.", flush=True)

    # ── 3. FP16 Baseline ──
    print("\n[3/5] Evaluating FP16 Baseline Perplexity...", flush=True)
    fp16_ppl, n_chunks = evaluate_wikitext2_ppl(
        model, test_ids, args.eval_seqlen,
        device=device, max_chunks=args.eval_max_chunks)
    print(f">> FP16 Baseline Perplexity = {fp16_ppl:.3f} "
          f"({n_chunks} chunks)", flush=True)

    results = {"FP16": fp16_ppl}
    quantized_models = {}

    # ── 4. POT-PTQ Quantization ──
    print("\n[4/5] Running POT-PTQ Quantization...", flush=True)
    configs = [
        (3, args.epochs_3bit),
        (2, args.epochs_2bit),
    ]

    for n_bits, epochs in configs:
        if n_bits in skip_bits:
            print(f"\nSkipping {n_bits}-bit (--skip_bits).", flush=True)
            continue

        print(f"\n{'─'*60}", flush=True)
        print(f"  {n_bits}-bit POT-PTQ (Step 1 + Step 2, {epochs} epochs)",
              flush=True)
        print(f"{'─'*60}", flush=True)

        qmodel = copy.deepcopy(model).to(device)
        t0 = time.time()
        qmodel, stats = quantize_model(
            model=qmodel,
            calib_batches=calib_batches,
            n_bits=n_bits,
            group_size=args.group_size,
            epochs=epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            init_method="step1",
            device=device,
            compute_dtype=model_dtype,
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"Quantization time: {elapsed:.1f}s ({elapsed/60:.1f} min)",
              flush=True)

        print(f"Evaluating {n_bits}-bit perplexity...", flush=True)
        ppl, _ = evaluate_wikitext2_ppl(
            qmodel, test_ids, args.eval_seqlen,
            device=device, max_chunks=args.eval_max_chunks)
        print(f">> POT-PTQ {n_bits}-bit Perplexity = {ppl:.3f}", flush=True)
        results[f"POT-PTQ {n_bits}-bit"] = ppl
        quantized_models[n_bits] = qmodel

    # ── 5. Results Summary (Table 1) ──
    print("\n" + "=" * 55, flush=True)
    print("  WIKITEXT-2 PERPLEXITY RESULTS (TABLE 1 STYLE)", flush=True)
    print(f"  Model: {args.model_name}", flush=True)
    print("=" * 55, flush=True)
    for k, v in results.items():
        print(f"  {k:<22}:  {v:8.3f}", flush=True)
    print("=" * 55, flush=True)

    # ── 6. Ablation (Table 4) ──
    if args.run_ablation:
        print("\n[Ablation Study @ 2-bit (Table 4)]", flush=True)

        print("  Step 1 Only (no fine-tuning)...", flush=True)
        qm1 = copy.deepcopy(model).to(device)
        qm1, _ = quantize_model(
            qm1, calib_batches, n_bits=2, group_size=args.group_size,
            epochs=0, lr=args.lr, weight_decay=args.weight_decay,
            init_method="step1", device=device,
            compute_dtype=model_dtype, verbose=False)
        ppl1, _ = evaluate_wikitext2_ppl(
            qm1, test_ids, args.eval_seqlen,
            device=device, max_chunks=args.eval_max_chunks)
        del qm1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("  Step 2 Only (naive b=1 init)...", flush=True)
        qm2 = copy.deepcopy(model).to(device)
        qm2, _ = quantize_model(
            qm2, calib_batches, n_bits=2, group_size=args.group_size,
            epochs=args.epochs_2bit, lr=args.lr,
            weight_decay=args.weight_decay,
            init_method="naive", device=device,
            compute_dtype=model_dtype, verbose=False)
        ppl2, _ = evaluate_wikitext2_ppl(
            qm2, test_ids, args.eval_seqlen,
            device=device, max_chunks=args.eval_max_chunks)
        del qm2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n  {'─'*50}", flush=True)
        print(f"  Step 1 Only                  :  {ppl1:.3f}", flush=True)
        print(f"  Step 2 Only (naive init)     :  {ppl2:.3f}", flush=True)
        full_ppl = results.get("POT-PTQ 2-bit", "N/A")
        print(f"  Step 1 + Step 2 (Full Method):  {full_ppl}", flush=True)
        print(f"  {'─'*50}", flush=True)

    # ── 7. Dequant Benchmark (Table 6) ──
    if args.run_benchmark:
        print("\n[Dequantization Microbenchmark (Section 4, Table 6)]",
              flush=True)
        n = 1_000_000
        scale = (torch.rand(n, device=device) * 0.05 + 1e-3).to(torch.float16)
        sign = torch.where(torch.rand(n, device=device) > 0.5,
                           1.0, -1.0).to(torch.float16)
        exponent = torch.randint(0, 4, (n,), device=device)

        w_naive = dequant_naive_fp(scale, sign, exponent)
        w_bitwise = dequant_bitwise(scale, sign, exponent)
        max_err = (w_naive.float() - w_bitwise.float()).abs().max().item()
        print(f"  Correctness: max error = {max_err:.6e} "
              f"(match={torch.allclose(w_naive.float(), w_bitwise.float())})",
              flush=True)

        def time_fn(fn, iters=50):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(scale, sign, exponent)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters

        t_fp = time_fn(dequant_naive_fp)
        t_bit = time_fn(dequant_bitwise)
        speedup = t_fp / max(t_bit, 1e-9)
        print(f"  Uniform (FP16 Multiply):  {t_fp * 1e3:.4f} ms", flush=True)
        print(f"  PoT (Bitwise + Int Add): {t_bit * 1e3:.4f} ms", flush=True)
        print(f"  Speedup:                  {speedup:.2f}x", flush=True)

    # ── 8. Text Generation ──
    if args.prompt and quantized_models:
        print("\n[Text Generation Quality Check]", flush=True)
        print(f'Prompt: "{args.prompt}"\n', flush=True)
        for n_bits, qm in quantized_models.items():
            sample = generate_sample(
                qm, tokenizer, prompt=args.prompt,
                max_new_tokens=60, device=device)
            print(f"--- {n_bits}-bit POT-PTQ ---", flush=True)
            print(sample, flush=True)
            print(flush=True)


if __name__ == "__main__":
    main()
