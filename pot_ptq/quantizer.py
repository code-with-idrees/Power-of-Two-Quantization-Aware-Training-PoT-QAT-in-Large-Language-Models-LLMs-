"""
Model-level Quantizer and Step-2 Layer-Sequential Calibration Driver.
Supports arbitrary Hugging Face causal LM architectures (LLaMA, Pythia, Qwen, OPT, Mistral, Gemma, Phi, etc.).
Based on Section 3.3, Algorithm 2 of "POT-PTQ: A Two-step Power-of-Two Post-training for LLMs".
"""

import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple

from .layers import PoTLinear


class StopForward(Exception):
    """Exception to halt forward pass during activation capture."""
    pass


def get_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """
    Locates the ModuleList containing transformer decoder blocks for common HF architectures.
    """
    candidate_paths = [
        "model.layers",          # LLaMA / Mistral / Qwen2 / Gemma / Phi-3 / DeepSeek
        "gpt_neox.layers",       # Pythia / GPT-NeoX
        "transformer.h",         # GPT-2 / GPT-J
        "model.decoder.layers",  # OPT (standard HF layout)
        "decoder.layers",        # OPT (alternate layout)
        "transformer.blocks",    # MPT / Falcon
        "model.transformer.h",   # Bloom / Starcoder
    ]
    for path in candidate_paths:
        obj = model
        found = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                found = False
                break
        if found and isinstance(obj, (nn.ModuleList, list)):
            return obj

    # Fallback: search named_modules for ModuleList of transformer blocks
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 1:
            first = mod[0]
            # Check if it has attention or mlp submodules
            has_attn = any("attn" in n.lower() or "self_attn" in n.lower() or "attention" in n.lower() for n, _ in first.named_modules())
            if has_attn:
                return mod

    raise ValueError(
        "Could not automatically locate the transformer block list. "
        f"Available top-level modules: {[n for n, _ in model.named_children()]}"
    )


def get_linear_submodules(block: nn.Module) -> Dict[str, nn.Linear]:
    """
    Finds all nn.Linear layers inside a transformer block, indexed by their relative dotted path.
    """
    return {
        name: mod
        for name, mod in block.named_modules()
        if isinstance(mod, nn.Linear) and not isinstance(mod, PoTLinear)
    }


def set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module):
    """
    Replaces a nested submodule in root specified by dotted_name with new_module.
    """
    parts = dotted_name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new_module)


def replace_linear_with_pot(
    block: nn.Module,
    n_bits: int,
    group_size: int = 128,
    compute_dtype: torch.dtype = torch.float16,
    init_method: str = "step1",
) -> Dict[str, PoTLinear]:
    """
    Replaces all nn.Linear layers within a block with PoTLinear modules.
    """
    linears = get_linear_submodules(block)
    pot_layers = {}
    for name, lin in linears.items():
        pot = PoTLinear(
            lin,
            n_bits=n_bits,
            group_size=group_size,
            compute_dtype=compute_dtype,
            init_method=init_method,
        )
        pot_layers[name] = pot
        set_submodule(block, name, pot)
    return pot_layers


@torch.no_grad()
def _detach_clone_io(x: Any) -> Any:
    """Recursively detaches and clones tensors in captured inputs."""
    if torch.is_tensor(x):
        return x.detach().clone()
    elif isinstance(x, tuple):
        return tuple(_detach_clone_io(v) for v in x)
    elif isinstance(x, list):
        return [_detach_clone_io(v) for v in x]
    elif isinstance(x, dict):
        return {k: _detach_clone_io(v) for k, v in x.items()}
    return x


def capture_first_block_inputs(
    model: nn.Module,
    first_block: nn.Module,
    calib_batches: List[torch.Tensor],
) -> List[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
    """
    Captures input activations (args, kwargs) to the very first transformer block (Block 0)
    for all calibration batches using a forward pre-hook.
    """
    captured_inputs = []

    def hook(module, args, kwargs):
        c_args = _detach_clone_io(args)
        c_kwargs = _detach_clone_io(kwargs)
        captured_inputs.append((c_args, c_kwargs))
        raise StopForward()

    handle = first_block.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for batch in calib_batches:
            try:
                # Disable KV-cache during calibration to ensure independent forward graphs
                model(batch, use_cache=False)
            except (TypeError, AttributeError):
                try:
                    model(batch)
                except StopForward:
                    pass
            except StopForward:
                pass
    finally:
        handle.remove()

    return captured_inputs


def quantize_model(
    model: nn.Module,
    calib_batches: List[torch.Tensor],
    n_bits: int,
    group_size: int = 128,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-1,
    init_method: str = "step1",
    device: torch.device = None,
    compute_dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """
    Executes full Two-Step POT-PTQ on the entire model:
    - Step 1: Parallel Data-Agnostic Scale Initialization (Algorithm 1)
    - Step 2: Data-Dependent Fine-Tuning with Learnable Residual Gamma (Algorithm 2)

    Uses memory-efficient layer-sequential block calibration (no duplicate full model needed).

    Args:
        model: Hugging Face CausalLM model
        calib_batches: List of input_id tensors for calibration
        n_bits: Quantization bit-width (e.g., 2 or 3)
        group_size: Weight group size (default: 128)
        epochs: Fine-tuning epochs per layer (paper default: 10 for 3-bit, 40 for 2-bit)
        lr: Learning rate for Gamma (paper default: 1e-3)
        weight_decay: L2 penalty lambda for Gamma (paper default: 1e-1)
        init_method: 'step1' for Algorithm 1 grid search, 'naive' for b=1 baseline
        device: Target execution device (cuda / cpu)
        compute_dtype: Model inference dtype (float16 / bfloat16)
        verbose: Whether to print progress per block

    Returns:
        quantized_model: In-place quantized model
        stats: Calibration loss & weight MSE statistics per transformer block
    """
    device = device or next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    blocks = get_transformer_blocks(model)
    n_blocks = len(blocks)

    if verbose:
        print(f"Starting POT-PTQ: {n_bits}-bit (groupsize={group_size}), {n_blocks} transformer blocks.", flush=True)
        print(f"Calibration: {len(calib_batches)} sequences, {epochs} epochs/block, lr={lr}, lambda={weight_decay}", flush=True)

    # 1. Capture inputs to the very first block
    current_inps = capture_first_block_inputs(model, blocks[0], calib_batches)

    stats = []

    # 2. Sequential block-by-block calibration
    for i, block in enumerate(blocks):
        # Deepcopy only this single block to serve as original reference (Eq. 14 H_orig)
        orig_block = copy.deepcopy(block).to(device)
        orig_block.eval()
        for p in orig_block.parameters():
            p.requires_grad_(False)

        # Compute original unquantized block outputs H_orig for all calibration batches
        H_orig_list = []
        with torch.no_grad():
            for args, kwargs in current_inps:
                # Ensure input tensors are on target device
                args_dev = tuple(a.to(device) if torch.is_tensor(a) else a for a in args)
                kwargs_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
                
                out = orig_block(*args_dev, **kwargs_dev)
                h_out = out[0] if isinstance(out, tuple) else out
                H_orig_list.append(h_out.detach().clone())

        # Free orig_block memory
        del orig_block
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Step 1: Replace linear layers with PoTLinear (performs Algorithm 1 grid search)
        pot_layers = replace_linear_with_pot(
            block,
            n_bits=n_bits,
            group_size=group_size,
            compute_dtype=compute_dtype,
            init_method=init_method,
        )
        block.to(device)

        # Step 2: Data-dependent fine-tuning of Gamma (Algorithm 2)
        params = [pot.Gamma for pot in pot_layers.values()]
        for p in params:
            p.requires_grad_(True)

        opt = torch.optim.Adam(params, lr=lr)
        last_rec_loss = 0.0

        if epochs > 0 and len(current_inps) > 0:
            for epoch in range(epochs):
                for idx, (args, kwargs) in enumerate(current_inps):
                    args_dev = tuple(a.detach().clone().to(device) if torch.is_tensor(a) else a for a in args)
                    kwargs_dev = {k: (tuple(x.detach().clone().to(device) for x in v) if isinstance(v, tuple) else (v.detach().clone().to(device) if torch.is_tensor(v) else v)) for k, v in kwargs.items()}
                    Horig = H_orig_list[idx]

                    # Forward pass through quantized block (Eq. 11, 16, 17)
                    Hquant = block(*args_dev, **kwargs_dev)
                    Hquant = Hquant[0] if isinstance(Hquant, tuple) else Hquant

                    # Loss: ||Horig - Hquant||_F^2 + (lambda / 2) * ||Gamma||_F^2 (Eq. 14)
                    rec_loss = F.mse_loss(Horig.float(), Hquant.float())
                    reg = sum((p.float() ** 2).mean() for p in params)
                    loss = rec_loss + (weight_decay / 2.0) * reg

                    opt.zero_grad()
                    loss.backward()  # Backprop using STE (Eq. 18)
                    opt.step()
                    last_rec_loss = rec_loss.item()

        avg_w_mse = sum(pot.weight_only_mse() for pot in pot_layers.values()) / max(len(pot_layers), 1)

        # Bake Gamma into S and freeze layer (frees original weight buffers)
        for pot in pot_layers.values():
            pot.freeze()

        # Compute output of the newly quantized block to pass to the next block
        next_inps = []
        with torch.no_grad():
            for args, kwargs in current_inps:
                args_dev = tuple(a.to(device) if torch.is_tensor(a) else a for a in args)
                kwargs_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
                out = block(*args_dev, **kwargs_dev)
                
                # Update hidden_states in args/kwargs
                if len(args_dev) > 0:
                    new_args = (out[0] if isinstance(out, tuple) else out,) + args_dev[1:]
                    new_kwargs = kwargs_dev
                else:
                    new_args = args_dev
                    new_kwargs = dict(kwargs_dev)
                    new_kwargs["hidden_states"] = out[0] if isinstance(out, tuple) else out

                next_inps.append((_detach_clone_io(new_args), _detach_clone_io(new_kwargs)))

        current_inps = next_inps
        del H_orig_list
        del opt
        del params

        stats.append({
            "block": i,
            "recon_loss": last_rec_loss,
            "weight_mse": avg_w_mse,
        })

        if verbose:
            print(f"  [block {i:2d}/{n_blocks-1}] output-recon MSE={last_rec_loss:.6e}  avg weight MSE={avg_w_mse:.6e}", flush=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    return model, stats
