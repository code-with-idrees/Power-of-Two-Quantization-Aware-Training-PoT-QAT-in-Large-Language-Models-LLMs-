"""
POT-PTQ: Power-of-Two Post-Training Quantization for Large Language Models.
Based on "POT-PTQ: A Two-step Power-of-Two Post-training for LLMs" (Wang et al., arXiv:2507.11959).
"""

from .core import (
    qmax_for_bits,
    ste_round_clamp,
    to_groups,
    from_groups,
    pot_quantize_dequantize,
    dequant_naive_fp,
    dequant_bitwise,
)
from .step1 import data_agnostic_scale_init, naive_scale_init
from .layers import PoTLinear
from .quantizer import quantize_model, get_transformer_blocks, replace_linear_with_pot
from .eval import evaluate_wikitext2_ppl, get_wikitext2_data, get_calibration_batches

__version__ = "1.0.0"
__all__ = [
    "qmax_for_bits",
    "ste_round_clamp",
    "to_groups",
    "from_groups",
    "pot_quantize_dequantize",
    "dequant_naive_fp",
    "dequant_bitwise",
    "data_agnostic_scale_init",
    "naive_scale_init",
    "PoTLinear",
    "quantize_model",
    "get_transformer_blocks",
    "replace_linear_with_pot",
    "evaluate_wikitext2_ppl",
    "get_wikitext2_data",
    "get_calibration_batches",
]
