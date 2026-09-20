from __future__ import annotations

import torch
import torch.nn.functional as functional


@torch.compile
def gelu_glu(gate_input: torch.Tensor, value_input: torch.Tensor) -> torch.Tensor:
    """Fused GELU-GLU activation: ``gelu(gate_input) * value_input``."""
    return functional.gelu(gate_input) * value_input


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "mp_bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[dtype_str]
