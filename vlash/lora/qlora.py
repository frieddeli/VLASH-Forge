#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""QLoRA: 4-bit Quantization Utilities.

This module provides functions for applying QLoRA (Quantized LoRA) to models
using bitsandbytes for 4-bit quantization.

QLoRA reduces memory usage by:
1. Quantizing base model weights to 4-bit (NF4 or FP4 format)
2. Keeping LoRA adapters in full precision for training
3. Using a higher precision compute dtype for forward pass

Usage:
    from vlash.lora.qlora import quantize_peft_model_4bit, get_compute_dtype
    
    compute_dtype = get_compute_dtype("bfloat16")
    quantize_peft_model_4bit(peft_model, compute_dtype=compute_dtype)
"""

from __future__ import annotations

import logging

import torch
from torch import nn


def get_compute_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype name to torch.dtype.
    
    Args:
        dtype_str: One of "float16", "bfloat16", or "float32".
        
    Returns:
        Corresponding torch.dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_str]


def get_parent_module(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Get parent module and child name from a dotted path.
    
    Example:
        get_parent_module(model, "encoder.layer.0.attention.self.query")
        -> (model.encoder.layer.0.attention.self, "query")
    
    Args:
        model: Root module to search from.
        name: Dotted path to the target module.
        
    Returns:
        Tuple of (parent_module, child_name).
    """
    parts = name.rsplit(".", 1)
    if len(parts) == 1:
        return model, name
    parent_name, child_name = parts
    parent = model.get_submodule(parent_name)
    return parent, child_name


def quantize_model_4bit(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Quantize Linear layers to 4-bit using bitsandbytes.
    
    This function replaces nn.Linear modules with bnb.nn.Linear4bit.
    Only layers matching target_modules patterns are quantized.

    
    Args:
        model: Model to quantize (must be on CPU).
        compute_dtype: Dtype for forward pass computation (bfloat16 recommended).
        quant_type: Quantization format, "nf4" (recommended) or "fp4".
        target_modules: List of module name patterns to quantize.
                        Only Linear layers whose names contain these patterns
                        will be quantized. If None, nothing is quantized.
    
    Returns:
        Quantized model on CUDA.
    """
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise ImportError(
            "QLoRA is enabled but `bitsandbytes` is not installed. "
            "Please install it with: pip install bitsandbytes"
        ) from exc

    logger = logging.getLogger(__name__)
    model = model.cpu()

    if not target_modules:
        logger.warning("[QLoRA] No target_modules specified, no layers will be quantized.")
        model = model.to("cuda")
        return model

    # Prepare patterns for case-insensitive matching
    target_patterns = [p.lower() for p in target_modules]

    def should_quantize(name: str) -> bool:
        """Check if module name matches any target pattern."""
        name_lower = name.lower()
        return any(p in name_lower for p in target_patterns)

    quantized_count = 0

    # Replace matching Linear layers with Linear4bit
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize(name):
            continue

        # Create 4-bit quantized replacement
        parent, child_name = get_parent_module(model, name)
        qlinear = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=(module.bias is not None),
            compute_dtype=compute_dtype,
            compress_statistics=True,  # Compress quantization statistics
            quant_type=quant_type,
        )
        qlinear.load_state_dict(module.state_dict())
        setattr(parent, child_name, qlinear)
        quantized_count += 1

    # Move to CUDA to trigger actual quantization
    model = model.to("cuda")

    logger.info(
        f"[QLoRA] Model quantized to 4-bit: {quantized_count} Linear layers quantized "
        f"(matching {target_modules}), quant_type={quant_type}, compute_dtype={compute_dtype}"
    )

    return model


def quantize_peft_model_4bit(
    peft_model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
) -> None:
    """Quantize base layers inside PEFT LoRA wrappers to 4-bit.
    
    This should be called AFTER PEFT has wrapped the model with LoRA.
    
    Structure before:
        LoraLinear
        ├── base_layer: nn.Linear (frozen, will be quantized)
        ├── lora_A: nn.Linear (trainable, stays fp32/bf16)
        └── lora_B: nn.Linear (trainable, stays fp32/bf16)
    
    Structure after:
        LoraLinear
        ├── base_layer: bnb.Linear4bit (quantized)
        ├── lora_A: nn.Linear (unchanged)
        └── lora_B: nn.Linear (unchanged)
    
    Memory optimization:
    1. Move entire model to CPU first (frees GPU memory)
    2. Replace base_layer with Linear4bit on CPU
    3. Move back to CUDA (triggers 4-bit packing)
    
    Args:
        peft_model: PEFT-wrapped model with LoRA layers.
        compute_dtype: Dtype for forward pass (bfloat16 recommended).
        quant_type: Quantization format, "nf4" or "fp4".
    """
    try:
        import bitsandbytes as bnb
        from peft.tuners.lora import Linear as LoraLinear
    except ImportError as exc:
        raise ImportError(
            "QLoRA requires `bitsandbytes` and `peft` to be installed."
        ) from exc

    logger = logging.getLogger(__name__)

    # Move to CPU to reduce peak GPU memory during quantization
    original_device = next(peft_model.parameters()).device
    logger.info("[QLoRA] Moving model to CPU for quantization...")
    peft_model.cpu()
    torch.cuda.empty_cache()

    quantized_count = 0

    # Find all LoRA wrappers and quantize their base_layer
    for name, module in list(peft_model.named_modules()):
        if not isinstance(module, LoraLinear):
            continue

        base_layer = getattr(module, "base_layer", None)
        if base_layer is None or not isinstance(base_layer, nn.Linear):
            continue

        # Create quantized replacement for base_layer
        state_dict = base_layer.state_dict()
        qlinear = bnb.nn.Linear4bit(
            base_layer.in_features,
            base_layer.out_features,
            bias=(base_layer.bias is not None),
            compute_dtype=compute_dtype,
            compress_statistics=True,
            quant_type=quant_type,
            device="cpu",
        )
        qlinear.load_state_dict(state_dict)

        # Replace base_layer in-place
        module.base_layer = qlinear
        quantized_count += 1

    # Move back to CUDA (triggers actual 4-bit packing)
    logger.info("[QLoRA] Moving quantized model back to CUDA...")
    peft_model.to("cuda")
    torch.cuda.empty_cache()

    logger.info(
        f"[QLoRA] Quantized {quantized_count} LoRA base layers to 4-bit, "
        f"compute_dtype={compute_dtype}, quant_type={quant_type}"
    )


def dequantize_model_4bit(model: nn.Module) -> nn.Module:
    """Dequantize Linear4bit layers back to standard nn.Linear.
    
    This is needed when:
    1. Merging LoRA weights into base model
    2. Saving checkpoint for inference without bitsandbytes
    3. Converting to other formats (ONNX, etc.)
    
    Args:
        model: Model with Linear4bit layers.
        
    Returns:
        Model with all Linear4bit layers replaced by nn.Linear.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        return model  # No bnb = nothing to dequantize

    logger = logging.getLogger(__name__)
    dequantized_count = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, bnb.nn.Linear4bit):
            continue

        # Dequantize packed weights back to full precision
        weight = module.weight
        dequantized_weight = weight.dequantize() if hasattr(weight, "dequantize") else weight.data

        # Create standard nn.Linear with dequantized weights
        linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=(module.bias is not None),
            device=dequantized_weight.device,
            dtype=dequantized_weight.dtype,
        )
        linear.weight.data = dequantized_weight
        if module.bias is not None:
            linear.bias.data = module.bias.data

        # Replace in parent module
        parent, child_name = get_parent_module(model, name)
        setattr(parent, child_name, linear)
        dequantized_count += 1

    if dequantized_count > 0:
        logger.info(f"[QLoRA] Dequantized {dequantized_count} Linear4bit layers")

    return model
