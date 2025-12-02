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
"""LoRA Status Logging Utilities.

This module provides functions for logging LoRA configuration status,
including parameter counts and trainability analysis.

Usage:
    from vlash.lora.logging import log_lora_status, count_parameters
    
    # Log detailed LoRA status
    log_lora_status(policy)
    
    # Count parameters
    total = count_parameters(policy)
    trainable = count_parameters(policy, only_trainable=True)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

from torch import nn


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """Count model parameters, correctly handling quantized layers.
        
    Args:
        model: The model to count parameters for.
        only_trainable: If True, only count parameters with requires_grad=True.
    """
    # Check if bitsandbytes is available for quantized layer handling
    try:
        import bitsandbytes as bnb
        has_bnb = True
    except ImportError:
        has_bnb = False

    total = 0
    counted_params = set()  # Track params we've already counted

    # First pass: handle quantized modules specially
    if has_bnb:
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                # For 4-bit linear, compute actual (uncompressed) param count
                weight_params = module.in_features * module.out_features
                bias_params = module.out_features if module.bias is not None else 0
                
                if only_trainable:
                    # Quantized weights are always frozen, only count trainable bias
                    if module.bias is not None and module.bias.requires_grad:
                        total += bias_params
                else:
                    total += weight_params + bias_params
                    
                # Mark these params as counted to avoid double-counting
                if module.weight is not None:
                    counted_params.add(id(module.weight))
                if module.bias is not None:
                    counted_params.add(id(module.bias))

    # Second pass: count remaining (non-quantized) parameters normally
    for p in model.parameters():
        if id(p) in counted_params:
            continue  # Skip already counted (quantized) params
        if only_trainable and not p.requires_grad:
            continue
        total += p.numel()

    return total


@dataclass
class ModuleLoRAStat:
    """Statistics for a group of modules with the same pattern.
    
    Used to aggregate information about layers that share a common
    structure (e.g., all attention layers across different positions).
    
    Attributes:
        count: Number of base parameters in this pattern group.
        total_params: Total parameter count (base params only, not LoRA).
        trainable_params: Number of trainable base parameters.
        has_lora: Whether any module in this group has LoRA adapters.
        requires_grad: Whether any base parameter requires gradients.
    """
    count: int = 0
    total_params: int = 0
    trainable_params: int = 0
    has_lora: bool = False
    requires_grad: bool = False


def patternize_name(name: str) -> str:
    """Collapse numeric indices in module names to create patterns.
    
    This groups modules that differ only in their position index,
    making it easier to see the overall model structure.
    
    Example:
        'paligemma.language_model.layers.0.self_attn'
        'paligemma.language_model.layers.1.self_attn'
        'paligemma.language_model.layers.2.self_attn'
    all become:
        'paligemma.language_model.layers.<n>.self_attn'
    
    Args:
        name: Full module name with numeric indices.
        
    Returns:
        Pattern with numeric indices replaced by '<n>'.
    """
    return re.sub(r"\.(\d+)(\.|$)", r".<n>\2", name)


def owner_from_param_name(param_name: str) -> str:
    """Extract the owning module name from a parameter name.
    
    Parameter names include the tensor type suffix (weight, bias, etc.)
    and PEFT wrapper prefixes (base_layer, lora_A, etc.). This function
    strips these to get the "raw" layer name.
    
    Examples:
        'model.layers.0.mlp.down_proj.base_layer.weight'
          -> 'model.layers.0.mlp.down_proj'
        
        'model.layers.0.mlp.down_proj.lora_A.default.weight'
          -> 'model.layers.0.mlp.down_proj'
        
        'model.lm_head.weight'
          -> 'model.lm_head'
    
    Args:
        param_name: Full parameter name from named_parameters().
        
    Returns:
        Module name without tensor suffix or PEFT wrapper parts.
    """
    parts = param_name.split(".")
    if not parts:
        return "<root>"

    # Remove the tensor name suffix (weight, bias, etc.)
    parts = parts[:-1]

    # Remove PEFT wrapper suffixes
    helper_suffixes = {"base_layer", "lora_a", "lora_b", "default"}
    while parts and parts[-1].lower() in helper_suffixes:
        parts = parts[:-1]

    return ".".join(parts) if parts else "<root>"


def log_lora_status(policy: nn.Module) -> None:
    """Log detailed LoRA status for debugging and verification.
    
    Outputs two types of information:
    1. Global summary: total and trainable parameter counts
    2. Per-pattern breakdown: which layer patterns have LoRA, 
       which are trainable, and their parameter counts
    
    Example output:
        [LoRA] GLOBAL total_params=3000000000 trainable_params=1000000
        [LoRA] pattern=model.layers.<n>.self_attn.q_proj count=32 has_lora=True ...
        [LoRA] pattern=model.layers.<n>.mlp.down_proj count=32 has_lora=True ...
        [LoRA] pattern=model.lm_head count=1 has_lora=False requires_grad=True ...
    
    Args:
        policy: The policy to analyze (with or without LoRA applied).
    """
    logger = logging.getLogger(__name__)

    # Collect statistics grouped by module pattern
    stats: dict[str, ModuleLoRAStat] = defaultdict(ModuleLoRAStat)

    # Log global parameter counts (handles quantized layers correctly)
    total_params_global = count_parameters(policy, only_trainable=False)
    trainable_params_global = count_parameters(policy, only_trainable=True)
    logger.info(
        "[LoRA] GLOBAL total_params=%d trainable_params=%d",
        total_params_global,
        trainable_params_global,
    )

    # Analyze each parameter
    for full_name, p in policy.named_parameters():
        # Get the owning module and its pattern
        owner = owner_from_param_name(full_name)
        pattern = patternize_name(owner)
        st = stats[pattern]

        # LoRA parameters: just mark the pattern as having LoRA
        if "lora" in full_name.lower():
            st.has_lora = True
            continue  # Don't count LoRA params in base param stats

        # Base (non-LoRA) parameters
        st.count += 1
        st.total_params += p.numel()
        if p.requires_grad:
            st.trainable_params += p.numel()
            st.requires_grad = True

    # Log per-pattern statistics sorted alphabetically
    for pattern, st in sorted(stats.items(), key=lambda x: x[0]):
        logger.info(
            "[LoRA] pattern=%s count=%d has_lora=%s requires_grad=%s "
            "total_params=%d trainable_params=%d",
            pattern,
            st.count,
            st.has_lora,
            st.requires_grad,
            st.total_params,
            st.trainable_params,
        )
