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
"""LoRA Application Module.

This module provides functions to apply Low-Rank Adaptation (LoRA) to
pretrained VLA policies for efficient fine-tuning.

Usage:
    from vlash.lora import apply_lora
    apply_lora(cfg.lora, policy, verbose=True)
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from vlash.configs.train_config import LoRAConfig
from vlash.lora.logging import log_lora_status
from vlash.lora.qlora import get_compute_dtype, quantize_peft_model_4bit


def cast_lora_adapters_to_base_dtype(peft_model: nn.Module) -> int:
    """Cast LoRA adapters to match their base layer dtype.
        
    Args:
        peft_model: A PEFT-wrapped model with LoRA layers.
        
    Returns:
        Number of LoRA modules that were cast.
    """
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except ImportError:
        return 0

    cast_count = 0
    for name, module in peft_model.named_modules():
        # Only process PEFT LoRA linear layers
        if not isinstance(module, LoraLinear):
            continue

        base_layer = getattr(module, "base_layer", None)
        if base_layer is None:
            continue

        # Get base layer dtype
        base_dtype = None
        if hasattr(base_layer, "weight") and base_layer.weight is not None:
            base_dtype = base_layer.weight.dtype

        if base_dtype is None:
            continue

        # Cast lora_A and lora_B to match base_layer dtype
        for adapter_name in module.lora_A:
            lora_a = module.lora_A[adapter_name]
            lora_b = module.lora_B[adapter_name]
            if lora_a.weight.dtype != base_dtype:
                lora_a.to(base_dtype)
            if lora_b.weight.dtype != base_dtype:
                lora_b.to(base_dtype)
        cast_count += 1

    return cast_count


def infer_unfreeze_modules_from_patterns(model: nn.Module, patterns: list[str]) -> list[str]:
    """Find module names to unfreeze based on parameter name patterns.
    
    When using LoRA, most parameters are frozen. However, some modules
    (e.g., layer norms, output heads) may benefit from being trainable.
    This function finds modules whose parameters match the given patterns.
    
    For example, if patterns=["lm_head"], this will find and return
    module names like "lm_head" that should be kept trainable.
        
    Args:
        model: The base model to search for matching modules.
        patterns: List of substrings to match against parameter names.
                  Case-insensitive matching is used.
                  
    Returns:
        List of module names to keep trainable, or None if no patterns.
    """
    if not patterns:
        return None

    lowered = [pat.lower() for pat in patterns]
    modules: set[str] = set()

    # Find all parameters matching any pattern
    for name, _ in model.named_parameters():
        lname = name.lower()
        if any(pat in lname for pat in lowered):
            # Extract module name by removing the final ".weight"/".bias"
            module_name = name.rsplit(".", 1)[0]
            modules.add(module_name)

    return list(modules)


def apply_lora(cfg: LoRAConfig, policy: nn.Module, verbose: bool = False) -> None:
    """Apply LoRA to a policy for efficient fine-tuning.
    
    Args:
        cfg: LoRA configuration with rank, alpha, target modules, etc.
        policy: The policy to apply LoRA to. Must have a .model attribute.
        verbose: If True, log detailed LoRA status after application.
        
    Raises:
        ImportError: If PEFT library is not installed.
        ValueError: If policy doesn't have a .model attribute.
    """
    if not cfg.enable:
        return

    # Import PEFT (optional dependency)
    try:
        from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise ImportError(
            "LoRA is enabled but the `peft` library is not installed. "
            "Please install `peft` into your environment."
        ) from exc

    logger = logging.getLogger(__name__)

    # Get the base model from the policy wrapper
    base_model = getattr(policy, "model", None)
    if base_model is None:
        raise ValueError("LoRA is enabled but this policy does not expose a `.model` attribute.")

    # Find modules to keep trainable based on extra_trainable_modules patterns
    # These modules will be fully trained (not just LoRA adapters)
    unfreeze_modules = infer_unfreeze_modules_from_patterns(base_model, cfg.extra_trainable_modules)

    # Create PEFT LoRA configuration
    peft_cfg = PeftLoraConfig(
        r=cfg.r,                          # LoRA rank (low-rank dimension)
        lora_alpha=cfg.alpha,             # LoRA scaling factor
        lora_dropout=cfg.dropout,         # Dropout on LoRA outputs
        target_modules=cfg.target_modules, # Which layers to adapt
        task_type=TaskType.FEATURE_EXTRACTION,  # Task type for PEFT
        modules_to_save=unfreeze_modules, # Modules to keep fully trainable
    )

    # Wrap base model with PEFT (injects LoRA layers)
    peft_model = get_peft_model(base_model, peft_cfg)
    
    # Store reference for checkpoint handling
    setattr(policy, "_peft_model", peft_model)

    # Fix dtype mismatch: PEFT creates fp32 adapters, but base may be fp16/bf16
    cast_count = cast_lora_adapters_to_base_dtype(peft_model)
    logger.info(f"[LoRA] Cast {cast_count} LoRA modules to match base layer dtype")

    # === QLoRA: Apply 4-bit quantization after LoRA injection ===
    # Quantizing AFTER LoRA ensures PEFT can detect target modules correctly
    if cfg.use_qlora:
        compute_dtype = get_compute_dtype(cfg.qlora_compute_dtype)

        logger.info("[QLoRA] Applying 4-bit quantization to LoRA base layers...")
        quantize_peft_model_4bit(
            peft_model,
            compute_dtype=compute_dtype,
            quant_type=cfg.qlora_quant_type,
        )

        # Cast LoRA adapters to compute_dtype to match quantized base layers
        for name, param in peft_model.named_parameters():
            if "lora_" in name and param.dtype != compute_dtype:
                param.data = param.data.to(compute_dtype)
        logger.info(f"[QLoRA] Cast LoRA adapters to {compute_dtype}")

        # Mark policy for QLoRA-aware checkpoint handling
        setattr(policy, "_qlora_enabled", True)
        setattr(policy, "_qlora_compute_dtype", cfg.qlora_compute_dtype)
        setattr(policy, "_qlora_quant_type", cfg.qlora_quant_type)

    if verbose:
        log_lora_status(policy)


def is_lora_policy(policy: nn.Module) -> bool:
    """Check if a policy has been wrapped with PEFT LoRA.
    
    This is used to determine whether special LoRA checkpoint handling
    is needed (e.g., merging adapters before saving).
    
    Args:
        policy: Policy to check.
        
    Returns:
        True if policy has a _peft_model that is a PeftModel instance.
    """
    peft_model = getattr(policy, "_peft_model", None)
    if peft_model is None:
        return False

    try:
        from peft import PeftModel
    except Exception:
        return False

    return isinstance(peft_model, PeftModel)


def is_qlora_policy(policy: nn.Module) -> bool:
    """
    Check whether the given policy uses QLoRA (4-bit quantized LoRA).

    Returns True if the policy has been quantized with bitsandbytes Linear4bit layers.
    """
    return getattr(policy, "_qlora_enabled", False)
