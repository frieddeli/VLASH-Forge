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
"""LoRA Checkpoint Utilities.

This module provides functions for saving and loading LoRA adapters,
as well as merging LoRA weights into base models for inference.

The checkpoint strategy:
1. During training: Keep LoRA adapters separate for efficient updates
2. At checkpoint: Save both raw adapters and merged model
3. For inference: Load merged model without PEFT dependency

Usage:
    # Merge for inference
    merge_lora_into_base(policy)
    
    # Resume training
    load_lora_adapters(policy, checkpoint_dir)
    
    # Create checkpoint
    merged = clone_and_merge_lora_policy(policy, lora_cfg)
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
from torch import nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

from vlash.configs.train_config import LoRAConfig
from vlash.lora.apply import apply_lora


def merge_lora_into_base(policy: nn.Module, verbose: bool = False) -> None:
    """Merge LoRA adapters into base model weights for efficient inference.
        
    Args:
        policy: Policy with LoRA applied (must have _peft_model attribute).
        verbose: If True, log the merge operation.
    """
    peft_model = getattr(policy, "_peft_model", None)
    if peft_model is None:
        raise ValueError(
            "No `_peft_model` attribute found on policy. "
            "Make sure `apply_lora` was called before attempting to merge."
        )

    try:
        from peft import PeftModel
    except Exception as exc:
        raise ImportError(
            "Merging LoRA weights requires the `peft` library to be installed. "
            "Please install `peft` into your environment."
        ) from exc

    if not isinstance(peft_model, PeftModel):
        raise TypeError(
            "Expected `policy._peft_model` to be a `PeftModel` when merging LoRA, "
            f"but got {type(peft_model)}. Ensure `apply_lora` was called and "
            "has not been altered."
        )

    if verbose:
        logging.getLogger(__name__).info("Merging LoRA adapters into base model for efficient inference.")

    # merge_and_unload() folds LoRA weights into base and returns unwrapped model
    merged = peft_model.merge_and_unload()
    policy.model = merged
    
    # Clean up PEFT wrapper reference
    if hasattr(policy, "_peft_model"):
        delattr(policy, "_peft_model")


def load_lora_adapters(policy: nn.Module, checkpoint_dir: Path) -> bool:
    """Load saved LoRA adapter weights for training resumption.
    
    When resuming LoRA training, we need to restore:
    1. LoRA adapter weights (lora_A, lora_B matrices)
    2. modules_to_save weights (fully trained modules like lm_head)
    
    This function loads both from the checkpoint's lora_adapters directory.
    
    Expected checkpoint structure:
        checkpoint_dir/
        └── pretrained_model/
            └── lora_adapters/
                ├── adapter_config.json
                └── adapter_model.safetensors (or .bin)
    
    Args:
        policy: Policy with LoRA already applied (via apply_lora).
        checkpoint_dir: Path to checkpoint directory.
        
    Returns:
        True if adapters loaded successfully, False if not found.
    """
    logger = logging.getLogger(__name__)

    # Verify LoRA was applied
    peft_model = getattr(policy, "_peft_model", None)
    if peft_model is None:
        logger.warning(
            "[LoRA] Cannot load adapters: policy does not have `_peft_model` attribute. "
            "Make sure `apply_lora` was called before loading adapters."
        )
        return False

    # Locate adapter directory
    lora_dir = checkpoint_dir / PRETRAINED_MODEL_DIR / "lora_adapters"
    if not lora_dir.exists():
        logger.warning(f"[LoRA] No lora_adapters directory found at {lora_dir}")
        return False

    # Check adapter weights exist
    adapter_weights_path = lora_dir / "adapter_model.safetensors"
    if not adapter_weights_path.exists():
        adapter_weights_path = lora_dir / "adapter_model.bin"
        if not adapter_weights_path.exists():
            logger.warning(f"[LoRA] No adapter weights found in {lora_dir}")
            return False

    try:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "Loading LoRA adapters requires `peft` and `safetensors` libraries. "
            "Please install them."
        ) from exc

    # Load adapter weights
    if adapter_weights_path.suffix == ".safetensors":
        adapter_weights = load_file(adapter_weights_path)
    else:
        adapter_weights = torch.load(adapter_weights_path, map_location="cpu", weights_only=True)

    # Restore weights using PEFT's utility (handles both LoRA and modules_to_save)
    set_peft_model_state_dict(peft_model, adapter_weights, adapter_name="default")

    # Log summary of loaded weights
    lora_keys = [k for k in adapter_weights.keys() if "lora_" in k]
    modules_to_save_keys = [k for k in adapter_weights.keys() if "modules_to_save" in k]
    logger.info(
        f"[LoRA] Loaded adapter weights from {lora_dir}: "
        f"{len(lora_keys)} LoRA params, {len(modules_to_save_keys)} modules_to_save params"
    )

    return True


def clone_and_merge_lora_policy(
    policy: PreTrainedPolicy,
    lora_cfg: LoRAConfig,
    *,
    lora_save_dir: Path | None = None,
) -> PreTrainedPolicy:
    """
    Create a new policy with LoRA weights merged.

    Args:
        policy: The LoRA-wrapped policy.
        lora_cfg: LoRA configuration (needed to recreate adapters).
        lora_save_dir: If provided, also save raw adapters here for resumption.
        
    Returns:
        New policy instance (on CPU) with LoRA merged into base weights.
        
    Raises:
        TypeError: If policy is not a PreTrainedPolicy.
        ValueError: If policy doesn't have _peft_model when saving adapters.
    """
    if not isinstance(policy, PreTrainedPolicy):
        raise TypeError(
            f"clone_and_merge_lora_policy expects a PreTrainedPolicy, got {type(policy)} instead."
        )

    logger = logging.getLogger(__name__)
    is_qlora = getattr(policy, "_qlora_enabled", False)

    # === Save raw LoRA adapters for training resumption ===
    if lora_save_dir is not None:
        peft_model = getattr(policy, "_peft_model", None)
        if peft_model is None:
            raise ValueError(
                "Expected `policy` to expose a `_peft_model` attribute when saving LoRA adapters. "
                "Make sure `apply_lora` was called before training when `cfg.lora.enable` is True."
            )

        try:
            from peft import PeftModel
        except Exception as exc:
            raise ImportError(
                "Saving LoRA adapters requires the `peft` library to be installed. "
                "Please install `peft` into your environment."
            ) from exc

        if not isinstance(peft_model, PeftModel):
            raise TypeError(
                "clone_and_merge_lora_policy was asked to save LoRA adapters, "
                "but `policy._peft_model` is not a `PeftModel` instance. "
                "Make sure `apply_lora` was called before training when `cfg.lora.enable` is True."
            )

        # Save adapters using PEFT's save_pretrained
        lora_dir = lora_save_dir / "lora_adapters"
        lora_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(lora_dir)

    # === Create merge target and merge adapters ===
    if is_qlora:
        # QLoRA requires special handling: quantized weights can't be directly copied
        logger.info("[QLoRA] Creating dequantized merge target for checkpoint...")

        # Create fresh non-quantized policy on CPU
        config_copy = copy.deepcopy(policy.config)
        policy_cls = type(policy)
        config_copy.device = "cpu"
        merged_policy = policy_cls(config_copy)

        # Apply LoRA WITHOUT quantization for the merge target
        lora_cfg_no_quant = copy.deepcopy(lora_cfg)
        lora_cfg_no_quant.use_qlora = False
        apply_lora(lora_cfg_no_quant, merged_policy, verbose=False)

        # Copy LoRA adapter weights (these are NOT quantized, so they transfer fine)
        peft_model = getattr(policy, "_peft_model", None)
        merged_peft_model = getattr(merged_policy, "_peft_model", None)

        if peft_model is not None and merged_peft_model is not None:
            source_state = peft_model.state_dict()
            target_state = merged_peft_model.state_dict()

            # Copy matching weights, skip base_layer (quantized vs non-quantized mismatch)
            for key in source_state:
                if key in target_state:
                    if "base_layer" in key:
                        continue  # Skip quantized base layer weights
                    if source_state[key].shape == target_state[key].shape:
                        target_state[key] = source_state[key].cpu()

            merged_peft_model.load_state_dict(target_state, strict=False)

        # Merge adapters into base weights
        merge_lora_into_base(merged_policy, verbose=False)

        return merged_policy

    else:
        # Standard LoRA: straightforward clone and merge
        config_copy = copy.deepcopy(policy.config)
        policy_cls = type(policy)
        config_copy.device = "cpu"
        
        # Create fresh policy and apply LoRA
        merged_policy = policy_cls(config_copy)
        apply_lora(lora_cfg, merged_policy, verbose=False)

        # Copy all weights from training policy
        merged_policy.load_state_dict(policy.state_dict())
        
        # Merge adapters into base weights
        merge_lora_into_base(merged_policy, verbose=False)

        return merged_policy
