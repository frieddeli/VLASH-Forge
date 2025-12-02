#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""VLASH Training Configuration.

This module defines configuration classes for VLASH training:
- LoRAConfig: Low-Rank Adaptation settings for efficient fine-tuning
- VLASHTrainConfig: Main training config extending LeRobot's TrainPipelineConfig
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.configs.train import TrainPipelineConfig


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation (LoRA).
        """

    # Enable/disable LoRA
    enable: bool = False

    # Backend implementation (currently only "peft" supported)
    backend: str = "peft"

    # LoRA hyperparameters
    r: int = 16  # Rank of low-rank matrices
    alpha: int = 16  # Scaling factor (effective lr = alpha/r * lr)
    dropout: float = 0.0  # Dropout on LoRA layers

    # Additional modules to keep trainable (pattern-based matching on param names)
    extra_trainable_modules: List[str] = field(default_factory=list)

    # Target modules for LoRA injection
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    )

    # QLoRA: 4-bit quantization of frozen weights
    use_qlora: bool = False
    qlora_quant_type: str = "nf4"  # "nf4" (normalized float4) or "fp4"
    qlora_compute_dtype: str = "bfloat16"  # Dtype for forward pass

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.backend not in {"peft"}:
            raise ValueError(f"Invalid LoRA backend: {self.backend!r}.")
        
        if self.use_qlora:
            if self.qlora_quant_type not in {"nf4", "fp4"}:
                raise ValueError(
                    f"Invalid qlora_quant_type: {self.qlora_quant_type!r}. "
                    "Supported values are 'nf4' and 'fp4'."
                )
            if self.qlora_compute_dtype not in {"float16", "bfloat16", "float32"}:
                raise ValueError(
                    f"Invalid qlora_compute_dtype: {self.qlora_compute_dtype!r}. "
                    "Supported values are 'float16', 'bfloat16', 'float32'."
                )


@dataclass
class VLASHTrainConfig(TrainPipelineConfig):
    """VLASH training configuration.

    Extends TrainPipelineConfig with VLASH-specific parameters.
    """

    # Temporal delay augmentation: random offset in [0, max_delay_steps]
    # Set to 0 to disable (standard training without delay)
    max_delay_steps: int = 0

    # Gradient accumulation steps
    grad_accum_steps: int = 1

    # LoRA configuration
    lora: LoRAConfig = field(default_factory=LoRAConfig)
