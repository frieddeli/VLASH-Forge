#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""PI0.5 Utility Functions.

This module provides utility functions for the PI0.5 model:
- Dtype handling for device compatibility
- Sinusoidal positional embeddings for flow matching
- Vector padding for variable-length inputs
- Attention mask construction
- Image resizing with aspect-ratio-preserving padding
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device) -> torch.dtype:
    """Get a device-compatible dtype, falling back if necessary.
    
    Some devices don't support float64:
    - MPS (Apple Silicon): No float64 support
    - Some Intel XPU: May lack FP64 capability
    
    Args:
        dtype: Requested dtype.
        device: Target device.
        
    Returns:
        The original dtype if supported, otherwise float32.
    """
    if isinstance(device, torch.device):
        device = device.type
        
    # MPS doesn't support float64
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    
    # Some Intel XPU devices lack FP64
    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            device_capability = torch.xpu.get_device_capability()
            if not device_capability.get("has_fp64", False):
                logging.warning(f"Device {device} does not support float64, using float32 instead.")
                return torch.float32
        else:
            logging.warning(
                f"Device {device} capability check failed. Assuming no support for float64, using float32 instead."
            )
            return torch.float32
        return dtype
    else:
        return dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Create sinusoidal positional embeddings for scalar timesteps.
    
    Used in flow matching to encode the diffusion timestep t ∈ [0, 1].
    Creates embeddings with frequencies spanning from min_period to max_period.
    
    The embedding formula:
        emb[i] = sin(t * 2π / period_i)  for i < dim/2
        emb[i] = cos(t * 2π / period_i)  for i >= dim/2
    
    where period_i = min_period * (max_period/min_period)^(i / (dim/2 - 1))
    
    Args:
        time: Scalar timesteps [batch_size].
        dimension: Embedding dimension (must be even).
        min_period: Minimum sinusoidal period.
        max_period: Maximum sinusoidal period.
        device: Target device.
        
    Returns:
        Positional embeddings [batch_size, dimension].
        
    Raises:
        ValueError: If dimension is odd or time is not 1D.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    # Use float64 for precision, with device compatibility fallback
    dtype = get_safe_dtype(torch.float64, device.type)
    
    # Create log-spaced frequencies from min to max period
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute sinusoidal embedding
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    
    return pos_emb


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad the last dimension of a vector to a target size.
    
    Useful for padding state/action vectors to a fixed maximum dimension
    for batching across different robot configurations.
    
    Args:
        vector: Input tensor, either [batch, features] or [batch, seq, features].
        new_dim: Target size for the last dimension.
        
    Returns:
        Padded tensor with last dimension == new_dim.
        Original values are preserved, padding is zeros.
    """
    if vector.shape[-1] == new_dim:
        return vector
        
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    
    return new_vector


def build_attention_mask_and_position_ids(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 4D attention mask and position IDs for transformer.
    
    Constructs attention patterns that support:
    - Padding: masked positions are ignored
    - Block-causal structure: tokens can only attend to earlier blocks
    
    The att_masks tensor encodes block boundaries. Tokens with the same
    cumulative sum value are in the same block and can attend to each other.
    
    Args:
        pad_masks: Boolean mask [B, N], True for real tokens, False for padding.
        att_masks: Block structure mask [B, N], non-zero marks block boundaries.
        dtype: Output dtype for attention mask.
        
    Returns:
        attention_mask: Additive attention mask [B, 1, N, N].
                       0 for allowed attention, -inf for blocked.
        position_ids: Position indices [B, N] for rotary embeddings.
        
    Raises:
        ValueError: If input tensors are not 2D.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # Build block-causal mask: token i can attend to j if cumsum[j] <= cumsum[i]
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    
    # Combine with padding mask
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks  # [B, N, N] bool

    # Position IDs: cumulative count of non-padding tokens
    position_ids = torch.cumsum(pad_masks, dim=1) - 1  # [B, N]

    # Convert to additive mask format (0 for attend, -inf for block)
    mask_value = torch.finfo(dtype).min
    attention_mask = torch.where(
        att_2d_masks,
        torch.zeros_like(att_2d_masks, dtype=dtype),
        torch.full_like(att_2d_masks, mask_value, dtype=dtype),
    )
    attention_mask = attention_mask.unsqueeze(1)  # [B, 1, N, N]

    return attention_mask, position_ids


def resize_with_pad(
    img: Tensor,
    width: int,
    height: int,
    pad_value: float = -1,
) -> Tensor:
    """Resize image to target size while preserving aspect ratio.
    
    The image is scaled to fit within the target dimensions, then
    padded on the left and top to reach the exact target size.
    This matches the preprocessing used by SigLIP in PI0.5.
    
    Args:
        img: Input image [B, C, H, W].
        width: Target width.
        height: Target height.
        pad_value: Value for padding pixels (default -1 for SigLIP range).
        
    Returns:
        Resized and padded image [B, C, height, width].
        
    Raises:
        ValueError: If input is not 4D.
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    # Scale to fit within target dimensions
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    # Pad to reach exact target size (pad left and top)
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    
    return padded_img
