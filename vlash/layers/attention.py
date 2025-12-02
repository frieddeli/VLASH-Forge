#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
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
"""Attention Layer with KV Cache.

This module provides a minimal scaled dot-product attention implementation
with optional KV caching for efficient autoregressive inference.

The KV cache strategy:
- First call with use_cache=True: Initialize cache with K/V (prefix prefill)
- Subsequent calls: Concatenate cached prefix K/V with new suffix K/V

"""

# TODO: use flashinfer kernels

import torch
from torch import nn


class Attention(nn.Module):
    """Scaled dot-product attention with optional KV cache.
    
    Computes: Attention(Q, K, V) = softmax(Q @ K^T / scale) @ V
    
    The KV cache enables efficient inference by storing prefix K/V
    and reusing them across multiple forward passes.
    """

    def __init__(self, scale: float):
        """Initialize attention layer.
        
        Args:
            scale: Scaling factor for attention scores, typically 1/sqrt(head_dim).
        """
        super().__init__()
        self.scale = scale
        
        # KV cache buffers: [B, H, L_prefix, D]
        self.k_cache = None
        self.v_cache = None

    def reset_cache(self):
        """Clear KV cache. Call when starting a new sequence."""
        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.
        
        Args:
            q: Query tensor [B, H, L_q, D].
            k: Key tensor [B, H, L_k_new, D].
            v: Value tensor [B, H, L_v_new, D].
            attention_mask: Additive mask [B, 1, L_q, L_k] or [B, H, L_q, L_k].
                           Use 0 for positions to attend, -inf for masked positions.
            use_cache: If True, use KV caching for prefix tokens.
            
        Returns:
            Output tensor [B, H, L_q, D].
        """
        # Handle KV cache
        if use_cache:
            if self.k_cache is None:
                # First call: initialize cache with prefix K/V
                self.k_cache = k.detach()
                self.v_cache = v.detach()
                k_full = k
                v_full = v
            else:
                # Subsequent calls: concatenate cached prefix with new suffix
                # Note: cache is not updated, always stores prefix only
                k_full = torch.cat([self.k_cache, k], dim=2)
                v_full = torch.cat([self.v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        # Compute attention scores: [B, H, L_q, L_k]
        attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale

        # Apply attention mask (additive)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask = attention_mask
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError(f"Unsupported attention_mask ndim: {attention_mask.ndim}")
            attn_scores = attn_scores + mask

        # Softmax and weighted sum
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v_full)
        
        return out
