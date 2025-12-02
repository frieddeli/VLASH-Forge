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
"""VLASH Dataset with Temporal Delay Augmentation.

This module extends LeRobotDataset to support VLASH's core training strategy:
temporal delay augmentation (async offset).

The key idea:
- During training, randomly delay the action chunk by [0, max_delay_steps]
- This teaches the policy to predict future actions from "stale" observations
- At inference, this enables asynchronous execution where the policy can
  start predicting the next chunk before the current one finishes

State handling with offset:
- offset == 0: Use original observation.state
- offset > 0: Use previous action (a_{offset-1}) as state

This matches the semantics used in nano-lerobot's apply_async_offset.
"""

from pathlib import Path
from typing import Callable
import random

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class VLASHDataset(LeRobotDataset):
    """Dataset with temporal delay augmentation for VLASH training.
    
    Extends LeRobotDataset to apply random temporal offsets to action chunks,
    teaching the model to handle timing variations and stale observations.
    
    Example with max_delay_steps=12, chunk_size=50:
        - Original: actions [t, t+1, ..., t+49]
        - With offset=5: actions [t+5, t+6, ..., t+54]
        - State becomes: action at t+4 (previous action before chunk start)
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        max_delay_steps: int = 0,
    ):
        """Initialize VLASH dataset.
        
        Args:
            repo_id: HuggingFace dataset repository ID.
            root: Local cache directory.
            episodes: List of episode indices to load (None = all).
            image_transforms: Optional image augmentation transforms.
            delta_timestamps: Timestamp offsets for each feature.
            tolerance_s: Tolerance for timestamp matching.
            revision: Dataset revision/version.
            force_cache_sync: Force re-download of cached data.
            download_videos: Whether to download video files.
            video_backend: Video decoding backend.
            batch_encoding_size: Batch size for video encoding.
            max_delay_steps: Maximum temporal delay for augmentation.
        """
        self.max_delay_steps = max_delay_steps

        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )

        # Track last offset for state construction in __getitem__
        self._last_offset: int = 0

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        """Get query indices with random temporal offset.
        
        Overrides parent to shift all delta_indices by a random offset
        sampled from [0, max_delay_steps], respecting episode boundaries.
        
        Args:
            idx: Sample index in dataset.
            ep_idx: Episode index.
            
        Returns:
            Tuple of (query_indices, padding_masks).
        """
        # Get episode boundaries
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Compute maximum valid offset (don't exceed episode boundary)
        max_delta = self.delta_indices["action"][-1]
        max_offset = min(self.max_delay_steps, max(0, ep_end - 1 - (idx + max_delta)))
        offset = random.randint(0, max_offset) if max_offset > 0 else 0

        # Store for use in __getitem__
        self._last_offset = offset

        # Build query indices with offset
        query_indices: dict[str, list[int]] = {}
        padding: dict[str, torch.BoolTensor] = {}

        for key, delta_idx in self.delta_indices.items():
            # Clamp indices to episode boundaries
            query_indices[key] = [
                max(ep_start, min(ep_end - 1, idx + delta + offset)) for delta in delta_idx
            ]
            # Mark padding for out-of-bounds indices
            padding[f"{key}_is_pad"] = torch.BoolTensor(
                [(idx + delta + offset < ep_start) | (idx + delta + offset >= ep_end) for delta in delta_idx]
            )

        return query_indices, padding

    def __getitem__(self, idx) -> dict:
        """Get sample with state constructed from previous action.
        
        When offset > 0, the observation.state is replaced with the
        action from the previous timestep (t + offset - 1). This matches
        the semantics of nano-lerobot's apply_async_offset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Sample dictionary with potentially modified observation.state.
        """
        item = super().__getitem__(idx)

        # No offset: return original item
        offset = getattr(self, "_last_offset", 0)
        if offset <= 0:
            return item

        # Get episode boundaries
        ep_idx = item["episode_index"].item() if "episode_index" in item else None
        if ep_idx is None:
            raise ValueError("episode_index not found in item")

        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Index of the previous action (before the offset action chunk starts)
        prev_idx = max(ep_start, min(ep_end - 1, idx + offset - 1))

        # Fetch previous action to use as state
        obs_state = item["observation.state"]
        prev_action = self.hf_dataset[prev_idx]["action"]

        # Validate dimensions
        if obs_state.dim() != 1 or prev_action.dim() != 1:
            raise ValueError("For now only support 1D state/action.")

        state_dim = obs_state.shape[0]
        action_dim = prev_action.shape[0]

        if state_dim == action_dim:
            # Dimensions match: use previous action as state
            new_state = prev_action
        else:
            raise ValueError(
                f"Unsupported state_dim != action_dim combination "
                "in VLASHDataset when applying async offset to observation.state. "
            )

        item["observation.state"] = new_state

        return item
