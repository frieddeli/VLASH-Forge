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
"""Inference Latency Benchmark.

"""

import json
import logging
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

from benchmarks.benchmark_config import BenchmarkConfig
from vlash.policies.factory import get_policy_class, make_policy


def load_dataset(cfg: BenchmarkConfig) -> tuple[LeRobotDataset, LeRobotDatasetMetadata]:
    """Load dataset for benchmarking.
    
    Uses standard LeRobotDataset without temporal augmentation.
    
    Args:
        cfg: Benchmark configuration.
        
    Returns:
        Tuple of (dataset, metadata).
    """
    logging.info(f"Loading dataset: {cfg.dataset.repo_id}")
    
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    
    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
    )
    
    logging.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_episodes} episodes")
    
    return dataset, ds_meta


def load_policy(cfg: BenchmarkConfig, ds_meta: LeRobotDatasetMetadata) -> PreTrainedPolicy:
    """Load pretrained policy for benchmarking.
    
    Uses VLASH's policy factory which handles both pretrained and new models.
    
    Args:
        cfg: Benchmark configuration.
        ds_meta: Dataset metadata.
        
    Returns:
        Policy in eval mode.
    """
    logging.info(f"Loading policy type: {cfg.policy.type}")
    
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    )
    
    policy.eval()
    
    device = get_safe_torch_device(cfg.policy.device)
    logging.info(f"Policy loaded successfully on device: {device}")
    
    return policy


def prepare_batch(batch: dict, device: torch.device) -> dict:
    """Move batch tensors to device.
    
    Also converts language_instruction to task field expected by policy.
    
    Args:
        batch: Input batch from dataloader.
        device: Target device.
        
    Returns:
        Prepared batch dictionary.
    """
    prepared = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            prepared[k] = v.to(device)
        else:
            prepared[k] = v
    
    if "language_instruction" in batch:
        prepared["task"] = batch["language_instruction"]
    
    return prepared


def warmup_model(
    policy: PreTrainedPolicy,
    dataloader: DataLoader,
    cfg: BenchmarkConfig,
):
    """Warm up model before benchmarking.
    """
    if cfg.warmup_steps <= 0:
        return
    
    logging.info(f"Warming up model for {cfg.warmup_steps} steps...")
    device = get_safe_torch_device(cfg.policy.device)
    
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= cfg.warmup_steps:
                break
            
            batch = prepare_batch(batch, device)
            _ = policy.predict_action_chunk(batch)
    
    # Ensure warmup is complete before starting benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    logging.info("Warmup complete")


def benchmark_inference_latency_impl(
    policy: PreTrainedPolicy,
    dataloader: DataLoader,
    cfg: BenchmarkConfig,
) -> dict:
    """Run inference latency measurement.
        """
    device = get_safe_torch_device(cfg.policy.device)
    latencies = []
    
    logging.info(f"Starting inference latency benchmarking with {cfg.num_samples} samples...")
    
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= cfg.num_samples:
                break
            
            batch = prepare_batch(batch, device)
            
            # Synchronize before timing for accurate GPU measurement
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = policy.predict_action_chunk(batch)
            
            # Synchronize after inference
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # ms
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                logging.info(f"Processed {i + 1}/{cfg.num_samples} samples...")
    
    # Compute statistics
    latencies = np.array(latencies)
    results = {
        "num_samples": len(latencies),
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "fps": float(1000.0 / np.mean(latencies)),
    }
    
    return results


def print_results(results: dict, cfg: BenchmarkConfig):
    """Print formatted benchmark results to console."""
    pretrained_path = getattr(cfg.policy, 'pretrained_path', None) or "N/A (new model)"
    
    print("\n" + "=" * 80)
    print("INFERENCE LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nPolicy Type: {cfg.policy.type}")
    print(f"Pretrained Path: {pretrained_path}")
    print(f"Dataset: {cfg.dataset.repo_id}")
    print(f"Device: {cfg.policy.device}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Compile: {cfg.policy.compile_model}")
    print(f"\nNumber of samples: {results['num_samples']}")
    print(f"\nLatency Statistics (milliseconds):")
    print(f"  Mean:   {results['mean_ms']:.2f} ms")
    print(f"  Median: {results['median_ms']:.2f} ms")
    print(f"  Std:    {results['std_ms']:.2f} ms")
    print(f"  Min:    {results['min_ms']:.2f} ms")
    print(f"  Max:    {results['max_ms']:.2f} ms")
    print(f"\nPercentiles:")
    print(f"  P50: {results['p50_ms']:.2f} ms")
    print(f"  P90: {results['p90_ms']:.2f} ms")
    print(f"  P95: {results['p95_ms']:.2f} ms")
    print(f"  P99: {results['p99_ms']:.2f} ms")
    print(f"\nThroughput:")
    print(f"  FPS: {results['fps']:.2f}")
    print("=" * 80 + "\n")


def save_results(results: dict, cfg: BenchmarkConfig):
    """Save benchmark results to JSON file.
    
    Includes both configuration and results for reproducibility.
    
    Args:
        results: Benchmark results.
        cfg: Configuration used for benchmark.
    """
    if cfg.output_file is None:
        return
    
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pretrained_path = getattr(cfg.policy, 'pretrained_path', None)
    
    output_data = {
        "config": {
            "benchmark_type": "inference_latency",
            "policy_type": cfg.policy.type,
            "policy_path": str(pretrained_path) if pretrained_path else None,
            "dataset_repo_id": cfg.dataset.repo_id,
            "device": cfg.policy.device,
            "batch_size": cfg.batch_size,
            "num_samples": cfg.num_samples,
            "warmup_steps": cfg.warmup_steps,
            "compile_model": cfg.policy.compile_model,
            "seed": cfg.seed,
        },
        "results": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Results saved to: {output_path}")


@parser.wrap()
def benchmark_inference_latency(cfg: BenchmarkConfig):
    """Main entry point for inference latency benchmark.
    """
    init_logging()
    logging.info("Starting inference latency benchmark")
        
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    
    set_seed(cfg.seed)
    
    # Load dataset and policy
    dataset, ds_meta = load_dataset(cfg)
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.policy.device == "cuda" else False,
    )
    
    policy = load_policy(cfg, ds_meta)
    
    # Warmup
    warmup_model(policy, dataloader, cfg)
    
    # Reset dataloader for actual benchmarking
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.policy.device == "cuda" else False,
    )
    
    # Benchmark
    results = benchmark_inference_latency_impl(policy, dataloader, cfg)
    
    # Output
    print_results(results, cfg)
    save_results(results, cfg)
    
    logging.info("Inference latency benchmark complete!")


def main():
    benchmark_inference_latency()


if __name__ == "__main__":
    main()
