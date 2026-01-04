#!/usr/bin/env python3
"""
Flux-1.dev Local Benchmarking Script

Benchmarks the actual flux-1.dev image inference pipeline used by the miner,
calling the same internal functions/modules. Supports both baseline (PyTorch)
and optimized (TensorRT) execution modes.

Usage:
    python benchmarks/benchmark_flux_local.py \
        --image_prompts_file prompts.json \
        --batch_size 1 \
        --iterations 10 \
        --warmup 3 \
        --output_csv results.csv \
        --label "fp16_baseline" \
        [--use_trt] \
        [--trt_engine_path /path/to/engine.plan]
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import traceback

import numpy as np
import torch
from diffusers import DiffusionPipeline

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from types import SimpleNamespace

# Import TRT modules only when needed (conditional import to avoid tensorrt dependency for baseline)
TRTTransformer = None
TRTInferenceServer = None
InferenceRequest = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_prompts(prompts_file: str) -> List[Dict[str, Any]]:
    """
    Load prompts from JSON file.

    Expected format:
    [
        {
            "prompt": "A cute anime girl...",
            "seed": 42,
            "guidance_scale": 7.5,
            "num_inference_steps": 28
        },
        ...
    ]

    Or simple text file with one prompt per line.
    """
    path = Path(prompts_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    if path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'prompts' in data:
                return data['prompts']
            else:
                raise ValueError("JSON must be a list or dict with 'prompts' key")
    else:
        # Treat as text file with one prompt per line
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        return [{"prompt": line} for line in lines]


def setup_baseline_pipeline(model_path: str, device: str = "cuda") -> DiffusionPipeline:
    """
    Setup baseline PyTorch pipeline (FP16/BF16).
    This matches the miner's baseline implementation.
    """
    logger.info(f"Loading baseline pipeline from: {model_path}")
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    logger.info("Baseline pipeline loaded successfully")
    return pipe


def setup_trt_pipeline(
    model_path: str,
    engine_path: str,
    device: str = "cuda"
) -> DiffusionPipeline:
    """
    Setup TensorRT-optimized pipeline.
    This matches the miner's optimized TRT implementation.
    """
    global TRTTransformer

    # Lazy import TRT modules only when needed
    if TRTTransformer is None:
        try:
            from trt import TRTTransformer as _TRTTransformer
            TRTTransformer = _TRTTransformer
        except ImportError as e:
            logger.error("Failed to import TensorRT modules. Please install tensorrt:")
            logger.error("  pip install tensorrt")
            logger.error(f"Error: {e}")
            raise

    logger.info(f"Loading TRT pipeline with engine: {engine_path}")

    # Load base pipeline
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    # Preserve transformer config
    transformer_config = pipe.transformer.config
    if hasattr(pipe.transformer, "encoder_hid_proj"):
        transformer_config.encoder_hid_proj = pipe.transformer.encoder_hid_proj
    else:
        transformer_config.encoder_hid_proj = SimpleNamespace(num_ip_adapters=0)

    # Replace with TRT transformer
    del pipe.transformer
    torch.cuda.empty_cache()

    pipe.transformer = TRTTransformer(
        engine_path,
        transformer_config,
        torch.device(device),
        max_batch_size=1
    )
    pipe.set_progress_bar_config(disable=True)

    logger.info("TRT pipeline loaded successfully")
    return pipe


def run_inference(
    pipe: DiffusionPipeline,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.5,
    seed: int = 42
) -> tuple[float, bool]:
    """
    Run a single inference and return (latency_seconds, success).
    """
    try:
        generator = torch.Generator(device="cuda").manual_seed(seed)

        start_time = time.perf_counter()
        _ = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images
        end_time = time.perf_counter()

        latency = end_time - start_time
        return latency, True

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.debug(traceback.format_exc())
        return 0.0, False


def compute_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Compute p50, p90, p95 latencies."""
    if not latencies:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0}

    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Flux-1.dev inference pipeline")
    parser.add_argument(
        "--image_prompts_file",
        type=str,
        required=True,
        help="Path to JSON or text file containing prompts"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (currently only batch_size=1 is supported)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of timed iterations per configuration"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label for this benchmark run (e.g., 'fp16_baseline' or 'trt_fp16')"
    )
    parser.add_argument(
        "--use_trt",
        action="store_true",
        help="Use TensorRT optimized pipeline"
    )
    parser.add_argument(
        "--trt_engine_path",
        type=str,
        default=None,
        help="Path to TensorRT engine file (.plan)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to base model (defaults to MODEL_PATH env var or 'black-forest-labs/FLUX.1-dev')"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.batch_size != 1:
        logger.warning("Currently only batch_size=1 is supported. Setting batch_size=1.")
        args.batch_size = 1

    if args.use_trt and not args.trt_engine_path:
        parser.error("--trt_engine_path is required when --use_trt is set")

    # Determine model path
    model_path = args.model_path or os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev")

    # Load prompts
    logger.info(f"Loading prompts from: {args.image_prompts_file}")
    prompts_data = load_prompts(args.image_prompts_file)
    logger.info(f"Loaded {len(prompts_data)} prompts")

    # Setup pipeline
    if args.use_trt:
        if not Path(args.trt_engine_path).exists():
            logger.error(f"TensorRT engine not found: {args.trt_engine_path}")
            sys.exit(1)
        pipe = setup_trt_pipeline(model_path, args.trt_engine_path)
    else:
        pipe = setup_baseline_pipeline(model_path)

    # Prepare CSV output
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'timestamp',
        'label',
        'batch_size',
        'iteration',
        'latency_seconds',
        'success'
    ])

    # Run warmup
    logger.info(f"Running {args.warmup} warmup iterations...")
    for i in range(args.warmup):
        prompt_data = prompts_data[i % len(prompts_data)]
        prompt = prompt_data.get("prompt", "")
        seed = prompt_data.get("seed", 42)
        guidance_scale = prompt_data.get("guidance_scale", 7.5)
        num_steps = prompt_data.get("num_inference_steps", 28)

        latency, success = run_inference(
            pipe,
            prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        logger.info(f"  Warmup {i+1}/{args.warmup}: {latency:.3f}s (success={success})")

    # Run benchmark iterations
    logger.info(f"Running {args.iterations} benchmark iterations...")
    latencies = []
    error_count = 0

    for i in range(args.iterations):
        # Cycle through prompts
        prompt_data = prompts_data[i % len(prompts_data)]
        prompt = prompt_data.get("prompt", "")
        seed = prompt_data.get("seed", 42 + i)  # Vary seed per iteration
        guidance_scale = prompt_data.get("guidance_scale", 7.5)
        num_steps = prompt_data.get("num_inference_steps", 28)

        latency, success = run_inference(
            pipe,
            prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

        if success:
            latencies.append(latency)
        else:
            error_count += 1

        # Write to CSV
        csv_writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            args.label,
            args.batch_size,
            i,
            f"{latency:.6f}",
            str(success)
        ])
        csv_file.flush()

        logger.info(f"  Iteration {i+1}/{args.iterations}: {latency:.3f}s (success={success})")

    csv_file.close()

    # Compute statistics
    if latencies:
        percentiles = compute_percentiles(latencies)
        avg_latency = np.mean(latencies)
        throughput = 1.0 / avg_latency if avg_latency > 0 else 0.0

        # Print summary
        print("\n" + "="*60)
        print(f"BENCHMARK SUMMARY: {args.label}")
        print("="*60)
        print(f"Label:              {args.label}")
        print(f"Batch Size:         {args.batch_size}")
        print(f"Iterations:         {args.iterations}")
        print(f"Successful:         {len(latencies)}")
        print(f"Failed:             {error_count}")
        print(f"Error Rate:         {error_count / args.iterations * 100:.1f}%")
        print(f"---")
        print(f"p50 Latency:        {percentiles['p50']:.3f} seconds")
        print(f"p90 Latency:        {percentiles['p90']:.3f} seconds")
        print(f"p95 Latency:        {percentiles['p95']:.3f} seconds")
        print(f"Average Latency:    {avg_latency:.3f} seconds")
        print(f"Throughput:         {throughput:.3f} images/second")
        print("="*60)
        print(f"Results saved to: {args.output_csv}")
        print("="*60 + "\n")
    else:
        logger.error("All inference attempts failed. No statistics to report.")
        sys.exit(1)


if __name__ == "__main__":
    main()
