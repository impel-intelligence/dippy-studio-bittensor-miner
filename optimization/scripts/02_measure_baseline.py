#!/usr/bin/env python3
"""
Baseline Performance Measurement Script

Measures the current (BF16) performance of FLUX Kontext pipeline to establish
a baseline for comparing against optimized versions.

Usage:
    python3 optimization/scripts/02_measure_baseline.py
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import psutil
import torch
from PIL import Image

# Import your existing Kontext pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from kontext_pipeline import KontextInferenceManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselinePerformanceMeasurer:
    """Measures baseline performance of BF16 Kontext pipeline."""

    def __init__(
        self,
        model_path: str = "black-forest-labs/FLUX.1-Kontext-dev",
        num_warmup: int = 3,
        num_benchmark: int = 10
    ):
        """
        Initialize performance measurer.

        Args:
            model_path: HuggingFace model path
            num_warmup: Number of warmup iterations
            num_benchmark: Number of benchmark iterations
        """
        self.model_path = model_path
        self.num_warmup = num_warmup
        self.num_benchmark = num_benchmark

        self.pipeline_manager = None
        self.results = {}

    def setup_pipeline(self):
        """Load and initialize the Kontext pipeline."""
        logger.info(f"Loading Kontext pipeline from {self.model_path}...")

        self.pipeline_manager = KontextInferenceManager(model_path=self.model_path)
        self.pipeline_manager.load_pipeline()

        logger.info("✓ Pipeline loaded successfully")

    def measure_memory_usage(self) -> Dict[str, float]:
        """
        Measure current GPU memory usage.

        Returns:
            Dictionary with memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        # Get memory stats
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)

        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "max_allocated_gb": round(max_allocated, 2),
            "total_gb": round(total_memory, 2),
            "utilization_pct": round((allocated / total_memory) * 100, 2)
        }

    def run_single_inference(
        self,
        prompt: str,
        image_path: str,
        seed: int
    ) -> Dict[str, float]:
        """
        Run single inference and measure performance.

        Args:
            prompt: Edit prompt
            image_path: Path to input image
            seed: Random seed

        Returns:
            Dictionary with timing and memory metrics
        """
        # Load image
        image = Image.open(image_path)

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # Measure inference time
        start_time = time.time()

        result_image = self.pipeline_manager.generate(
            prompt=prompt,
            image=image,
            seed=seed,
            guidance_scale=2.5,
            num_inference_steps=28
        )

        end_time = time.time()

        # Calculate metrics
        latency = end_time - start_time
        memory_stats = self.measure_memory_usage()

        return {
            "latency_sec": round(latency, 3),
            "latency_ms": round(latency * 1000, 1),
            "memory": memory_stats
        }

    def run_warmup(self, test_prompt: str, test_image: str):
        """
        Run warmup iterations to stabilize performance.

        Args:
            test_prompt: Prompt for warmup
            test_image: Image path for warmup
        """
        logger.info(f"Running {self.num_warmup} warmup iterations...")

        for i in range(self.num_warmup):
            logger.info(f"Warmup {i+1}/{self.num_warmup}...")
            self.run_single_inference(
                prompt=test_prompt,
                image_path=test_image,
                seed=42 + i
            )

        logger.info("✓ Warmup complete\n")

    def run_benchmark(
        self,
        calibration_dataset_path: str = "optimization/calibration/data/calibration_dataset.json"
    ) -> Dict:
        """
        Run benchmark on calibration dataset.

        Args:
            calibration_dataset_path: Path to calibration dataset JSON

        Returns:
            Benchmark results dictionary
        """
        # Load calibration dataset
        with open(calibration_dataset_path) as f:
            dataset = json.load(f)

        # Use first N samples for benchmark
        benchmark_samples = dataset[:self.num_benchmark]

        logger.info(f"Running benchmark on {len(benchmark_samples)} samples...")
        logger.info("="*60)

        latencies = []
        peak_memory = 0

        for i, sample in enumerate(benchmark_samples):
            logger.info(f"Benchmark {i+1}/{len(benchmark_samples)}: {sample['prompt'][:50]}...")

            metrics = self.run_single_inference(
                prompt=sample['prompt'],
                image_path=sample['image_path'],
                seed=sample['seed']
            )

            latencies.append(metrics['latency_sec'])
            peak_memory = max(peak_memory, metrics['memory']['max_allocated_gb'])

            logger.info(f"  Latency: {metrics['latency_ms']} ms")

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        median_latency = sorted(latencies)[len(latencies) // 2]

        # Get final memory usage
        final_memory = self.measure_memory_usage()

        results = {
            "model": "FLUX.1-Kontext-dev",
            "precision": "BF16",
            "optimization": "None (Baseline)",
            "num_samples": len(benchmark_samples),
            "latency": {
                "avg_sec": round(avg_latency, 3),
                "avg_ms": round(avg_latency * 1000, 1),
                "min_sec": round(min_latency, 3),
                "max_sec": round(max_latency, 3),
                "median_sec": round(median_latency, 3),
                "all_latencies_sec": [round(l, 3) for l in latencies]
            },
            "memory": {
                "peak_allocated_gb": round(peak_memory, 2),
                "final_state": final_memory
            },
            "throughput": {
                "images_per_minute": round(60 / avg_latency, 2),
                "avg_time_per_step_ms": round((avg_latency / 28) * 1000, 2)  # 28 inference steps
            }
        }

        return results

    def print_results(self, results: Dict):
        """
        Print benchmark results in a formatted way.

        Args:
            results: Results dictionary
        """
        logger.info("\n" + "="*60)
        logger.info("BASELINE PERFORMANCE RESULTS")
        logger.info("="*60)
        logger.info(f"Model: {results['model']}")
        logger.info(f"Precision: {results['precision']}")
        logger.info(f"Optimization: {results['optimization']}")
        logger.info("-"*60)
        logger.info("LATENCY:")
        logger.info(f"  Average: {results['latency']['avg_ms']} ms ({results['latency']['avg_sec']} sec)")
        logger.info(f"  Median:  {results['latency']['median_sec']} sec")
        logger.info(f"  Min:     {results['latency']['min_sec']} sec")
        logger.info(f"  Max:     {results['latency']['max_sec']} sec")
        logger.info("-"*60)
        logger.info("MEMORY:")
        logger.info(f"  Peak Allocated: {results['memory']['peak_allocated_gb']} GB")
        logger.info(f"  GPU Utilization: {results['memory']['final_state']['utilization_pct']}%")
        logger.info("-"*60)
        logger.info("THROUGHPUT:")
        logger.info(f"  Images/minute: {results['throughput']['images_per_minute']}")
        logger.info(f"  Time per step: {results['throughput']['avg_time_per_step_ms']} ms")
        logger.info("="*60 + "\n")

    def save_results(self, results: Dict, output_path: str = "optimization/logs/baseline_results.json"):
        """
        Save results to JSON file.

        Args:
            results: Results dictionary
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Results saved to {output_file}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("FLUX Kontext Baseline Performance Measurement")
    logger.info("="*60 + "\n")

    # Create measurer
    measurer = BaselinePerformanceMeasurer(
        model_path="black-forest-labs/FLUX.1-Kontext-dev",
        num_warmup=3,
        num_benchmark=10
    )

    # Setup pipeline
    measurer.setup_pipeline()

    # Run warmup
    measurer.run_warmup(
        test_prompt="Add a red hat to the person",
        test_image="character_1.png"
    )

    # Run benchmark
    results = measurer.run_benchmark()

    # Print results
    measurer.print_results(results)

    # Save results
    measurer.save_results(results)

    logger.info("✓ Baseline measurement complete!")
    logger.info("Next step: Create FP8 quantization config (03_create_fp8_config.py)")


if __name__ == "__main__":
    main()
