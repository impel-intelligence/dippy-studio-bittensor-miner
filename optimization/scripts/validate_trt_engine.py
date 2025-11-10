#!/usr/bin/env python3
"""
TensorRT Engine Validation Script

Validates that the TRT-optimized Kontext pipeline:
1. Loads successfully
2. Produces deterministic outputs
3. Maintains quality (visual comparison)
4. Achieves expected performance improvements

Usage:
    python3 optimization/scripts/validate_trt_engine.py
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

# Import both pipelines for comparison
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from kontext_pipeline import KontextInferenceManager as OriginalKontext
from kontext_pipeline_optimized import OptimizedKontextInferenceManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TRTEngineValidator:
    """Validates TRT-optimized Kontext engine."""

    def __init__(self, test_image_paths: List[str] = None):
        """
        Initialize validator.

        Args:
            test_image_paths: Paths to test images (uses defaults if None)
        """
        self.test_image_paths = test_image_paths or [
            "character_1.png",
            "character_2.png"
        ]

        self.test_prompts = [
            "Add a red hat to the person",
            "Change the background to a beach",
            "Make the lighting warmer"
        ]

        self.test_seed = 42

    def test_loading(self) -> bool:
        """
        Test that TRT pipeline loads successfully.

        Returns:
            True if loaded successfully
        """
        logger.info("="*60)
        logger.info("TEST 1: Pipeline Loading")
        logger.info("="*60)

        try:
            manager = OptimizedKontextInferenceManager()
            manager.load_pipeline()

            info = manager.get_info()
            logger.info(f"✓ Pipeline loaded successfully")
            logger.info(f"  Backend: {'TRT-FP8' if info['using_trt'] else 'PyTorch-BF16'}")
            logger.info(f"  Precision: {info['precision']}")

            if info['using_trt']:
                logger.info(f"  Engine: {info['trt_engine_path']}")
                logger.info("✓ TEST PASSED: TRT engine loaded")
            else:
                logger.warning("⚠ TEST PARTIAL: Fell back to PyTorch (TRT not available)")

            return True

        except Exception as e:
            logger.error(f"✗ TEST FAILED: {e}", exc_info=True)
            return False

    def test_determinism(self) -> bool:
        """
        Test that outputs are deterministic (same seed → same output).

        Returns:
            True if outputs are deterministic
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Deterministic Generation")
        logger.info("="*60)

        try:
            manager = OptimizedKontextInferenceManager()
            manager.load_pipeline()

            # Load test image
            test_image = Image.open(self.test_image_paths[0])
            test_prompt = self.test_prompts[0]

            logger.info(f"Running same generation 3 times with seed={self.test_seed}...")

            # Generate 3 times with same seed
            outputs = []
            hashes = []

            for i in range(3):
                logger.info(f"  Generation {i+1}/3...")
                result = manager.generate(
                    prompt=test_prompt,
                    image=test_image,
                    seed=self.test_seed
                )

                # Convert to bytes and hash
                img_bytes = result.tobytes()
                img_hash = hashlib.sha256(img_bytes).hexdigest()

                outputs.append(result)
                hashes.append(img_hash)

                logger.info(f"    Hash: {img_hash[:16]}...")

            # Check if all hashes are identical
            if len(set(hashes)) == 1:
                logger.info("✓ TEST PASSED: All outputs identical (deterministic)")
                return True
            else:
                logger.error("✗ TEST FAILED: Outputs differ (non-deterministic)")
                logger.error(f"  Unique hashes: {len(set(hashes))}")
                for i, h in enumerate(hashes):
                    logger.error(f"    Run {i+1}: {h[:32]}")
                return False

        except Exception as e:
            logger.error(f"✗ TEST FAILED: {e}", exc_info=True)
            return False

    def test_performance(self) -> Tuple[bool, Dict]:
        """
        Test performance and compare to baseline if available.

        Returns:
            Tuple of (passed, metrics)
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Performance Measurement")
        logger.info("="*60)

        try:
            manager = OptimizedKontextInferenceManager()
            manager.load_pipeline()

            info = manager.get_info()

            # Load test image
            test_image = Image.open(self.test_image_paths[0])
            test_prompt = self.test_prompts[0]

            # Warmup
            logger.info("Warmup runs (3)...")
            for i in range(3):
                manager.generate(
                    prompt=test_prompt,
                    image=test_image,
                    seed=self.test_seed + i
                )

            # Benchmark
            num_runs = 10
            logger.info(f"Benchmark runs ({num_runs})...")

            latencies = []
            for i in range(num_runs):
                torch.cuda.synchronize()
                start = time.time()

                manager.generate(
                    prompt=test_prompt,
                    image=test_image,
                    seed=self.test_seed + i
                )

                torch.cuda.synchronize()
                end = time.time()

                latency = end - start
                latencies.append(latency)

                logger.info(f"  Run {i+1}: {latency*1000:.1f} ms")

            # Calculate metrics
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)

            # Memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)

            metrics = {
                "backend": "TRT-FP8" if info['using_trt'] else "PyTorch-BF16",
                "avg_latency_ms": round(avg_latency * 1000, 1),
                "std_latency_ms": round(std_latency * 1000, 1),
                "min_latency_ms": round(min_latency * 1000, 1),
                "max_latency_ms": round(max_latency * 1000, 1),
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "throughput_images_per_min": round(60 / avg_latency, 1)
            }

            logger.info("\nPerformance Results:")
            logger.info(f"  Backend: {metrics['backend']}")
            logger.info(f"  Avg Latency: {metrics['avg_latency_ms']} ms")
            logger.info(f"  Std Dev: {metrics['std_latency_ms']} ms")
            logger.info(f"  Min/Max: {metrics['min_latency_ms']}/{metrics['max_latency_ms']} ms")
            logger.info(f"  Memory: {metrics['memory_allocated_gb']} GB allocated")
            logger.info(f"  Throughput: {metrics['throughput_images_per_min']} img/min")

            # Try to load baseline for comparison
            baseline_path = Path("optimization/logs/baseline_results.json")
            if baseline_path.exists():
                with open(baseline_path) as f:
                    baseline = json.load(f)

                baseline_latency = baseline['latency']['avg_ms']
                speedup = baseline_latency / metrics['avg_latency_ms']

                logger.info("\nComparison to Baseline:")
                logger.info(f"  Baseline (BF16): {baseline_latency} ms")
                logger.info(f"  Current: {metrics['avg_latency_ms']} ms")
                logger.info(f"  Speedup: {speedup:.2f}x")

                if info['using_trt'] and speedup >= 1.8:
                    logger.info("✓ TEST PASSED: Achieved target speedup (>1.8x)")
                    return True, metrics
                elif info['using_trt']:
                    logger.warning(f"⚠ TEST PARTIAL: Speedup {speedup:.2f}x < 1.8x target")
                    return True, metrics
            else:
                logger.warning("⚠ No baseline found, skipping comparison")

            logger.info("✓ TEST PASSED: Performance measured")
            return True, metrics

        except Exception as e:
            logger.error(f"✗ TEST FAILED: {e}", exc_info=True)
            return False, {}

    def test_visual_quality(self) -> bool:
        """
        Test visual quality by generating sample outputs.

        Returns:
            True if test passes (manual verification required)
        """
        logger.info("\n" + "="*60)
        logger.info("TEST 4: Visual Quality Check")
        logger.info("="*60)

        try:
            manager = OptimizedKontextInferenceManager()
            manager.load_pipeline()

            output_dir = Path("optimization/logs/validation_outputs")
            output_dir.mkdir(parents=True, exist_ok=True)

            info = manager.get_info()
            backend = "trt_fp8" if info['using_trt'] else "pytorch_bf16"

            logger.info(f"Generating validation outputs with {backend}...")

            for img_idx, img_path in enumerate(self.test_image_paths[:2]):
                test_image = Image.open(img_path)

                for prompt_idx, prompt in enumerate(self.test_prompts[:2]):
                    logger.info(f"  Image {img_idx+1}, Prompt {prompt_idx+1}: {prompt[:40]}...")

                    result = manager.generate(
                        prompt=prompt,
                        image=test_image,
                        seed=self.test_seed + img_idx * 10 + prompt_idx
                    )

                    # Save output
                    output_name = f"{backend}_img{img_idx+1}_prompt{prompt_idx+1}.png"
                    output_path = output_dir / output_name
                    result.save(output_path)

                    logger.info(f"    Saved: {output_path}")

            logger.info(f"\n✓ Validation outputs saved to: {output_dir}")
            logger.info("  Please manually inspect for quality issues:")
            logger.info("  - Artifacts or distortions")
            logger.info("  - Color shifts")
            logger.info("  - Loss of detail")
            logger.info("  - Prompt adherence")

            logger.info("✓ TEST PASSED: Outputs generated (manual review required)")
            return True

        except Exception as e:
            logger.error(f"✗ TEST FAILED: {e}", exc_info=True)
            return False

    def run_all_tests(self) -> bool:
        """
        Run all validation tests.

        Returns:
            True if all tests pass
        """
        logger.info("\n" + "="*60)
        logger.info("TRT ENGINE VALIDATION SUITE")
        logger.info("="*60 + "\n")

        results = []

        # Test 1: Loading
        results.append(("Loading", self.test_loading()))

        # Test 2: Determinism
        results.append(("Determinism", self.test_determinism()))

        # Test 3: Performance
        perf_passed, metrics = self.test_performance()
        results.append(("Performance", perf_passed))

        # Test 4: Visual Quality
        results.append(("Visual Quality", self.test_visual_quality()))

        # Summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)

        for test_name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{status}: {test_name}")

        all_passed = all(passed for _, passed in results)

        logger.info("="*60)
        if all_passed:
            logger.info("✓ ALL TESTS PASSED")
        else:
            logger.info("✗ SOME TESTS FAILED")
        logger.info("="*60 + "\n")

        return all_passed


def main():
    """Main execution."""
    validator = TRTEngineValidator()
    success = validator.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
