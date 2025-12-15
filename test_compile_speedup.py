#!/usr/bin/env python3
"""
Compare inference speed: original vs torch.compile()

Usage:
    python test_compile_speedup.py
"""
import time
from PIL import Image

# Test both implementations
from kontext_pipeline import KontextInferenceManager
from kontext_pipeline_fast import KontextFastInferenceManager


# Production parameters (must match fal.ai prod query)
PROD_SEED = 618854
PROD_GUIDANCE_SCALE = 4
PROD_NUM_INFERENCE_STEPS = 20
PROD_PROMPT = "Transform into a powerful fire mage with glowing orange eyes, flames dancing around the hands, and a flowing crimson cape billowing in magical wind"


def benchmark(manager, name: str, image: Image.Image, num_runs: int = 3):
    """Run multiple inferences and measure time."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    print(f"  seed={PROD_SEED}, guidance={PROD_GUIDANCE_SCALE}, steps={PROD_NUM_INFERENCE_STEPS}")

    times = []

    for i in range(num_runs):
        start = time.time()

        output = manager.generate(
            prompt=PROD_PROMPT,
            image=image,
            seed=PROD_SEED,
            guidance_scale=PROD_GUIDANCE_SCALE,
            num_inference_steps=PROD_NUM_INFERENCE_STEPS
        )

        elapsed = time.time() - start
        times.append(elapsed)

        print(f"  Run {i+1}: {elapsed:.2f}s")

        # Save first output for comparison
        if i == 0:
            output.save(f"output_{name.replace(' ', '_')}.png")

    # Stats (skip first run for compiled version as it includes compilation)
    if "Fast" in name and len(times) > 1:
        avg_time = sum(times[1:]) / len(times[1:])
        print(f"\n  Average (excluding first/compile run): {avg_time:.2f}s")
    else:
        avg_time = sum(times) / len(times)
        print(f"\n  Average: {avg_time:.2f}s")

    return times


def main():
    print("Loading test image...")
    image = Image.open("ninja.png")
    print(f"Image size: {image.size}")

    # Benchmark original
    print("\n" + "="*60)
    print("ORIGINAL IMPLEMENTATION (no compile)")
    print("="*60)
    original = KontextInferenceManager()
    original_times = benchmark(original, "Original", image, num_runs=2)

    # Clean up memory
    del original
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Benchmark compiled
    print("\n" + "="*60)
    print("FAST IMPLEMENTATION (torch.compile)")
    print("="*60)
    fast = KontextFastInferenceManager(compile_transformer=True)
    fast_times = benchmark(fast, "Fast (compiled)", image, num_runs=3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    original_avg = sum(original_times) / len(original_times)
    # For compiled, skip first run (includes compilation)
    fast_avg = sum(fast_times[1:]) / len(fast_times[1:]) if len(fast_times) > 1 else fast_times[0]

    print(f"Original average:    {original_avg:.2f}s")
    print(f"Compiled average:    {fast_avg:.2f}s (excluding compile run)")
    print(f"Speedup:             {original_avg/fast_avg:.2f}x")

    print("\nOutputs saved to:")
    print("  - output_Original.png")
    print("  - output_Fast_(compiled).png")
    print("\nCompare these images to verify determinism is preserved!")


if __name__ == "__main__":
    main()
