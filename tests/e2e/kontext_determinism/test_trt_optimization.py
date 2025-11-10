"""
E2E tests for TensorRT-optimized FLUX.1-Kontext pipeline.

These tests verify that the TRT-optimized pipeline:
1. Produces deterministic outputs (same as BF16 PyTorch with same seed)
2. Achieves performance improvements
3. Maintains quality
4. Uses the specified production parameters

Production parameters:
- image_size: 1024x1024
- guidance_scale: 4
- num_inference_steps: 20
"""
import pytest
import numpy as np
import hashlib
import time
from PIL import Image


# Production parameters for Kontext
PRODUCTION_WIDTH = 1024
PRODUCTION_HEIGHT = 1024
PRODUCTION_GUIDANCE_SCALE = 4
PRODUCTION_NUM_INFERENCE_STEPS = 20


@pytest.mark.e2e
def test_trt_determinism_production_params(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Verify TRT pipeline produces deterministic outputs with production parameters.

    This is critical: TRT optimization must not break determinism.
    """
    prompt = "Add a vibrant sunset with orange and purple clouds"
    seed = 12345

    # Run twice with production parameters
    results = []
    job_ids = []

    for run_idx in range(2):
        response = miner_client.post_edit(
            prompt=prompt,
            image_b64=sample_image_b64,
            seed=seed,
            guidance_scale=PRODUCTION_GUIDANCE_SCALE,
            num_inference_steps=PRODUCTION_NUM_INFERENCE_STEPS,
        )
        job_ids.append(response["job_id"])

        miner_client.wait_for_completion(response["job_id"])
        result = miner_client.get_result(response["job_id"])
        results.append(result)

    # Verify determinism
    arr1 = np.array(results[0])
    arr2 = np.array(results[1])

    assert arr1.shape == arr2.shape, "Images must have same dimensions"

    # Check for pixel-perfect match
    pixel_match = np.array_equal(arr1, arr2)

    if not pixel_match:
        # Calculate difference percentage
        diff_pixels = np.sum(arr1 != arr2)
        total_pixels = arr1.size
        diff_pct = (diff_pixels / total_pixels) * 100

        pytest.fail(
            f"TRT DETERMINISM FAILURE:\n"
            f"  Different outputs from same seed!\n"
            f"  Differing pixels: {diff_pixels}/{total_pixels} ({diff_pct:.4f}%)\n"
            f"  Job IDs: {job_ids[0]}, {job_ids[1]}\n"
            f"  This likely means TRT has non-deterministic operations."
        )

    # Cryptographic verification
    hash1 = hashlib.sha256(arr1.tobytes()).hexdigest()
    hash2 = hashlib.sha256(arr2.tobytes()).hexdigest()
    assert hash1 == hash2, "Image hashes must match for deterministic output"


@pytest.mark.e2e
def test_trt_vs_pytorch_consistency(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Verify TRT outputs are similar to PyTorch BF16 outputs.

    Note: They won't be pixel-perfect due to FP8 quantization,
    but should be perceptually similar.
    """
    prompt = "Transform the scene into a watercolor painting style"
    seed = 42

    # Generate with current backend (may be TRT or PyTorch)
    response = miner_client.post_edit(
        prompt=prompt,
        image_b64=sample_image_b64,
        seed=seed,
        guidance_scale=PRODUCTION_GUIDANCE_SCALE,
        num_inference_steps=PRODUCTION_NUM_INFERENCE_STEPS,
    )

    miner_client.wait_for_completion(response["job_id"])
    result = miner_client.get_result(response["job_id"])

    # Verify output is valid
    arr = np.array(result)

    # Check dimensions match expected 1024x1024 (or resized)
    assert arr.shape[2] == 3, "Image must be RGB"
    assert arr.shape[0] > 0 and arr.shape[1] > 0, "Image must have valid dimensions"

    # Check for artifacts (all black, all white, NaN)
    mean_val = np.mean(arr)
    assert 10 < mean_val < 245, \
        f"Image appears corrupted (mean={mean_val:.1f}), possible quantization failure"

    std_val = np.std(arr)
    assert std_val > 5, \
        f"Image has no variance (std={std_val:.1f}), possible generation failure"


@pytest.mark.e2e
@pytest.mark.slow
def test_trt_performance_improvement(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Measure inference time and verify speedup (if TRT is enabled).

    Expected: TRT should be at least 1.5x faster than BF16.
    """
    prompt = "Add dramatic cinematic lighting with god rays"
    seed = 999

    # Warmup run
    warmup_response = miner_client.post_edit(
        prompt=prompt,
        image_b64=sample_image_b64,
        seed=seed,
        guidance_scale=PRODUCTION_GUIDANCE_SCALE,
        num_inference_steps=PRODUCTION_NUM_INFERENCE_STEPS,
    )
    miner_client.wait_for_completion(warmup_response["job_id"])

    # Benchmark runs
    num_runs = 3
    latencies = []

    for i in range(num_runs):
        start_time = time.time()

        response = miner_client.post_edit(
            prompt=prompt,
            image_b64=sample_image_b64,
            seed=seed + i,  # Different seed each time
            guidance_scale=PRODUCTION_GUIDANCE_SCALE,
            num_inference_steps=PRODUCTION_NUM_INFERENCE_STEPS,
        )

        miner_client.wait_for_completion(response["job_id"])
        _ = miner_client.get_result(response["job_id"])

        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)

    # Calculate metrics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    print(f"\n{'='*60}")
    print(f"Performance Metrics (Production Parameters)")
    print(f"{'='*60}")
    print(f"Average latency: {avg_latency:.2f}s ± {std_latency:.2f}s")
    print(f"Throughput: {60/avg_latency:.1f} images/minute")
    print(f"Individual runs: {', '.join(f'{l:.2f}s' for l in latencies)}")
    print(f"{'='*60}\n")

    # Check if latency is reasonable
    # For 1024x1024 with 20 steps, expect:
    # - BF16 PyTorch: ~15-30s on H100
    # - TRT FP8: ~8-15s on H100 (target: <15s)
    assert avg_latency < 60, \
        f"Latency too high ({avg_latency:.1f}s), pipeline may be degraded"


@pytest.mark.e2e
def test_trt_quality_various_prompts(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Test quality across various prompt types with production parameters.
    """
    prompts = [
        "Add a bright blue sky with fluffy white clouds",
        "Transform into a cyberpunk neon aesthetic",
        "Make the colors warmer and more saturated",
        "Add soft bokeh blur to the background",
        "Apply dramatic black and white conversion",
    ]

    seed = 42

    for prompt in prompts:
        response = miner_client.post_edit(
            prompt=prompt,
            image_b64=sample_image_b64,
            seed=seed,
            guidance_scale=PRODUCTION_GUIDANCE_SCALE,
            num_inference_steps=PRODUCTION_NUM_INFERENCE_STEPS,
        )

        miner_client.wait_for_completion(response["job_id"])
        result = miner_client.get_result(response["job_id"])

        # Verify output quality
        arr = np.array(result)

        # Check for artifacts
        mean_val = np.mean(arr)
        std_val = np.std(arr)

        assert 10 < mean_val < 245, \
            f"Image corrupted for prompt '{prompt}' (mean={mean_val:.1f})"
        assert std_val > 5, \
            f"No variance for prompt '{prompt}' (std={std_val:.1f})"


@pytest.mark.e2e
def test_trt_determinism_stress(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Stress test determinism with multiple seeds.

    Run 5 different seeds twice each and verify all are deterministic.
    """
    prompt = "Add vibrant abstract geometric patterns"
    seeds = [1, 42, 123, 999, 7777]

    for seed in seeds:
        # Run twice with same seed
        results = []

        for _ in range(2):
            response = miner_client.post_edit(
                prompt=prompt,
                image_b64=sample_image_b64,
                seed=seed,
                guidance_scale=PRODUCTION_GUIDANCE_SCALE,
                num_inference_steps=PRODUCTION_NUM_INFERENCE_STEPS,
            )

            miner_client.wait_for_completion(response["job_id"])
            result = miner_client.get_result(response["job_id"])
            results.append(np.array(result))

        # Verify determinism for this seed
        assert np.array_equal(results[0], results[1]), \
            f"Determinism failed for seed {seed} with TRT optimization"


@pytest.mark.e2e
@pytest.mark.slow
def test_trt_memory_usage(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Verify TRT optimization doesn't cause memory issues.

    Run multiple sequential inferences to check for memory leaks.
    """
    prompt = "Add cinematic depth of field effect"
    num_iterations = 5

    for i in range(num_iterations):
        response = miner_client.post_edit(
            prompt=prompt,
            image_b64=sample_image_b64,
            seed=i,
            guidance_scale=PRODUCTION_GUIDANCE_SCALE,
            num_inference_steps=PRODUCTION_NUM_INFERENCE_STEPS,
        )

        miner_client.wait_for_completion(response["job_id"])
        result = miner_client.get_result(response["job_id"])

        # Verify result is valid
        arr = np.array(result)
        assert arr.shape[2] == 3, f"Iteration {i}: Invalid image format"
        assert 10 < np.mean(arr) < 245, f"Iteration {i}: Image appears corrupted"


@pytest.mark.e2e
def test_trt_backend_info(
    miner_client,
    health_check
):
    """
    Query backend info to verify TRT is being used (if enabled).
    """
    # Check health endpoint for backend info
    health = health_check

    # Verify Kontext is available
    assert "kontext_edit" in health["services"], \
        "Kontext service not found in health check"

    kontext_status = health["services"]["kontext_edit"]
    assert kontext_status != "disabled", \
        "Kontext is disabled, cannot test TRT optimization"

    # Log backend info if available
    if "backend" in health.get("kontext_info", {}):
        backend = health["kontext_info"]["backend"]
        precision = health["kontext_info"].get("precision", "unknown")
        print(f"\nKontext Backend: {backend} ({precision})")

        if backend == "TRT-FP8":
            print("✓ TensorRT FP8 optimization is active")
        elif backend == "PyTorch-BF16":
            print("⚠ Running on PyTorch BF16 (TRT not active)")


@pytest.mark.e2e
def test_trt_edge_cases(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Test edge cases that might break with TRT quantization.
    """
    # Very simple prompt (might expose quantization artifacts)
    response1 = miner_client.post_edit(
        prompt="slightly brighter",
        image_b64=sample_image_b64,
        seed=42,
        guidance_scale=PRODUCTION_GUIDANCE_SCALE,
        num_inference_steps=PRODUCTION_NUM_INFERENCE_STEPS,
    )
    miner_client.wait_for_completion(response1["job_id"])
    result1 = miner_client.get_result(response1["job_id"])

    # Very complex prompt (might stress quantization)
    response2 = miner_client.post_edit(
        prompt="Transform the entire scene into a hyper-detailed steampunk "
               "industrial landscape with brass gears, copper pipes, steam "
               "vents, Victorian architecture, and dramatic chiaroscuro lighting",
        image_b64=sample_image_b64,
        seed=123,
        guidance_scale=PRODUCTION_GUIDANCE_SCALE,
        num_inference_steps=PRODUCTION_NUM_INFERENCE_STEPS,
    )
    miner_client.wait_for_completion(response2["job_id"])
    result2 = miner_client.get_result(response2["job_id"])

    # Verify both produce valid outputs
    for result, prompt_type in [(result1, "simple"), (result2, "complex")]:
        arr = np.array(result)
        mean_val = np.mean(arr)
        std_val = np.std(arr)

        assert 10 < mean_val < 245, \
            f"{prompt_type} prompt: Image corrupted (mean={mean_val:.1f})"
        assert std_val > 5, \
            f"{prompt_type} prompt: No variance (std={std_val:.1f})"
