"""
Core determinism verification tests for FLUX.1-Kontext-dev.

These tests verify that the same seed produces identical outputs.
"""
import pytest
import numpy as np
import hashlib
from PIL import Image


@pytest.mark.e2e
def test_same_seed_produces_identical_output(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Critical determinism test: same seed must produce pixel-perfect identical output.

    This is the core requirement for deterministic inference.
    """
    prompt = "Add a bright red circle in the center"
    seed = 42

    # First generation
    response1 = miner_client.post_edit(
        prompt=prompt,
        image_b64=sample_image_b64,
        seed=seed,
        num_inference_steps=10,  # Faster for testing
        guidance_scale=2.5
    )
    job_id_1 = response1["job_id"]

    # Wait for completion
    miner_client.wait_for_completion(job_id_1)
    result1 = miner_client.get_result(job_id_1)

    # Second generation with same seed
    response2 = miner_client.post_edit(
        prompt=prompt,
        image_b64=sample_image_b64,
        seed=seed,
        num_inference_steps=10,
        guidance_scale=2.5
    )
    job_id_2 = response2["job_id"]

    miner_client.wait_for_completion(job_id_2)
    result2 = miner_client.get_result(job_id_2)

    # Convert to numpy arrays
    arr1 = np.array(result1)
    arr2 = np.array(result2)

    # Verify pixel-perfect match
    assert arr1.shape == arr2.shape, "Images must have same dimensions"
    assert np.array_equal(arr1, arr2), \
        "DETERMINISM FAILURE: Same seed produced different outputs"

    # Verify with cryptographic hash
    hash1 = hashlib.sha256(arr1.tobytes()).hexdigest()
    hash2 = hashlib.sha256(arr2.tobytes()).hexdigest()
    assert hash1 == hash2, "Image hashes must match for deterministic output"


@pytest.mark.e2e
def test_different_seeds_produce_different_outputs(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Verify that different seeds actually affect the output.

    This ensures the seed is being used correctly.
    """
    prompt = "Add colorful abstract shapes"

    # Generation with seed 42
    response1 = miner_client.post_edit(
        prompt=prompt,
        image_b64=sample_image_b64,
        seed=42,
        num_inference_steps=10
    )
    miner_client.wait_for_completion(response1["job_id"])
    result1 = miner_client.get_result(response1["job_id"])

    # Generation with seed 123
    response2 = miner_client.post_edit(
        prompt=prompt,
        image_b64=sample_image_b64,
        seed=123,
        num_inference_steps=10
    )
    miner_client.wait_for_completion(response2["job_id"])
    result2 = miner_client.get_result(response2["job_id"])

    # Convert to numpy
    arr1 = np.array(result1)
    arr2 = np.array(result2)

    # Should be different
    assert not np.array_equal(arr1, arr2), \
        "Different seeds should produce different outputs"

    # At least 1% of pixels should differ
    diff_ratio = np.mean(arr1 != arr2)
    assert diff_ratio > 0.01, \
        f"Only {diff_ratio*100:.2f}% pixels differ - seed may not be working"


@pytest.mark.e2e
def test_determinism_with_different_prompts(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Verify determinism holds across different prompts.
    """
    seed = 999

    prompts = [
        "Add a blue sky with clouds",
        "Make the colors more vibrant",
        "Add dramatic lighting"
    ]

    for prompt in prompts:
        # Run twice with same seed
        jobs = []
        for _ in range(2):
            response = miner_client.post_edit(
                prompt=prompt,
                image_b64=sample_image_b64,
                seed=seed,
                num_inference_steps=10
            )
            miner_client.wait_for_completion(response["job_id"])
            result = miner_client.get_result(response["job_id"])
            jobs.append(np.array(result))

        # Verify determinism for this prompt
        assert np.array_equal(jobs[0], jobs[1]), \
            f"Determinism failed for prompt: '{prompt}'"


@pytest.mark.e2e
@pytest.mark.slow
def test_determinism_with_varying_parameters(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Test determinism with different guidance_scale, steps, strength.
    """
    prompt = "Add warm sunset colors"
    seed = 777

    param_sets = [
        {"guidance_scale": 2.0, "num_inference_steps": 10, "strength": 0.5},
        {"guidance_scale": 3.5, "num_inference_steps": 20, "strength": 0.8},
        {"guidance_scale": 2.5, "num_inference_steps": 28, "strength": 1.0},
    ]

    for params in param_sets:
        # Run twice with same params and seed
        results = []
        for _ in range(2):
            response = miner_client.post_edit(
                prompt=prompt,
                image_b64=sample_image_b64,
                seed=seed,
                **params
            )
            miner_client.wait_for_completion(response["job_id"])
            result = miner_client.get_result(response["job_id"])
            results.append(np.array(result))

        # Verify determinism
        assert np.array_equal(results[0], results[1]), \
            f"Determinism failed for params: {params}"
