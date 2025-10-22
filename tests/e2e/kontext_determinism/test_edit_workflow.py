"""
E2E tests for edit workflow: chaining, reference jobs, error handling.
"""
import pytest
import base64
from io import BytesIO


@pytest.mark.e2e
def test_reference_job_chaining(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Test chaining edits: base → edit1 → edit2 → edit3
    """
    seed = 100

    # Edit 1: Add red tint
    response1 = miner_client.post_edit(
        prompt="Add a warm red tint",
        image_b64=sample_image_b64,
        seed=seed,
        num_inference_steps=10
    )
    miner_client.wait_for_completion(response1["job_id"])

    # Edit 2: Chain from edit 1 (with fallback)
    result1 = miner_client.get_result(response1["job_id"])
    buffer = BytesIO()
    result1.save(buffer, format="PNG")
    fallback_b64 = base64.b64encode(buffer.getvalue()).decode()

    response2 = miner_client.post_edit(
        prompt="Add blue highlights",
        reference_job_id=response1["job_id"],
        image_b64=fallback_b64,  # Fallback
        seed=seed + 1,
        num_inference_steps=10
    )
    miner_client.wait_for_completion(response2["job_id"])

    # Edit 3: Chain from edit 2
    result2 = miner_client.get_result(response2["job_id"])
    buffer = BytesIO()
    result2.save(buffer, format="PNG")
    fallback_b64_2 = base64.b64encode(buffer.getvalue()).decode()

    response3 = miner_client.post_edit(
        prompt="Increase contrast",
        reference_job_id=response2["job_id"],
        image_b64=fallback_b64_2,
        seed=seed + 2,
        num_inference_steps=10
    )
    status3 = miner_client.wait_for_completion(response3["job_id"])

    assert status3["status"] == "completed"
    assert "reference" in status3["request"]["image_source"]


@pytest.mark.e2e
def test_reference_missing_uses_fallback(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Test that missing reference_job_id falls back to image_b64.
    """
    response = miner_client.post_edit(
        prompt="Add purple glow",
        reference_job_id="definitely-does-not-exist",
        image_b64=sample_image_b64,  # Should use this
        seed=42,
        num_inference_steps=10
    )

    status = miner_client.wait_for_completion(response["job_id"])

    assert status["status"] == "completed"
    assert "fallback" in status["request"]["image_source"]


@pytest.mark.e2e
def test_missing_both_image_sources_fails(
    miner_client,
    health_check
):
    """
    Test that missing both reference_job_id and image_b64 fails.
    """
    import requests

    response = requests.post(
        f"{miner_client.base_url}/edit",
        json={
            "prompt": "This should fail",
            "seed": 42
            # No image_b64 or reference_job_id
        }
    )

    assert response.status_code == 400


@pytest.mark.e2e
def test_missing_seed_fails_validation(
    miner_client,
    sample_image_b64,
    health_check
):
    """
    Test that missing seed is rejected (Pydantic validation).
    """
    import requests

    response = requests.post(
        f"{miner_client.base_url}/edit",
        json={
            "prompt": "Missing seed",
            "image_b64": sample_image_b64
            # No seed
        }
    )

    assert response.status_code == 422  # Pydantic validation error
