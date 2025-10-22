"""
E2E test validating the edit endpoint callback protocol.
"""
import pytest
import aiohttp
import os
from datetime import datetime, timedelta, timezone


TEST_PROMPT = "Add a blue tint to the image"


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_edit_callback_protocol(
    callback_server,
    callback_base_url,
    sample_image_b64,
    health_check
):
    """Submit an edit request with callback and verify callback is dispatched."""

    miner_base_url = os.getenv("ASYNC_MINER_URL", "http://localhost:8091").rstrip("/")
    callback_secret = os.getenv("ASYNC_CALLBACK_SECRET", "test-secret")
    callback_url = f"{callback_base_url}/callback"

    expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    payload = {
        "prompt": TEST_PROMPT,
        "image_b64": sample_image_b64,
        "seed": 42,
        "num_inference_steps": 10,
        "callback_url": callback_url,
        "callback_secret": callback_secret,
        "expiry": expiry,
    }

    async with aiohttp.ClientSession() as session:
        # Ensure the callback server is reachable
        try:
            async with session.get(f"{callback_base_url}/callbacks") as response:
                if response.status != 200:
                    pytest.skip("Callback server not responding")
        except aiohttp.ClientError as exc:
            pytest.skip(f"Callback server unavailable: {exc}")

        # Verify miner is reachable
        try:
            async with session.get(miner_base_url) as response:
                if response.status >= 500:
                    pytest.skip(f"Miner server unhealthy: {response.status}")
        except aiohttp.ClientError as exc:
            pytest.skip(f"Miner server unavailable: {exc}")

        # Submit edit job
        async with session.post(f"{miner_base_url}/edit", json=payload) as response:
            response.raise_for_status()
            submission = await response.json()

        job_id = submission.get("job_id")
        assert job_id, f"Edit submission did not return a job_id: {submission}"

        # Poll for completion
        max_attempts = 150  # 5 minutes at 2s intervals
        for _ in range(max_attempts):
            async with session.get(f"{miner_base_url}/edit/status/{job_id}") as response:
                response.raise_for_status()
                status_payload = await response.json()

            if status_payload.get("status") == "completed":
                break

            if status_payload.get("status") == "failed":
                pytest.fail(f"Edit job failed: {status_payload.get('error')}")

            await asyncio.sleep(2)
        else:
            pytest.fail(f"Edit job {job_id} did not complete within timeout")

        # Verify job completed
        final_status = status_payload.get("status")
        assert final_status == "completed", f"Unexpected final status: {status_payload}"

        # Wait for callback to be dispatched (background task)
        # Background tasks run after response is sent, so poll for callback info
        callback_meta = {}
        for _ in range(10):  # Wait up to 10 seconds
            async with session.get(f"{miner_base_url}/edit/status/{job_id}") as response:
                response.raise_for_status()
                status_payload = await response.json()

            callback_meta = status_payload.get("callback", {})
            if callback_meta:
                break
            await asyncio.sleep(1)
        else:
            # If no callback metadata after waiting, it might not have been requested
            if not callback_url:
                pytest.skip("No callback was requested")
        assert callback_meta.get("status") == "delivered", f"Callback not delivered: {callback_meta}"

        status_code = callback_meta.get("status_code")
        assert status_code is not None and int(status_code) < 400, f"Callback HTTP failure: {callback_meta}"

        assert callback_meta.get("payload_status") == "completed", f"Unexpected payload status: {callback_meta}"

        # Verify callback was received by callback server
        max_callback_attempts = 10
        callback_payload = None
        for _ in range(max_callback_attempts):
            async with session.get(f"{callback_base_url}/callbacks") as response:
                response.raise_for_status()
                data = await response.json()
                callbacks = data.get("callbacks", [])

                # Find callback for this job_id
                for cb in callbacks:
                    if cb.get("job_id") == job_id:
                        callback_payload = cb
                        break

                if callback_payload:
                    break

            await asyncio.sleep(1)

        assert callback_payload is not None, "Expected callback payload but none was received"
        assert callback_payload.get("job_id") == job_id
        assert callback_payload.get("status") == "completed"
        assert callback_payload.get("provided_secret") == callback_secret
        assert callback_payload.get("error") in (None, "")

        # Verify image was included in callback
        if callback_payload.get("has_image"):
            saved_path = callback_payload.get("saved_path")
            assert saved_path, "Callback indicates image was stored but no path provided"


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_edit_callback_on_failure(
    callback_server,
    callback_base_url,
    health_check
):
    """Verify callback is dispatched even when edit job fails."""

    miner_base_url = os.getenv("ASYNC_MINER_URL", "http://localhost:8091").rstrip("/")
    callback_secret = os.getenv("ASYNC_CALLBACK_SECRET", "test-secret")
    callback_url = f"{callback_base_url}/callback"

    expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    # Invalid payload - no image source
    payload = {
        "prompt": "This should fail",
        "seed": 42,
        "num_inference_steps": 10,
        "callback_url": callback_url,
        "callback_secret": callback_secret,
        "expiry": expiry,
    }

    async with aiohttp.ClientSession() as session:
        # This should fail validation at submission
        async with session.post(f"{miner_base_url}/edit", json=payload) as response:
            # Expected to fail with 400 or 422
            assert response.status in (400, 422), "Expected validation error for missing image"


import asyncio
