"""End-to-end test validating the async inference callback protocol."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import aiohttp
import pytest

from .callback_server import CALLBACK_SECRET_HEADER
from .client import poll_job_status, submit_inference_job, wait_for_callback


TEST_PROMPT = "A cute anime girl with blue hair"


@pytest.mark.asyncio
async def test_async_inference_protocol(callback_server, callback_base_url):
    """Submit an async inference request and wait for the callback payload."""

    miner_base_url = os.getenv("ASYNC_MINER_URL", "http://localhost:8091").rstrip("/")
    callback_secret = os.getenv("ASYNC_CALLBACK_SECRET", "test-secret")
    callback_url = f"{callback_base_url}/callback"

    expiry = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    payload = {
        "prompt": TEST_PROMPT,
        "callback_url": callback_url,
        "callback_secret": callback_secret,
        "expiry": expiry,
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 20,
        "seed": 42,
    }

    async with aiohttp.ClientSession() as session:
        # Ensure the callback server is reachable before submitting jobs.
        try:
            async with session.get(f"{callback_base_url}/callbacks") as response:
                if response.status != 200:
                    pytest.skip("Callback server did not respond with HTTP 200 during preflight check")
        except aiohttp.ClientError as exc:
            pytest.skip(f"Callback server unavailable: {exc}")

        # Quick connectivity check to the miner API to surface friendly errors.
        try:
            async with session.get(miner_base_url) as response:
                if response.status >= 500:
                    pytest.skip(f"Miner server unhealthy: {response.status}")
        except aiohttp.ClientError as exc:
            pytest.skip(f"Miner server unavailable: {exc}")

        submission = await submit_inference_job(session, miner_base_url, payload)
        job_id = submission.get("job_id")
        assert job_id, f"Inference submission did not return a job_id: {submission}"

        status_payload = await poll_job_status(session, miner_base_url, job_id)
        final_status = status_payload.get("status")
        assert final_status == "completed", f"Unexpected final status: {status_payload}"

        callback_meta = status_payload.get("callback", {})
        assert callback_meta.get("status") == "delivered", f"Unexpected callback status: {callback_meta}"
        status_code = callback_meta.get("status_code")
        assert status_code is not None and int(status_code) < 400, f"Callback HTTP failure: {callback_meta}"
        assert callback_meta.get("payload_status") == "completed", f"Unexpected payload status: {callback_meta}"

        callback_payload = await wait_for_callback(session, callback_base_url, job_id)
        assert callback_payload is not None, "Expected callback payload but none was received"
        assert callback_payload.get("job_id") == job_id
        assert callback_payload.get("status") == "completed"
        assert callback_payload.get("provided_secret") == callback_secret
        assert callback_payload.get("error") in (None, "")

        # Some miners upload the generated image file; verify presence when advertised.
        if callback_payload.get("has_image"):
            saved_path = callback_payload.get("saved_path")
            assert saved_path, "Callback indicates an image was stored but no path was provided"
