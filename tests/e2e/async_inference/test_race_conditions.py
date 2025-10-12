"""Stress test to reproduce race conditions causing incomplete images."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import aiohttp
import pytest
import numpy as np

from .callback_server import CALLBACK_SECRET_HEADER
from .client import submit_inference_job, poll_job_status, wait_for_callback
from .image_validator import ImageValidator, detect_partial_render


# Test LoRA paths - adjust these to your actual test LoRAs
TEST_LORAS = [
    "",  # Base model (no LoRA)
    # Add your test LoRA paths here if available
    # "path/to/test_lora_1.safetensors",
    # "path/to/test_lora_2.safetensors",
]

TEST_PROMPTS = [
    "A cute anime girl with blue hair",
    "A serene mountain landscape at sunset",
    "A futuristic cyberpunk city with neon lights",
    "A medieval castle on a cliff overlooking the ocean",
    "A cozy coffee shop interior with warm lighting",
]


@pytest.mark.asyncio
async def test_rapid_concurrent_inference(callback_server, callback_base_url):
    """
    Stress test: Submit multiple concurrent LoRA inference requests rapidly.

    This test attempts to trigger race conditions #1, #2, #3:
    - File system race (file exists before write completes)
    - Missing CUDA sync (image saved before GPU ops finish)
    - Engine queue race (engine returned before save completes)
    """
    miner_base_url = os.getenv("ASYNC_MINER_URL", "http://localhost:8091").rstrip("/")
    callback_secret = os.getenv("ASYNC_CALLBACK_SECRET", "test-secret")
    callback_url = f"{callback_base_url}/callback"

    num_concurrent = int(os.getenv("STRESS_TEST_CONCURRENT", "10"))

    print(f"\n{'='*80}")
    print(f"STRESS TEST: Submitting {num_concurrent} concurrent inference requests")
    print(f"{'='*80}\n")

    validator = ImageValidator(
        expected_width=1024,
        expected_height=1024,
        max_blank_percentage=0.5,
        min_variance=100.0
    )

    async with aiohttp.ClientSession() as session:
        # Preflight checks
        try:
            async with session.get(f"{callback_base_url}/callbacks") as response:
                if response.status != 200:
                    pytest.skip("Callback server not available")
        except aiohttp.ClientError as exc:
            pytest.skip(f"Callback server unavailable: {exc}")

        try:
            async with session.get(miner_base_url) as response:
                if response.status >= 500:
                    pytest.skip(f"Miner server unhealthy: {response.status}")
        except aiohttp.ClientError as exc:
            pytest.skip(f"Miner server unavailable: {exc}")

        # Submit jobs concurrently
        expiry = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

        tasks = []
        for i in range(num_concurrent):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            lora_path = TEST_LORAS[i % len(TEST_LORAS)]

            payload = {
                "prompt": prompt,
                "callback_url": callback_url,
                "callback_secret": callback_secret,
                "expiry": expiry,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 20,
                "seed": 42 + i,  # Unique seed per job
                "lora_path": lora_path,
            }

            # Remove empty lora_path
            if not lora_path:
                payload.pop("lora_path", None)

            tasks.append(submit_inference_job(session, miner_base_url, payload))

        # Submit all at once
        print(f"Submitting {num_concurrent} jobs simultaneously...")
        submissions = await asyncio.gather(*tasks, return_exceptions=True)

        job_ids = []
        for i, result in enumerate(submissions):
            if isinstance(result, Exception):
                print(f"  ❌ Job {i} submission failed: {result}")
            else:
                job_id = result.get("job_id")
                if job_id:
                    job_ids.append(job_id)
                    print(f"  ✅ Job {i} submitted: {job_id}")
                else:
                    print(f"  ❌ Job {i} missing job_id: {result}")

        assert len(job_ids) > 0, "No jobs were successfully submitted"
        print(f"\nSuccessfully submitted {len(job_ids)} jobs\n")

        # Poll for completion
        print("Waiting for all jobs to complete...")
        poll_tasks = [poll_job_status(session, miner_base_url, job_id) for job_id in job_ids]
        statuses = await asyncio.gather(*poll_tasks, return_exceptions=True)

        # Analyze results
        results = []
        for i, (job_id, status_result) in enumerate(zip(job_ids, statuses)):
            if isinstance(status_result, Exception):
                results.append({
                    "job_id": job_id,
                    "status": "error",
                    "error": str(status_result),
                    "validation": None
                })
                print(f"  ❌ Job {job_id}: {status_result}")
                continue

            final_status = status_result.get("status")

            # Get callback data
            try:
                callback_payload = await wait_for_callback(session, callback_base_url, job_id)
            except Exception as e:
                callback_payload = None
                print(f"  ⚠️  Job {job_id}: No callback received ({e})")

            # Validate image if available
            validation_result = None
            if callback_payload and callback_payload.get("has_image"):
                saved_path = Path(callback_payload.get("saved_path"))
                if saved_path.exists():
                    validation_result = validator.validate(saved_path)

                    # Also check for partial render
                    is_partial, partial_reason = detect_partial_render(saved_path)

                    if validation_result.is_valid and not is_partial:
                        print(f"  ✅ Job {job_id}: Valid image ({validation_result.file_size} bytes, variance={validation_result.variance:.1f}, blank={validation_result.blank_percentage*100:.1f}%)")
                    elif is_partial:
                        print(f"  ❌ Job {job_id}: PARTIAL RENDER - {partial_reason}")
                        validation_result.is_valid = False
                        validation_result.reason = partial_reason
                    else:
                        print(f"  ❌ Job {job_id}: INVALID - {validation_result.reason}")
                else:
                    print(f"  ❌ Job {job_id}: Image file not found at {saved_path}")

            results.append({
                "job_id": job_id,
                "status": final_status,
                "validation": validation_result,
                "callback": callback_payload
            })

        # Summary
        print(f"\n{'='*80}")
        print("STRESS TEST RESULTS")
        print(f"{'='*80}\n")

        completed = sum(1 for r in results if r["status"] == "completed")
        valid_images = sum(1 for r in results if r["validation"] and r["validation"].is_valid)
        invalid_images = sum(1 for r in results if r["validation"] and not r["validation"].is_valid)
        no_images = sum(1 for r in results if not r.get("callback") or not r["callback"].get("has_image"))

        print(f"Total jobs: {len(job_ids)}")
        print(f"Completed: {completed}")
        print(f"Valid images: {valid_images}")
        print(f"Invalid/incomplete images: {invalid_images}")
        print(f"No images: {no_images}")
        print()

        if invalid_images > 0:
            print("INVALID IMAGES DETECTED:")
            for r in results:
                if r["validation"] and not r["validation"].is_valid:
                    print(f"  - {r['job_id']}: {r['validation'].reason}")
            print()

        # Fail test if any images are invalid
        assert invalid_images == 0, f"Found {invalid_images} invalid/incomplete images out of {len(job_ids)} total"


@pytest.mark.asyncio
async def test_rapid_sequential_different_loras(callback_server, callback_base_url):
    """
    Stress test: Submit LoRA inference requests rapidly in sequence.

    This test attempts to trigger race condition #3:
    - Engine queue race (next job starts while previous image still saving)
    """
    miner_base_url = os.getenv("ASYNC_MINER_URL", "http://localhost:8091").rstrip("/")
    callback_secret = os.getenv("ASYNC_CALLBACK_SECRET", "test-secret")
    callback_url = f"{callback_base_url}/callback"

    num_sequential = int(os.getenv("STRESS_TEST_SEQUENTIAL", "5"))

    print(f"\n{'='*80}")
    print(f"STRESS TEST: Submitting {num_sequential} sequential inference requests with minimal delay")
    print(f"{'='*80}\n")

    validator = ImageValidator()

    async with aiohttp.ClientSession() as session:
        # Preflight
        try:
            async with session.get(f"{callback_base_url}/callbacks") as response:
                if response.status != 200:
                    pytest.skip("Callback server not available")
        except aiohttp.ClientError:
            pytest.skip("Callback server unavailable")

        expiry = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

        job_ids = []
        for i in range(num_sequential):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            lora_path = TEST_LORAS[i % len(TEST_LORAS)]

            payload = {
                "prompt": prompt,
                "callback_url": callback_url,
                "callback_secret": callback_secret,
                "expiry": expiry,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 20,
                "seed": 100 + i,
                "lora_path": lora_path,
            }

            if not lora_path:
                payload.pop("lora_path", None)

            submission = await submit_inference_job(session, miner_base_url, payload)
            job_id = submission.get("job_id")
            if job_id:
                job_ids.append(job_id)
                print(f"  ✅ Submitted job {i+1}/{num_sequential}: {job_id}")

            # Very short delay to maximize race condition chance
            await asyncio.sleep(0.1)

        print(f"\nSubmitted {len(job_ids)} jobs sequentially\n")

        # Wait for all to complete
        print("Waiting for completion...")
        poll_tasks = [poll_job_status(session, miner_base_url, job_id) for job_id in job_ids]
        statuses = await asyncio.gather(*poll_tasks, return_exceptions=True)

        # Validate
        results = []
        for job_id, status_result in zip(job_ids, statuses):
            if isinstance(status_result, Exception):
                results.append({"job_id": job_id, "valid": False, "error": str(status_result)})
                continue

            try:
                callback_payload = await wait_for_callback(session, callback_base_url, job_id)
                if callback_payload and callback_payload.get("has_image"):
                    saved_path = Path(callback_payload.get("saved_path"))
                    if saved_path.exists():
                        validation = validator.validate(saved_path)
                        is_partial, partial_reason = detect_partial_render(saved_path)

                        results.append({
                            "job_id": job_id,
                            "valid": validation.is_valid and not is_partial,
                            "validation": validation,
                            "partial": is_partial,
                            "partial_reason": partial_reason
                        })

                        if validation.is_valid and not is_partial:
                            print(f"  ✅ {job_id}: Valid")
                        elif is_partial:
                            print(f"  ❌ {job_id}: PARTIAL RENDER - {partial_reason}")
                        else:
                            print(f"  ❌ {job_id}: INVALID - {validation.reason}")
                    else:
                        results.append({"job_id": job_id, "valid": False, "error": "File not found"})
                else:
                    results.append({"job_id": job_id, "valid": False, "error": "No callback"})
            except Exception as e:
                results.append({"job_id": job_id, "valid": False, "error": str(e)})

        # Summary
        valid_count = sum(1 for r in results if r.get("valid"))
        invalid_count = len(results) - valid_count

        print(f"\n{'='*80}")
        print(f"Valid: {valid_count}/{len(results)}")
        print(f"Invalid: {invalid_count}/{len(results)}")
        print(f"{'='*80}\n")

        assert invalid_count == 0, f"Found {invalid_count} invalid images"


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("ENABLE_TRAINING", "true").lower() == "false",
    reason="Training disabled, cannot test interference"
)
async def test_training_interference(callback_server, callback_base_url):
    """
    Stress test: Submit inference, then immediately trigger training.

    This attempts to reproduce race condition where training job
    unloads TRT server mid-inference.
    """
    miner_base_url = os.getenv("ASYNC_MINER_URL", "http://localhost:8091").rstrip("/")
    callback_secret = os.getenv("ASYNC_CALLBACK_SECRET", "test-secret")
    callback_url = f"{callback_base_url}/callback"

    print(f"\n{'='*80}")
    print("STRESS TEST: Training interference with in-flight inference")
    print(f"{'='*80}\n")

    validator = ImageValidator()

    async with aiohttp.ClientSession() as session:
        expiry = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

        # Submit inference with long inference steps
        inference_payload = {
            "prompt": TEST_PROMPTS[0],
            "callback_url": callback_url,
            "callback_secret": callback_secret,
            "expiry": expiry,
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50,  # Long inference
            "seed": 999,
        }

        submission = await submit_inference_job(session, miner_base_url, inference_payload)
        job_id = submission.get("job_id")
        assert job_id, "Failed to submit inference job"
        print(f"  ✅ Submitted long inference job: {job_id}")

        # Wait briefly then trigger training
        await asyncio.sleep(2)

        # Submit minimal training job (this may fail if no training config available)
        # This is just to trigger the unload_trt_server() call
        try:
            training_payload = {
                "job_id": "stress_test_training",
                "prompt": "test training",
                "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 transparent PNG
            }
            async with session.post(f"{miner_base_url}/train", json=training_payload) as resp:
                print(f"  ⚠️  Triggered training job: {resp.status}")
        except Exception as e:
            print(f"  ⚠️  Training trigger failed (expected): {e}")

        # Wait for inference to complete
        print("\nWaiting for inference job to complete...")
        status = await poll_job_status(session, miner_base_url, job_id)

        # Validate
        callback_payload = await wait_for_callback(session, callback_base_url, job_id)

        if callback_payload and callback_payload.get("has_image"):
            saved_path = Path(callback_payload.get("saved_path"))
            if saved_path.exists():
                validation = validator.validate(saved_path)
                is_partial, partial_reason = detect_partial_render(saved_path)

                print(f"\n{'='*80}")
                if validation.is_valid and not is_partial:
                    print("✅ PASS: Image completed despite training interference")
                elif is_partial:
                    print(f"❌ FAIL: Partial render detected - {partial_reason}")
                    pytest.fail(f"Training interference caused partial render: {partial_reason}")
                else:
                    print(f"❌ FAIL: Invalid image - {validation.reason}")
                    pytest.fail(f"Training interference caused invalid image: {validation.reason}")
                print(f"{'='*80}\n")
            else:
                pytest.fail(f"Image file not found: {saved_path}")
        else:
            pytest.fail("No callback or image received")
