"""HTTP helpers shared by the async inference E2E tests."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import aiohttp


async def submit_inference_job(
    session: aiohttp.ClientSession,
    base_url: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    async with session.post(f"{base_url}/inference", json=payload) as response:
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"Inference request failed ({response.status}): {text}")
        return await response.json()


async def poll_job_status(
    session: aiohttp.ClientSession,
    base_url: str,
    job_id: str,
    *,
    max_attempts: int = 120,
    interval: float = 2.0,
) -> Dict[str, Any]:
    for _ in range(max_attempts):
        await asyncio.sleep(interval)
        async with session.get(f"{base_url}/inference/status/{job_id}") as response:
            if response.status != 200:
                continue
            payload = await response.json()
            status = payload.get("status")
            if status in {"completed", "failed", "timeout"}:
                return payload
    raise TimeoutError(f"Status polling exceeded {max_attempts * interval:.0f}s for job {job_id}")


async def wait_for_callback(
    session: aiohttp.ClientSession,
    callback_base: str,
    job_id: str,
    *,
    timeout: float = 90.0,
    interval: float = 1.0,
) -> Optional[Dict[str, Any]]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(interval)
        try:
            async with session.get(f"{callback_base}/callbacks") as response:
                if response.status != 200:
                    continue
                payload = await response.json()
        except aiohttp.ClientError:
            continue

        for item in payload.get("callbacks", []):
            if item.get("job_id") == job_id:
                return item
    return None
