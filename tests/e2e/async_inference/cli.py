"""Command-line entry point for async inference end-to-end checks."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp

from .callback_server import CallbackServer
from .client import poll_job_status, submit_inference_job, wait_for_callback


DEFAULT_PROMPT = "A cute anime girl with blue hair"


@dataclass
class TestCase:
    name: str
    payload: Dict[str, Any]
    expect_callback: bool = True


async def _run_test_case(
    session: aiohttp.ClientSession,
    miner_url: str,
    callback_base: str,
    callback_secret: str,
    case: TestCase,
) -> bool:
    print(f"\n📝 {case.name}")

    try:
        submission = await submit_inference_job(session, miner_url, case.payload)
    except Exception as exc:
        print(f"   ❌ Submission failed: {exc}")
        return False

    job_id = submission.get("job_id")
    status = submission.get("status")
    message = submission.get("message")

    print(f"   ✅ Job submitted: {job_id}")
    print(f"   📋 Status: {status}")
    if message:
        print(f"   💬 Message: {message}")

    try:
        status_payload = await poll_job_status(session, miner_url, job_id)
    except Exception as exc:
        print(f"   ❌ Polling failed: {exc}")
        return False

    final_status = status_payload.get("status")
    print(f"   🎯 Final status: {final_status}")

    if case.expect_callback:
        callback_payload = await wait_for_callback(session, callback_base, job_id)
        if callback_payload:
            print("   📬 Callback received")
            print(f"      ↳ Secret header: {callback_payload.get('provided_secret')}")
            if callback_payload.get("has_image"):
                print(f"      ↳ Image stored at: {callback_payload.get('saved_path')}")
        else:
            print("   ❌ Expected callback but none received within timeout")
            return False
    else:
        callback_meta = status_payload.get("callback")
        if callback_meta:
            print(f"   ℹ️ Callback metadata: {callback_meta}")
        else:
            print("   ℹ️ No callback metadata recorded")

    return final_status == "completed"


async def run_tests(
    *,
    miner_url: str,
    callback_base: str,
    callback_secret: str,
    prompt: str,
    auto_callback: bool,
    bind_host: str,
    expiry_minutes: int,
) -> None:
    callback_url = f"{callback_base}/callback"
    now_utc = datetime.now(timezone.utc)
    expiry = (now_utc + timedelta(minutes=expiry_minutes)).isoformat()

    test_cases = [
        TestCase(
            name="Callback delivered before expiry",
            payload={
                "prompt": prompt,
                "callback_url": callback_url,
                "callback_secret": callback_secret,
                "expiry": expiry,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 20,
                "seed": 42,
            },
        )
    ]

    server: Optional[CallbackServer] = None
    if auto_callback:
        print("🔄 Starting local callback server...")
        host, port = _parse_host_port(callback_base)
        server = CallbackServer(host=bind_host, port=port, probe_host=host)
        server.start()
        print(f"   ↳ Listening at {callback_base}")

    print("🚀 Testing Async Inference with Callbacks")
    print(f"   Miner Server: {miner_url}")
    print(f"   Callback URL: {callback_url}")

    results: list[bool] = []

    async with aiohttp.ClientSession() as session:
        try:
            for case in test_cases:
                result = await _run_test_case(session, miner_url, callback_base, callback_secret, case)
                results.append(result)
        finally:
            if server:
                server.stop()

    print("\n📊 Test Summary:")
    print(f"   - Total tests: {len(test_cases)}")
    passed = sum(1 for ok in results if ok)
    print(f"   - Passed: {passed}")
    print(f"   - Callback dashboard: {callback_base}")


def _parse_host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80
    return host, port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async inference E2E helper")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("callback-server", help="Run the reference callback server")
    serve.add_argument("--host", default="0.0.0.0", help="Host/IP to bind callback server")
    serve.add_argument("--port", type=int, default=8092, help="Port for callback server")

    run = subparsers.add_parser("run", help="Execute the async inference test")
    run.add_argument("--miner-url", default=os.getenv("ASYNC_MINER_URL", "http://localhost:8091"))
    run.add_argument("--callback-base", default=os.getenv("ASYNC_CALLBACK_BASE_URL", "http://127.0.0.1:8092"))
    run.add_argument("--callback-secret", default=os.getenv("ASYNC_CALLBACK_SECRET", "test-secret"))
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.add_argument("--no-auto-callback", action="store_true", help="Do not launch a local callback server")
    run.add_argument("--callback-bind-host", default=os.getenv("ASYNC_CALLBACK_BIND_HOST", "0.0.0.0"))
    run.add_argument("--expiry-minutes", type=int, default=5)

    parser.set_defaults(command="run")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "callback-server":
        server = CallbackServer(host=args.host, port=args.port, probe_host="127.0.0.1")
        try:
            server.start()
            print(f"Callback server listening on http://{args.host}:{args.port}")
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("Stopping callback server...")
        finally:
            server.stop()
        return

    if args.command == "run":
        auto_callback = not args.no_auto_callback
        asyncio.run(
            run_tests(
                miner_url=args.miner_url.rstrip("/"),
                callback_base=args.callback_base.rstrip("/"),
                callback_secret=args.callback_secret,
                prompt=args.prompt,
                auto_callback=auto_callback,
                bind_host=args.callback_bind_host,
                expiry_minutes=args.expiry_minutes,
            )
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
