"""Pytest fixtures shared across async inference end-to-end tests."""

from __future__ import annotations

import os
from typing import Iterator, Optional
from urllib.parse import urlparse

import pytest

from .callback_server import CallbackServer


def _resolve_callback_base_url() -> str:
    base = os.getenv("ASYNC_CALLBACK_BASE_URL", "http://127.0.0.1:8092")
    return base.rstrip("/")


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


@pytest.fixture(scope="session")
def callback_base_url() -> str:
    return _resolve_callback_base_url()


@pytest.fixture(scope="session")
def callback_server(callback_base_url: str) -> Iterator[Optional[CallbackServer]]:
    """Start the reference callback server unless an external one is provided."""

    use_external = os.getenv("ASYNC_USE_EXTERNAL_CALLBACK", "").lower() in {"1", "true", "yes"}
    if use_external:
        yield None
        return

    host, port = _parse_host_port(callback_base_url)
    bind_host = os.getenv("ASYNC_CALLBACK_BIND_HOST", host)

    server = CallbackServer(host=bind_host, port=port, probe_host=host)
    server.start()
    try:
        yield server
    finally:
        server.stop()
