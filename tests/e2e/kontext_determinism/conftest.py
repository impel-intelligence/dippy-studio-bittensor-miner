"""
Shared fixtures for Kontext determinism E2E tests.
"""
import pytest
import base64
import requests
import time
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlparse
import sys

# Add parent directory to import callback server
sys.path.insert(0, str(Path(__file__).parent.parent))
from async_inference.callback_server import CallbackServer

# Import production configuration
from test_config import PRODUCTION_CONFIG, TEST_TIMEOUTS, PERFORMANCE_TARGETS

# Miner URL from environment
import os
MINER_URL = os.getenv("ASYNC_MINER_URL", "http://localhost:8091")


@pytest.fixture
def sample_image():
    """Load ninja.png for testing"""
    # Navigate from tests/e2e/kontext_determinism/ to project root
    ninja_path = Path(__file__).parent.parent.parent.parent / "ninja.png"
    if not ninja_path.exists():
        # Fallback to generated image if ninja.png not found
        return Image.new("RGB", (512, 512), (128, 128, 200))
    return Image.open(ninja_path)


@pytest.fixture
def sample_image_b64(sample_image):
    """Create base64 encoded sample image"""
    buffer = BytesIO()
    sample_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def miner_client():
    """Client for interacting with miner API"""
    class MinerClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def post_edit(self, **kwargs):
            """POST /edit"""
            response = requests.post(f"{self.base_url}/edit", json=kwargs)
            response.raise_for_status()
            return response.json()

        def get_status(self, job_id):
            """GET /edit/status/{job_id}"""
            response = requests.get(f"{self.base_url}/edit/status/{job_id}")
            response.raise_for_status()
            return response.json()

        def get_result(self, job_id):
            """GET /edit/result/{job_id} - returns PIL Image"""
            response = requests.get(f"{self.base_url}/edit/result/{job_id}")
            response.raise_for_status()
            return Image.open(BytesIO(response.content))

        def wait_for_completion(self, job_id, timeout=300, poll_interval=2):
            """Poll status until completed or timeout"""
            start = time.time()
            while time.time() - start < timeout:
                status = self.get_status(job_id)
                if status["status"] == "completed":
                    return status
                elif status["status"] == "failed":
                    raise RuntimeError(f"Job failed: {status.get('error')}")
                time.sleep(poll_interval)
            raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    return MinerClient(MINER_URL)


@pytest.fixture
def health_check(miner_client):
    """Verify miner is healthy before running tests"""
    response = requests.get(f"{miner_client.base_url}/health")
    response.raise_for_status()
    health = response.json()

    # Verify Kontext is enabled
    assert health["services"]["kontext_edit"] != "disabled", \
        "Kontext editing must be enabled for these tests"

    return health


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


# Production parameter fixtures
@pytest.fixture
def production_params():
    """Production-invariant parameters for Kontext."""
    return PRODUCTION_CONFIG.copy()


@pytest.fixture
def production_image_size(production_params):
    """Production image size (1024x1024)."""
    return production_params["image_size"]


@pytest.fixture
def production_guidance_scale(production_params):
    """Production guidance scale (4)."""
    return production_params["guidance_scale"]


@pytest.fixture
def production_num_steps(production_params):
    """Production inference steps (20)."""
    return production_params["num_inference_steps"]


@pytest.fixture
def performance_targets():
    """Expected performance targets for TRT optimization."""
    return PERFORMANCE_TARGETS.copy()


@pytest.fixture
def test_timeouts():
    """Test timeout configuration."""
    return TEST_TIMEOUTS.copy()


@pytest.fixture
def large_test_image():
    """
    Create a large (1024x1024) test image for production testing.

    Uses character_1.png if available, otherwise generates a test pattern.
    """
    # Try to load character_1.png (from project root)
    char1_path = Path(__file__).parent.parent.parent.parent / "character_1.png"
    if char1_path.exists():
        img = Image.open(char1_path)
        # Resize to 1024x1024 if needed
        if img.size != (1024, 1024):
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        return img

    # Fallback: generate test pattern
    img = Image.new("RGB", (1024, 1024))
    pixels = img.load()

    # Create gradient pattern
    for y in range(1024):
        for x in range(1024):
            r = int((x / 1024) * 255)
            g = int((y / 1024) * 255)
            b = int(((x + y) / 2048) * 255)
            pixels[x, y] = (r, g, b)

    return img


@pytest.fixture
def large_test_image_b64(large_test_image):
    """Base64-encoded large test image."""
    buffer = BytesIO()
    large_test_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
