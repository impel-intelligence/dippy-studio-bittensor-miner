# Kontext Determinism E2E Tests

End-to-end tests for FLUX.1-Kontext image editing with deterministic seeds and callback support.

## Prerequisites

1. **Miner server running** (either locally or in Docker)
2. **Kontext editing enabled** in the miner configuration

## Running the Tests

### Option 1: Local Miner (No Docker)

If running the miner directly on your host machine:

```bash
# Start the miner server
python3 miner_server.py

# In another terminal, run the tests
pytest tests/e2e/kontext_determinism/ -v -m e2e
```

### Option 2: Docker Miner (Recommended)

If running the miner in Docker (using `make run-dev` or docker-compose):

```bash
# Start the miner
make run-dev

# Find the Docker network gateway IP
docker network inspect dippy-studio-bittensor-miner_default | grep Gateway
# Example output: "Gateway": "172.18.0.1"

# Set environment variables for callback server
export ASYNC_CALLBACK_BASE_URL=http://172.18.0.1:8092
export ASYNC_CALLBACK_BIND_HOST=0.0.0.0

# Run the tests
pytest tests/e2e/kontext_determinism/ -v -m e2e
```

**Why these environment variables?**

- `ASYNC_CALLBACK_BASE_URL`: The callback server URL that the **miner** will send callbacks to. Must use the Docker bridge gateway IP (e.g., `172.18.0.1`) so the container can reach the host.
- `ASYNC_CALLBACK_BIND_HOST`: The interface the callback server binds to. Set to `0.0.0.0` to accept connections from Docker containers.

## Test Descriptions

### `test_determinism.py`

Tests deterministic image generation with the same seed:

- `test_same_seed_produces_identical_output` - Verifies same seed + prompt = identical images

### `test_edit_workflow.py`

Tests edit job workflows:

- `test_reference_job_chaining` - Chain multiple edits together
- `test_reference_missing_uses_fallback` - Fallback to image_b64 when reference_job_id doesn't exist
- `test_missing_both_image_sources_fails` - Validates error handling for missing image sources
- `test_missing_seed_fails_validation` - Validates seed is required

### `test_edit_callback.py`

Tests callback protocol for async edit jobs:

- `test_edit_callback_protocol` - Verifies successful callback delivery for completed jobs
- `test_edit_callback_on_failure` - Verifies validation errors are handled correctly

## Troubleshooting

### "Callback not delivered" error

**Symptom:** Test fails with `AssertionError: Callback not delivered: {}`

**Cause:** The miner container cannot reach the callback server at `127.0.0.1:8092`

**Solution:** Set `ASYNC_CALLBACK_BASE_URL` to use the Docker gateway IP:

```bash
# Find your gateway IP
docker network inspect dippy-studio-bittensor-miner_default | grep Gateway

# Set the environment variable (replace with your gateway IP)
export ASYNC_CALLBACK_BASE_URL=http://172.18.0.1:8092
export ASYNC_CALLBACK_BIND_HOST=0.0.0.0

# Re-run the tests
pytest tests/e2e/kontext_determinism/test_edit_callback.py -v
```

### "Kontext editing must be enabled" error

**Cause:** The miner has `enable_kontext_edit=False` in configuration

**Solution:** Set the environment variable:

```bash
export ENABLE_KONTEXT_EDIT=true
```

Or update your `.env` file to include:

```
ENABLE_KONTEXT_EDIT=true
```

### Connection refused to miner

**Symptom:** `pytest.skip: Miner server unavailable`

**Cause:** The miner server is not running or not reachable

**Solution:**

1. Check the miner is running: `docker ps | grep miner`
2. Verify port 8091 is accessible: `curl http://localhost:8091/health`
3. If using Docker, ensure port mapping is correct in docker-compose.yml

## Running Specific Tests

```bash
# Run only determinism tests
pytest tests/e2e/kontext_determinism/test_determinism.py -v

# Run only callback tests
pytest tests/e2e/kontext_determinism/test_edit_callback.py -v

# Run a specific test
pytest tests/e2e/kontext_determinism/test_edit_callback.py::test_edit_callback_protocol -v -s

# Run with full output (including logs)
pytest tests/e2e/kontext_determinism/ -v -s
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ASYNC_MINER_URL` | `http://localhost:8091` | Miner server URL for tests to connect to |
| `ASYNC_CALLBACK_BASE_URL` | `http://127.0.0.1:8092` | Callback server URL (use Docker gateway IP when miner is in Docker) |
| `ASYNC_CALLBACK_BIND_HOST` | `127.0.0.1` | Callback server bind interface (use `0.0.0.0` for Docker) |
| `ASYNC_CALLBACK_SECRET` | `test-secret` | Secret header for callback authentication |
| `ASYNC_USE_EXTERNAL_CALLBACK` | `false` | Use external callback server instead of starting one in tests |
