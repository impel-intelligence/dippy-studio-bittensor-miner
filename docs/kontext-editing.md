# FLUX.1-Kontext-dev Image Editing

## Overview

The miner supports deterministic image editing using FLUX.1-Kontext-dev, a 12B parameter image-to-image editing model.

**Key Features:**
- Deterministic editing with explicit seed control
- Text-based image modification instructions
- Reference job chaining for iterative edits
- Async job processing with callbacks

## Configuration

### Enable Kontext Editing

Add to `.env`:

```bash
ENABLE_KONTEXT_EDIT=true
KONTEXT_MODEL_PATH=black-forest-labs/FLUX.1-Kontext-dev
```

### Determinism Settings

These environment variables enforce deterministic behavior:

```bash
PYTHONHASHSEED=0                    # Deterministic Python hash
CUBLAS_WORKSPACE_CONFIG=:4096:8     # Deterministic CUDA operations
```

## API Usage

### POST /edit

Create an image editing job.

**Request:**

```json
{
  "prompt": "Add a red hat to the person",
  "image_b64": "base64_encoded_png_or_jpeg",
  "seed": 42,
  "guidance_scale": 2.5,
  "num_inference_steps": 28
}
```

**Parameters:**

- `prompt` (required): Text instruction for editing
- `image_b64` (optional): Base64 encoded input image
- `reference_job_id` (optional): Job ID to chain from
- `seed` (required): Random seed for determinism
- `guidance_scale` (optional, default 2.5): Prompt adherence (1.0-20.0)
- `num_inference_steps` (optional, default 28): Quality control (1-100)
- `callback_url` (optional): Webhook URL for async completion notification
- `callback_secret` (optional): Secret sent in `X-Callback-Secret` header
- `expiry` (optional): ISO timestamp for callback expiration

**Note:** The `strength` parameter is not supported by FluxKontextPipeline.

**Response:**

```json
{
  "accepted": true,
  "job_id": "edit-abc123",
  "status": "queued",
  "status_url": "/edit/status/edit-abc123",
  "result_url": "/edit/result/edit-abc123"
}
```

### GET /edit/status/{job_id}

Check job status.

**Response:**

```json
{
  "id": "edit-abc123",
  "status": "completed",
  "queued_at": "2025-10-14T12:00:00Z",
  "completed_at": "2025-10-14T12:00:15Z",
  "result_url": "/edit/result/edit-abc123",
  "image_url": "http://miner:8091/images/edits/edit-abc123.png",
  "callback": {
    "status": "delivered",
    "status_code": 200,
    "attempted_at": "2025-10-14T12:00:15Z",
    "payload_status": "completed"
  }
}
```

**Callback field statuses:**
- `delivered` - Callback successfully sent
- `failed` - Callback delivery failed (includes error details)
- `expired` - Callback not sent due to expiry
- `skipped` - No callback_url provided

### GET /edit/result/{job_id}

Download edited image (PNG).

## Async Callbacks

Edit jobs support webhook callbacks for async completion notification:

```python
# Submit edit with callback
response = requests.post("/edit", json={
    "prompt": "Add a red hat",
    "image_b64": base_image,
    "seed": 42,
    "callback_url": "https://your-server.com/webhook",
    "callback_secret": "your-secret-key",
    "expiry": "2025-10-22T12:00:00Z"
})

# Your webhook will receive:
# POST https://your-server.com/webhook
# Headers:
#   X-Callback-Secret: your-secret-key
# Form Data:
#   job_id: edit-abc123
#   status: completed
#   completed_at: 2025-10-22T11:45:00Z
#   image_url: http://miner:8091/images/edits/edit-abc123.png
#   image: <binary PNG file>
```

**Callback behavior:**
- Sent after job completion (success or failure)
- Includes `X-Callback-Secret` header for authentication
- Delivers the edited image as multipart form data
- Will not retry if delivery fails
- Skipped if `expiry` timestamp has passed

## Chaining Edits

Use `reference_job_id` to chain edits:

```python
# Edit 1: Add hat
response1 = requests.post("/edit", json={
    "prompt": "Add a red hat",
    "image_b64": base_image,
    "seed": 42
})

# Edit 2: Chain from edit 1
response2 = requests.post("/edit", json={
    "prompt": "Make the hat blue",
    "reference_job_id": response1["job_id"],
    "image_b64": base_image,  # Fallback
    "seed": 43
})
```

**Best Practice:** Always include `image_b64` as fallback when using `reference_job_id`.

## Determinism Guarantees

**Same seed → Identical output:**

- Pixel-perfect reproduction across multiple runs
- Same SHA256 hash
- Works across server restarts (same GPU/environment)

**Cross-GPU determinism:** Not guaranteed due to hardware differences.

## Performance

**PyTorch Implementation:**
- Latency: ~5-15s per edit (512x512, 28 steps)
- VRAM: ~14GB
- Can run alongside FLUX.1-dev inference on H100

## Testing

### Running E2E Tests

For **local miner** (not in Docker):

```bash
export ENABLE_KONTEXT_EDIT=true
export ASYNC_MINER_URL=http://localhost:8091
pytest tests/e2e/kontext_determinism/ -v -m e2e
```

For **Docker-based miner**:

```bash
# Find Docker gateway IP
docker network inspect dippy-studio-bittensor-miner_default | grep Gateway

# Set environment variables (replace 172.18.0.1 with your gateway IP)
export ENABLE_KONTEXT_EDIT=true
export ASYNC_MINER_URL=http://localhost:8091
export ASYNC_CALLBACK_BASE_URL=http://172.18.0.1:8092
export ASYNC_CALLBACK_BIND_HOST=0.0.0.0

# Run tests
pytest tests/e2e/kontext_determinism/ -v -m e2e
```

**Test suites:**
- `test_determinism.py` - Deterministic generation with same seeds
- `test_edit_workflow.py` - Reference job chaining and error handling
- `test_edit_callback.py` - Async callback delivery verification

See [tests/e2e/kontext_determinism/README.md](../tests/e2e/kontext_determinism/README.md) for detailed testing documentation.

## Troubleshooting

**Edit outputs are non-deterministic:**
- Check `PYTHONHASHSEED=0` is set
- Verify `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set
- Ensure same PyTorch/CUDA versions

**Reference job not found:**
- Image may have been cleaned up (24h retention)
- Use `image_b64` fallback for resilience

**Out of memory:**
- Reduce concurrent edits
- Disable inference (`ENABLE_INFERENCE=false`)
