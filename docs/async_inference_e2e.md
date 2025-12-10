# Async Inference End-to-End Test

Use this suite to verify a miner correctly implements the async inference protocol (job submission, status polling, and callback delivery).

## Prerequisites

- The miner API is running locally and reachable (default `http://localhost:8091`).
- Python dependencies from `requirements_test_async.txt` are installed in your environment.
- GPU/driver requirements for the model are satisfied so that a test job can complete.

## Running the Test

### Option 1: Pytest (recommended)

```bash
pip install -r requirements_test_async.txt
pytest tests/e2e/async_inference/test_async_inference.py -s
```

The test automatically launches a local callback server on port `8092` unless you set `ASYNC_USE_EXTERNAL_CALLBACK=1`.

### Option 2: CLI helper

```bash
pip install -r requirements_test_async.txt
python -m tests.e2e.async_inference.cli run
```

The CLI reports progress for each test case, starts the callback server by default, and stores callback payloads under `mock_callbacks/` for inspection. To re-use an existing callback endpoint, pass `--no-auto-callback` and point `--callback-base` at your server.

### Callback server only

Start the reference callback server in its own terminal if you prefer to manage it manually:

```bash
python -m tests.e2e.async_inference.cli callback-server --host 0.0.0.0 --port 8092
```

Callbacks are mirrored at `http://<host>:<port>/callbacks` and saved to `mock_callbacks/` by job id.

## Environment Variables / Flags

| Name | Purpose | Default |
| --- | --- | --- |
| `ASYNC_MINER_URL` | Miner API base URL | `http://localhost:8091` |
| `ASYNC_CALLBACK_BASE_URL` | Public base URL for callbacks | `http://127.0.0.1:8092` |
| `ASYNC_CALLBACK_BIND_HOST` | Host/IP for the local callback server when auto-started | Pytest fixture: derived from base URL (default `127.0.0.1`); CLI helper default `0.0.0.0` |
| `ASYNC_CALLBACK_SECRET` | Shared secret expected in callback requests | `test-secret` |
| `ASYNC_USE_EXTERNAL_CALLBACK` | Skip starting the local callback server (`1`, `true`, `yes`) | unset |

CLI flags mirror these values (run `python -m tests.e2e.async_inference.cli --help`).

## Protocol Overview

1. **Client submits a job** via `POST /inference` with payload fields:
   - `prompt`, `width`, `height`, `num_inference_steps`, `seed`
   - `callback_url`: full URL to the callback endpoint (e.g. `http://127.0.0.1:8092/callback`)
   - `callback_secret`: secret token to validate the callback
   - `expiry`: ISO8601 timestamp after which the job should not emit callbacks

   Example request body:

   ```json
   {
     "prompt": "A cute anime girl with blue hair",
     "width": 1024,
     "height": 1024,
     "num_inference_steps": 20,
     "seed": 42,
     "callback_url": "http://127.0.0.1:8092/callback",
     "callback_secret": "test-secret",
     "expiry": "2024-01-01T12:34:56Z"
   }
   ```

2. **Miner responds immediately** with a JSON body containing:
   - `job_id`: unique identifier for the inference run
   - `status`: current state (`queued`, `processing`, etc.)
   - `message`: optional human-readable detail

   Example response:

   ```json
   {
     "job_id": "123e4567-e89b-12d3-a456-426614174000",
     "status": "queued",
     "message": "Job accepted"
   }
   ```

3. **Client polls job status** via `GET /inference/status/{job_id}` until it reaches a terminal state (`completed`, `failed`, or `timeout`). The final response includes a `callback` section with delivery metadata (`status`, `status_code`, `payload_status`, `response_preview`, `attempted_at`).

   ```json
   {
     "job_id": "123e4567-e89b-12d3-a456-426614174000",
     "status": "completed",
     "result_url": "/inference/result/123e4567-e89b-12d3-a456-426614174000",
     "callback": {
       "status": "delivered",
       "status_code": 200,
       "response_preview": "OK",
       "attempted_at": "2024-01-01T12:35:12Z",
       "payload_status": "completed"
     }
   }
   ```

4. **Miner performs the callback** by issuing a `POST` (multipart form) to the provided URL with fields:
   - `job_id`, `status`, `completed_at`
   - `image_url`: optional link if the artifact is hosted elsewhere
   - `error`: optional string when the job failed
   - `image`: optional binary payload of the generated image
   - Header `X-Callback-Secret` matching the submitted secret

   Example callback (simplified form-data view):

   ```text
   POST /callback HTTP/1.1
   Host: 127.0.0.1:8092
   Content-Type: multipart/form-data; boundary=...
   X-Callback-Secret: test-secret

   --...
   Content-Disposition: form-data; name="job_id"

   123e4567-e89b-12d3-a456-426614174000
   --...
   Content-Disposition: form-data; name="status"

   completed
   --...
   Content-Disposition: form-data; name="completed_at"

   2024-01-01T12:35:12Z
   --...
   Content-Disposition: form-data; name="image"; filename="result.png"
   Content-Type: image/png

   <binary data>
   --...--
   ```

5. **Client stores or validates** the callback payload. The reference callback server persists the multipart request to `mock_callbacks/images/<job_id>.png` and exposes JSON metadata at `/callbacks`.

### Detailed Field Reference

#### `POST /inference` request body

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `prompt` | string | yes | Text prompt for generation |
| `width` / `height` | int | optional | Defaults 1024; must be within model limits |
| `num_inference_steps` | int | optional | Defaults 28 |
| `guidance_scale` | float | optional | Defaults 7.0 |
| `seed` | int | optional | Use to reproduce generations |
| `callback_url` | string | optional | Provide to enable async delivery |
| `callback_secret` | string | optional | Forwarded as `X-Callback-Secret` header |
| `expiry` | ISO8601 | optional | Skip callback if current time is past expiry |

#### Immediate response (`POST /inference`)

| Field | Type | Notes |
| --- | --- | --- |
| `accepted` | bool | `true` when the job entered the queue |
| `job_id` | string | Unique identifier used for polling |
| `status` | string | Initial job status (`queued`) |
| `message` | string | Human readable acknowledgement |
| `status_url` | string | Relative path for polling |
| `result_url` | string | Relative path to retrieve the generated image |

#### Polling response (`GET /inference/status/{job_id}`)

| Field | Type | Notes |
| --- | --- | --- |
| `status` | string | `queued`, `processing`, `completed`, `failed`, `timeout` |
| `completed_at` | ISO8601 | Present when finished |
| `result_url` / `image_url` | string | Paths to download or externally reference the artifact |
| `error` | string | Present when job fails |
| `callback` | object | Delivery metadata (see next table) |

`callback` object:

| Field | Type | Meaning |
| --- | --- | --- |
| `status` | string | `delivered`, `failed`, `expired`, `skipped`, `error` |
| `status_code` | int | Present when the callback target returned an HTTP response |
| `response_preview` | string | First 500 chars of the callback response body |
| `attempted_at` | ISO8601 | Timestamp of delivery attempt |
| `payload_status` | string | Reflects the payload sent (`completed` on success, `failed` when the callback body describes a failure) |
| `error` | string | Present when delivery failed before an HTTP response (for example, connection errors) |

#### Callback payload (`POST callback_url`)

Multipart form fields:

| Field | Type | Notes |
| --- | --- | --- |
| `job_id` | string | Matches the original job id |
| `status` | string | Same terminal status reported by polling |
| `completed_at` | ISO8601 | Completion timestamp |
| `image_url` | string | Optional direct URL if the miner hosts the artifact (included when an image file is attached) |
| `error` | string | Optional failure detail |
| `image` | file | Optional binary payload of the generated image |

HTTP header:

| Header | Value |
| --- | --- |
| `X-Callback-Secret` | Echoes `callback_secret` from submission |

#### Recommended Validation Flow

1. Submit a job and capture the returned `job_id`.
2. Poll `GET /inference/status/{job_id}` until `status` is terminal. Confirm:
   - `status == "completed"` (or expected failure mode)
   - `callback.status == "delivered"`
   - `callback.status_code < 400` (when present)
   - `payload_status == "completed"`
3. Fetch callback metadata (`/callbacks`) to verify the secret, error field, and optional image path.
4. (Optional) Download the artifact via `GET /inference/result/{job_id}` or the `image_url` to ensure file availability.

### Failure Modes

- `status == "failed"`: Model execution or callback dispatch failed. Inspect `error` and callback metadata.
- `status == "timeout"`: Job exceeded the configured inference timeout. Callback metadata typically reports `payload_status == "failed"` because the callback delivers a failure payload.
- `callback.status == "failed"`: Delivery to the callback endpoint failed (connection error, HTTP 4xx/5xx). Retry manually after fixing connectivity.
- `callback.status == "expired"`: The current time passed the supplied `expiry`; no callback was sent.

### Networking Notes

- Ensure the miner can reach the callback URL. When the miner runs in Docker, set `ASYNC_CALLBACK_BASE_URL` to a host IP accessible from the container (e.g. `http://172.17.0.1:8092`) and bind the callback server with `ASYNC_CALLBACK_BIND_HOST=0.0.0.0`.
- The callback server exposed by the test suite saves binary payloads under `mock_callbacks/images/` and exposes metadata at `/callbacks` for easy inspection.

## Troubleshooting Tips

- If the test skips with "Miner server unavailable", ensure `miner_server.py` (or your deployment) is running and the port matches `ASYNC_MINER_URL`.
- The callback server must be reachable from the miner container/host. When running in Docker, set `ASYNC_CALLBACK_BASE_URL` to an address the miner can reach (e.g. `http://host.docker.internal:8092`) and adjust `ASYNC_CALLBACK_BIND_HOST` accordingly.
- Clear old callback artifacts by deleting `mock_callbacks/` before re-running tests if needed.
- Increase the default expiry window with `--expiry-minutes` if jobs take longer than five minutes.
