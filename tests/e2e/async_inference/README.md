# Async Inference E2E Tests with Race Condition Detection

This directory contains end-to-end tests for the async inference system, including specialized stress tests to detect race conditions that cause incomplete image generation.

## Test Files

### `test_async_inference.py`
Basic E2E test validating the async callback protocol works correctly.

### `test_race_conditions.py` ⚠️ NEW
Stress tests designed to reproduce race conditions causing partial/incomplete images:

1. **`test_rapid_concurrent_inference`** - Submits many concurrent LoRA inference requests to trigger race conditions
2. **`test_rapid_sequential_different_loras`** - Submits sequential requests with minimal delay to trigger engine queue races

### `image_validator.py` ⚠️ NEW
Automated validation to detect incomplete/corrupted images:
- File size validation
- PIL integrity checks
- Dimension verification
- **Blank region detection** (detects partial renders)
- **Pixel variance analysis** (detects solid colors/corruption)
- **Partial render detection** (horizontal cutoff detection)

### `callback_server.py`
Mock callback server for receiving inference results.

### `client.py`
Client utilities for submitting jobs and polling status.

---

## Setup

### 1. Install Test Dependencies

```bash
pip install -r requirements_test_async.txt
```

New dependencies added:
- `numpy` - For pixel-level image analysis
- `pillow` - For image validation (PIL)

### 2. Start Miner Service

```bash
# Deploy inference-only mode
make setup-inference
make up

# Or deploy with docker compose
docker-compose up -d
```

### 3. Configure Environment Variables

```bash
# Required
export ASYNC_MINER_URL=http://localhost:8091
export ASYNC_CALLBACK_BASE_URL=http://127.0.0.1:8092

# For Docker miner, use docker0 gateway IP
export ASYNC_CALLBACK_BASE_URL=http://172.17.0.1:8092
export ASYNC_CALLBACK_BIND_HOST=0.0.0.0

# Optional - test configuration
export STRESS_TEST_CONCURRENT=10   # Number of concurrent jobs in stress test
export STRESS_TEST_SEQUENTIAL=5     # Number of sequential jobs
export ASYNC_CALLBACK_SECRET=test-secret
```

---

## Running Tests

### Basic E2E Test (Validates Protocol Works)

```bash
pytest tests/e2e/async_inference/test_async_inference.py -s -v
```

### Stress Tests (Detect Race Conditions)

```bash
# Run all stress tests
pytest tests/e2e/async_inference/test_race_conditions.py -s -v

# Run specific test
pytest tests/e2e/async_inference/test_race_conditions.py::test_rapid_concurrent_inference -s -v

# Increase concurrency for higher reproduction chance
STRESS_TEST_CONCURRENT=20 pytest tests/e2e/async_inference/test_race_conditions.py::test_rapid_concurrent_inference -s -v
```

### Run All Tests

```bash
pytest tests/e2e/async_inference/ -s -v
```

---

## Understanding Test Results

### ✅ Success (No Issues Found)

```
STRESS TEST RESULTS
================================================================================

Total jobs: 10
Completed: 10
Valid images: 10
Invalid/incomplete images: 0
No images: 0
```

All images passed validation - race conditions not triggered.

### ❌ Failure (Race Condition Detected)

```
STRESS TEST RESULTS
================================================================================

Total jobs: 10
Completed: 10
Valid images: 7
Invalid/incomplete images: 3
No images: 0

INVALID IMAGES DETECTED:
  - job_003: Partial render detected: 85.3% of bottom rows are blank
  - job_007: Image has 52.4% blank/black pixels (threshold: 50.0%)
  - job_009: File too small (89432 bytes, expected >100000)
```

**This indicates race conditions are present!** The test has successfully reproduced the issue.

---

## Image Validation Details

The `ImageValidator` performs these checks:

1. **File Exists** - Image file was created
2. **File Size** - Between 100KB and 10MB for 1024x1024 PNG
3. **PIL Verification** - File is valid PNG format
4. **Dimensions** - Image is exactly 1024x1024
5. **Blank Region Detection** - <50% of pixels are black (detects partial renders)
6. **Variance Check** - Pixel variance >100 (detects solid colors/corruption)

### Detecting Partial Renders

The `detect_partial_render()` function specifically checks if the **bottom 30% of the image** is blank/black, which is the typical symptom of interrupted generation.

---

## Reproducing Production Issues

If you're experiencing incomplete images in production:

### Option 1: Validate Existing Images

```python
from pathlib import Path
from tests.e2e.async_inference.image_validator import ImageValidator, detect_partial_render

validator = ImageValidator()

# Check single image
result = validator.validate(Path("/app/output/job_123.png"))
if not result.is_valid:
    print(f"Invalid: {result.reason}")

# Check for partial render
is_partial, reason = detect_partial_render(Path("/app/output/job_123.png"))
if is_partial:
    print(f"Partial render: {reason}")
```

### Option 2: Run Stress Test with High Concurrency

```bash
# Increase concurrent jobs to 50+
STRESS_TEST_CONCURRENT=50 pytest tests/e2e/async_inference/test_race_conditions.py::test_rapid_concurrent_inference -s -v
```

Higher concurrency = higher chance of triggering race conditions.

### Option 3: Monitor Production Callbacks

Modify `callback_server.py` to run against your production validator and inspect received images:

```bash
# Point to production miner
export ASYNC_MINER_URL=http://your-miner-ip:8091

# Run callback server
python tests/e2e/async_inference/callback_server.py

# Submit test jobs and check callback dashboard
# Open http://localhost:8092 in browser
```

---

## Expected Behavior After Fixes

After implementing the Phase 1 fixes (atomic file writes, CUDA sync, engine queue ordering):

1. **All stress tests should pass** with 0 invalid images
2. **File sizes should be consistent** (typically 2-8MB for 1024x1024 PNG)
3. **No blank regions** detected in bottom of images
4. **Variance >100** for all images

If tests still fail after fixes, additional investigation needed for remaining race conditions.

---

## Troubleshooting

### "Callback server unavailable"

```bash
# Check if port 8092 is available
lsof -i :8092

# Try different port
export ASYNC_CALLBACK_BASE_URL=http://127.0.0.1:9999
```

### "Miner server unavailable"

```bash
# Check miner status
make logs

# Check health endpoint
curl http://localhost:8091/health
```

### "ModuleNotFoundError: No module named 'numpy'"

```bash
# Install test dependencies
pip install -r requirements_test_async.txt
```

### Tests timeout

```bash
# Increase pytest timeout
pytest tests/e2e/async_inference/test_race_conditions.py -s -v --timeout=600
```

### Docker networking issues

For Docker miner, the callback server must be reachable from inside the container:

```bash
# Find docker0 gateway IP
ip addr show docker0 | grep inet

# Use that IP in callback URL
export ASYNC_CALLBACK_BASE_URL=http://172.17.0.1:8092
export ASYNC_CALLBACK_BIND_HOST=0.0.0.0
```

---

## Next Steps

1. **Run stress tests** to confirm you can reproduce the issue
2. **Implement Phase 1 fixes** (atomic writes, CUDA sync, queue ordering)
3. **Re-run stress tests** to validate fixes
4. **Increase test concurrency** to stress test further
5. **Deploy to production** once all tests pass

For implementation details of the fixes, see the main investigation report.
