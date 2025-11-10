"""
Production configuration for Kontext E2E tests.

These are the invariant parameters used in production.
"""

# Production parameters - these should not change
PRODUCTION_CONFIG = {
    "image_size": {
        "width": 1024,
        "height": 1024
    },
    "guidance_scale": 4,
    "num_inference_steps": 20,
}

# Test timeouts
TEST_TIMEOUTS = {
    "job_completion": 300,  # 5 minutes max per job
    "health_check": 10,      # 10 seconds for health check
}

# Performance expectations
PERFORMANCE_TARGETS = {
    "max_latency_bf16": 30,    # BF16 should complete in <30s
    "max_latency_fp8": 15,     # FP8 target: <15s
    "min_speedup_fp8": 1.5,    # FP8 should be at least 1.5x faster
    "max_memory_bf16_gb": 14,  # BF16 uses ~14GB
    "max_memory_fp8_gb": 8,    # FP8 target: ~7-8GB
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "min_mean_pixel": 10,     # Avoid all-black images
    "max_mean_pixel": 245,    # Avoid all-white images
    "min_std_dev": 5,         # Minimum variance to detect generation
}
