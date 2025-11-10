#!/bin/bash
set -e

#
# FLUX Kontext TensorRT Engine Build Script
#
# This script is designed to run on Docker startup to build the optimized
# FP8 TensorRT engine for FLUX Kontext if it doesn't already exist.
#
# It checks for an existing engine and only builds if needed, ensuring
# the miner can start quickly if the engine is already available.
#

echo "========================================"
echo "Kontext TensorRT Engine Build Script"
echo "========================================"

# Configuration
ENGINE_PATH="${TRT_KONTEXT_ENGINE_PATH:-/trt-cache/kontext_fp8.trt}"
FORCE_REBUILD="${FORCE_TRT_REBUILD:-0}"
SKIP_TRT_BUILD="${SKIP_KONTEXT_TRT_BUILD:-0}"

echo "Configuration:"
echo "  Engine path: $ENGINE_PATH"
echo "  Force rebuild: $FORCE_REBUILD"
echo "  Skip build: $SKIP_TRT_BUILD"
echo ""

# Check if TRT build should be skipped
if [ "$SKIP_TRT_BUILD" = "1" ]; then
    echo "⊘ TRT build skipped (SKIP_KONTEXT_TRT_BUILD=1)"
    echo "   Kontext will run in PyTorch BF16 mode"
    exit 0
fi

# Check if Kontext editing is enabled
if [ "${ENABLE_KONTEXT_EDIT}" != "true" ]; then
    echo "⊘ Kontext editing disabled (ENABLE_KONTEXT_EDIT != true)"
    echo "   Skipping TRT engine build"
    exit 0
fi

# Check if engine already exists
if [ -f "$ENGINE_PATH" ] && [ "$FORCE_REBUILD" != "1" ]; then
    echo "✓ TensorRT engine already exists: $ENGINE_PATH"
    echo "  File size: $(du -h $ENGINE_PATH | cut -f1)"
    echo "  Modified: $(stat -c %y $ENGINE_PATH)"
    echo ""
    echo "To rebuild, set FORCE_TRT_REBUILD=1"
    exit 0
fi

# Engine needs to be built
echo "Building TensorRT engine..."
echo ""

# Step 1: Prepare calibration data (if not exists)
CALIB_DATA="/workspace/optimization/calibration/data/calibration_dataset.json"

if [ ! -f "$CALIB_DATA" ]; then
    echo "→ Preparing calibration dataset..."
    python3 /workspace/optimization/scripts/01_prepare_calibration_data.py
    echo ""
else
    echo "✓ Calibration dataset exists: $CALIB_DATA"
    echo ""
fi

# Step 2: Measure baseline performance (optional, for comparison)
if [ "${MEASURE_BASELINE:-0}" = "1" ]; then
    echo "→ Measuring baseline performance..."
    python3 /workspace/optimization/scripts/02_measure_baseline.py
    echo ""
fi

# Step 3: Build TensorRT engine
echo "→ Building FP8 TensorRT engine..."
echo "  This may take 20-40 minutes on first run..."
echo ""

python3 /workspace/optimization/scripts/build_kontext_trt_fp8.py \
    --model-path "${KONTEXT_MODEL_PATH:-black-forest-labs/FLUX.1-Kontext-dev}" \
    --output-engine "$ENGINE_PATH" \
    --precision fp8 \
    --calibration-samples "${TRT_CALIBRATION_SAMPLES:-100}" \
    2>&1 | tee /workspace/optimization/logs/trt_build.log

# Check if build succeeded
if [ $? -eq 0 ] && [ -f "$ENGINE_PATH" ]; then
    echo ""
    echo "✓ TensorRT engine built successfully!"
    echo "  Location: $ENGINE_PATH"
    echo "  Size: $(du -h $ENGINE_PATH | cut -f1)"
    echo ""
    echo "The miner will now use the optimized FP8 engine for Kontext editing."
    echo "Expected performance: ~2x faster than BF16, ~50% memory reduction"
else
    echo ""
    echo "✗ TensorRT engine build failed!"
    echo "  Check logs at: /workspace/optimization/logs/trt_build.log"
    echo ""
    echo "The miner will fall back to PyTorch BF16 mode."
    exit 1
fi

echo "========================================"
echo "Build complete!"
echo "========================================"
