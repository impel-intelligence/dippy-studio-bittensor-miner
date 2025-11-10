#!/bin/bash
set -e

#
# Docker Entrypoint with TRT Build Support
#
# This script runs before the main miner server starts. It:
# 1. Checks if TRT engine needs to be built
# 2. Builds it if necessary (or skips if exists)
# 3. Starts the miner server
#

echo "========================================"
echo "Dippy Studio Bittensor Miner - Starting"
echo "========================================"
echo "Timestamp: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo ""

# Build TRT engine if needed (for Kontext)
if [ "${ENABLE_KONTEXT_EDIT}" = "true" ]; then
    echo "Kontext editing enabled - checking TRT engine..."
    /workspace/scripts/build_kontext_trt_on_startup.sh || {
        echo "⚠ TRT build failed or skipped, continuing with PyTorch fallback"
    }
    echo ""
fi

# Start the miner server
echo "Starting miner server..."
echo "========================================"
echo ""

exec python3 /workspace/miner_server.py "$@"
