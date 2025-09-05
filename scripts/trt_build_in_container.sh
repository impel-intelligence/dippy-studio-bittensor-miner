#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "TRT Engine Builder Container"
echo "========================================="

# Check GPU availability first
if ! nvidia-smi > /dev/null 2>&1; then
  echo "ERROR: GPU not visible in container. Ensure --gpus=all is set."
  exit 1
fi

# Read environment exactly like the miner does
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/[ ()]/_/g')
CUDA_VERSION=$(python3 - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)
TRT_VERSION=$(python3 - <<'PY'
import tensorrt as trt
print(trt.__version__)
PY
)

ENGINE_HASH="${GPU_NAME}_cu${CUDA_VERSION}_trt${TRT_VERSION}_fp16"
TRT_CACHE_DIR="${TRT_CACHE_DIR:-/trt-cache}"
ENGINE_PATH="${TRT_CACHE_DIR}/${ENGINE_HASH}/transformer.plan"

echo ""
echo "Build Configuration:"
echo "  GPU: $GPU_NAME"
echo "  Torch CUDA: $CUDA_VERSION"
echo "  TensorRT: $TRT_VERSION"
echo "  Engine path: $ENGINE_PATH"
echo ""

# Check if engine already exists
if [[ -f "$ENGINE_PATH" ]]; then
  echo "✅ Engine already exists at $ENGINE_PATH"
  echo "   Size: $(du -h "$ENGINE_PATH" | cut -f1)"
  echo "   Use 'docker compose run --rm trt-builder --force' to rebuild"
  if [[ "${1:-}" != "--force" ]]; then
    exit 0
  fi
  echo ""
  echo "Force rebuild requested, continuing..."
fi

# Create cache directory
mkdir -p "$(dirname "$ENGINE_PATH")"

# Optional knobs (mirror production behavior)
: "${TRT_BUILD_SAFE:=0}"
: "${USE_FLASH_ATTENTION:=1}"
: "${XFORMERS_DISABLED:=0}"

echo "Build settings:"
echo "  TRT_BUILD_SAFE: $TRT_BUILD_SAFE"
echo "  USE_FLASH_ATTENTION: $USE_FLASH_ATTENTION"
echo "  XFORMERS_DISABLED: $XFORMERS_DISABLED"
echo ""

# Apply safe mode if requested
if [[ "$TRT_BUILD_SAFE" == "1" ]]; then
  echo "Applying safe build mode (disabling optional fused kernels)..."
  export USE_FLASH_ATTENTION=0
  export DISABLE_FLASH_ATTENTION=1
  export XFORMERS_DISABLED=1
  export CUDA_FORCE_PTX_JIT=1
fi

# Set CUDA environment variables
export CUDA_MODULE_LOADING=LAZY
export CUDA_CACHE_MAXSIZE=2147483648
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDA_ARCH_LIST="90;90+PTX"
export TRT_SUPPRESS_CUDA_WARNINGS=1

# Set environment for trt.py
export MODEL_PATH="${MODEL_PATH:-black-forest-labs/FLUX.1-dev}"
export LORA_PATH=""  # No LoRA for base engine
export ONNX_EXPORT_DIR="$(dirname "$ENGINE_PATH")/onnx"
export ENGINE_EXPORT_DIR="$(dirname "$ENGINE_PATH")"

echo "Starting TRT engine build..."
echo "This will take 20-30 minutes on first run..."
echo ""

cd /workspace && python3 trt.py

# Move the generated files to the expected location
if [[ -f "./trt/transformer.plan" ]]; then
  echo ""
  echo "Moving generated engine to cache directory..."
  mkdir -p "$(dirname "$ENGINE_PATH")"
  mv ./trt/transformer.plan "$ENGINE_PATH"
  
  # Also move the mapping file if it exists
  if [[ -f "./trt/mapping.json" ]]; then
    mv ./trt/mapping.json "$(dirname "$ENGINE_PATH")/mapping.json"
  fi
  
  # Move ONNX files if needed
  if [[ -d "./onnx" ]]; then
    mv ./onnx "$(dirname "$ENGINE_PATH")/onnx"
  fi
  
  # Clean up the trt directory
  rm -rf ./trt
  echo "Files moved to: $(dirname "$ENGINE_PATH")"
fi

# Verify the engine was created
if [[ ! -s "$ENGINE_PATH" ]]; then
  echo ""
  echo "❌ ERROR: Engine file not created at $ENGINE_PATH"
  exit 3
fi

echo ""
echo "========================================="
echo "✅ TRT Engine Built Successfully!"
echo "========================================="
echo "  Path: $ENGINE_PATH"
echo "  Size: $(du -h "$ENGINE_PATH" | cut -f1)"
echo ""
echo "The miner can now use this cached engine."