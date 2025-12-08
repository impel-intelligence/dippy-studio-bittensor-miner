#!/usr/bin/env bash
#
# run_flux_benchmarks.sh
#
# Runs a comprehensive benchmark suite for flux-1.dev image inference,
# comparing local and cloud implementations:
#
# LOCAL:
#   1. HuggingFace Default (stock, unmodified FLUX.1-dev from HF)
#   2. Miner Baseline (PyTorch FP16 with miner modifications)
#   3. Miner TensorRT (TensorRT-optimized with miner modifications)
#
# CLOUD APIs (optional, requires API keys):
#   4. Replicate API ($0.030/image)
#   5. FAL.ai API ($0.025/image @ 1024x1024)
#   6. Together.ai API ($0.035/image @ 1024x1024)
#
# Usage:
#   ./benchmarks/run_flux_benchmarks.sh
#
# Environment variables:
#   MODEL_PATH          - Path to base model (default: black-forest-labs/FLUX.1-dev)
#   TRT_ENGINE_PATH     - Path to TensorRT engine (.plan file)
#   PROMPTS_FILE        - Path to prompts JSON file (default: benchmarks/prompts.json)
#   ITERATIONS          - Number of iterations per config (default: 10)
#   WARMUP              - Number of warmup iterations (default: 3)
#   RESULTS_DIR         - Output directory for CSV files (default: benchmarks/results)
#   SKIP_API_BENCHMARKS - Set to "true" to skip API provider benchmarks (default: false)
#
# API Keys (optional, set to enable API benchmarks):
#   REPLICATE_API_TOKEN - Replicate API token
#   FAL_KEY             - FAL.ai API key
#   TOGETHER_API_KEY    - Together.ai API key
#

set -euo pipefail

# Configuration
MODEL_PATH="${MODEL_PATH:-black-forest-labs/FLUX.1-dev}"
TRT_ENGINE_PATH="${TRT_ENGINE_PATH:-./trt/transformer.plan}"
PROMPTS_FILE="${PROMPTS_FILE:-benchmarks/prompts.json}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-3}"
RESULTS_DIR="${RESULTS_DIR:-benchmarks/results}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-1024}"
IMAGE_WIDTH="${IMAGE_WIDTH:-1024}"
SKIP_API_BENCHMARKS="${SKIP_API_BENCHMARKS:-false}"

# Ensure we're in the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================="
echo "Flux-1.dev Benchmarking Suite"
echo "=============================================="
echo "Model Path:       $MODEL_PATH"
echo "TRT Engine:       $TRT_ENGINE_PATH"
echo "Prompts File:     $PROMPTS_FILE"
echo "Iterations:       $ITERATIONS"
echo "Warmup:           $WARMUP"
echo "Results Dir:      $RESULTS_DIR"
echo "Image Size:       ${IMAGE_HEIGHT}x${IMAGE_WIDTH}"
echo "=============================================="
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: Prompts file not found: $PROMPTS_FILE"
    echo "Creating example prompts file..."

    cat > "$PROMPTS_FILE" <<'EOF'
[
    {
        "prompt": "A cute anime girl wearing a hoodie, sitting on a sofa in the living room sipping coffee",
        "seed": 42,
        "guidance_scale": 7.5,
        "num_inference_steps": 28
    },
    {
        "prompt": "A futuristic cityscape at night with neon lights reflecting on wet streets",
        "seed": 123,
        "guidance_scale": 7.5,
        "num_inference_steps": 28
    },
    {
        "prompt": "A serene mountain landscape with a crystal clear lake and pine trees",
        "seed": 456,
        "guidance_scale": 7.0,
        "num_inference_steps": 28
    },
    {
        "prompt": "A steampunk robot playing chess in a Victorian library",
        "seed": 789,
        "guidance_scale": 8.0,
        "num_inference_steps": 28
    },
    {
        "prompt": "A magical forest with glowing mushrooms and fireflies at twilight",
        "seed": 1011,
        "guidance_scale": 7.5,
        "num_inference_steps": 28
    }
]
EOF
    echo "Created example prompts file at: $PROMPTS_FILE"
fi

# Batch sizes to test
BATCH_SIZES=(1)

echo ""
echo "=============================================="
echo "Starting Benchmark Runs"
echo "=============================================="
echo ""

# Run HuggingFace Default benchmarks (stock, unmodified model)
echo ">>> Running HUGGINGFACE DEFAULT (Stock, Unmodified) benchmarks..."
for bs in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "--- HuggingFace Default (Stock), Batch Size: $bs ---"
    python3 benchmarks/flux_hf_default.py \
        --image_prompts_file "$PROMPTS_FILE" \
        --batch_size "$bs" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --output_csv "$RESULTS_DIR/flux_hf_default_bs${bs}.csv" \
        --label "hf_default_fp16_bs${bs}" \
        --model_path "$MODEL_PATH" \
        --height "$IMAGE_HEIGHT" \
        --width "$IMAGE_WIDTH"
    echo ""
done

# Run baseline FP16 benchmarks (miner-modified code)
echo ">>> Running MINER BASELINE (PyTorch FP16, Modified Code) benchmarks..."
for bs in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "--- Baseline FP16, Batch Size: $bs ---"
    python3 benchmarks/benchmark_flux_local.py \
        --image_prompts_file "$PROMPTS_FILE" \
        --batch_size "$bs" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --output_csv "$RESULTS_DIR/flux_fp16_bs${bs}.csv" \
        --label "fp16_baseline_bs${bs}" \
        --model_path "$MODEL_PATH" \
        --height "$IMAGE_HEIGHT" \
        --width "$IMAGE_WIDTH"
    echo ""
done

# Run TensorRT benchmarks (only if engine exists)
if [ -f "$TRT_ENGINE_PATH" ]; then
    echo ">>> Running MINER TENSORRT (Optimized, Modified Code) benchmarks..."
    for bs in "${BATCH_SIZES[@]}"; do
        echo ""
        echo "--- TensorRT FP16, Batch Size: $bs ---"
        python3 benchmarks/benchmark_flux_local.py \
            --image_prompts_file "$PROMPTS_FILE" \
            --batch_size "$bs" \
            --iterations "$ITERATIONS" \
            --warmup "$WARMUP" \
            --output_csv "$RESULTS_DIR/flux_trt_bs${bs}.csv" \
            --label "trt_fp16_bs${bs}" \
            --use_trt \
            --trt_engine_path "$TRT_ENGINE_PATH" \
            --model_path "$MODEL_PATH" \
            --height "$IMAGE_HEIGHT" \
            --width "$IMAGE_WIDTH"
        echo ""
    done
else
    echo ""
    echo "WARNING: TensorRT engine not found at: $TRT_ENGINE_PATH"
    echo "Skipping TensorRT benchmarks."
    echo "To build a TensorRT engine, run:"
    echo "  python trt.py"
    echo ""
fi

# Run API provider benchmarks (optional)
if [ "$SKIP_API_BENCHMARKS" != "true" ]; then
    echo ""
    echo "=============================================="
    echo "Cloud API Provider Benchmarks"
    echo "=============================================="
    echo ""

    # Replicate API
    if [ -n "${REPLICATE_API_TOKEN:-}" ]; then
        echo ">>> Running REPLICATE API benchmarks..."
        for bs in "${BATCH_SIZES[@]}"; do
            echo ""
            echo "--- Replicate API, Batch Size: $bs ---"
            python3 benchmarks/flux_replicate.py \
                --image_prompts_file "$PROMPTS_FILE" \
                --batch_size "$bs" \
                --iterations "$ITERATIONS" \
                --output_csv "$RESULTS_DIR/flux_replicate_bs${bs}.csv" \
                --label "replicate_api_bs${bs}" \
                --height "$IMAGE_HEIGHT" \
                --width "$IMAGE_WIDTH"
            echo ""
        done
    else
        echo ">>> Skipping Replicate API (REPLICATE_API_TOKEN not set)"
    fi

    # FAL.ai API
    if [ -n "${FAL_KEY:-}" ]; then
        echo ">>> Running FAL.AI API benchmarks..."
        for bs in "${BATCH_SIZES[@]}"; do
            echo ""
            echo "--- FAL.ai API, Batch Size: $bs ---"
            python3 benchmarks/flux_fal.py \
                --image_prompts_file "$PROMPTS_FILE" \
                --batch_size "$bs" \
                --iterations "$ITERATIONS" \
                --output_csv "$RESULTS_DIR/flux_fal_bs${bs}.csv" \
                --label "fal_api_bs${bs}" \
                --height "$IMAGE_HEIGHT" \
                --width "$IMAGE_WIDTH"
            echo ""
        done
    else
        echo ">>> Skipping FAL.ai API (FAL_KEY not set)"
    fi

    # Together.ai API
    if [ -n "${TOGETHER_API_KEY:-}" ]; then
        echo ">>> Running TOGETHER.AI API benchmarks..."
        for bs in "${BATCH_SIZES[@]}"; do
            echo ""
            echo "--- Together.ai API, Batch Size: $bs ---"
            python3 benchmarks/flux_together.py \
                --image_prompts_file "$PROMPTS_FILE" \
                --batch_size "$bs" \
                --iterations "$ITERATIONS" \
                --output_csv "$RESULTS_DIR/flux_together_bs${bs}.csv" \
                --label "together_api_bs${bs}" \
                --height "$IMAGE_HEIGHT" \
                --width "$IMAGE_WIDTH"
            echo ""
        done
    else
        echo ">>> Skipping Together.ai API (TOGETHER_API_KEY not set)"
    fi

    echo ""
else
    echo ""
    echo "Skipping API provider benchmarks (SKIP_API_BENCHMARKS=true)"
    echo ""
fi

echo ""
echo "=============================================="
echo "All Benchmarks Complete"
echo "=============================================="
echo ""

# Generate summary if Python script exists
if [ -f "benchmarks/summarize_results.py" ]; then
    echo ">>> Generating summary report..."
    python3 benchmarks/summarize_results.py "$RESULTS_DIR"/*.csv
else
    echo ">>> Summary script not found. Skipping summary generation."
fi

echo ""
echo "Results saved to: $RESULTS_DIR"
echo "=============================================="
