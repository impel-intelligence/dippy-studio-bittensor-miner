# Dippy Studio Bittensor Miner

A specialized Bittensor subnet miner (subnet 231 on testnet) for AI model training and inference, featuring LoRA fine-tuning of FLUX.1-dev with TensorRT acceleration for high-performance inference.

## Prerequisites

You will need access to a VPS (or similar setup) with a capable enough GPu. We recommend at least 1 A100 80GB (PCIe or SXM will do fine).
For managing python installation, we recommend [uv](https://docs.astral.sh/uv/getting-started/installation/) and only officially support this configuration.


Before running this miner, you must:

1. **Set up a HuggingFace account and token**:
   - Create an account at [HuggingFace](https://huggingface.co)
   - Generate an access token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - The token must have write permissions to create and upload models

2. **Accept FLUX.1-dev model license**:
   - Visit the [FLUX.1-dev model page](https://huggingface.co/black-forest-labs/FLUX.1-dev)
   - Read and accept the license agreement
   - Without accepting the license, the model download will fail

## Quick Start

### Choose Your Deployment Mode

The miner can run in two modes:
- **Inference Mode**: Serves pre-trained models with TensorRT acceleration
- **Training Mode**: Fine-tunes LoRA models on FLUX.1-dev

```bash
# 1. Clone the repository
git clone https://github.com/dippy-ai/dippy-studio-bittensor-miner
cd dippy-studio-bittensor-miner

# 2. Create .env file with your credentials
cp .env.example .env
# Edit .env and add your HF_TOKEN

# 3. Choose your deployment mode:

# For INFERENCE only (requires TRT engines)
make setup-inference

# For TRAINING only (requires base model)
make setup-training

# 4. Check logs
make logs
```

The miner server will be available at `http://localhost:8091`.

### Available Make Commands

```bash
# Deployment Modes
make setup-inference  # Deploy inference-only server (auto-builds TRT if needed)
make setup-training   # Deploy training-only server

# Building & Management
make build            # Build Docker images
make trt-build        # Build TRT engine (20-30 min)
make trt-rebuild      # Force rebuild TRT engine
make up               # Start miner service
make down             # Stop miner service
make logs             # Follow miner logs
make restart          # Restart miner service

# Maintenance
make clean-cache      # Remove all cached TRT engines
make help             # Show all available commands
```

## Architecture

The system consists of three main components:

### 1. Reverse Proxy (Public Entrypoint)
The reverse proxy handles Bittensor authentication and routes requests to internal services.

**Setup:**
1. Register on testnet: `btcli s register --netuid 231 --subtensor.network test`
2. Transfer 0.01 testnet TAO to `5FU2csPXS5CZfMVd2Ahdis9DNaYmpTCX4rsN11UW7ghdx24A` for mining permit
3. Configure environment variables in `reverse_proxy/.env`
4. Install and run:
   ```bash
   cd reverse_proxy
   uv pip install -e .[dev]
   python server.py
   ```

### 2. Miner Server (Separate Training & Inference Modes)
A FastAPI server (`miner_server.py`) that can run in either training or inference mode.

**Features:**
- **Training Mode**: Background LoRA fine-tuning with HuggingFace upload
- **Inference Mode**: TensorRT-accelerated image generation with LoRA support and automatic engine preloading
- **Static file serving**: Direct image URL access

**Endpoints:**
- `POST /train` - Submit LoRA training job
- `POST /inference` - Generate image (with optional LoRA)
- `GET /training/status/{job_id}` - Check training status
- `GET /inference/status/{job_id}` - Check inference status
- `GET /inference/result/{job_id}` - Download generated image
- `GET /health` - Health check

### 3. TensorRT Engine
High-performance inference engine for FLUX.1-dev model.

**Building the Engine:**
```bash
# Using make (recommended)
make trt-build     # Build if not exists
make trt-rebuild   # Force rebuild

# Or Docker directly
docker compose run --rm trt-builder
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
HF_TOKEN=your_huggingface_token_here        # HuggingFace token with write permissions

# Mode Configuration (set based on deployment choice)
ENABLE_TRAINING=true                        # Set to false for inference-only mode
ENABLE_INFERENCE=true                       # Set to false for training-only mode
MODEL_PATH=black-forest-labs/FLUX.1-dev    # Base model path
OUTPUT_DIR=/app/output                      # Output directory in container (mapped to ./output on host)
MINER_SERVER_PORT=8091                      # Server port
MINER_SERVER_HOST=0.0.0.0                   # Server host
SERVICE_URL=http://localhost:8091           # Public URL for image serving
```

**Note:** Training outputs and generated images are persisted in the `./output` directory on the host, which is mapped to `/app/output` in the container.

For the reverse proxy, create `reverse_proxy/.env`:

```bash
# Required
MINER_HOTKEY=your_miner_hotkey_here         # Bittensor miner hotkey

# Service endpoints (internal)
TRAINING_SERVER_URL=http://localhost:8091   # Miner server for training
INFERENCE_SERVER_URL=http://localhost:8091  # Miner server for inference
```


## How It Works

The miner provides two main services:

### Training Service
1. **Receives training requests** via POST to `/train`:
   - `job_type`: "lora_training"
   - `params`: Including prompt, image_b64, seed
   - `job_id`: Unique identifier
   - `validator_endpoint`: Callback URL for results

2. **Processes training jobs** in background:
   - Generates configuration from request parameters
   - Performs LoRA fine-tuning on FLUX.1-dev
   - Uses provided prompts and images for training

3. **Uploads trained models** to HuggingFace:
   - Creates a new repository for each training job
   - Uploads LoRA weights and metadata
   - Makes models publicly accessible

4. **Reports completion** back to validator endpoint

### Inference Service
1. **Receives generation requests** via POST to `/inference`:
   - `prompt`: Text description for image generation
   - `lora_path`: Optional path to LoRA weights
   - `width/height`: Image dimensions
   - `num_inference_steps`: Quality control
   - `guidance_scale`: Prompt adherence strength
   - `seed`: For reproducibility

2. **Generates images** using TensorRT:
   - Uses pre-built TRT engine for fast inference
   - Supports dynamic LoRA switching via refitting
   - Returns image URL immediately after generation

3. **Serves generated images** via static file server:
   - Images accessible at `/images/{job_id}.png`
   - Direct URL access for validator retrieval

## API Examples

### Submit Training Job
```bash
curl -X POST http://localhost:8091/train \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "lora_training",
    "job_id": "test-training-001",
    "params": {
      "prompt": "A cute anime girl",
      "image_b64": "...",
      "seed": 42
    }
  }'
```

### Generate Image with Base Model
```bash
curl -X POST http://localhost:8091/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 7.5,
    "seed": 42
  }'
```

### Generate Image with LoRA
```bash
curl -X POST http://localhost:8091/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A portrait in anime style",
    "lora_path": "/app/models/anime_lora.safetensors",
    "width": 1024,
    "height": 1024
  }'
```

### Check Job Status
```bash
# Training status
curl http://localhost:8091/training/status/{job_id}

# Inference status
curl http://localhost:8091/inference/status/{job_id}
```

## System Requirements

- **GPU**: NVIDIA GPU with at least 24GB VRAM (recommended: A100, H100)
- **CUDA**: Version 11.8 or higher
- **RAM**: 32GB minimum
- **Storage**: 100GB+ for model weights and training data
- **Docker**: Latest version with nvidia-container-toolkit

## Monitoring and Logs

- **API logs**: Check `docker compose logs -f`
- **Training progress**: Monitor individual job outputs in logs
- **GPU usage**: Use `nvidia-smi` to monitor GPU utilization

## Troubleshooting

### TensorRT Engine Issues
If inference fails with TRT errors:
1. Rebuild the engine: `docker compose --profile build up trt-builder --force-recreate`
2. Check GPU compatibility (requires compute capability 7.0+)
3. Ensure sufficient GPU memory (24GB+ recommended)

### LoRA Loading Issues
If LoRA weights don't apply correctly:
1. Verify LoRA was trained for FLUX.1-dev (not SDXL or other models)
2. Check file path is accessible within container
3. Ensure LoRA file is in .safetensors format

### Model Download Issues
If you encounter permission errors downloading FLUX.1-dev:
1. Ensure you've accepted the model license on HuggingFace
2. Verify your HF_TOKEN is correctly set
3. Check that your token has read permissions

### GPU Memory Issues
If training fails with CUDA out of memory:
1. Use separate deployment modes (inference OR training, not both)
2. Reduce batch size in configuration
3. Enable gradient checkpointing
4. Use a GPU with more VRAM

### Docker Issues
If container fails to start:
1. Check nvidia-docker is installed: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
2. Verify Docker has GPU access
3. Check port 8091 is not in use: `lsof -i :8091`

### Connection Issues
If the miner can't connect to validators:
1. Check firewall settings for port 8091
2. Ensure Docker networking is properly configured
3. Verify validator endpoints are accessible
4. Check SERVICE_URL environment variable for production

## Additional Resources

- [Bittensor Documentation](https://docs.bittensor.com)
- [FLUX.1 Model Card](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [LoRA Training Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)

## Support

For issues and questions:
- Check existing issues in the repository
- Join the Bittensor Discord community
- Review validator documentation for integration details