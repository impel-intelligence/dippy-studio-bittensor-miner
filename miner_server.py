import os
import sys
import json
import asyncio
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor
import subprocess
import requests

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from config_generator import generate_training_config, cleanup_job_files
from lora_generate_image import TRTInferenceServer, InferenceRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dippy Studio Bittensor Miner Server",
    description="Combined LoRA training and TensorRT inference server",
    version="1.0.0"
)

trt_server: Optional[TRTInferenceServer] = None
training_executor = ThreadPoolExecutor(max_workers=2)
inference_jobs: Dict[str, Dict[str, Any]] = {}
training_jobs: Dict[str, Dict[str, Any]] = {}

class TrainingRequest(BaseModel):
    job_type: str = "lora_training"
    params: Dict[str, Any]
    job_id: str
    validator_endpoint: Optional[str] = None

class InferenceRequestModel(BaseModel):
    prompt: str
    lora_path: Optional[str] = None  # Path to LoRA weights (.safetensors)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    num_inference_steps: int = Field(default=28, ge=1, le=100)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    seed: Optional[int] = None
    job_id: Optional[str] = None

class ServerConfig(BaseModel):
    enable_training: bool = Field(default=True, description="Enable training endpoints")
    enable_inference: bool = Field(default=True, description="Enable TRT inference endpoints")
    training_port: int = Field(default=8091, description="Port for training API")
    inference_port: int = Field(default=8092, description="Port for inference API")
    trt_engine_path: Optional[str] = Field(default=None, description="Path to TRT engine")
    model_path: str = Field(default="black-forest-labs/FLUX.1-dev", description="Base model path")
    output_dir: str = Field(default="./output", description="Output directory")

def load_trt_server():
    """Lazily load TRT server when needed"""
    global trt_server

    if trt_server is not None:
        return  # Already loaded

    config = app.state.config
    if config.enable_inference and config.trt_engine_path:
        try:
            logger.info("Lazy-loading TRT inference server...")
            trt_server = TRTInferenceServer(
                base_model_path=config.model_path,
                engine_path=config.trt_engine_path,
                mapping_path="./trt/mapping.json"
            )
            # Start the inference server in background
            trt_server.start()
            logger.info("TRT inference server loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TRT inference server: {e}")
            raise

def unload_trt_server():
    """Unload TRT server to free GPU memory"""
    global trt_server

    if trt_server is not None:
        logger.info("Unloading TRT server to free GPU memory...")
        try:
            trt_server.stop()
            trt_server = None

            # Force GPU memory cleanup
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            logger.info("TRT server unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading TRT server: {e}")

async def initialize_servers():
    # Store config globally but don't initialize TRT yet
    config = ServerConfig(
        enable_training=os.getenv("ENABLE_TRAINING", "true").lower() == "true",
        enable_inference=os.getenv("ENABLE_INFERENCE", "true").lower() == "true",
        trt_engine_path=os.getenv("TRT_ENGINE_PATH", "/app/trt/transformer.plan"),
        model_path=os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev"),
        output_dir=os.getenv("OUTPUT_DIR", "./output")
    )

    # Create output directory if it doesn't exist
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Store config for later use
    app.state.config = config
    app.state.OUTPUT_DIR = config.output_dir
    # Get service URL from environment or construct from host/port
    service_url = os.getenv("SERVICE_URL")
    if not service_url:
        host = os.getenv("MINER_SERVER_HOST", "0.0.0.0")
        port = os.getenv("MINER_SERVER_PORT", "8091")
        # In production, this should be set to your actual domain
        service_url = f"http://localhost:{port}"
    app.state.SERVICE_URL = service_url

    # Mount static files for serving images
    app.mount("/images", StaticFiles(directory=config.output_dir), name="images")

    logger.info(f"Server initialized. Output directory: {config.output_dir}")
    logger.info(f"Service URL: {app.state.SERVICE_URL}")
    logger.info("TRT will be loaded on first inference request.")

    return config

@app.post("/train", status_code=201)
async def train(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start LoRA training asynchronously and return immediately"""

    if request.job_type != "lora_training":
        raise HTTPException(status_code=400, detail="Only lora_training supported")

    try:
        job_id = request.job_id  # must be present
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")

        # Store job info
        training_jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "type": request.job_type,
            "created_at": datetime.now().isoformat(),
            "params": request.params
        }

        # Start training in background
        background_tasks.add_task(
            run_training_job,
            job_id,
            request.params,
            request.validator_endpoint
        )

        # Return immediately with 201 OK (empty response body matches training_server.py)
        return

    except Exception as e:
        logger.error(f"Error submitting training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_training_job(job_id: str, params: Dict[str, Any], validator_endpoint: Optional[str]):
    """Background task to run training and post results to validator"""

    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now().isoformat()

        # Unload TRT server before training to free GPU memory
        unload_trt_server()

        # Generate config from params
        config_path = generate_training_config(job_id, params)

        # Run the actual training
        result = subprocess.run([
            "python3", "run.py", config_path, "--seed", str(params.get("seed", 42))
        ], capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Training failed for job {job_id}: {result.stderr}")
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = result.stderr
            training_jobs[job_id]["failed_at"] = datetime.now().isoformat()
            return

        # Upload to HuggingFace and get model URL
        huggingface_url = upload_to_huggingface(job_id)

        # Update job status
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        training_jobs[job_id]["huggingface_url"] = huggingface_url

        # Post results to validator
        if validator_endpoint:
            post_results_to_validator(job_id, huggingface_url, validator_endpoint)

    except Exception as e:
        logger.error(f"Error in background training for job {job_id}: {e}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["failed_at"] = datetime.now().isoformat()
    finally:
        cleanup_job_files(job_id)
        # logger.info(f"Keeping job files for debugging: job_{job_id}")

def upload_to_huggingface(job_id: str) -> str:
    """Upload trained model to HuggingFace and return model URL"""
    from huggingface_hub import HfApi, upload_folder

    model_dir = Path(f"output/job_{job_id}")
    if not model_dir.exists():
        model_dir = Path(app.state.OUTPUT_DIR) / f"job_{job_id}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model output not found for job {job_id} at {model_dir}")

    # Create unique model name
    model_name = f"lora_{job_id}"

    # Upload to HuggingFace (uses token's associated user)
    api = HfApi()
    repo_id = api.create_repo(repo_id=model_name, exist_ok=True).repo_id
    upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model"
    )

    # Return the full HuggingFace URL
    return f"https://huggingface.co/{repo_id}"

def post_results_to_validator(job_id: str, huggingface_url: str, validator_endpoint: str):
    """Post completion results to validator"""
    import requests

    payload = {
        "job_id": job_id,
        "hugging_face_url": huggingface_url
    }

    try:
        response = requests.post(
            validator_endpoint,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        logger.info(f"Successfully posted results for job {job_id}")

    except requests.RequestException as e:
        logger.error(f"Failed to post results to validator: {e}")

async def notify_validator(endpoint: str, job_id: str, status: str, output_path: str = None, error: str = None):
    import aiohttp

    payload = {
        "job_id": job_id,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

    if output_path:
        payload["output_path"] = output_path
    if error:
        payload["error"] = error

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                if response.status != 200:
                    logger.warning(f"Validator notification failed: {response.status}")
    except Exception as e:
        logger.error(f"Failed to notify validator: {e}")

@app.post("/inference")
async def generate_image(request: InferenceRequestModel):
    # Lazy-load TRT server on first inference request
    if trt_server is None:
        try:
            load_trt_server()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize TRT inference server: {str(e)}"
            )

    try:
        job_id = request.job_id or str(uuid.uuid4())

        # Generate output path for this job using global OUTPUT_DIR
        output_path = str(Path(app.state.OUTPUT_DIR) / f"{job_id}.png")

        # Create inference request
        inference_req = InferenceRequest(
            prompt=request.prompt,
            output_path=output_path,
            lora_path=request.lora_path or "",
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed or 42
        )

        # Store job info
        inference_jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "request": request.dict()
        }

        # Submit to TRT server
        trt_server.submit(inference_req)

        # Wait for inference to complete
        max_wait_time = 60  # seconds
        poll_interval = 0.5  # seconds
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            output_path = Path(app.state.OUTPUT_DIR) / f"{job_id}.png"
            if output_path.exists():
                # Image is ready
                image_url = f"{app.state.SERVICE_URL}/images/{job_id}.png"

                # Update job status
                inference_jobs[job_id]["status"] = "completed"
                inference_jobs[job_id]["output_path"] = str(output_path)
                inference_jobs[job_id]["completed_at"] = datetime.now().isoformat()

                return {
                    "success": True,
                    "job_id": job_id,
                    "status": "completed",
                    "message": "Image generated successfully",
                    "result_url": f"/inference/result/{job_id}",
                    "image_url": image_url  # Full URL to the image
                }

            # Wait before checking again
            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval

        # Timeout occurred
        inference_jobs[job_id]["status"] = "timeout"
        raise HTTPException(
            status_code=504,
            detail=f"Inference timeout after {max_wait_time} seconds"
        )

    except Exception as e:
        logger.error(f"Error submitting inference job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inference/status/{job_id}")
async def get_inference_status(job_id: str):
    if job_id not in inference_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = inference_jobs[job_id]

    # Check if output exists using global OUTPUT_DIR
    output_path = Path(app.state.OUTPUT_DIR) / f"{job_id}.png"
    if output_path.exists():
        job["status"] = "completed"
        job["output_path"] = str(output_path)
        job["result_url"] = f"/inference/result/{job_id}"
        job["image_url"] = f"{app.state.SERVICE_URL}/images/{job_id}.png"

    return job

@app.get("/inference/result/{job_id}")
async def get_inference_result(job_id: str):
    if job_id not in inference_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    output_path = Path(app.state.OUTPUT_DIR) / f"{job_id}.png"

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found. Job may still be processing.")

    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename=f"{job_id}.png"
    )

@app.get("/training/status/{job_id}")
async def get_training_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return training_jobs[job_id]

@app.get("/jobs")
async def list_jobs(job_type: Optional[str] = None):
    if job_type == "training":
        return {"training_jobs": list(training_jobs.values())}
    elif job_type == "inference":
        return {"inference_jobs": list(inference_jobs.values())}
    else:
        return {
            "training_jobs": list(training_jobs.values()),
            "inference_jobs": list(inference_jobs.values())
        }

@app.get("/health")
async def health_check():
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "training": "enabled",
            "inference": "enabled" if trt_server is not None else "disabled"
        }
    }

    if trt_server is not None:
        status["services"]["trt_engine"] = "loaded"

    return status

@app.get("/")
async def root():
    return {
        "name": "Dippy Studio Bittensor Miner Server",
        "version": "1.0.0",
        "endpoints": {
            "training": {
                "POST /train": "Submit training job",
                "GET /training/status/{job_id}": "Get training job status"
            },
            "inference": {
                "POST /inference": "Generate image with TRT",
                "GET /inference/status/{job_id}": "Get inference job status",
                "GET /inference/result/{job_id}": "Download generated image"
            },
            "general": {
                "GET /jobs": "List all jobs",
                "GET /health": "Health check"
            }
        }
    }

async def startup_event():
    await initialize_servers()

async def shutdown_event():
    global trt_server
    if trt_server:
        trt_server.stop()
    training_executor.shutdown(wait=True)

def main():
    app.add_event_handler("startup", startup_event)
    app.add_event_handler("shutdown", shutdown_event)

    port = int(os.getenv("MINER_SERVER_PORT", "8091"))
    host = os.getenv("MINER_SERVER_HOST", "0.0.0.0")

    logger.info(f"Starting Dippy Studio Bittensor Miner Server on {host}:{port}")

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()