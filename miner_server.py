import os
import asyncio
import uuid
import time
import importlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum
import aiohttp
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image

from lora_generate_image import TRTInferenceServer, InferenceRequest
from kontext_pipeline import KontextInferenceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dippy Studio Bittensor Miner Server",
    description="TensorRT inference and Kontext editing server",
    version="1.0.0"
)

trt_server: Optional[TRTInferenceServer] = None
kontext_manager: Optional[KontextInferenceManager] = None
inference_jobs: Dict[str, Dict[str, Any]] = {}
edit_jobs: Dict[str, Dict[str, Any]] = {}

class InferenceRequestModel(BaseModel):
    prompt: str
    lora_path: Optional[str] = None  # Path to LoRA weights (.safetensors)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    num_inference_steps: int = Field(default=28, ge=1, le=100)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    seed: Optional[int] = None
    job_id: Optional[str] = None
    callback_url: Optional[str] = Field(
        default=None,
        description="HTTP endpoint that receives the generated image"
    )
    callback_secret: Optional[str] = Field(
        default=None,
        description="Secret forwarded via X-Callback-Secret header"
    )
    expiry: Optional[datetime] = Field(
        default=None,
        description="ISO 8601 timestamp (UTC) after which callbacks are skipped"
    )

class EditRequest(BaseModel):
    """Request model for image editing with FLUX.1-Kontext-dev"""
    prompt: str = Field(..., description="Text instruction for editing the image")
    image_url: Optional[str] = Field(
        default=None,
        description="HTTP(S) URL to fetch input image from"
    )
    image_b64: Optional[str] = Field(
        default=None,
        description="Base64 encoded input image (PNG/JPEG)"
    )
    seed: int = Field(..., description="Random seed (required for determinism)")
    guidance_scale: float = Field(
        default=2.5,
        ge=1.0,
        le=20.0,
        description="How strongly to follow the prompt"
    )
    num_inference_steps: int = Field(
        default=28,
        ge=1,
        le=100,
        description="Number of denoising steps"
    )
    job_id: Optional[str] = Field(
        default=None,
        description="Optional job ID (auto-generated if not provided)"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="HTTP endpoint that receives the edited image"
    )
    callback_secret: Optional[str] = Field(
        default=None,
        description="Secret forwarded via X-Callback-Secret header"
    )
    expiry: Optional[datetime] = Field(
        default=None,
        description="ISO 8601 timestamp (UTC) after which callbacks are skipped"
    )

class ServerConfig(BaseModel):
    enable_inference: bool = Field(default=True, description="Enable TRT inference endpoints")
    enable_kontext_edit: bool = Field(default=False, description="Enable Kontext image editing endpoints")
    trt_engine_path: Optional[str] = Field(default=None, description="Path to TRT engine")
    model_path: str = Field(default="black-forest-labs/FLUX.1-dev", description="Base model path")
    output_dir: str = Field(default="./output", description="Output directory")
    preload_engines: bool = Field(default=True, description="Preload TRT engines at startup")


class CallbackStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    INFERENCE = "inference"
    EDIT = "edit"


@dataclass
class EditTaskParams:
    prompt: str
    image_url: Optional[str]
    image_b64: Optional[str]
    seed: int
    guidance_scale: float
    num_inference_steps: int


@dataclass
class QueuedTask:
    job_id: str
    task_type: TaskType
    inference_request: Optional[InferenceRequest]
    edit_params: Optional[EditTaskParams]
    callback_url: Optional[str]
    callback_secret: Optional[str]
    expiry: Optional[datetime]
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def normalize_expiry(expiry: Optional[datetime]) -> Optional[datetime]:
    if expiry is None:
        return None
    if expiry.tzinfo is None:
        return expiry.replace(tzinfo=timezone.utc)
    return expiry.astimezone(timezone.utc)


def load_job_record_hook() -> Optional[Callable[[str, Dict[str, Any]], None]]:
    """
    Optional hook loader for persisting job records before cleanup.
    Set JOB_RECORD_HOOK to module:function to enable.
    """
    hook_path = os.getenv("JOB_RECORD_HOOK")
    if not hook_path:
        return None

    try:
        module_path, func_name = hook_path.split(":")
        module = importlib.import_module(module_path)
        hook = getattr(module, func_name)
        if not callable(hook):
            raise TypeError(f"{hook_path} is not callable")
        logger.info(f"Loaded job record hook: {hook_path}")
        return hook
    except Exception as exc:
        logger.error(f"Failed to load JOB_RECORD_HOOK '{hook_path}': {exc}")
        return None


def persist_job_record(job_type: TaskType, job_record: Dict[str, Any]) -> None:
    """Invoke optional user-provided hook for persisting job records."""
    hook: Optional[Callable[[str, Dict[str, Any]], None]] = getattr(app.state, "job_record_hook", None)
    if not hook:
        return

    try:
        hook(job_type.value, job_record)
    except Exception as exc:
        logger.error(f"Job record hook failed for {job_type.value}:{job_record.get('id')}: {exc}")


async def _cleanup_job_record(job_type: TaskType, job_id: str, job_record: Dict[str, Any]):
    """Remove job record after retention period and invoke persistence hook."""
    ttl_seconds = getattr(app.state, "job_record_ttl", 300)
    if ttl_seconds > 0:
        await asyncio.sleep(ttl_seconds)

    persist_job_record(job_type, job_record)

    if job_type == TaskType.INFERENCE:
        removed = inference_jobs.pop(job_id, None)
    else:
        removed = edit_jobs.pop(job_id, None)

    if removed:
        logger.info(f"Cleaned up job record for {job_type.value} job {job_id}")


def schedule_job_cleanup(job_type: TaskType, job_id: str, job_record: Dict[str, Any]):
    """Schedule asynchronous cleanup without blocking the worker."""
    record_copy = dict(job_record)
    asyncio.create_task(_cleanup_job_record(job_type, job_id, record_copy))


async def wait_for_image(output_path: Path, timeout: Optional[float], poll_interval: float = 0.5) -> Path:
    start_time = time.monotonic()
    while True:
        if output_path.exists():
            return output_path

        if timeout and timeout > 0 and (time.monotonic() - start_time) > timeout:
            raise TimeoutError(
                f"Timed out after {timeout} seconds waiting for image at {output_path}"
            )

        await asyncio.sleep(poll_interval)


async def dispatch_callback(
    task: QueuedTask,
    status: CallbackStatus,
    *,
    image_path: Optional[Path] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    attempted_at = datetime.now(timezone.utc)
    attempt_iso = attempted_at.isoformat()

    if not task.callback_url:
        return {
            "status": "skipped",
            "reason": "missing_callback_url",
            "attempted_at": attempt_iso,
            "payload_status": status.value
        }

    expiry = task.expiry
    if expiry and attempted_at > expiry:
        return {
            "status": "expired",
            "attempted_at": attempt_iso,
            "expiry": expiry.isoformat(),
            "payload_status": status.value
        }

    callback_timeout = getattr(app.state, "callback_timeout", 30)
    headers = {}
    if task.callback_secret:
        headers["X-Callback-Secret"] = task.callback_secret

    form = aiohttp.FormData()
    form.add_field("job_id", task.job_id)
    form.add_field("status", status.value)
    form.add_field("completed_at", attempt_iso)
    if error:
        form.add_field("error", error)

    image_bytes: Optional[bytes] = None
    if image_path is not None:
        try:
            image_bytes = await asyncio.to_thread(image_path.read_bytes)
        except FileNotFoundError:
            logger.warning(f"Image path not found for job {task.job_id}: {image_path}")
        except Exception as exc:
            logger.error(f"Failed to read generated image for job {task.job_id}: {exc}")
            return {
                "status": "failed",
                "error": f"Unable to read image: {exc}",
                "attempted_at": attempt_iso,
                "payload_status": status.value
            }

    if image_bytes is not None:
        form.add_field("image", image_bytes, filename=image_path.name, content_type="image/png")
        form.add_field("image_url", f"{app.state.SERVICE_URL}/images/{image_path.stem}.png")

    timeout_config = aiohttp.ClientTimeout(total=callback_timeout)

    try:
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.post(task.callback_url, data=form, headers=headers) as response:
                response_text = await response.text()
                return {
                    "status": "delivered" if response.status < 400 else "error",
                    "status_code": response.status,
                    "response_preview": response_text[:500],
                    "attempted_at": attempt_iso,
                    "payload_status": status.value
                }
    except Exception as exc:
        logger.error(f"Callback delivery failed for job {task.job_id}: {exc}")
        return {
            "status": "failed",
            "error": str(exc),
            "attempted_at": attempt_iso,
            "payload_status": status.value
        }


def decode_base64_image(b64_data: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.

    Args:
        b64_data: Base64 encoded image data

    Returns:
        PIL Image

    Raises:
        HTTPException: If base64 is invalid or not an image
    """
    try:
        image_bytes = base64.b64decode(b64_data)
        image = Image.open(BytesIO(image_bytes))
        return image
    except base64.binascii.Error as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 encoding: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {str(e)}"
        )


async def load_edit_input_image(
    image_url: Optional[str],
    image_b64: Optional[str],
    output_dir: str
) -> tuple[Image.Image, str]:
    """
    Load input image from URL or base64.

    Priority: image_url > image_b64
    Falls back to image_b64 if URL fetch fails.

    Args:
        image_url: Optional HTTP(S) URL to fetch image from
        image_b64: Optional base64 encoded fallback image
        output_dir: Output directory containing job images (unused, kept for compatibility)

    Returns:
        Tuple of (image, source) where source describes the image origin

    Raises:
        HTTPException: If no valid image source available
    """
    # Try URL first (preferred method)
    if image_url:
        try:
            logger.info(f"Fetching image from URL: {image_url[:100]}...")
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {response.reason}")
                    
                    image_bytes = await response.read()
                    logger.info(f"Downloaded {len(image_bytes)} bytes from URL")
                    
                    # Try to open as image
                    image = Image.open(BytesIO(image_bytes))
                    image.load()  # Verify it's a valid image
                    return image, "url"
        except Exception as e:
            error_msg = f"Failed to fetch image from URL: {str(e)}"
            logger.warning(error_msg)
            
            if image_b64:
                logger.info("Falling back to image_b64 after URL fetch failure")
                return decode_base64_image(image_b64), "url_fallback_to_base64"
            else:
                raise HTTPException(status_code=400, detail=error_msg)
    
    # Use base64
    if image_b64:
        logger.info("Loading image from base64 data")
        return decode_base64_image(image_b64), "base64"

    # No valid source
    raise HTTPException(
        status_code=400,
        detail="Must provide either image_url or image_b64"
    )

def load_trt_server():
    """Load TRT server (called at startup or lazily)"""
    global trt_server

    if trt_server is not None:
        return trt_server  # Already loaded

    config = app.state.config
    if config.enable_inference and config.trt_engine_path:
        try:
            logger.info("Initializing TRT inference server...")
            engine_path = Path(config.trt_engine_path)
            mapping_env = os.getenv("TRT_MAPPING_PATH")
            if mapping_env:
                mapping_path = Path(mapping_env)
            else:
                mapping_candidate = engine_path.with_name("mapping.json")
                mapping_path = mapping_candidate if mapping_candidate.exists() else Path("./trt/mapping.json")
            trt_server = TRTInferenceServer(
                base_model_path=config.model_path,
                engine_path=str(engine_path),
                mapping_path=str(mapping_path)
            )
            # Start the inference server in background
            trt_server.start()
            logger.info("TRT inference server loaded successfully")
            return trt_server
        except Exception as e:
            logger.error(f"Failed to load TRT inference server: {e}")
            raise
    return None

def get_kontext_manager() -> Optional[KontextInferenceManager]:
    """Get or create Kontext inference manager (lazy loading)."""
    global kontext_manager

    if kontext_manager is not None:
        return kontext_manager

    config = app.state.config
    if not config.enable_kontext_edit:
        logger.info("Kontext editing is disabled")
        return None

    try:
        logger.info("Initializing Kontext inference manager...")
        model_path = os.getenv("KONTEXT_MODEL_PATH", "black-forest-labs/FLUX.1-Kontext-dev")
        kontext_manager = KontextInferenceManager(model_path=model_path)
        logger.info("Kontext inference manager initialized (lazy load)")
        return kontext_manager
    except Exception as e:
        logger.error(f"Failed to initialize Kontext manager: {e}")
        raise

async def preload_trt_engines():
    """Preload TRT engines at startup"""
    config = app.state.config
    
    if not config.preload_engines or not config.enable_inference:
        logger.info("Engine preloading disabled, will load on first request")
        return
    
    if not config.trt_engine_path:
        logger.warning("No TRT engine path specified, skipping preload")
        return
        
    try:
        logger.info("Preloading TRT engines at startup...")
        start_time = time.time()
        
        # Load TRT server synchronously during startup
        server = load_trt_server()
        if server is None:
            logger.warning("TRT server could not be loaded")
            return
        
        logger.info(f"TRT engines preloaded in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to preload TRT engines: {e}")

async def initialize_servers():
    # Store config globally
    config = ServerConfig(
        enable_inference=os.getenv("ENABLE_INFERENCE", "true").lower() == "true",
        enable_kontext_edit=os.getenv("ENABLE_KONTEXT_EDIT", "false").lower() == "true",
        trt_engine_path=os.getenv("TRT_ENGINE_PATH", "/app/trt/transformer.plan"),
        model_path=os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev"),
        output_dir=os.getenv("OUTPUT_DIR", "./output"),
        preload_engines=os.getenv("PRELOAD_ENGINES", "true").lower() == "true"
    )

    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir).joinpath("edits").mkdir(parents=True, exist_ok=True)

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
    
    # Configure queues and workers
    app.state.inference_timeout = int(os.getenv("INFERENCE_TIMEOUT", "300"))
    app.state.callback_timeout = int(os.getenv("CALLBACK_TIMEOUT", "30"))
    app.state.job_record_ttl = max(0, int(os.getenv("JOB_RECORD_TTL_SECONDS", "300")))
    app.state.job_record_hook = load_job_record_hook()

    queue_maxsize_env = int(os.getenv("INFERENCE_QUEUE_MAXSIZE", "0"))
    queue_maxsize = max(0, queue_maxsize_env)
    worker_count = max(1, int(os.getenv("INFERENCE_WORKERS", "1")))
    queue_enabled = config.enable_inference or config.enable_kontext_edit

    if queue_enabled:
        app.state.job_queue = asyncio.Queue(maxsize=queue_maxsize)
        app.state.job_worker_tasks = [
            asyncio.create_task(job_queue_worker(idx))
            for idx in range(worker_count)
        ]
        queue_desc = "unbounded" if queue_maxsize == 0 else queue_maxsize
        logger.info(
            f"Job queue ready (workers={worker_count}, maxsize={queue_desc}, timeout={app.state.inference_timeout}s)"
        )
    else:
        app.state.job_queue = None
        app.state.job_worker_tasks = []

    # Start TRT engine preloading if enabled
    await preload_trt_engines()

    return config


async def handle_inference_task(task: QueuedTask, worker_id: int):
    job_id = task.job_id
    job_record = inference_jobs.get(job_id, {})
    job_record.update({
        "id": job_id,
        "status": "processing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "worker_id": worker_id
    })

    inference_jobs[job_id] = job_record

    try:
        server = load_trt_server()
        if server is None:
            raise RuntimeError("TRT inference server is not available")

        if task.inference_request is None:
            raise RuntimeError("Missing inference request payload")

        server.submit(task.inference_request)
        output_path = Path(task.inference_request.output_path)

        try:
            await wait_for_image(output_path, getattr(app.state, "inference_timeout", 300))
        except TimeoutError as exc:
            callback_info = await dispatch_callback(task, CallbackStatus.FAILED, error=str(exc))
            job_record.update({
                "status": "timeout",
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "callback": callback_info
            })
            return

        completed_at = datetime.now(timezone.utc).isoformat()
        job_record.update({
            "status": "completed",
            "completed_at": completed_at,
            "output_path": str(output_path),
            "result_url": f"/inference/result/{job_id}",
            "image_url": f"{app.state.SERVICE_URL}/images/{job_id}.png"
        })

        callback_info = await dispatch_callback(task, CallbackStatus.COMPLETED, image_path=output_path)
        job_record["callback"] = callback_info

    except Exception as exc:
        logger.error(f"Inference worker {worker_id} failed for job {job_id}: {exc}", exc_info=True)
        callback_info = await dispatch_callback(task, CallbackStatus.FAILED, error=str(exc))
        job_record.update({
            "status": "failed",
            "error": str(exc),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "callback": callback_info
        })
    finally:
        inference_jobs[job_id] = job_record
        schedule_job_cleanup(TaskType.INFERENCE, job_id, job_record)


async def handle_edit_task(task: QueuedTask, worker_id: int):
    job_id = task.job_id
    job_record = edit_jobs.get(job_id, {})
    job_record.update({
        "id": job_id,
        "status": "processing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "worker_id": worker_id
    })
    edit_jobs[job_id] = job_record

    try:
        if task.edit_params is None:
            raise RuntimeError("Missing edit parameters")

        image, image_source = await load_edit_input_image(
            image_url=task.edit_params.image_url,
            image_b64=task.edit_params.image_b64,
            output_dir=app.state.OUTPUT_DIR
        )
        job_record.setdefault("request", {})
        job_record["request"]["image_source"] = image_source

        manager = get_kontext_manager()
        if manager is None:
            raise RuntimeError("Kontext manager not available")

        logger.info(f"Processing edit job {job_id} with seed={task.edit_params.seed}")
        result_image = await asyncio.to_thread(
            manager.generate,
            prompt=task.edit_params.prompt,
            image=image,
            seed=task.edit_params.seed,
            guidance_scale=task.edit_params.guidance_scale,
            num_inference_steps=task.edit_params.num_inference_steps
        )

        output_dir = Path(app.state.OUTPUT_DIR) / "edits"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}.png"
        result_image.save(output_path)

        completed_at = datetime.now(timezone.utc).isoformat()
        job_record.update({
            "status": "completed",
            "completed_at": completed_at,
            "output_path": str(output_path),
            "result_url": f"/edit/result/{job_id}",
            "image_url": f"{app.state.SERVICE_URL}/images/edits/{job_id}.png"
        })

        if task.callback_url:
            logger.info(f"Dispatching callback for edit job {job_id} to {task.callback_url}")
            callback_info = await dispatch_callback(task, CallbackStatus.COMPLETED, image_path=output_path)
            job_record["callback"] = callback_info

    except Exception as exc:
        logger.error(f"Edit job {job_id} failed: {exc}", exc_info=True)
        job_record.update({
            "status": "failed",
            "error": str(exc),
            "failed_at": datetime.now(timezone.utc).isoformat()
        })

        if task.callback_url:
            logger.info(f"Dispatching failure callback for edit job {job_id} to {task.callback_url}")
            callback_info = await dispatch_callback(task, CallbackStatus.FAILED, error=str(exc))
            job_record["callback"] = callback_info
            logger.info(f"Failure callback dispatched for edit job {job_id}: {callback_info}")
    finally:
        edit_jobs[job_id] = job_record
        schedule_job_cleanup(TaskType.EDIT, job_id, job_record)


async def job_queue_worker(worker_id: int):
    queue: asyncio.Queue = app.state.job_queue
    logger.info(f"Job worker {worker_id} started")

    while True:
        task = await queue.get()

        if task is None:
            logger.info(f"Job worker {worker_id} shutting down")
            queue.task_done()
            break

        try:
            if task.task_type == TaskType.INFERENCE:
                await handle_inference_task(task, worker_id)
            elif task.task_type == TaskType.EDIT:
                await handle_edit_task(task, worker_id)
            else:
                logger.error(f"Unknown task type for job {task.job_id}: {task.task_type}")
        except Exception as exc:
            logger.error(f"Job worker {worker_id} encountered an error for job {getattr(task, 'job_id', 'unknown')}: {exc}", exc_info=True)
        finally:
            queue.task_done()

@app.post("/inference")
async def generate_image(request: InferenceRequestModel):
    config = app.state.config
    if not config.enable_inference:
        raise HTTPException(status_code=503, detail="Inference is currently disabled")

    job_queue = getattr(app.state, "job_queue", None)
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Job queue is not ready")

    job_id = request.job_id or str(uuid.uuid4())

    existing_job = inference_jobs.get(job_id)
    if existing_job and existing_job.get("status") not in {"completed", "failed", "timeout"}:
        raise HTTPException(status_code=409, detail=f"Job {job_id} is already {existing_job.get('status')}")

    output_path = Path(app.state.OUTPUT_DIR) / f"{job_id}.png"

    inference_req = InferenceRequest(
        prompt=request.prompt,
        output_path=str(output_path),
        lora_path=request.lora_path or "",
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed or 42
    )

    expiry = normalize_expiry(request.expiry)

    task = QueuedTask(
        job_id=job_id,
        task_type=TaskType.INFERENCE,
        inference_request=inference_req,
        edit_params=None,
        callback_url=request.callback_url,
        callback_secret=request.callback_secret,
        expiry=expiry
    )

    now_iso = datetime.now(timezone.utc).isoformat()

    inference_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "queued_at": now_iso,
        "request": {
            "prompt": request.prompt,
            "lora_path": request.lora_path,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "callback_url": request.callback_url,
            "callback_secret_provided": bool(request.callback_secret),
            "expiry": expiry.isoformat() if expiry else None
        }
    }

    try:
        await job_queue.put(task)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(f"Failed to enqueue inference job {job_id}: {exc}")
        inference_jobs[job_id]["status"] = "failed"
        inference_jobs[job_id]["error"] = str(exc)
        inference_jobs[job_id]["failed_at"] = datetime.now(timezone.utc).isoformat()
        raise HTTPException(status_code=500, detail="Failed to queue inference job")

    logger.info(f"Queued inference job {job_id} for prompt '{request.prompt[:50]}...'")

    return {
        "accepted": True,
        "job_id": job_id,
        "status": "queued",
        "message": "Inference job queued",
        "status_url": f"/inference/status/{job_id}",
        "result_url": f"/inference/result/{job_id}"
    }

@app.get("/inference/status/{job_id}")
async def get_inference_status(job_id: str):
    output_path = Path(app.state.OUTPUT_DIR) / f"{job_id}.png"
    job = inference_jobs.get(job_id)

    if job is None:
        if output_path.exists():
            return {
                "id": job_id,
                "status": "completed",
                "output_path": str(output_path),
                "result_url": f"/inference/result/{job_id}",
                "image_url": f"{app.state.SERVICE_URL}/images/{job_id}.png",
                "note": "job record cleaned up"
            }
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found (it may have been cleaned up)")

    # Check if output exists using global OUTPUT_DIR
    if output_path.exists() and job.get("status") in {"queued", "processing"}:
        job.update({
            "status": "completed",
            "output_path": str(output_path),
            "result_url": f"/inference/result/{job_id}",
            "image_url": f"{app.state.SERVICE_URL}/images/{job_id}.png",
            "completed_at": job.get("completed_at", datetime.now(timezone.utc).isoformat())
        })
        inference_jobs[job_id] = job

    return job

@app.get("/inference/result/{job_id}")
async def get_inference_result(job_id: str):
    output_path = Path(app.state.OUTPUT_DIR) / f"{job_id}.png"

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found. Job may still be processing.")

    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename=f"{job_id}.png"
    )

@app.post("/edit")
async def create_edit_job(request: EditRequest):
    """
    Create image editing job with FLUX.1-Kontext-dev.

    Requires deterministic seed for reproducible edits.
    Supports image_url (recommended) or image_b64 input.
    """
    config = app.state.config
    if not config.enable_kontext_edit:
        raise HTTPException(
            status_code=503,
            detail="Kontext image editing is currently disabled"
        )

    job_queue = getattr(app.state, "job_queue", None)
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Job queue is not ready")

    # Generate job ID if not provided
    job_id = request.job_id or f"edit-{str(uuid.uuid4())}"

    # Check for duplicate job
    if job_id in edit_jobs:
        existing = edit_jobs[job_id]
        if existing.get("status") not in {"completed", "failed"}:
            raise HTTPException(
                status_code=409,
                detail=f"Job {job_id} already exists with status {existing.get('status')}"
            )

    # Create job record
    now_iso = datetime.now(timezone.utc).isoformat()
    edit_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "queued_at": now_iso,
        "request": {
            "prompt": request.prompt,
            "image_url": request.image_url,
            "image_source": None,
            "seed": request.seed,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "callback_url": request.callback_url,
            "callback_secret_provided": bool(request.callback_secret),
            "expiry": request.expiry.isoformat() if request.expiry else None
        }
    }

    edit_params = EditTaskParams(
        prompt=request.prompt,
        image_url=request.image_url,
        image_b64=request.image_b64,
        seed=request.seed,
        guidance_scale=request.guidance_scale,
        num_inference_steps=request.num_inference_steps
    )

    task = QueuedTask(
        job_id=job_id,
        task_type=TaskType.EDIT,
        inference_request=None,
        edit_params=edit_params,
        callback_url=request.callback_url,
        callback_secret=request.callback_secret,
        expiry=normalize_expiry(request.expiry)
    )

    try:
        await job_queue.put(task)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(f"Failed to enqueue edit job {job_id}: {exc}")
        edit_jobs[job_id]["status"] = "failed"
        edit_jobs[job_id]["error"] = str(exc)
        edit_jobs[job_id]["failed_at"] = datetime.now(timezone.utc).isoformat()
        raise HTTPException(status_code=500, detail="Failed to queue edit job")

    logger.info(f"Queued edit job {job_id} with seed={request.seed}, source=pending_fetch")

    return {
        "accepted": True,
        "job_id": job_id,
        "status": "queued",
        "message": "Edit job queued",
        "status_url": f"/edit/status/{job_id}",
        "result_url": f"/edit/result/{job_id}"
    }

@app.get("/edit/status/{job_id}")
async def get_edit_status(job_id: str):
    """Get status of edit job."""
    output_path = Path(app.state.OUTPUT_DIR) / "edits" / f"{job_id}.png"
    job = edit_jobs.get(job_id)

    if job is None:
        if output_path.exists():
            return {
                "id": job_id,
                "status": "completed",
                "output_path": str(output_path),
                "result_url": f"/edit/result/{job_id}",
                "image_url": f"{app.state.SERVICE_URL}/images/edits/{job_id}.png",
                "note": "job record cleaned up"
            }
        raise HTTPException(status_code=404, detail=f"Edit job {job_id} not found (it may have been cleaned up)")

    return job


@app.get("/edit/result/{job_id}")
async def get_edit_result(job_id: str):
    """Download edited image."""
    output_path = Path(app.state.OUTPUT_DIR) / "edits" / f"{job_id}.png"

    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Output file not found"
        )

    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename=f"{job_id}.png"
    )

@app.get("/jobs")
async def list_jobs(job_type: Optional[str] = None):
    if job_type == "inference":
        return {"inference_jobs": list(inference_jobs.values())}
    elif job_type == "edit":
        return {"edit_jobs": list(edit_jobs.values())}
    else:
        return {
            "inference_jobs": list(inference_jobs.values()),
            "edit_jobs": list(edit_jobs.values())
        }

@app.get("/health")
async def health_check():
    config = app.state.config
    trt_status = "disabled"

    if config.enable_inference:
        if trt_server is not None:
            trt_status = "loaded"
        elif config.preload_engines:
            trt_status = "preloading"  # May still be loading in background
        else:
            trt_status = "lazy_load_enabled"

    kontext_status = "disabled"
    if config.enable_kontext_edit:
        if kontext_manager is not None:
            kontext_status = "loaded"
        else:
            kontext_status = "lazy_load_enabled"

    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "inference": trt_status,
            "kontext_edit": kontext_status
        },
        "config": {
            "preload_engines": config.preload_engines,
            "engine_path": config.trt_engine_path
        }
    }

    if trt_server is not None:
        status["services"]["trt_engine"] = "loaded"

    if kontext_manager is not None:
        status["services"]["kontext_manager"] = "loaded"

    return status


@app.get("/")
async def root():
    return {
        "name": "Dippy Studio Bittensor Miner Server",
        "version": "1.0.0",
        "endpoints": {
            "inference": {
                "POST /inference": "Generate image with TRT",
                "GET /inference/status/{job_id}": "Get inference job status",
                "GET /inference/result/{job_id}": "Download generated image"
            },
            "editing": {
                "POST /edit": "Edit image with Kontext",
                "GET /edit/status/{job_id}": "Get edit job status",
                "GET /edit/result/{job_id}": "Download edited image"
            },
            "general": {
                "GET /jobs": "List all jobs",
                "GET /health": "Health check",
                "GET /engine/status": "Detailed engine status"
            }
        }
    }

@app.on_event("startup")
async def startup_event():
    await initialize_servers()

@app.on_event("shutdown")
async def shutdown_event():
    global trt_server
    job_queue = getattr(app.state, "job_queue", None)
    worker_tasks = getattr(app.state, "job_worker_tasks", [])

    if job_queue:
        # Wait for in-flight tasks to complete before shutting down workers
        await job_queue.join()
        for _ in worker_tasks:
            await job_queue.put(None)

        await asyncio.gather(*worker_tasks, return_exceptions=True)

    if trt_server:
        trt_server.stop()

def main():
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
