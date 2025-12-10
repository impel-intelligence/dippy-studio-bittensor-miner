from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import subprocess
import requests
import os
import threading
from pathlib import Path
from config_generator import generate_training_config, cleanup_job_files

app = FastAPI()

# Configuration

class TrainingRequest(BaseModel):
    job_type: str
    params: dict
    job_id: str
    validator_endpoint: str

@app.post("/train", status_code=201)
async def train_lora(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start LoRA training asynchronously and return immediately"""

    if request.job_type != "lora_training":
        raise HTTPException(status_code=400, detail="Only lora_training supported")

    try:
        # Generate config and prepare files
        config_path = generate_training_config(request.job_id, request.params)

        # Start training in background
        background_tasks.add_task(run_training, request.job_id, config_path, request.params, request.validator_endpoint)

        # Return immediately with 201 OK (empty response)
        return

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_training(job_id: str, config_path: str, params: dict, validator_endpoint: str):
    """Background task to run training and post results to validator"""

    try:
        # Run training (using current directory)
        result = subprocess.run([
            "python", "run.py", config_path, "--seed", str(params.get("seed", 42))
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Training failed for job {job_id}: {result.stderr}")
            return

        # Upload to HuggingFace and get model URL
        huggingface_url = upload_to_huggingface(job_id)

        # Post results to validator
        post_results_to_validator(job_id, huggingface_url, validator_endpoint)

    except Exception as e:
        print(f"Error in background training for job {job_id}: {e}")
    finally:
        # Cleanup temporary files
        cleanup_job_files(job_id)

def upload_to_huggingface(job_id: str) -> str:
    """Upload trained model to HuggingFace and return model URL"""
    from huggingface_hub import HfApi, upload_folder

    # Find the trained model
    model_dir = Path(f"./output/job_{job_id}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model output not found for job {job_id}")

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
        print(f"Successfully posted results for job {job_id}")

    except requests.RequestException as e:
        print(f"Failed to post results to validator: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091)