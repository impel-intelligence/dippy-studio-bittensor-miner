import yaml
import base64
import tempfile
import os
from pathlib import Path

def generate_training_config(job_id: str, params: dict) -> str:
    """Generate YAML config from API request parameters"""

    # Validate required parameters
    if 'prompt' not in params or not isinstance(params['prompt'], str) or not params['prompt'].strip():
        raise ValueError("params['prompt'] must be a non-empty string")
    if 'image_b64' not in params:
        raise ValueError("params['image_b64'] is required")

    # Clean the prompt
    prompt = params['prompt'].strip()

    # Create job-specific directory
    job_dir = Path(f"/app/datasets/job_{job_id}")
    job_dir.mkdir(parents=True, exist_ok=True)

    # Decode and save image
    image_data = base64.b64decode(params['image_b64'])
    image_path = job_dir / "image.jpg"
    with open(image_path, 'wb') as f:
        f.write(image_data)

    # Create caption file
    caption_path = job_dir / "image.txt"
    with open(caption_path, 'w') as f:
        f.write(prompt)

    # Use base config template
    config = {
        "job": "extension",
        "config": {
            "name": f"job_{job_id}",
            "process": [{
                "type": "sd_trainer",
                "training_folder": "output",
                "device": "cuda:0",
                "network": {
                    "type": "lora",
                    "linear": 16,
                    "linear_alpha": 16
                },
                "save": {
                    "dtype": "float16",
                    "save_every": 250,
                    "max_step_saves_to_keep": 1
                },
                "datasets": [{
                    "folder_path": str(job_dir),
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "cache_latents_to_disk": True,
                    "resolution": [512, 768, 1024]
                }],
                "train": {
                    "batch_size": 1,
                    "steps": 1000,
                    "gradient_accumulation_steps": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit",
                    "lr": 1e-4,
                    "dtype": "bf16",
                    "ema_config": {
                        "use_ema": True,
                        "ema_decay": 0.99
                    }
                },
                "model": {
                    "name_or_path": "black-forest-labs/FLUX.1-dev",
                    "is_flux": True,
                    "quantize": False
                },
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": 250,
                    "width": 1024,
                    "height": 1024,
                    "prompts": [prompt],
                    "seed": params.get('seed', 42),
                    "guidance_scale": 4,
                    "sample_steps": 20
                }
            }]
        }
    }

    # Save config file
    config_path = f"/app/config/job_{job_id}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path

def encode_model_to_base64(model_dir: str) -> str:
    """Find and encode the trained LoRA model to base64"""
    model_path = None

    # Find the .safetensors file
    for file in Path(model_dir).glob("*.safetensors"):
        model_path = file
        break

    if not model_path:
        raise FileNotFoundError("No trained model found")

    with open(model_path, 'rb') as f:
        model_data = f.read()

    return base64.b64encode(model_data).decode('utf-8')

def cleanup_job_files(job_id: str):
    """Clean up temporary files for a job"""
    import shutil

    # Remove dataset directory
    job_dir = Path(f"/app/datasets/job_{job_id}")
    if job_dir.exists():
        shutil.rmtree(job_dir)

    # Remove config file
    config_path = Path(f"/app/config/job_{job_id}.yaml")
    if config_path.exists():
        config_path.unlink()

    # Remove output directory
    output_dir = Path(f"/app/output/job_{job_id}")
    if output_dir.exists():
        shutil.rmtree(output_dir)