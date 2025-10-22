"""
FLUX.1-Kontext-dev deterministic image editing pipeline manager.

Provides deterministic image-to-image editing using PyTorch/diffusers.
"""
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import FluxKontextPipeline
from PIL import Image

logger = logging.getLogger(__name__)


class KontextInferenceManager:
    """Manages FLUX.1-Kontext-dev pipeline with deterministic execution."""

    def __init__(self, model_path: str = "black-forest-labs/FLUX.1-Kontext-dev"):
        """
        Initialize Kontext pipeline manager.

        Args:
            model_path: HuggingFace model path or local directory
        """
        self.model_path = model_path
        self.pipeline = None
        self.device = "cuda"

    def load_pipeline(self):
        """Lazy load pipeline with deterministic settings."""
        if self.pipeline is not None:
            return  # Already loaded

        logger.info(f"Loading Kontext pipeline from {self.model_path}")

        self.pipeline = FluxKontextPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,  # Native FLUX precision
            use_safetensors=True
        ).to(self.device)

        # Disable progress bar for cleaner logs
        self.pipeline.set_progress_bar_config(disable=True)

        logger.info("Kontext pipeline loaded successfully")

    def set_deterministic_mode(self, seed: int) -> torch.Generator:
        """
        Configure all RNG sources for determinism.

        Args:
            seed: Random seed for reproducibility

        Returns:
            PyTorch generator configured with seed
        """
        # Python random module
        random.seed(seed)

        # Numpy random
        np.random.seed(seed)

        # PyTorch CPU random
        torch.manual_seed(seed)

        # PyTorch CUDA random (all GPUs) - only if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Enable deterministic CUDA operations
        # warn_only=True allows operations that don't have deterministic implementations
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Disable cuDNN benchmarking (non-deterministic)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Create and return generator for pipeline
        generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.debug(f"Deterministic mode enabled with seed={seed}")
        return generator

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Deterministic image preprocessing.

        Args:
            image: Input PIL image

        Returns:
            Preprocessed PIL image (RGB, resized if needed)
        """
        # Convert to RGB if needed (handles RGBA, L, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if larger than max size (deterministic LANCZOS resampling)
        max_size = 1024
        if max(image.size) > max_size:
            # Calculate new size maintaining aspect ratio
            ratio = max_size / max(image.size)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)

            # LANCZOS is deterministic and high-quality
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to {new_width}x{new_height}")

        return image

    def generate(
        self,
        prompt: str,
        image: Image.Image,
        seed: int,
        guidance_scale: float = 2.5,
        num_inference_steps: int = 28
    ) -> Image.Image:
        """
        Run deterministic image edit.

        Args:
            prompt: Text instruction for editing
            image: Input image to edit
            seed: Random seed (required for determinism)
            guidance_scale: How strongly to follow prompt (default: 2.5)
            num_inference_steps: Number of denoising steps (default: 28)
            strength: Edit strength 0.0-1.0 (default: 1.0, ignored for Kontext)

        Returns:
            Edited PIL image
        """
        # Ensure pipeline is loaded
        self.load_pipeline()

        # Preprocess input image
        image = self.preprocess_image(image)

        # Set deterministic mode and get generator
        generator = self.set_deterministic_mode(seed)

        logger.info(f"Generating edit with seed={seed}, steps={num_inference_steps}")

        # Run inference in deterministic mode
        # Note: FluxKontextPipeline does not support 'strength' parameter
        with torch.inference_mode():
            output = self.pipeline(
                prompt=prompt,
                image=image,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="pil"
            )

        result_image = output.images[0]
        logger.info(f"Edit generated successfully: {result_image.size}")

        return result_image
