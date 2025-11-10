"""
FLUX.1-Kontext-dev OPTIMIZED deterministic image editing pipeline manager.

This is an enhanced version of kontext_pipeline.py that supports:
- TensorRT FP8/FP4 optimized inference
- Automatic fallback to PyTorch BF16
- Same API as original KontextInferenceManager

Usage:
    # Will automatically use TRT engine if available
    manager = OptimizedKontextInferenceManager()
    manager.load_pipeline()

    result = manager.generate(
        prompt="Add a red hat",
        image=input_image,
        seed=42
    )
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


class OptimizedKontextInferenceManager:
    """
    Manages FLUX.1-Kontext-dev pipeline with TensorRT optimization support.

    Automatically uses TensorRT engine if available, falls back to PyTorch.
    """

    def __init__(
        self,
        model_path: str = "black-forest-labs/FLUX.1-Kontext-dev",
        trt_engine_path: Optional[str] = None,
        use_trt: bool = True
    ):
        """
        Initialize optimized Kontext pipeline manager.

        Args:
            model_path: HuggingFace model path or local directory
            trt_engine_path: Path to TensorRT engine (auto-detected if None)
            use_trt: Whether to use TensorRT if available
        """
        self.model_path = model_path
        self.device = "cuda"

        # Pipeline state
        self.pipeline = None
        self.using_trt = False

        # TensorRT configuration
        self.use_trt = use_trt and torch.cuda.is_available()
        self.trt_engine_path = self._resolve_trt_engine_path(trt_engine_path)

        # TensorRT components (loaded on demand)
        self.trt_engine = None
        self.trt_context = None

    def _resolve_trt_engine_path(self, provided_path: Optional[str]) -> Optional[Path]:
        """
        Resolve TensorRT engine path from environment or default locations.

        Args:
            provided_path: Explicitly provided engine path

        Returns:
            Path to engine if found, None otherwise
        """
        if provided_path:
            path = Path(provided_path)
            if path.exists():
                return path
            logger.warning(f"Provided TRT engine not found: {path}")
            return None

        # Try environment variable
        env_path = os.getenv("TRT_KONTEXT_ENGINE_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                logger.info(f"Found TRT engine from environment: {path}")
                return path

        # Try default location
        default_path = Path("/trt-cache/kontext_fp8.trt")
        if default_path.exists():
            logger.info(f"Found TRT engine at default location: {default_path}")
            return default_path

        logger.info("No TRT engine found, will use PyTorch BF16 mode")
        return None

    def load_pipeline(self):
        """
        Lazy load pipeline with automatic TRT/PyTorch selection.

        Tries to load TensorRT engine first, falls back to PyTorch if unavailable.
        """
        if self.pipeline is not None:
            return  # Already loaded

        # Try TensorRT first if enabled and available
        if self.use_trt and self.trt_engine_path:
            try:
                self._load_trt_pipeline()
                logger.info("✓ Using TensorRT optimized pipeline (FP8)")
                self.using_trt = True
                return
            except Exception as e:
                logger.warning(f"Failed to load TRT engine: {e}")
                logger.info("Falling back to PyTorch BF16 mode")

        # Fallback to PyTorch
        self._load_pytorch_pipeline()
        logger.info("✓ Using PyTorch BF16 pipeline")
        self.using_trt = False

    def _load_pytorch_pipeline(self):
        """Load standard PyTorch BF16 pipeline."""
        logger.info(f"Loading PyTorch Kontext pipeline from {self.model_path}")

        self.pipeline = FluxKontextPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        ).to(self.device)

        # Disable progress bar for cleaner logs
        self.pipeline.set_progress_bar_config(disable=True)

    def _load_trt_pipeline(self):
        """
        Load TensorRT optimized pipeline.

        Uses hybrid approach:
        - TensorRT for transformer (96% of compute)
        - PyTorch for text encoder and VAE
        """
        logger.info(f"Loading TensorRT engine from {self.trt_engine_path}")

        try:
            # Import TRT engine wrapper
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent / "optimization"))
            from trt_engine_wrapper import TRTEngineWrapper

            # Load TRT engine
            self.trt_engine = TRTEngineWrapper(
                engine_path=str(self.trt_engine_path),
                device=self.device
            )
            self.trt_engine.load()
            self.trt_engine.activate()

            # Allocate buffers with default shapes
            # These will be reallocated dynamically if needed
            self.trt_engine.allocate_buffers()

            logger.info("✓ TRT engine loaded and activated")

            # Load PyTorch pipeline for non-transformer components
            logger.info("Loading PyTorch components (text encoder, VAE)...")
            self.pipeline = FluxKontextPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            ).to(self.device)

            # Disable progress bar
            self.pipeline.set_progress_bar_config(disable=True)

            # Store original transformer forward for reference
            self._original_transformer_forward = self.pipeline.transformer.forward

            # Replace transformer forward with TRT wrapper
            self._wrap_transformer_with_trt()

            logger.info("✓ Hybrid TRT pipeline ready (TRT transformer + PyTorch text/VAE)")

        except Exception as e:
            logger.error(f"Failed to load TRT pipeline: {e}")
            raise

    def _wrap_transformer_with_trt(self):
        """
        Wrap the transformer's forward method to use TRT engine.

        This replaces the PyTorch transformer forward pass with TRT inference
        while maintaining compatibility with the pipeline.
        """
        def trt_forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            pooled_projections: torch.FloatTensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[dict] = None,
            return_dict: bool = True,
            **kwargs
        ):
            """
            TRT-accelerated transformer forward pass.

            Maintains same interface as PyTorch transformer.
            """
            # Handle optional conditioning image
            # Kontext passes this via joint_attention_kwargs or directly
            image = kwargs.get("image", None)
            if image is None and joint_attention_kwargs:
                image = joint_attention_kwargs.get("image", None)

            # Prepare inputs for TRT
            # Ensure all tensors are on correct device and dtype
            feed_dict = {
                "hidden_states": hidden_states.to(self.device, dtype=torch.float16),
                "encoder_hidden_states": encoder_hidden_states.to(self.device, dtype=torch.float16),
                "pooled_projections": pooled_projections.to(self.device, dtype=torch.float16),
                "timestep": timestep.to(self.device, dtype=torch.float16),
                "img_ids": img_ids.to(self.device, dtype=torch.float16),
                "txt_ids": txt_ids.to(self.device, dtype=torch.float16),
                "guidance": guidance.to(self.device, dtype=torch.float16),
            }

            # Add image if present (Kontext-specific)
            if image is not None:
                feed_dict["image"] = image.to(self.device, dtype=torch.float16)

            # Run TRT inference
            outputs = self.trt_engine.infer(feed_dict)

            # Get output tensor
            output_hidden_states = outputs["output_hidden_states"]

            # Convert back to expected dtype
            output_hidden_states = output_hidden_states.to(hidden_states.dtype)

            # Return in expected format
            if return_dict:
                from diffusers.utils import BaseOutput
                return BaseOutput(sample=output_hidden_states)
            else:
                return (output_hidden_states,)

        # Replace the transformer's forward method
        self.pipeline.transformer.forward = trt_forward
        logger.debug("✓ Transformer forward method wrapped with TRT")

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

        # PyTorch CUDA random (all GPUs)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Enable deterministic CUDA operations
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
        # Convert to RGB if needed
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
        Run deterministic image edit with automatic TRT/PyTorch selection.

        Args:
            prompt: Text instruction for editing
            image: Input image to edit
            seed: Random seed (required for determinism)
            guidance_scale: How strongly to follow prompt (default: 2.5)
            num_inference_steps: Number of denoising steps (default: 28)

        Returns:
            Edited PIL image
        """
        # Ensure pipeline is loaded
        self.load_pipeline()

        # Preprocess input image
        image = self.preprocess_image(image)

        # Set deterministic mode and get generator
        generator = self.set_deterministic_mode(seed)

        inference_mode_str = "TRT-FP8" if self.using_trt else "PyTorch-BF16"
        logger.info(
            f"Generating edit with {inference_mode_str}: "
            f"seed={seed}, steps={num_inference_steps}"
        )

        # Run inference
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

    def get_info(self) -> dict:
        """
        Get information about current pipeline configuration.

        Returns:
            Dictionary with configuration details
        """
        return {
            "model_path": str(self.model_path),
            "using_trt": self.using_trt,
            "trt_engine_path": str(self.trt_engine_path) if self.trt_engine_path else None,
            "device": self.device,
            "precision": "FP8" if self.using_trt else "BF16",
            "pipeline_loaded": self.pipeline is not None
        }


# Backward compatibility: alias to original class name for drop-in replacement
KontextInferenceManager = OptimizedKontextInferenceManager
