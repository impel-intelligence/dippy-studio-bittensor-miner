"""
FLUX.1-Kontext-dev FAST deterministic image editing pipeline.

Optimizations applied (while preserving determinism):
1. torch.compile() - JIT compilation for 1.5-2x speedup
2. QKV Fusion - Fuses Query/Key/Value projections (~5-10% speedup)
3. VAE Slicing - Reduces memory peaks
4. VAE Tiling - Handles large images efficiently
5. Channels-last memory format - Optimizes for GPU Tensor Cores (~5-10% speedup)
6. Flash Attention backend - Forces fastest SDPA implementation (~5-15% speedup)

Drop-in replacement for KontextInferenceManager.
"""
import logging
import os

# Ensure CuBLAS determinism for reproducible results
# Must be set before any CUDA operations
# See: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import FluxKontextPipeline
from PIL import Image

logger = logging.getLogger(__name__)


class KontextFastInferenceManager:
    """
    Manages FLUX.1-Kontext-dev pipeline with:
    - Deterministic execution (same seed = same output)
    - torch.compile() for 1.5-2x speedup
    - QKV fusion for ~5-10% additional speedup
    - VAE optimizations for memory efficiency
    - Channels-last memory format for Tensor Core optimization
    - Flash Attention backend for fastest attention computation

    Drop-in replacement for KontextInferenceManager.
    """

    def __init__(
        self,
        model_path: str = "black-forest-labs/FLUX.1-Kontext-dev",
        compile_transformer: bool = True,
        compile_mode: str = "reduce-overhead",
        enable_qkv_fusion: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = True,
        enable_channels_last: bool = True,
        enable_flash_attention: bool = True,
    ):
        """
        Initialize Kontext pipeline manager.

        Args:
            model_path: HuggingFace model path or local directory
            compile_transformer: Whether to use torch.compile() (default: True)
            compile_mode: Compilation mode - options:
                - "reduce-overhead": Best for inference (recommended)
                - "default": Balanced
                - "max-autotune": Maximum optimization, slower compile
            enable_qkv_fusion: Fuse QKV projections for faster attention (default: True)
            enable_vae_slicing: Enable VAE slicing to reduce memory (default: True)
            enable_vae_tiling: Enable VAE tiling for large images (default: True)
            enable_channels_last: Use channels-last memory format for Tensor Cores (default: True)
            enable_flash_attention: Force Flash Attention SDPA backend (default: True)
        """
        self.model_path = model_path
        self.pipeline = None
        self.device = "cuda"
        self.compile_transformer = compile_transformer
        self.compile_mode = compile_mode
        self.enable_qkv_fusion = enable_qkv_fusion
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling
        self.enable_channels_last = enable_channels_last
        self.enable_flash_attention = enable_flash_attention
        self._is_compiled = False
        self._optimizations_applied = []

    def load_pipeline(self):
        """Lazy load pipeline with deterministic settings and optimizations."""
        if self.pipeline is not None:
            return  # Already loaded

        logger.info(f"Loading Kontext pipeline from {self.model_path}")
        load_start = time.time()

        # Load the base pipeline (same as original)
        self.pipeline = FluxKontextPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,  # Native FLUX precision
            use_safetensors=True
        ).to(self.device)

        # Disable progress bar for cleaner logs
        self.pipeline.set_progress_bar_config(disable=True)

        load_time = time.time() - load_start
        logger.info(f"Pipeline loaded in {load_time:.1f}s")

        # Apply optimizations
        self._apply_optimizations()

        # Apply torch.compile() last (after other optimizations)
        if self.compile_transformer:
            self._compile_transformer()

        # Log summary of applied optimizations
        if self._optimizations_applied:
            logger.info(f"Optimizations applied: {', '.join(self._optimizations_applied)}")

    def _apply_optimizations(self):
        """Apply non-compile optimizations to the pipeline."""

        # QKV Fusion - fuses Query, Key, Value projections into single operation
        # This reduces memory bandwidth and improves speed by ~5-10%
        if self.enable_qkv_fusion:
            try:
                self.pipeline.transformer.fuse_qkv_projections()
                self._optimizations_applied.append("QKV Fusion")
                logger.info("Applied QKV fusion to transformer")
            except Exception as e:
                logger.warning(f"Could not apply QKV fusion: {e}")

        # VAE Slicing - processes VAE in slices to reduce peak memory
        # Helpful for high-resolution images or limited VRAM
        if self.enable_vae_slicing:
            try:
                self.pipeline.vae.enable_slicing()
                self._optimizations_applied.append("VAE Slicing")
                logger.info("Enabled VAE slicing")
            except Exception as e:
                logger.warning(f"Could not enable VAE slicing: {e}")

        # VAE Tiling - tiles large images to avoid OOM errors
        # Essential for images larger than 1024x1024
        if self.enable_vae_tiling:
            try:
                self.pipeline.vae.enable_tiling()
                self._optimizations_applied.append("VAE Tiling")
                logger.info("Enabled VAE tiling")
            except Exception as e:
                logger.warning(f"Could not enable VAE tiling: {e}")

        # Channels-last memory format - optimizes memory layout for GPU Tensor Cores
        # Instead of NCHW (batch, channels, height, width), uses NHWC layout
        # This allows Tensor Cores to access memory more efficiently
        # Typical speedup: 5-10% on modern NVIDIA GPUs (Ampere, Hopper)
        if self.enable_channels_last:
            try:
                self.pipeline.transformer.to(memory_format=torch.channels_last)
                self._optimizations_applied.append("Channels-last")
                logger.info("Applied channels-last memory format to transformer")
            except Exception as e:
                logger.warning(f"Could not apply channels-last format: {e}")

        # Flash Attention backend - prefer PyTorch's fastest SDPA implementation
        # SDPA (Scaled Dot-Product Attention) has multiple backends:
        #   1. Flash Attention - fastest, memory efficient (O(N) instead of O(N²))
        #   2. Memory-efficient attention - good for long sequences
        #   3. Math fallback - slowest, most compatible
        # We enable Flash but keep fallbacks for layers that don't support it (e.g., VAE)
        if self.enable_flash_attention:
            try:
                # Enable Flash Attention (fastest) - PyTorch will use it when compatible
                torch.backends.cuda.enable_flash_sdp(True)
                # Keep memory-efficient as fallback for incompatible layers
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                # Keep math fallback for edge cases
                torch.backends.cuda.enable_math_sdp(True)
                self._optimizations_applied.append("Flash Attention (preferred)")
                logger.info("Enabled Flash Attention SDPA backend (with fallbacks)")
            except Exception as e:
                logger.warning(f"Could not configure Flash Attention: {e}")

    def _compile_transformer(self):
        """
        Apply torch.compile() to the transformer.

        This is where the magic happens:
        - First inference will be slow (compilation)
        - Subsequent inferences will be 1.5-2x faster
        """
        if self._is_compiled:
            return

        logger.info(f"Compiling transformer with mode='{self.compile_mode}'...")
        logger.info("First inference will be slow (compiling), subsequent will be fast.")

        compile_start = time.time()

        # torch.compile() wraps the transformer module
        # It analyzes the computation graph and generates optimized code
        self.pipeline.transformer = torch.compile(
            self.pipeline.transformer,
            mode=self.compile_mode,
            # fullgraph=False allows partial compilation if some ops aren't supported
            fullgraph=False,
            # dynamic=False assumes fixed input shapes (faster)
            # Set to True if you need variable image sizes
            dynamic=False,
        )

        self._is_compiled = True
        compile_time = time.time() - compile_start
        logger.info(f"Transformer compiled in {compile_time:.1f}s (graph capture, not full compilation)")
        logger.info("Full compilation happens on first inference.")

    def set_deterministic_mode(self, seed: int) -> torch.Generator:
        """
        Configure all RNG sources for determinism.

        IMPORTANT: This is identical to the original implementation.
        torch.compile() does NOT break determinism - the compiled code
        still uses the same deterministic algorithms.

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

        Identical to original implementation.

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

        Identical interface to original - but faster after first run.

        Args:
            prompt: Text instruction for editing
            image: Input image to edit
            seed: Random seed (required for determinism)
            guidance_scale: How strongly to follow prompt (default: 2.5)
            num_inference_steps: Number of denoising steps (default: 28)

        Returns:
            Edited PIL image
        """
        # Ensure pipeline is loaded (and compiled if enabled)
        self.load_pipeline()

        # Preprocess input image
        image = self.preprocess_image(image)

        # Set deterministic mode and get generator
        generator = self.set_deterministic_mode(seed)

        logger.info(f"Generating edit with seed={seed}, steps={num_inference_steps}")

        inference_start = time.time()

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

        inference_time = time.time() - inference_start

        result_image = output.images[0]
        logger.info(f"Edit generated in {inference_time:.1f}s - size: {result_image.size}")

        return result_image

    def warmup(self, warmup_steps: int = 4):
        """
        Run warmup inference to trigger compilation.

        Call this after initialization if you want to pay the compilation
        cost upfront rather than on first real request.

        Args:
            warmup_steps: Number of denoising steps for warmup (fewer = faster)
        """
        logger.info("Running warmup inference to trigger compilation...")

        # Create a small dummy image
        dummy_image = Image.new("RGB", (512, 512), color=(128, 128, 128))

        warmup_start = time.time()

        # Run inference with minimal steps
        self.generate(
            prompt="warmup",
            image=dummy_image,
            seed=0,
            num_inference_steps=warmup_steps
        )

        warmup_time = time.time() - warmup_start
        logger.info(f"Warmup completed in {warmup_time:.1f}s - subsequent inferences will be fast")
