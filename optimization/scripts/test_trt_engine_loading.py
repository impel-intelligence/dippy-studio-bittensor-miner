#!/usr/bin/env python3
"""
Quick test script to verify TRT engine can be loaded and used.

Usage:
    python3 optimization/scripts/test_trt_engine_loading.py \
        --engine-path /path/to/kontext_fp8.trt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_engine_loading(engine_path: str):
    """Test TRT engine loading."""
    logger.info("="*60)
    logger.info("TRT Engine Loading Test")
    logger.info("="*60)

    # Import wrapper
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from trt_engine_wrapper import TRTEngineWrapper

    logger.info(f"Testing engine: {engine_path}")

    # Load engine
    logger.info("\n1. Loading engine...")
    engine = TRTEngineWrapper(engine_path=engine_path)
    engine.load()

    # Activate
    logger.info("\n2. Creating execution context...")
    engine.activate()

    # Get tensor info
    logger.info("\n3. Engine tensor information:")
    logger.info(f"  Input tensors: {engine.get_input_names()}")
    logger.info(f"  Output tensors: {engine.get_output_names()}")

    # Allocate buffers
    logger.info("\n4. Allocating buffers...")
    engine.allocate_buffers()

    logger.info("\n5. Creating dummy inputs...")
    # Create dummy inputs matching expected shapes
    dummy_inputs = {
        "hidden_states": torch.randn(1, 4096, 16, dtype=torch.float16).cuda(),
        "encoder_hidden_states": torch.randn(1, 512, 4096, dtype=torch.float16).cuda(),
        "pooled_projections": torch.randn(1, 768, dtype=torch.float16).cuda(),
        "timestep": torch.randn(1, dtype=torch.float16).cuda(),
        "img_ids": torch.randn(4096, 3, dtype=torch.float16).cuda(),
        "txt_ids": torch.randn(512, 3, dtype=torch.float16).cuda(),
        "guidance": torch.full((1,), 3.5, dtype=torch.float16).cuda(),
        "image": torch.randn(1, 16, 64, 64, dtype=torch.float16).cuda(),
    }

    logger.info("\n6. Running test inference...")
    try:
        outputs = engine.infer(dummy_inputs)
        logger.info(f"✓ Inference successful!")
        logger.info(f"  Output shapes:")
        for name, tensor in outputs.items():
            logger.info(f"    {name}: {tensor.shape}")

        logger.info("\n" + "="*60)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("="*60)
        return True

    except Exception as e:
        logger.error(f"✗ Inference failed: {e}", exc_info=True)
        logger.info("\n" + "="*60)
        logger.info("✗ TESTS FAILED")
        logger.info("="*60)
        return False


def test_optimized_pipeline(engine_path: str):
    """Test OptimizedKontextInferenceManager with TRT."""
    logger.info("\n" + "="*60)
    logger.info("Optimized Pipeline Test")
    logger.info("="*60)

    from kontext_pipeline_optimized import OptimizedKontextInferenceManager
    from PIL import Image

    logger.info("1. Creating OptimizedKontextInferenceManager...")
    manager = OptimizedKontextInferenceManager(
        trt_engine_path=engine_path,
        use_trt=True
    )

    logger.info("2. Loading pipeline (this may take a while)...")
    manager.load_pipeline()

    info = manager.get_info()
    logger.info(f"3. Pipeline info:")
    logger.info(f"  Using TRT: {info['using_trt']}")
    logger.info(f"  Backend: {info['precision']}")
    logger.info(f"  Engine: {info['trt_engine_path']}")

    if not info['using_trt']:
        logger.warning("⚠ Pipeline fell back to PyTorch!")
        return False

    logger.info("\n4. Creating test image...")
    test_image = Image.new("RGB", (512, 512), (128, 128, 200))

    logger.info("5. Running test inference...")
    try:
        result = manager.generate(
            prompt="Add a red circle",
            image=test_image,
            seed=42,
            num_inference_steps=1  # Just 1 step for quick test
        )

        logger.info(f"✓ Generation successful!")
        logger.info(f"  Output size: {result.size}")

        logger.info("\n" + "="*60)
        logger.info("✓ PIPELINE TEST PASSED")
        logger.info("="*60)
        return True

    except Exception as e:
        logger.error(f"✗ Generation failed: {e}", exc_info=True)
        logger.info("\n" + "="*60)
        logger.info("✗ PIPELINE TEST FAILED")
        logger.info("="*60)
        return False


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="Test TRT engine loading")
    parser.add_argument(
        "--engine-path",
        type=str,
        required=True,
        help="Path to TRT engine file"
    )
    parser.add_argument(
        "--test-pipeline",
        action="store_true",
        help="Also test full pipeline (requires model download)"
    )

    args = parser.parse_args()

    # Check engine exists
    engine_path = Path(args.engine_path)
    if not engine_path.exists():
        logger.error(f"Engine not found: {engine_path}")
        sys.exit(1)

    # Test 1: Engine loading
    success = test_engine_loading(str(engine_path))

    if not success:
        sys.exit(1)

    # Test 2: Full pipeline (optional)
    if args.test_pipeline:
        success = test_optimized_pipeline(str(engine_path))
        if not success:
            sys.exit(1)

    logger.info("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    main()
