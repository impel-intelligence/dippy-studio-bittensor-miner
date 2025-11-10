"""
TensorRT Engine Wrapper for FLUX Kontext Transformer.

Provides a clean interface for loading and running TRT engines.
"""
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import tensorrt as trt
from polygraphy.backend.trt import engine_from_bytes
from polygraphy.backend.common import bytes_from_path

logger = logging.getLogger(__name__)


# Type mappings
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}


class TRTEngineWrapper:
    """
    Wrapper for TensorRT engine providing simplified inference interface.

    Handles engine loading, buffer allocation, and inference execution.
    """

    def __init__(self, engine_path: str, device: str = "cuda"):
        """
        Initialize TRT engine wrapper.

        Args:
            engine_path: Path to TRT engine file
            device: Device to run on (default: cuda)
        """
        self.engine_path = Path(engine_path)
        self.device = device
        self.engine = None
        self.context = None
        self.tensors = OrderedDict()
        self.stream = None

        if not self.engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {self.engine_path}")

    def load(self):
        """Load TensorRT engine from file."""
        logger.info(f"Loading TensorRT engine: {self.engine_path}")

        try:
            # Load engine
            engine_bytes = bytes_from_path(self.engine_path)
            self.engine = engine_from_bytes(engine_bytes)

            if not self.engine:
                raise RuntimeError("Failed to deserialize TRT engine")

            logger.info(f"✓ TRT engine loaded successfully")
            logger.info(f"  Number of I/O tensors: {self.engine.num_io_tensors}")

            # Log tensor info
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                mode = self.engine.get_tensor_mode(name)
                logger.debug(f"  Tensor {i}: {name} {shape} {dtype} {mode}")

        except Exception as e:
            logger.error(f"Failed to load TRT engine: {e}")
            raise

    def activate(self):
        """Create execution context for inference."""
        if not self.engine:
            raise RuntimeError("Engine not loaded. Call load() first.")

        logger.info("Creating TRT execution context...")
        self.context = self.engine.create_execution_context()

        if not self.context:
            raise RuntimeError("Failed to create execution context")

        # Create CUDA stream
        self.stream = torch.cuda.Stream()

        logger.info("✓ TRT execution context created")

    def allocate_buffers(self, shape_dict: Optional[Dict] = None):
        """
        Allocate GPU buffers for inputs and outputs.

        Args:
            shape_dict: Optional dict mapping tensor names to shapes.
                       If None, uses engine's default shapes.
        """
        if not self.context:
            raise RuntimeError("Context not created. Call activate() first.")

        logger.info("Allocating TRT buffers...")

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)

            # Get shape
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)

            # Get dtype
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = numpy_to_torch_dtype_dict.get(dtype, torch.float32)

            # Set input shape if dynamic
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)

            # Allocate tensor
            tensor = torch.empty(
                tuple(shape),
                dtype=torch_dtype,
                device=self.device
            )
            self.tensors[name] = tensor

            logger.debug(f"  Allocated: {name} {shape} {torch_dtype}")

        logger.info(f"✓ Allocated {len(self.tensors)} tensors")

    def infer(self, feed_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run inference with given inputs.

        Args:
            feed_dict: Dictionary mapping input names to torch tensors

        Returns:
            Dictionary mapping output names to result tensors
        """
        if not self.context:
            raise RuntimeError("Context not created. Call activate() first.")

        # Copy input tensors
        for name, input_tensor in feed_dict.items():
            if name not in self.tensors:
                raise ValueError(f"Unknown input tensor: {name}")
            self.tensors[name].copy_(input_tensor)

        # Set tensor addresses
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Execute
        with torch.cuda.stream(self.stream):
            success = self.context.execute_async_v3(self.stream.cuda_stream)

        if not success:
            raise RuntimeError("TRT inference execution failed")

        # Wait for completion
        self.stream.synchronize()

        # Return outputs (filter out inputs)
        outputs = {}
        for name, tensor in self.tensors.items():
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT:
                outputs[name] = tensor

        return outputs

    def get_input_names(self):
        """Get list of input tensor names."""
        if not self.engine:
            raise RuntimeError("Engine not loaded")

        inputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(name)
        return inputs

    def get_output_names(self):
        """Get list of output tensor names."""
        if not self.engine:
            raise RuntimeError("Engine not loaded")

        outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        return outputs

    def __del__(self):
        """Cleanup resources."""
        if self.tensors:
            self.tensors.clear()
        if self.context:
            del self.context
        if self.engine:
            del self.engine
