import os
import gc
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
from functools import partial
import time
import hashlib
import json
import resource
import re

import numpy as np
import onnx
import PIL.Image
import tensorrt as trt
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import validate_hf_hub_args
from onnx import shape_inference, numpy_helper
from packaging import version
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.onnx.loader import fold_constants
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.mod.trt_importer import lazy_import_trt
trt = lazy_import_trt()

from diffusers import DiffusionPipeline, FluxPipeline, FluxTransformer2DModel
from diffusers.configuration_utils import FrozenDict, deprecate
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.normalization import RMSNorm

import modelopt.torch.quantization as mtq


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


import contextlib
@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


def export_fused_transformer(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    lora_scale: float = 1.0
):
    """
    Loads a base diffusion pipeline, fuses a LoRA into it, and saves the
    resulting fused pipeline to a new directory.

    Args:
        base_model_path (str):
            Path to the pre-trained base model (e.g., './baseflux').
        lora_path (str):
            Path to the LoRA weights file (e.g., 'lora.safetensors').
        output_path (str):
            Path to the file where the fused transformer state-dict will be saved.
        lora_scale (float, optional):
            The weight to apply to the LoRA during fusion. Defaults to 1.0.
    """
    print(f"\n[1/3] Loading base model '{base_model_path}' onto GPU...")
    pipe = DiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    print("      Base model loaded successfully.")

    if lora_path:
        print(f"\n[2/3] Loading and fusing LoRA '{os.path.basename(lora_path)}'...")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)
        pipe.unload_lora_weights()
        print(f"      LoRA fused successfully with a scale of {lora_scale}.")

    print(f"\n[3/3] Saving fused pipeline to '{output_path}'...")
    torch.save(pipe.transformer.state_dict(), output_path)
    print("      Fused pipeline saved successfully.")

    del pipe
    torch.cuda.empty_cache()
    print("\n--- LoRA fusion export finished successfully! ---")


class ONNXSafeRMSNorm(torch.nn.Module):
    def __init__(self, module: RMSNorm):
        super().__init__()
        self.weight = module.weight
        self.eps = module.eps

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states_norm = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states_scaled = self.weight * hidden_states_norm
        return hidden_states_scaled.to(original_dtype)

def patch_rms_norm(model):
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            parent_path = name.rsplit('.', 1)
            if len(parent_path) == 1: # The module is at the top level
                parent_module = model
                child_name = parent_path[0]
            else:
                parent_name, child_name = parent_path
                parent_module = model.get_submodule(parent_name)

            new_module = ONNXSafeRMSNorm(module)
            setattr(parent_module, child_name, new_module)
            #logger.warning(f"Patched RMSNorm at: {name}")
    return model



# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


class Engine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray)]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_all_tactics=False,
        timing_cache=None,
    ):
        logger.warning(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        extra_build_args = {
            "memory_pool_limits": {
                trt.MemoryPoolType.WORKSPACE: 2**33,
                trt.MemoryPoolType.TACTIC_DRAM: 2**33,
            },
            "tactic_sources": [
                trt.TacticSource.CUBLAS,
                trt.TacticSource.CUBLAS_LT,
                trt.TacticSource.CUDNN,
                trt.TacticSource.EDGE_MASK_CONVOLUTIONS,
                trt.TacticSource.JIT_CONVOLUTIONS,
            ],
            "refittable": True,
            #"strip_plan": False,
            "precision_constraints": "obey",
            #"tf32": False,
            #"fp8": False,
            #"builder_optimization_level": 3,
        }

        builder, network, parser = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])

        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in [trt.LayerType.REDUCE]:
                layer.precision = trt.float32
            if layer.type == trt.LayerType.ELEMENTWISE and '/Pow' in layer.name:
                layer.precision = trt.float32

        engine = engine_from_network(
            (builder, network, parser),
            config=CreateConfig(
                fp16=fp16,
                profiles=[p],
                load_timing_cache=timing_cache,
                **extra_build_args
            ),
            save_timing_cache=timing_cache,
        )
        save_engine(engine, path=self.engine_path)

    def load(self):
        logger.warning(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self):
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[name] = tensor

    def infer(self, feed_dict, stream):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")

        return self.tensors


class BaseModel:
    def __init__(self, model, fp16=False, device="cuda", max_batch_size=16, embedding_dim=768, text_maxlen=77):
        self.model = model
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        return self.model

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


def getOnnxPath(model_name, onnx_dir):
    return os.path.join(onnx_dir, model_name + ".onnx")


def getEnginePath(model_name, engine_dir):
    return os.path.join(engine_dir, model_name + ".plan")


def build_engines(
    models: dict,
    engine_dir,
    onnx_dir,
    onnx_opset,
    opt_image_height,
    opt_image_width,
    opt_batch_size=1,
    force_engine_rebuild=False,
    static_batch=False,
    static_shape=True,
    enable_all_tactics=False,
):
    built_engines = {}
    if not os.path.isdir(onnx_dir):
        os.makedirs(onnx_dir)
    if not os.path.isdir(engine_dir):
        os.makedirs(engine_dir)

    # Export models to ONNX
    for model_name, model_obj in models.items():
        engine_path = getEnginePath(model_name, engine_dir)
        if force_engine_rebuild or not os.path.exists(engine_path):
            logger.warning("Building Engines...")
            logger.warning("Engine build can take a while to complete")
            onnx_path = getOnnxPath(model_name, onnx_dir)
            if force_engine_rebuild or not os.path.exists(onnx_path):
                logger.warning(f"Exporting model: {onnx_path}")
                model = model_obj.get_model()
                with torch.inference_mode():#, torch.autocast("cuda"):
                    inputs = model_obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
                    torch.onnx.export(
                        model,
                        inputs,
                        onnx_path,
                        opset_version=onnx_opset,
                        export_params=True,
                        do_constant_folding=False,
                        keep_initializers_as_inputs=True,
                        use_external_data_format=True,
                        input_names=model_obj.get_input_names(),
                        output_names=model_obj.get_output_names(),
                        dynamic_axes=model_obj.get_dynamic_axes(),
                        #operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                    )
                    print("ONNX len keys:", len({w.name: tuple(w.dims) for w in onnx.load(onnx_path).graph.initializer}.keys()))
                del model
                torch.cuda.empty_cache()
                gc.collect()
            else:
                logger.warning(f"Found cached model: {onnx_path}")

    # Build TensorRT engines
    for model_name, model_obj in models.items():
        engine_path = getEnginePath(model_name, engine_dir)
        engine = Engine(engine_path)
        onnx_path = getOnnxPath(model_name, onnx_dir)

        if force_engine_rebuild or not os.path.exists(engine.engine_path):
            engine.build(
                onnx_path,
                fp16=True,
                input_profile=model_obj.get_input_profile(
                    opt_batch_size,
                    opt_image_height,
                    opt_image_width,
                    static_batch=static_batch,
                    static_shape=static_shape,
                ),
                timing_cache=str(Path(engine_path).with_suffix(".timing")),
            )
        built_engines[model_name] = engine

    # Load and activate TensorRT engines
    for model_name, model_obj in models.items():
        engine = built_engines[model_name]
        engine.load()
        engine.activate()

    return built_engines


def runEngine(engine, feed_dict, stream):
    return engine.infer(feed_dict, stream)



class DiffusionTransformer(BaseModel):
    """
    Exporter class for a FluxTransformer2DModel.
    This version is updated to precisely match the `FullArgSpec` provided by the user.
    """
    def __init__(
        self,
        model,
        fp16=False,
        device="cuda",
        max_batch_size=4,
        in_channels=16,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        text_seq_len=512,
    ):
        super(DiffusionTransformer, self).__init__(
            model=model,
            fp16=fp16,
            device=device,
            max_batch_size=max_batch_size,
        )
        self.name = "Transformer"
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.text_seq_len = text_seq_len

    def get_input_names(self):
        # The exact names and order from the provided `FullArgSpec`
        return ["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep", "img_ids", "txt_ids", "guidance"]

    def get_output_names(self):
        return ["output_hidden_states"]

    def get_dynamic_axes(self):
        # B = batch_size, H = latent_height, W = latent_width
        return {
            "hidden_states": {0: "B"},
            "encoder_hidden_states": {0: "B"},
            "pooled_projections": {0: "B"},
            "timestep": {0: "B"},
            "guidance": {0: "B"},
            "output_hidden_states": {0: "B"},
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32

        # --- Create sample inputs that match the full signature ---
        hidden_states = torch.randn(
            batch_size, self.joint_attention_dim, self.in_channels,
            dtype=dtype, device=self.device
        )

        encoder_hidden_states = torch.randn(
            batch_size, self.text_seq_len, self.joint_attention_dim,
            dtype=dtype, device=self.device
        )
        pooled_projections = torch.randn(
            batch_size, self.pooled_projection_dim,
            dtype=dtype, device=self.device
        )
        timestep = torch.randn(batch_size, device=self.device, dtype=dtype)

        txt_ids = torch.randn(self.text_seq_len, 3, device=self.device, dtype=dtype)
        img_ids = torch.randn(self.joint_attention_dim, 3, device=self.device, dtype=dtype)

        # Create guidance tensor (for CFG)
        guidance = torch.full((batch_size,), 7.5, dtype=dtype, device=self.device)

        return (hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance)

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch, max_batch, _, _, _, _,
            min_latent_height, max_latent_height,
            min_latent_width, max_latent_width,
        ) = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            "hidden_states": [
                (min_batch, self.joint_attention_dim, self.in_channels),
                (batch_size, self.joint_attention_dim, self.in_channels),
                (max_batch, self.joint_attention_dim, self.in_channels),
            ],
            "encoder_hidden_states": [
                (min_batch, self.text_seq_len, self.joint_attention_dim),
                (batch_size, self.text_seq_len, self.joint_attention_dim),
                (max_batch, self.text_seq_len, self.joint_attention_dim),
            ],
            "pooled_projections": [
                (min_batch, self.pooled_projection_dim),
                (batch_size, self.pooled_projection_dim),
                (max_batch, self.pooled_projection_dim),
            ],
            "timestep": [(min_batch,), (batch_size,), (max_batch,)],
            "img_ids": [
                (self.joint_attention_dim, 3),
                (self.joint_attention_dim, 3),
                (self.joint_attention_dim, 3),
            ],
            "txt_ids": [
                (self.text_seq_len, 3),
                (self.text_seq_len, 3),
                (self.text_seq_len, 3),
            ],
            "guidance": [(min_batch,), (batch_size,), (max_batch,)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "hidden_states": (batch_size, self.joint_attention_dim, self.in_channels),
            "encoder_hidden_states": (batch_size, self.text_seq_len, self.joint_attention_dim),
            "pooled_projections": (batch_size, self.pooled_projection_dim),
            "timestep": (batch_size,),
            "img_ids": (self.joint_attention_dim, 3),
            "txt_ids": (self.text_seq_len, 3),
            "guidance": (batch_size,),
            "output_hidden_states": (batch_size, self.joint_attention_dim, self.in_channels),
        }



class TRTRefitter:
    def __init__(self, engine, flux_path):
        self.refitter = trt.Refitter(engine, TRT_LOGGER)
        transformer_sd = DiffusionPipeline.from_pretrained(flux_path, torch_dtype=torch.float16).transformer.state_dict()
        for trt_weight_name in self.refitter.get_all_weights():
            pyt_weight_name = trt_weight_name.replace('transformer.', '').replace('base_layer.', '')
            if pyt_weight_name in transformer_sd:
                self.refitter.set_named_weights(
                    trt_weight_name,
                    trt.Weights(transformer_sd[pyt_weight_name].numpy())
                )

    def prepare_lora_refit(self, lora_path):
        from safetensors.torch import load_file
        lora_sd = load_file(lora_path)
        start_time = time.time()
        for trt_weight_name in self.refitter.get_all_weights():
            if '.lora.' in trt_weight_name:
                pyt_weight_name = trt_weight_name.replace('.lora.weight', '.weight')
                self.refitter.set_named_weights(
                    trt_weight_name,
                    trt.Weights(lora_sd[pyt_weight_name].numpy())
                )
        print(f"--- LoRA prep in {time.time() - start_time:.4f} seconds ---")

    def prepare_refit(self, state_dict):
        print("--- Starting TensorRT engine refit ---")
        start_time = time.time()

        for trt_name, pt_name in self.weights_name_map.items():
            if pt_name in state_dict:
                # Weights must be contiguous, on CPU, and in numpy format for the refitter
                new_weight = state_dict[pt_name].clone(memory_format=torch.contiguous_format).cpu().numpy()
                self.refitter.set_named_weights(trt_name, trt.Weights(new_weight))
            else:
                print(f"State dict '{pt_name}' for TRT '{trt_name}' missing")

        end_time = time.time()
        print(f"--- Engine refit prepared in {end_time - start_time:.4f} seconds ---")

    def commit_refit(self):
        start_time = time.time()
        if not self.refitter.refit_cuda_engine():
            print(f"Failed to refit, missing weights:\n{self.refitter.get_missing_weights()}")
            raise RuntimeError(f"Failed to refit TensorRT engine")
        end_time = time.time()
        print(f"--- Engine refit committed in {end_time - start_time:.4f} seconds ---")



class TRTTransformer(torch.nn.Module):
    """
    A stand-in for the FluxTransformer2DModel that uses a TensorRT engine for its forward pass.
    It copies necessary attributes from the original model to ensure compatibility with the pipeline.
    """
    def __init__(self, engine_path: str, transformer_config, device: torch.device, max_batch_size):
        super().__init__()
        self.device = device
        self.engine = Engine(engine_path)
        self.stream = cuda.Stream()

        self.config = transformer_config
        self.encoder_hid_proj = self.config.encoder_hid_proj
#        if hasattr(original_transformer, "encoder_hid_proj"):
#             self.encoder_hid_proj = original_transformer.encoder_hid_proj
#        else:
#             self.encoder_hid_proj = SimpleNamespace(num_ip_adapters=0)

        self.engine.load()
        self.engine.activate()

        # Define the expected shapes for buffer allocation
        # Note: 'max_batch_size' is implicitly 1 because we built a static engine.
        #joint_attention_dim = original_transformer.config.joint_attention_dim
        #pooled_projection_dim = original_transformer.config.pooled_projection_dim
        text_seq_len = 512

        shape_dict = {
            "hidden_states": (max_batch_size, self.config.joint_attention_dim, self.config.in_channels),
            "encoder_hidden_states": (max_batch_size, text_seq_len, self.config.joint_attention_dim),
            "pooled_projections": (max_batch_size, self.config.pooled_projection_dim),
            "timestep": (max_batch_size,),
            "img_ids": (self.config.joint_attention_dim, 3),
            "txt_ids": (text_seq_len, 3),
            "guidance": (max_batch_size,),
            "output_hidden_states": (max_batch_size, self.config.joint_attention_dim, self.config.in_channels),
        }

        print("Allocating memory for TensorRT engine...")
        self.engine.allocate_buffers(shape_dict, self.device)
        print("TRT Transformer wrapper initialized.")

    @property
    def dtype(self) -> torch.dtype:
        # The engine was built with float16
        return torch.float16

    def forward(
        self,
        hidden_states,
        timestep,
        guidance,
        pooled_projections,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        joint_attention_kwargs=None, # This is passed by the pipeline, so it must be in the signature
        return_dict=False # Same for this argument
    ):
        assert(not return_dict)

        hidden_states = hidden_states.to(self.device, dtype=self.dtype)
        encoder_hidden_states = encoder_hidden_states.to(self.device, dtype=self.dtype)
        pooled_projections = pooled_projections.to(self.device, dtype=self.dtype)
        guidance = guidance.to(self.device, dtype=self.dtype)
        txt_ids = txt_ids.to(self.device, dtype=self.dtype)
        img_ids = img_ids.to(self.device, dtype=self.dtype)

        feed_dict = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance,
        }

        outputs = self.engine.infer(feed_dict, self.stream.ptr)
        self.stream.synchronize()

        output_hidden_states = outputs["output_hidden_states"]
        return (output_hidden_states,)


def do_calibrate(pipe, model):
    pipe.transformer = model
    prompt = "A cute anime girl wearing a hoodie, sitting on a sofa in the living room sipping coffee"
    image = pipe(
        prompt,
        guidance_scale=7.5,
        height=1024,
        width=1024,
        num_inference_steps=28,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]


def fp8_filter_func(name: str) -> bool:
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|x_embedder).*"
    )
    return pattern.match(name) is not None


def build_transformer_engine_from_pipeline(
    model_path: str,
    lora_path: str,
    onnx_dir: str,
    engine_dir: str,
    force_engine_rebuild: bool = False,
    max_batch_size: int = 2,
    image_height: int = 1024,
    image_width: int = 1024,
    onnx_opset: int = 18,
    static_batch: bool = False,
    static_shape: bool = False
):
    """
    Loads a fused Diffusers pipeline, extracts its transformer component,
    and builds a TensorRT engine for it.
    """
    logger.warning(f"Loading fused pipeline from: {model_path}")
    pipe = DiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to("cuda")
    if lora_path != "":
        pipe.load_lora_weights(lora_path, adapter_name="lora")
        pipe.set_adapters(["lora"], adapter_weights=[1.0])

    transformer_model = pipe.transformer
    transformer_model = patch_rms_norm(transformer_model)
    #mtq.quantize(transformer_model, mtq.FP8_DEFAULT_CFG, partial(do_calibrate, pipe))
    #mtq.disable_quantizer(transformer_model, fp8_filter_func)

    # This wrapper ensures the model is called with the exact signature required for export
    # and forces it to return a simple tensor tuple, which is ideal for ONNX.
    class TransformerWrapper(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

        def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance):
            # ONNX export requires a tuple of tensors as output.
            # We call the model with all required arguments and `return_dict=False`.
            output = self.transformer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False
            )
            # The model returns a tuple, we only need the first element (the output hidden states)
            return output[0]

    transformer_to_export = TransformerWrapper(transformer_model)
    device = pipe.device

    transformer_exporter = DiffusionTransformer(
        model=transformer_to_export,
        fp16=True,
        device=device,
        max_batch_size=max_batch_size,
        in_channels=transformer_model.config.in_channels,
        joint_attention_dim=transformer_model.config.joint_attention_dim,
        pooled_projection_dim=transformer_model.config.pooled_projection_dim,
    )

    models_to_build = {"transformer": transformer_exporter}

    logger.warning("Starting ONNX export and TensorRT engine build for the Transformer...")
    built_engines = build_engines(
        models=models_to_build,
        engine_dir=engine_dir,
        onnx_dir=onnx_dir,
        onnx_opset=onnx_opset,
        opt_image_height=image_height,
        opt_image_width=image_width,
        opt_batch_size=max_batch_size,
        force_engine_rebuild=force_engine_rebuild,
        static_batch=static_batch,
        static_shape=static_shape,
    )

    torch_to_onnx_map(transformer_to_export.transformer, f'{onnx_dir}/transformer.onnx', f'{engine_dir}/mapping.json')

    logger.warning(f"Successfully built engine for transformer: {built_engines['transformer'].engine_path}")

    del pipe, transformer_model, transformer_to_export
    torch.cuda.empty_cache()
    gc.collect()

    return built_engines



def torch_to_onnx_map(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    output_map_path: str,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Compares a PyTorch model's state_dict with an ONNX model's initializers by
    hashing their byte content. It creates a name mapping, reports on the
    matching process, and saves the successful mapping to a JSON file.

    This hashing approach is significantly faster and more memory-efficient
    for large models than using raw byte strings as dictionary keys.

    Args:
        pytorch_model (torch.nn.Module):
            The PyTorch model instance that corresponds to the ONNX file.
        onnx_path (str):
            The file path to the ONNX model.
        output_map_path (str):
            The file path where the resulting JSON mapping will be saved.
            The format will be: {"onnx_initializer_name": "pytorch_state_dict_key"}
        verbose (bool, optional):
            If True, prints a detailed report of the matching process. Defaults to True.

    Returns:
        Dict[str, str]:
            The generated mapping dictionary.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")

    # --- 1. Load models and prepare data structures using hashing ---
    if verbose:
        print("--- Starting PyTorch to ONNX Weight Mapper (Hashed) ---")
        print(f"Loading ONNX model from: {onnx_path}")
    model_proto = onnx.load(onnx_path)
    state_dict = pytorch_model.state_dict()

    # Create a map from a weight's SHA256 hash to its metadata
    def get_hash(byte_data: bytes) -> str:
        return hashlib.sha256(byte_data).hexdigest()

    onnx_weights_map = {
        get_hash(init.raw_data): {'name': init.name, 'shape': tuple(init.dims)}
        for init in model_proto.graph.initializer
    }

    torch_weights_map = {
        get_hash(tensor.contiguous().cpu().numpy().tobytes()): {'name': name, 'shape': tuple(tensor.shape)}
        for name, tensor in state_dict.items()
    }

    # --- 2. Hash Matching ---
    final_mapping = {}
    mapped_onnx_names = set()
    mapped_torch_names = set()

    # Iterate through the smaller map for efficiency
    for pt_hash, pt_meta in torch_weights_map.items():
        if pt_hash in onnx_weights_map:
            onnx_meta = onnx_weights_map[pt_hash]
            final_mapping[onnx_meta['name']] = pt_meta['name']
            mapped_onnx_names.add(onnx_meta['name'])
            mapped_torch_names.add(pt_meta['name'])

    # --- 3. Report the results (identical to before) ---
    if verbose:
        print("\n--- Mapping Report ---")
        print(f"Total PyTorch state_dict keys: {len(state_dict)}")
        print(f"Total ONNX initializers:       {len(model_proto.graph.initializer)}")
        print(f"Successfully matched weights:  {len(final_mapping)}")

        unmapped_torch_keys = set(state_dict.keys()) - mapped_torch_names
        all_onnx_names = {init.name for init in model_proto.graph.initializer}
        unmapped_onnx_initializers = all_onnx_names - mapped_onnx_names

        if unmapped_torch_keys:
            print(f"\n[INFO] {len(unmapped_torch_keys)} PyTorch keys had NO match in ONNX:")
            # To avoid flooding the console, maybe print only a few
            for i, key in enumerate(sorted(list(unmapped_torch_keys))):
                if i < 10: print(f"  - {key}")
            if len(unmapped_torch_keys) > 10: print(f"  - ... and {len(unmapped_torch_keys)-10} more.")
            breakpoint()

        if unmapped_onnx_initializers:
            print(f"\n[INFO] {len(unmapped_onnx_initializers)} ONNX initializers had NO match in PyTorch:")
            for i, name in enumerate(sorted(list(unmapped_onnx_initializers))):
                if i < 10: print(f"  - {name}")
            if len(unmapped_onnx_initializers) > 10: print(f"  - ... and {len(unmapped_onnx_initializers)-10} more.")
            breakpoint()

        print("-" * 25)

    # --- 4. Save the mapping to a file (identical to before) ---
    try:
        os.makedirs(os.path.dirname(output_map_path), exist_ok=True)
        with open(output_map_path, 'w') as f:
            json.dump(final_mapping, f, indent=4, sort_keys=True)
        if verbose:
            print(f"\nSuccessfully saved weight map to: {output_map_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save mapping file: {e}")

    return final_mapping



if __name__ == "__main__":
    # Get model path from environment variable or use HuggingFace model ID
    MODEL_PATH = os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev")

    # Check if MODEL_PATH is a local directory that exists
    if os.path.isdir(MODEL_PATH):
        logger.warning(f"Using local model directory: {MODEL_PATH}")
    else:
        logger.warning(f"Using HuggingFace model: {MODEL_PATH}")

    # LORA_PATH = "./output/lora_run_1/lora_run_1.safetensors"
    LORA_PATH = os.getenv("LORA_PATH", "")
    ONNX_EXPORT_DIR = os.getenv("ONNX_EXPORT_DIR", "./onnx")
    ENGINE_EXPORT_DIR = os.getenv("ENGINE_EXPORT_DIR", "./trt")

    # increase max files open limit (for onnx parsing)
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_limit = min(65536, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))

    logger.warning("--- Starting Fused Transformer to TensorRT Export ---")

    # Build engine (function handles both local directories and HuggingFace IDs)
    build_transformer_engine_from_pipeline(
        model_path=MODEL_PATH,
        lora_path=LORA_PATH,
        onnx_dir=ONNX_EXPORT_DIR,
        engine_dir=ENGINE_EXPORT_DIR,
        force_engine_rebuild=True,
        max_batch_size=1,
        image_height=1024,
        image_width=1024,
        static_batch=False,
        static_shape=True,
        onnx_opset=18,
    )
    logger.warning("--- Transformer Export Finished ---")
