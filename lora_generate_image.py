import contextlib
import threading
import queue
import time
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict
from types import SimpleNamespace
import json
from pathlib import Path

os.environ['TQDM_DISABLE'] = '1'

import torch
from diffusers import DiffusionPipeline

from trt import TRTTransformer, TRTRefitter


if __name__ == '__main__' and False:
    prompt = "A cute anime girl wearing a hoodie, sitting on a sofa in the living room sipping coffee"

    pipe = DiffusionPipeline.from_pretrained(
        "/ephemeral/baseflux",
        torch_dtype=torch.float16,
        # use_auth_token=True
    ).to("cuda")

    transformer_config = pipe.transformer.config
    if hasattr(pipe.transformer, "encoder_hid_proj"):
         transformer_config.encoder_hid_proj = pipe.transformer.encoder_hid_proj
    else:
         transformer_config.encoder_hid_proj = SimpleNamespace(num_ip_adapters=0)
    del pipe.transformer
    torch.cuda.empty_cache()

    pipe.transformer = TRTTransformer("./trt/transformer.plan", transformer_config, torch.device("cuda"), max_batch_size=1)

    refitter = TRTRefitter(pipe.transformer.engine.engine, './baseflux')

    b = time.time()
    refitter.prepare_lora_refit('./output/lora_run_1/lora_run_1.safetensors')
    refitter.commit_refit()
    print(f"refit: {(time.time() - b):.3f} secs")

    b = time.time()
    generator = torch.Generator(device="cuda").manual_seed(42)
    images = pipe(prompt, num_inference_steps=28, guidance_scale=7.5, height=1024, width=1024, generator=generator).images
    print(f"trt: {(time.time() - b):.3f} secs")
    [img.save(f"./trt_lora1_{i}.png") for i, img in enumerate(images)]

    b = time.time()
    refitter.prepare_lora_refit('./output/lora_run_1/lora_run_1_000001000.safetensors')
    refitter.commit_refit()
    print(f"refit: {(time.time() - b):.3f} secs")

    b = time.time()
    generator = torch.Generator(device="cuda").manual_seed(42)
    images = pipe(prompt, num_inference_steps=28, guidance_scale=7.5, height=1024, width=1024, generator=generator).images
    print(f"trt: {(time.time() - b):.3f} secs")
    [img.save(f"./trt_lorab_{i}.png") for i, img in enumerate(images)]

    b = time.time()
    refitter.prepare_lora_refit('./output/lora_run_1/lora_run_1.safetensors')
    refitter.commit_refit()
    print(f"refit: {(time.time() - b):.3f} secs")

    b = time.time()
    generator = torch.Generator(device="cuda").manual_seed(42)
    images = pipe(prompt, num_inference_steps=28, guidance_scale=7.5, height=1024, width=1024, generator=generator).images
    print(f"trt: {(time.time() - b):.3f} secs")
    [img.save(f"./trt_lora2_{i}.png") for i, img in enumerate(images)]


    import sys; sys.exit()



@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


@dataclass
class InferenceRequest:
    prompt: str
    output_path: str
    lora_path: str = None

    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    seed: int = 42

    @property
    def name(self):
        return Path(self.lora_path).name


class TRTInferenceServer:
    def __init__(self, base_model_path: str, engine_path: str, mapping_path: str):
        self.request_queue = queue.Queue()
        self.engine_queue = queue.Queue()
        self.inference_queue = queue.Queue()
        self.base_model_path = base_model_path

        print("[Server] Initializing...")
        self.pipe, self.transformer_config = self._initialize_pipeline(base_model_path)

        # two TRT engines
        self.engine_queue.put(self._create_trt_transformer(engine_path, self.transformer_config))
        self.engine_queue.put(self._create_trt_transformer(engine_path, self.transformer_config))

        self.loader_thread = threading.Thread(
            target=self._weight_loader_worker, daemon=True
        )
        self.inference_thread = threading.Thread(
            target=self._inference_worker, daemon=True
        )
        print("[Server] Initialization complete.")

    def _create_trt_transformer(self, engine_path, transformer_config):
        transformer = TRTTransformer(engine_path, transformer_config, torch.device("cuda"), max_batch_size=1)
        refitter = TRTRefitter(transformer.engine.engine, self.base_model_path)
        return transformer, refitter

    def _initialize_pipeline(self, base_model_path):
        """Loads the base pipeline and replaces the transformer with its TRT equivalent."""
        print(f"[Server] Loading base pipeline from: {base_model_path}")
        pipe = DiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
        ).to("cuda")

        # Preserve config before deleting the original transformer
        transformer_config = pipe.transformer.config
        if hasattr(pipe.transformer, "encoder_hid_proj"):
            transformer_config.encoder_hid_proj = pipe.transformer.encoder_hid_proj
        else:
            transformer_config.encoder_hid_proj = SimpleNamespace(num_ip_adapters=0)

        del pipe.transformer
        torch.cuda.empty_cache()

        return pipe, transformer_config

    def start(self):
        """Starts the worker threads."""
        print("[Server] Starting worker threads...")
        self.loader_thread.start()
        self.inference_thread.start()

    def stop(self):
        """Waits for all jobs to complete and shuts down the workers."""
        print("[Server] Shutting down... Waiting for queues to empty.")
        self.request_queue.put(None) # Sentinel to stop the loader
        self.request_queue.join()
        self.inference_queue.join()
        self.loader_thread.join()
        self.inference_thread.join()
        print("[Server] Shutdown complete.")

    def submit(self, request: InferenceRequest):
        """Submits a new inference request to the queue."""
        self.request_queue.put(request)

    def _weight_loader_worker(self):
        """
        [Producer Thread]
        Pulls requests, loads the corresponding LoRA weights (if any) and
        applies them to the base model, then passes it to the inference queue.
        """
        while True:
            transformer, refitter = self.engine_queue.get()

            request = self.request_queue.get()
            if request is None:
                self.inference_queue.put(None) # Pass sentinel to inference worker
                self.engine_queue.task_done()
                self.request_queue.task_done()
                break

            print(f"[Loader] Preparing weights for {request.lora_path if request.lora_path else 'base model'}")
            with nvtx_range(f"prepare_{request.name}"):
                # Apply LoRA if provided, otherwise use base weights
                refitter.prepare_lora_refit(request.lora_path, lora_scale=1.0)
                refitter.commit_refit()

            self.inference_queue.put((request, transformer, refitter))
            self.engine_queue.task_done()
            self.request_queue.task_done()
        print("[Loader] Worker finished.")

    def _inference_worker(self):
        """
        [Consumer Thread]
        Pulls (request, state_dict) jobs, prepares the refit on CPU,
        then commits the refit and runs inference on GPU.
        """
        while True:
            job = self.inference_queue.get()
            if job is None:
                self.inference_queue.task_done()
                break

            request, transformer, refitter = job
            # Commit refit and run inference (GPU-bound).
            print(f"[Inference] Running inference for {request.lora_path if request.lora_path else 'base model'}")
            inference_start_time = time.time()

            with nvtx_range(f"swap_{request.name}"):
                self.pipe.transformer = transformer
            with nvtx_range(f"infer_{request.name}"):
                generator = torch.Generator(device="cuda").manual_seed(request.seed)
                image = self.pipe(
                    prompt=request.prompt,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    height=request.height,
                    width=request.width,
                    generator=generator
                ).images[0]

            inference_end_time = time.time()
            print(f"[Inference] Inference for {request.lora_path} took {inference_end_time - inference_start_time:.3f}s")

            self.engine_queue.put((transformer, refitter))

            # Save output (I/O-bound)
            with nvtx_range(f"save_{request.name}"):
                os.makedirs(os.path.dirname(request.output_path), exist_ok=True)
                image.save(request.output_path)
            print(f"[Inference] Saved output to '{request.output_path}'")

            self.inference_queue.task_done()
        print("[Inference] Worker finished.")


if __name__ == "__main__":
    # Get model path from environment variable or use HuggingFace model ID
    import os
    base_model_path = os.getenv("MODEL_PATH", "black-forest-labs/FLUX.1-dev")
    
    # Check if MODEL_PATH is a local directory that exists
    if os.path.isdir(base_model_path):
        print(f"Using local model directory: {base_model_path}")
    else:
        print(f"Using HuggingFace model: {base_model_path}")
    
    server = TRTInferenceServer(
        base_model_path=base_model_path,
        engine_path="./trt/transformer.plan",
        mapping_path="./trt/mapping.json",
    )
    server.start()

    requests = [
        InferenceRequest(
            prompt = "A cute anime girl wearing a hoodie, sitting on a sofa in the living room sipping coffee",
            output_path="output/prompt_1_base.png",
            # lora_path="./output/lora_run_1/lora_run_1.safetensors",
            lora_path="",
            seed=42,
        ),
        # InferenceRequest(
        #     prompt = "A cute anime girl wearing a hoodie, sitting on a sofa in the living room sipping coffee",
        #     output_path="output/prompt_1_lora1.png",
        #     lora_path="./output/lora_run_1/lora_run_1_000001000.safetensors",
        #     seed=42,
        # ),
        # InferenceRequest(
        #     prompt = "A cute anime girl wearing a hoodie",
        #     output_path="output/prompt_2_base.png",
        #     lora_path="./output/lora_run_1/lora_run_1_000001000.safetensors",
        #     seed=42,
        # ),
        # InferenceRequest(
        #     prompt = "A cute anime girl wearing a hoodie",
        #     output_path="output/prompt_2_lora1.png",
        #     lora_path="./output/lora_run_1/lora_run_1.safetensors",
        #     seed=42,
        # ),
    ]

    for req in requests:
        server.submit(req)

    server.stop()
    print("\n--- All jobs completed successfully. ---")

