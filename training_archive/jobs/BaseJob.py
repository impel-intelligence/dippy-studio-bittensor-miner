import importlib
from collections import OrderedDict
from typing import List

from jobs.process import BaseProcess
import torch
import numpy as np
import random
import os

def setup_deterministic_training(seed: int = 123):
    """
    Setup comprehensive deterministic training environment for LoRA training.
    
    Args:
        seed: Random seed for reproducibility
    """
    print(f"Setting up deterministic training with seed: {seed}")
    
    # PyTorch seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # NumPy seeding (used in data augmentations and processing)
    np.random.seed(seed)
    
    # Python random module seeding (used throughout the codebase)
    random.seed(seed)
    
    # Python hash seed for reproducible hashing
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # cuDNN deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable PyTorch deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception as e:
        print(f"Warning: Could not enable deterministic algorithms: {e}")
    
    # CUDA deterministic settings
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Disable expandable segments in PyTorch CUDA allocator to avoid fragmentation issues
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
    
    # Set training seed in environment for processes to pick up
    os.environ['TRAINING_SEED'] = str(seed)
    
    print("Deterministic training environment configured")



class BaseJob:

    def __init__(self, config: OrderedDict):
        if not config:
            raise ValueError('config is required')
        self.process: List[BaseProcess]

        self.config = config['config']
        self.raw_config = config
        self.job = config['job']
        self.torch_profiler = self.get_conf('torch_profiler', False)
        self.name = self.get_conf('name', required=True)
        if 'meta' in config:
            self.meta = config['meta']
        else:
            self.meta = OrderedDict()

    def get_conf(self, key, default=None, required=False):
        if key in self.config:
            return self.config[key]
        elif required:
            raise ValueError(f'config file error. Missing "config.{key}" key')
        else:
            return default

    def run(self):
        print("")
        print(f"#############################################")
        print(f"# Running job: {self.name}")
        print(f"#############################################")
        print("")
        setup_deterministic_training()

        # implement in child class
        # be sure to call super().run() first
        pass

    def load_processes(self, process_dict: dict):
        # only call if you have processes in this job type
        if 'process' not in self.config:
            raise ValueError('config file is invalid. Missing "config.process" key')
        if len(self.config['process']) == 0:
            raise ValueError('config file is invalid. "config.process" must be a list of processes')

        module = importlib.import_module('jobs.process')
        setup_deterministic_training()

        # add the processes
        self.process = []
        for i, process in enumerate(self.config['process']):
            if 'type' not in process:
                raise ValueError(f'config file is invalid. Missing "config.process[{i}].type" key')

            # check if dict key is process type
            if process['type'] in process_dict:
                if isinstance(process_dict[process['type']], str):
                    ProcessClass = getattr(module, process_dict[process['type']])
                else:
                    # it is the class
                    ProcessClass = process_dict[process['type']]
                self.process.append(ProcessClass(i, self, process))
            else:
                raise ValueError(f'config file is invalid. Unknown process type: {process["type"]}')

    def cleanup(self):
        # if you implement this in child clas,
        # be sure to call super().cleanup() LAST
        del self
