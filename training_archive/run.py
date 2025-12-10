import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
from typing import Union, OrderedDict
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc
import torch
import numpy as np
import random

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def setup_deterministic_training(seed: int):
    """
    Setup comprehensive deterministic training environment for LoRA training.
    
    Args:
        seed: Random seed for reproducibility
    """
    print_acc(f"Setting up deterministic training with seed: {seed}")
    
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
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print_acc(f"Warning: Could not enable deterministic algorithms: {e}")
    
    # CUDA deterministic settings
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Set training seed in environment for processes to pick up
    os.environ['TRAINING_SEED'] = str(seed)
    
    print_acc("Deterministic training environment configured")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Make training deterministic'
    )

    args = parser.parse_args()
    
    if args.seed is not None:
        setup_deterministic_training(args.seed)

    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e

    print_end_message(jobs_completed, jobs_failed)


if __name__ == '__main__':
    main()