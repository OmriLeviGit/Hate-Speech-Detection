import io
import os
import random
import sys

import numpy as np
import torch


def capture_output(func, *args, **kwargs):
    buffer = io.StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = buffer
        func(*args, **kwargs)
    finally:
        sys.stdout = original_stdout
    return buffer.getvalue()

def reset_seeds(seed=42):
    """Reset all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_header(model_name, total_length=80):
    name_length = len(model_name)
    side_length = (total_length - name_length - 2) // 2

    extra_dash = 1 if (total_length - name_length - 2) % 2 != 0 else 0

    print("\n", "-" * side_length + f" {model_name} " + "-" * (side_length + extra_dash))


def format_duration(seconds):
    """ Convert duration from seconds to hh:mm:ss or mm:ss format """
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
    return f"{int(minutes)}:{int(seconds):02}"


def check_device():
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        
        try:
            tensor = torch.randn(3, 3).cuda()
            print(f"Tensor on GPU: {tensor}")
        except Exception as e:
            print(f"Failed to allocate tensor on GPU: {e}")
    else:
        print("CUDA is not available, running on CPU.")