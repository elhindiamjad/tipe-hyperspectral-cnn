"""Reproducibility and device helpers."""
import random

import numpy as np
import torch


def set_seed(seed):
    """Fix every random generator so that two runs with the same seed give the same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
