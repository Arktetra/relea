from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np
import random
import torch
import torch.nn as nn

def has_instance(list: Sequence, type: Any):
    for o in list:
        if isinstance(o, type):
            return True
    return False

def to_cpu(x):
    if isinstance(x, Mapping):
        return {k: to_cpu(v) for k, v in x}
    if isinstance(x, list):
        return [to_cpu(o) for o in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(list(x)))
    return x.detach().cpu()

def get_trainable_parameters(model: nn.Module):
    n_params = 0

    for param in model.parameters():
        if param.requires_grad:
            n_params += param.numel()

    return n_params

def get_total_parameters(model: nn.Module):
    n_params = 0

    for param in model.parameters():
        n_params += param.numel()

    return n_params

def apply_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)
        