from collections.abc import Mapping
from typing import Any, Sequence

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