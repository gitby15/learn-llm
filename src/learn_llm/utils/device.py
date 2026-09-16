
import torch

_CACHE_DEVICE = None
def get_cached_device():
    global _CACHE_DEVICE
    if _CACHE_DEVICE is not None:
        return _CACHE_DEVICE

    if torch.cuda.is_available():
        _CACHE_DEVICE = torch.device("cuda")
    elif torch.mps.is_available():
        _CACHE_DEVICE = torch.device("mps")
    else:
        _CACHE_DEVICE = torch.device("cpu")
    return _CACHE_DEVICE