import torch

def run_mps_profiler(callback):
    with torch.mps.profiler.profile():
        callback()
    torch.mps.synchronize()
    pass