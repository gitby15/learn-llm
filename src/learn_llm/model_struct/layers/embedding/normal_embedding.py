import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.n_embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.n_embedding(x)




def profile():
    from learn_llm.utils.device import get_cached_device
    from tim_benchmark.profile import run_mps_profiler

    device = get_cached_device()
    embedding = Embedding(100, 128).to(device)
    inputs = torch.randint(0, 100, (100, 10), device=device)
    run_mps_profiler(lambda: embedding(inputs))

if __name__ == "__main__":
    profile()
    

