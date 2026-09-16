import torch
from torch import nn


class Rope(nn.Module):
    def __init__(self, base: float = 10000.0):
        super().__init__()
        # base 是用T计算角度时候的一个旋转率
        self.base = base

    # x是[B, T, D] D 需要是偶数，这里为了简单，我就不检查了
    def forward(self, x: torch.Tensor):
        B, T, C = x.shapec
        # 用T算出角度，然后用这个角度去旋转D里面的数据
        if C % 2 != 0:
            raise ValueError("Hidden dim must be even")

        # 按照T拿到所有的位置下标
        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        # 旋转率，每个维度用不同的旋转率，Todo： 为啥不可以用相同的旋转率？
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, C, 2, device=x.device, dtype=torch.float32) / C)
        )
        # 按照位置和旋转率，去计算每个位置的旋转角度，这是一个二维的角度：
        # 每个位置，旋转角度一样，同一个位置的hidden dim，角度一样，但是旋转率越往后越大
        theta = torch.outer(positions, inv_freq)

        cos = theta.cos()
        sin = theta.sin()
        
        x_even = x[:, :, ::2]
        x_odd = x[:, :, 1::2]
        
        # 两两一组，转TMD
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
        return out


if __name__ == "__main__":
    from learn_llm.utils.device import get_cached_device
    
    device = get_cached_device()
    
    
    B, T, C = (1, 10, 1024)
    x = torch.ones(B, T, C, device=device)
    rope = Rope()
    rope(x)
    