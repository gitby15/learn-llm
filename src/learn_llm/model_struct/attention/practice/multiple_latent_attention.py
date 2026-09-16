
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadLatentAttention(nn.Module):
    def __init__(self, hidden_size: int, sub_head_dim: int, num_heads: int, latent_dim: int):
        super().__init__()
        assert sub_head_dim * num_heads == hidden_size
        self.hidden_size = hidden_size
        self.sub_head_dim = sub_head_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim


        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv_down_proj = nn.Linear(hidden_size, latent_dim, bias=False)
        
        self.kv_down_proj = nn.Linear(latent_dim, hidden_size, bias=False)



    def _attention(self, query, key, value):
        B, T, C = query.shape

        query = query.reshape(B, T, self.num_heads, self.sub_head_dim).transpose(1, 2)
        key = key.reshape(B, T, self.num_heads, self.sub_head_dim).transpose(1, 2)
        value = value.reshape(B, T, self.num_heads, self.sub_head_dim).transpose(1, 2)

        # 这里的形状会是[B, T, T]
        score = query @ key.transpose(-2, -1) / (C**0.5)
        latent_score = self.to_latent(score)
        score = F.softmax(latent_score, dim=-1)

        latent_attention = score @ value
        attention = self.to_hidden(latent_attention)
        attention = attention.transpose(1, 2).reshape(B, T, C)
        return attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)
        
        output = self.w_o(self._attention(query, key, value))
        return output