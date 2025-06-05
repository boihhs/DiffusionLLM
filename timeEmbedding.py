"""Sinusoidal time embedding used for diffusion steps."""
import math
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, d_in: int, max_period: int = 10000, dropout_prob: float = 0.1):
        super().__init__()
        if d_in % 2 != 0:
            raise ValueError("Embedding dimension must be even")
        self.half = d_in // 2
        exponents = torch.arange(self.half, dtype=torch.float32)
        self.register_buffer('omega', torch.exp(-math.log(max_period) * exponents / self.half))
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.SiLU(),
            nn.Linear(d_in, d_in),
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, t, xin):
        B, L, _ = xin.shape
        args = t.unsqueeze(1) * self.omega.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        emb = self.dropout(self.mlp(emb))
        return emb.unsqueeze(1).expand(-1, L, -1)
