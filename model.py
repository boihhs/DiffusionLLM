import torch
import torch.nn as nn
from timeEmbedding import TimeEmbedding
from mulitHeadAttention import MultiHeadBlock

class Network(nn.Module):
    def __init__(self, N: int, d: int = 384, num_blocks: int = 20):
        super().__init__()
        self.N = N
        self.d = d
        self.in_embedding = nn.Linear(N, d)
        self.timing = TimeEmbedding(d)
        self.blocks = nn.ModuleList([MultiHeadBlock(8, d) for _ in range(num_blocks)])
        self.out_embedding = nn.Linear(d, N)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xin, t):
        time = self.timing(t, xin)
        latent = self.in_embedding(xin)
        for block in self.blocks:
            residual = latent
            latent = 0.7 * block(latent + time)
            latent = latent + residual
        logits = self.out_embedding(latent)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, :, 0] = True
        return logits.masked_fill(mask, -1e9)
