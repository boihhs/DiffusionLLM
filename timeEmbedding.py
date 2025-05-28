'''
This is to get the time embedding for the diffusion process for the model
The input is the time t which is size B and the output is (B, L, d_in)
This gets added onto the input later on
'''


import math
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, d_in, max_period=10000, dropout_prob=0.1):
        super(TimeEmbedding, self).__init__()
        if d_in % 2 != 0:
            raise ValueError(f"Embedding dimension (D={d_in}) must be even.")
        self.d_in = d_in
        self.half = d_in // 2
        # frequencies are exp(-ln(max_period) * i / half)
        exponents = torch.arange(self.half, dtype=torch.float32)
        self.register_buffer(
            'omega', 
            torch.exp(-math.log(max_period) * exponents / self.half)
        )
        # optional small MLP to project embedding to d_in
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.SiLU(),
            nn.Linear(d_in, d_in)
        )

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, t, xin):
        """
        t: torch.Tensor of shape (B,) with values in [0,1]
        xin: torch.Tensor of shape (B, L, *)
        returns: torch.Tensor of shape (B, L, d_in)
        """
        B, L, _ = xin.shape
        # compute sinusoidal embeddings: (B, half)
        args = t.unsqueeze(1) * self.omega.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d_in)
        # project through MLP (optional but recommended)
        emb = self.dropout(self.mlp(emb))  # (B, d_in)

        # expand to match sequence length: (B, L, d_in)
        emb = emb.unsqueeze(1).expand(-1, L, -1)

        return emb




