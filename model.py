"""
Import is just xin which is (B, L, N) where B is batch size, L is lenght of tokens, N is total number choices for tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timeEmbedding import TimeEmbedding
from mulitHeadAttention import MultiHeadBlock

class Network(nn.Module):
    def __init__(self, N, d = 256):
        super(Network, self).__init__()
        
        self.d = d
        self.N = N
        self.embedding = nn.Linear(self.N, self.d, bias=False)
        
        self.timing = TimeEmbedding(self.d)

        self.MHB_1 = MultiHeadBlock(8, self.d)
        self.MHB_2 = MultiHeadBlock(8, self.d)
        self.MHB_3 = MultiHeadBlock(8, self.d)
        self.MHB_4 = MultiHeadBlock(8, self.d)
        self.MHB_5 = MultiHeadBlock(8, self.d)
        self.MHB_6 = MultiHeadBlock(8, self.d)
        self.MHB_7 = MultiHeadBlock(8, self.d)


        

    def forward(self, xin, t):
        time = self.timing(t, xin)
        latent = self.embedding(xin)
        latent = self.MHB_1(latent + time) + latent
        latent = self.MHB_2(latent + time) + latent
        latent = self.MHB_3(latent + time) + latent
        latent = self.MHB_4(latent + time) + latent
        latent = self.MHB_5(latent + time) + latent
        latent = self.MHB_6(latent + time) + latent
        latent = self.MHB_7(latent + time) + latent
        out = F.linear(latent, self.embedding.weight.t())
        out[:, :, 0] = float('-inf')
        out = torch.softmax(out, dim=-1)

        return out

# 1) make your model
# model = model(N=100)  # example N

# # 2) total parameters (all weights & biases)
# total_params = sum(p.numel() for p in model.parameters())

# # 3) trainable parameters only
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total params: {total_params:,}")
# print(f"Trainable params: {trainable_params:,}")
