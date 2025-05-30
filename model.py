<<<<<<< HEAD
"""
Import is just xin which is (B, L, N) where B is batch size, L is lenght of tokens, N is total number choices for tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timeEmbedding import TimeEmbedding
from mulitHeadAttention import MultiHeadBlock

class Network(nn.Module):
    def __init__(self, N, d = 384, num_blocks=20):
        super(Network, self).__init__()
        
        self.d = d
        self.N = N
        self.in_embedding = nn.Linear(self.N, self.d)
        
        self.timing = TimeEmbedding(self.d)

        self.blocks      = nn.ModuleList(
            [ MultiHeadBlock(8, d) for _ in range(num_blocks) ]
        )
    
    
        self.out_embedding = nn.Linear(self.d, self.N)

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
            latent   = .7*block(latent + time)
            latent   = latent + residual
       
      
        logits = self.out_embedding(latent)              # [B,L,N]
        # build a mask for token-0
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, :, 0] = True
        # create a _new_ tensor, leaving `logits`’ history intact
        logits = logits.masked_fill(mask, -1e9)          # very negative but not inf
        return logits


# # 1) make your model
# model = Network(N=1022)  # example N

# # 2) total parameters (all weights & biases)
# total_params = sum(p.numel() for p in model.parameters())

# # 3) trainable parameters only
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total params: {total_params:,}")
# print(f"Trainable params: {trainable_params:,}")
=======
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
>>>>>>> 5ef3fe9be88c49d2ae7bd10b46754207eb21c73c
