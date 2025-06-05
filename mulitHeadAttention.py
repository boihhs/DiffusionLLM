"""Multi-head attention block used by the model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHead(nn.Module):
    def __init__(self, heads: int, d_in: int, dropout_prob: float = 0.1):
        super().__init__()
        assert d_in % heads == 0, "d_in must be divisible by number of heads"
        self.h = heads
        self.d_in = d_in
        self.d_h = d_in // heads
        self.W_QKV = nn.Linear(d_in, d_in * 3, bias=False)
        self.W_O = nn.Linear(d_in, d_in, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, xin):
        b, L, _ = xin.shape
        QKV = self.W_QKV(xin).reshape(b, 3, self.h, L, self.d_h)
        Q, K, V = QKV[:,0], QKV[:,1], QKV[:,2]
        scores = (Q @ K.transpose(-2,-1)) * (self.d_h ** -0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ V
        out = out.transpose(1,2).reshape(b, L, self.d_in)
        return self.W_O(out)

class MultiHeadBlock(nn.Module):
    def __init__(self, heads: int, d_in: int, dropout_prob: float = 0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_in)
        self.layer_norm2 = nn.LayerNorm(d_in)
        self.mha = MultiHead(heads, d_in, dropout_prob)
        self.fc1 = nn.Linear(d_in, d_in*4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_in*4, d_in)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, xin):
        xin = self.mha(self.layer_norm1(xin)) + xin
        xin = self.dropout(self.fc2(self.gelu(self.fc1(self.layer_norm2(xin))))) + xin
        return xin
