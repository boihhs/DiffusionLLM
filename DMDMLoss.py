"""
Loss used for the discrete mask diffusion model.
GT and xin are one-hot tensors of shape (B, L, N).
logits are raw model outputs of the same shape.
t contains the noise level for each sample in (0,1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DMDMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, GT, logits, t, xin, eps: float = 1e-8):
        log_probs = F.log_softmax(logits, dim=-1)
        log_p_true = (log_probs * GT).sum(dim=-1)
        mask_flag = xin[:, :, 0].float()
        w = 1.0 / (t.clamp_min(eps)).unsqueeze(-1)
        per_token = -w * mask_flag * log_p_true
        denom = mask_flag.sum().clamp_min(1.0)
        return per_token.sum() / denom
