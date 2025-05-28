"""
This is the loss for the Discrete Mask Diffustion Model (DMDM).
It takes in the ground truth token values (B, L, N) where B is batch size and L is sequence lenght and N is the number of tokens(these are one hot),
it also takes in the predicted token values which is also size (B, L, N) but this is contionus and is from the model,
lastly it also takes in the time t which is size of B which values varies from [0, 1].

Of course the output is a single value which is the mean of the losses of all the values
"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class DMDMLoss(nn.Module):

    def __init__(self):
        super(DMDMLoss, self).__init__()
        

    def forward(self, GT, P, t, xin):
        """
        GT:   [B, L, N] one-hot ground truth
        P:    [B, L, N] predicted probabilities
        t:    [B]       noise levels
        xin:  [B, L, N] one-hot inputs (mask token is index 0)
        """

        # 1) extract only the log-prob of the true class
        #    (P * GT).sum(-1) is [B, L], log it safely:
        true_prob = (P * GT).sum(dim=2).clamp(min=1e-8)
        log_p     = torch.log(true_prob)               # [B, L]

        # 2) build a mask of where xin == mask_token (class 0):
        mask_flag = xin[:, :, 0]                        # [B, L], 1.0 where masked, 0.0 else

        # 3) sum log-probs only over masked positions:
        sum_log   = (mask_flag * log_p).sum(dim=1)      # [B]

        # 4) weight by your factor and take **mean over batch**:
        #    per-sample loss = - sum_log / (t_i)
        loss_per_sample = - sum_log / (t).clamp(min=1e-8)  # [B]

        # 5) final scalar:
        return loss_per_sample.mean()

        





