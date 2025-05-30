<<<<<<< HEAD
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
        

    def forward(self, GT, logits, t, xin, eps: float = 1e-8):
        """
        GT     : (B, L, N) one-hot ground truth x
        logits : (B, L, N) raw model outputs
        t      : (B,)       noise levels in (0,1]
        xin    : (B, L, N)  one-hot z_t  (token 0 is the mask)
        """
        # 1) log-probabilities (stable)
        log_probs = F.log_softmax(logits, dim=-1)          # (B,L,N)

        # 2) log-p of the correct class
        log_p_true = (log_probs * GT).sum(dim=-1)          # (B,L)

        # 3) mask flag (1 where the token *was* masked)
        mask_flag = xin[:, :, 0].float()                   # (B,L)

        # 4) weight 1/t  for α_t = 1−t   (broadcast to (B,1))
        w = 1.0 / (t.clamp_min(eps)).unsqueeze(-1)         # (B,1)

        # 5) per-token loss, only on masked positions
        per_token = - w * mask_flag * log_p_true           # (B,L)

        # 6) **average over masked tokens**  (stabilises scale)
        denom = mask_flag.sum().clamp_min(1.0)
        return per_token.sum() / denom

=======
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

        





>>>>>>> 5ef3fe9be88c49d2ae7bd10b46754207eb21c73c
