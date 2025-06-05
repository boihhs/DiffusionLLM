import torch
import torch.nn.functional as F
from model import Network
from ByteTokenizer import tok

class Inference:
    def __init__(self, model_path, tokenizer, device='cuda'):
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.model = Network(N=self.vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def denoise(self, text, T, eta_cap=0.02, top_p=0.99, ton=0.55, toff=0.01):
        states_ids = []
        ids = torch.tensor(self.tokenizer.encode(text), device=self.device)
        xin = F.one_hot(ids, num_classes=self.vocab_size).float()
        L, K = xin.shape
        mask_id = self.tokenizer.mask_id
        first = True
        i = T
        while i > 0:
            a_t = 1 - i / (T + 1)
            a_s = 1 - (i - 1) / (T + 1)
            t_val = torch.tensor([i / T], device=self.device)
            x_theta = F.softmax(self.model(xin.unsqueeze(0), t_val), dim=-1).squeeze(0)
            if t_val > ton:
                sigma = 0.0
                alpha_t, alpha_s = a_t, a_s
            elif t_val > toff and first:
                i0 = max(1, int(ton * T))
                alpha_loop = 1 - i0 / (T + 1)
                alpha_t = alpha_s = alpha_loop
                sigma = min(eta_cap, (1 - alpha_s) / alpha_t)
            else:
                if first:
                    i = int(ton * T)
                    a_t = 1 - i / (T + 1)
                    a_s = 1 - (i - 1) / (T + 1)
                    first = False
                sigma = 0.0
                alpha_t, alpha_s = a_t, a_s
            mask_pos = ids == mask_id
            v = torch.zeros_like(xin)
            v[~mask_pos, 1:] = (1 - sigma) * xin[~mask_pos, 1:]
            v[~mask_pos, 0] = sigma
            v[mask_pos, 1:] = (alpha_s - (1 - sigma) * alpha_t)/(1 - alpha_t) * x_theta[mask_pos, 1:]
            v[mask_pos, 0] = (1 - alpha_s - sigma * alpha_t)/(1 - alpha_t)
            probs = v
            sorted_vals, indices = probs.sort(dim=-1, descending=True)
            cumsum = sorted_vals.cumsum(dim=-1)
            cutoff = (cumsum > top_p).to(self.device)
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_vals[cutoff] = 0.0
            row_sums = sorted_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            sorted_vals = sorted_vals / row_sums
            next_in_sorted = torch.multinomial(sorted_vals, num_samples=1).squeeze(-1)
            ids = indices.gather(dim=-1, index=next_in_sorted.unsqueeze(-1)).squeeze(-1)
            xin = F.one_hot(ids, num_classes=K).float()
            states_ids.append(ids.tolist())
            i -= 1
        return self.tokenizer.decode(ids.tolist()), states_ids

tokenizer = tok
