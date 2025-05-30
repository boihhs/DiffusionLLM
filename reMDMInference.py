import torch
import torch.nn.functional as F
from model import Network
from dataload import CharTokenizer, chars_by_freq
from visulaizing import animate_diffusion_states
from BPETokenizer import tok

class Inference:
    def __init__(self, model_path, tokenizer, device='cuda'):
        self.device = device
        self.tokenizer = tokenizer
        self.vocab_size = 1022
        # Initialize and load the model
        self.model = Network(N=1022).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set to evaluation moder

    def denoise(
        self,
        text,
        T,
        eta_cap: float = 0.02,     # max-cap for σ_t
        top_p: float   = 0.99,      # nucleus threshold
        ton: float     = 0.55,     # loop “on” fraction
        toff: float    = 0.01,     # loop “off” fraction
    ):
        states_ids = []
        # 1) Tokenize & initial one-hot
        tokens = self.tokenizer.encode(text)                   # [L]
        ids    = torch.tensor(tokens, device=self.device)      # [L]
        xin    = F.one_hot(ids, num_classes=self.vocab_size).float().to(self.device)  # [L,K]

        L, K       = xin.shape
        mask_id    = self.tokenizer.mask_id

        first = True
        i = T
        while i > 0:
            # 2) compute “a” schedule

            a_t = 1 - i    / (T + 1)
            a_s = 1 - (i-1) / (T + 1)
         

            # 3) model prediction
            t_val   = torch.tensor([(i/T)], device=self.device)
            x_theta = F.softmax(self.model(xin.unsqueeze(0), t_val),dim=-1).squeeze(0)
            # (by SUBS, x_theta[:,0] == 0)

            # # 4) decide phase & σ_t
            if t_val > ton:
                # Phase 1: no remasking
                sigma = 0.0
                αt, αs = a_t, a_s

            elif t_val > toff and first:
                # Phase 2: loop with constant α(ton) and max-capped σ
                # approximate α(ton) by the “a” at step floor(ton*T)
                i0     = max(1, int(ton * T))
                α_loop = 1 - i0 / (T + 1)
                αt = αs = α_loop
                sigma = min(eta_cap, (1 - αs) / αt)


            else:
                if first:
                    i = int(ton*T)
                
                    a_t = 1 - i    / (T + 1)
                    a_s = 1 - (i-1) / (T + 1)
                    first = False   
                # Phase 3: back to MDLM
                sigma = 0.0
                αt, αs = a_t, a_s

            # 5) build the reverse‐cat logits v = [(1-σ)x_θ + σ m] ·scaled by (αs-αt)/(1-αt)
            mask_pos   = (ids == mask_id)
            v        = torch.zeros_like(xin)
            # for z_t ≠ [MASK], mass to x_theta

            v[~mask_pos, 1:]  = (1 - sigma) * xin[~mask_pos,1:]
            v[~mask_pos, 0]  = sigma
            # for [MASK] channel, mass = (1 - αs) + σ·(something)
            # here we inject remasking mass σ onto the mask channel:
            v[mask_pos, 1:]  = (αs - (1 - sigma)*αt)/(1-αt) * x_theta[mask_pos,1:]
            v[mask_pos, 0]  = (1-αs-sigma*αt)/(1-αt)

            # normalize
            probs    = v   # now ∑_k probs[l,k] = 1
            sorted, indeces = probs.sort(dim=-1, descending=True)
            cumsum = sorted.cumsum(dim=-1)
            cutoff = (cumsum > top_p).to(self.device)
            cutoff[..., 1:] = cutoff[..., :-1].clone() # Shift mask
            cutoff[..., 0]  = False
            sorted[cutoff] = 0.0
            row_sums = sorted.sum(dim=-1, keepdim=True)     
            row_sums = row_sums.clamp(min=1e-8)              
            sorted   = sorted / row_sums

            next_in_sorted = torch.multinomial(sorted, num_samples=1).squeeze(-1)  


            # 8) sample new IDs
            ids = indeces.gather(dim=-1, index=next_in_sorted.unsqueeze(-1)).squeeze(-1)
            xin             = F.one_hot(ids, num_classes=K).float().to(self.device)
            print("Percent of masks: ", (ids == 0).sum().item() / L)
            print((αs - (1 - sigma)*αt)/(1-αt))
            # (optional) debug print
            print(i, self.tokenizer.decode(ids.tolist()))
            states_ids.append(ids.tolist())

            i = i - 1

        return self.tokenizer.decode(ids.tolist()), states_ids

tokenizer = CharTokenizer(chars_by_freq)
testing = Inference('model_weights.pth', tok)

word = '<mask>'*10


tee, states_ids = testing.denoise(word, 1000)
print(tee)
animate_diffusion_states(states_ids, tok, interval=5, tokens_per_line=25, cell_w=0.1, cell_h= .2)