from model import Network
from dataload import CharTokenizer, chars_by_freq
import torch
import torch.nn.functional as F
from visulaizing import animate_diffusion_states
from BPETokenizer import tok

import torch, torch.nn.functional as F
from model import Network
import re, torch, torch.nn.functional as F


class Inference:
    def __init__(self, model_path, tokenizer, device='cuda'):
        self.device      = torch.device(device)
        self.tokenizer   = tokenizer
        self.vocab_size  = tokenizer.vocab_size        # use real size
        self.model       = Network(N=self.vocab_size).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def denoise(
        self,
        text: str,
        T: int,
        freeze_words: int = 0,        # how many leading words to freeze
    ):
        """
        Reverse-diffusion sampler.

        • first `freeze_words` space-separated words are immutable
        • no σ-loop; tiny remask_eps keeps later correction possible
        """
        # ---------- prepare ids ---------------------------------------
        ids = torch.tensor(self.tokenizer.encode(text),
                           device=self.device, dtype=torch.long)

        # ---- compute how many *tokens* belong to the first N words ---
        freeze_n = 0
        if freeze_words > 0:
            words = re.split(r'\s+', text, maxsplit=freeze_words)
            prefix = " ".join(words[:freeze_words])
            freeze_n = len(self.tokenizer.encode(prefix))

        freeze_n = min(freeze_n, ids.numel())
        freeze_pos = torch.zeros_like(ids, dtype=torch.bool)
        if freeze_n > 0:
            freeze_pos[:freeze_n] = True               # hard-frozen rows

        xin = F.one_hot(ids, num_classes=self.vocab_size).float()

        # hyper-params
        remask_eps   = 0.02
        tau_start, tau_end = 1.2, 0.9
        top_p_start, top_p_end = 0.95, 0.98
        rep_penalty  = 0.6
        min_top_k    = 5
        history, max_hist = [], 32
        mask_id      = self.tokenizer.mask_id
        eps          = 1e-9
        states_ids   = []

        # ---------- reverse diffusion loop ----------------------------
        for i in range(T, 0, -1):
            a_t = 1 -  i    / (T + 1)
            a_s = 1 - (i-1) / (T + 1)

            t_val = torch.tensor([i / T], device=self.device)
            x_theta = F.softmax(self.model(xin.unsqueeze(0), t_val), dim=-1
                               ).squeeze(0)

            # cat probs
            v = torch.zeros_like(xin)
            v[:, 1:] = (a_s - a_t) * x_theta[:, 1:]
            v[:, 0]  = 1.0 - a_s
            probs = (v + eps) / (1 - a_t + eps)

            # finished rows: leave 1-remask_eps on current id
            mask_pos   = (ids == mask_id)
            unmask_pos = ~mask_pos
            if unmask_pos.any():
                probs[unmask_pos] *= (1 - remask_eps)
                probs[unmask_pos, mask_id] = remask_eps
                probs[unmask_pos, ids[unmask_pos]] += (1 - remask_eps)

            # *hard* freeze first N tokens
            if freeze_pos.any():
                probs[freeze_pos] = 0.0
                probs[freeze_pos, ids[freeze_pos]] = 1.0

            # sample only genuinely masked & unfrozen rows
            sample_rows = mask_pos & (~freeze_pos)
            if sample_rows.any():
                prog  = 1 - i / T
                tau   = tau_start  + (tau_end   - tau_start) * prog
                p_cut = top_p_start + (top_p_end - top_p_start) * prog

                logits = probs[sample_rows].log() / tau

                # repetition penalty
                if history:
                    hist_ids = torch.tensor(history, device=self.device)
                    logits[:, hist_ids] *= rep_penalty

                # nucleus + ensure at least min_top_k choices
                sorted, idx = logits.sort(-1, descending=True)
                cdf = sorted.softmax(-1).cumsum(-1)
                cut = cdf > p_cut
                cut[..., :min_top_k] = False
                sorted = sorted.masked_fill(cut, -float('inf'))
                logits = sorted.scatter(-1, idx, sorted)

                new_ids = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
                ids[sample_rows] = new_ids
                history.extend(new_ids.tolist())
                if len(history) > max_hist:
                    history = history[-max_hist:]

            # refresh
            xin = F.one_hot(ids, num_classes=self.vocab_size).float()
            states_ids.append(ids.tolist())
            print(self.tokenizer.decode(ids.tolist()))

        return self.tokenizer.decode(ids.tolist()), states_ids



tokenizer = CharTokenizer(chars_by_freq)


sampler = Inference("model_weights.pth", tok)

prompt = "Once upon a time " + "<mask>"*20
text, trace = sampler.denoise(prompt, T=100, freeze_words=4)
print(text)
 




            


    