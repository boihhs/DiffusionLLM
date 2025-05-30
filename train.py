<<<<<<< HEAD
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from dataload import CharDataset
from dataload import tokenizer
from BPETokenizer import tok

from dataload import DataLoader
from dataload import collate_fn
from model import Network
from DMDMLoss import DMDMLoss


# … your CharDataset, collate_fn, DMDMLoss, Network, tokenizer …
vocab_size = 1022
batch_size = 100
dataset    = CharDataset("TinyStoriesV2-GPT4-train.txt", tok, 100)
loader     = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model   = Network(N=vocab_size).to(device)
total_params = sum(p.numel() for p in model.parameters())



print(f"Total params: {total_params:,}")
# model.load_state_dict(torch.load('model_weights.pth'))

# --- 1.  optimiser -----------------------------------------------------------
base_lr = 7e-6                                       # 🔧 lower
opt      = torch.optim.AdamW(model.parameters(),
                             lr=base_lr,
                             weight_decay=1e-2)

# --- 2.  scheduling parameters ----------------------------------------------
steps_per_epoch = len(loader)                        # e.g. 10 000
warmup_frac     = 0.10                               # 🔧 10 %
warmup_steps    = int(warmup_frac * steps_per_epoch)

# --- 3.  warm-up scheduler: 10 % of steps ------------------------------------
warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=0.10,                           # 0.1 × base_lr
        end_factor=1.00,                             # reach base_lr
        total_iters=warmup_steps
)

# --- 4.  cosine decay for the remaining 90 % ---------------------------------
cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max = steps_per_epoch - warmup_steps,      # whole rest of epoch
        eta_min = 1e-7                               # final LR 1 e-6
)

loss_fn = DMDMLoss()
global_step = 0
ACCUM_STEPS = 1

for epoch in range(1, 3):
    epoch_losses = []
    for batch in loader:
        batch    = batch.to(device)         # [B, L]
        B, L     = batch.shape

        # 1) sample noise level
        Bf = float(B)
        interval = 1.0 / Bf

        # 1) a little jitter within each sub-interval
        u = torch.rand(B, device=device) * interval    # [0, 1/B)

        # 2) base offsets 0/B, 1/B, 2/B, …, (B-1)/B
        i = torch.arange(B, device=device, dtype=torch.float)

        # 3) final t values in [i/B, (i+1)/B)
        t = i * interval + u                           # shape [B]
        t = t.clamp_min(1e-3)
        a_t = (1 - t).view(B,1,1)           # [B,1,1]

        # 2) one-hot GT
        GT    = F.one_hot(batch, num_classes=vocab_size).float()  # [B,L,88]

        # 3) build mixed-noise distribution
        probs = GT * a_t
        probs[..., 0] = 1 - a_t.squeeze(-1)               # mask token

        # 4) sample categorically
        flat = probs.view(-1, vocab_size)                         # [B*L,88]
        idx  = torch.multinomial(flat, num_samples=1)     # [B*L,1]
        idx  = idx.view(B, L)                             # [B,L]
        xin  = F.one_hot(idx, num_classes=vocab_size).float()     # [B,L,88]

        # 5) forward + loss
        logits = model(xin, t)
        raw_loss = loss_fn(GT, logits, t, xin)

        loss = raw_loss / ACCUM_STEPS
        epoch_losses.append(raw_loss.item())

        # 2) backward every mini‐batch
        
        loss.backward()
     
    


        # 3) only step & clip once every ACCUM_STEPS
        if (global_step + 1) % ACCUM_STEPS == 0:
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            # optimizer step
            opt.step()
            # warm-up scheduler step
            if global_step < warmup_steps:
                warmup_sched.step()
            else:
                cosine_sched.step()
            # zero grads for next accumulation block
            opt.zero_grad()

        # (optional) logging on raw loss
        if global_step % 100 == 0:
            print(f"Step {global_step:5d} | loss {raw_loss.item():.1f}")

        global_step += 1

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch:>2} | avg loss: {avg_loss:0.1f} | LR: {opt.param_groups[0]['lr']:.2E}")
    print(epoch_losses)
    # Save
    torch.save(model.state_dict(), 'model_weights.pth')


=======
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from dataload import CharDataset
from dataload import tokenizer
from dataload import DataLoader
from dataload import collate_fn
from model import Network
from DMDMLoss import DMDMLoss


# … your CharDataset, collate_fn, DMDMLoss, Network, tokenizer …

batch_size = 512
dataset    = CharDataset("TinyStoriesV2-GPT4-train.txt", tokenizer)
loader     = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = Network(N=88).to(device)

base_lr = 5e-5
opt     = AdamW(model.parameters(),
                lr=base_lr,
                weight_decay=1e-2)

# ───  Warm-up scheduler  ────────────────────────────────────────────────────────
# Ramp linearly from 10% → 100% of base_lr over warmup_steps
warmup_epochs = 1
steps_per_epoch = len(loader)
print(steps_per_epoch)
warmup_steps    = warmup_epochs * steps_per_epoch

warmup_scheduler = LinearLR(
    opt,
    start_factor=0.1,     # start at 0.1 * base_lr
    end_factor=1.0,       # end at 1.0 * base_lr
    total_iters=warmup_steps
)

# ───  Plateau scheduler  ───────────────────────────────────────────────────────
plateau_scheduler = ReduceLROnPlateau(
    opt,
    mode="min",
    factor=0.5,
    patience=3,
    verbose=True
)

loss_fn = DMDMLoss()
global_step = 0

for epoch in range(1, 11):
    epoch_losses = []
    for batch in loader:
        batch    = batch.to(device)         # [B, L]
        B, L     = batch.shape

        # 1) sample noise level
        t   = torch.rand(B, device=device)  # [B]
        a_t = (1 - t).view(B,1,1)           # [B,1,1]

        # 2) one-hot GT
        GT    = F.one_hot(batch, num_classes=88).float()  # [B,L,88]

        # 3) build mixed-noise distribution
        probs = GT * a_t
        probs[..., 0] = 1 - a_t.squeeze(-1)               # mask token

        # 4) sample categorically
        flat = probs.view(-1, 88)                         # [B*L,88]
        idx  = torch.multinomial(flat, num_samples=1)     # [B*L,1]
        idx  = idx.view(B, L)                             # [B,L]
        xin  = F.one_hot(idx, num_classes=88).float()     # [B,L,88]

        # 5) forward + loss
        P = model(xin, t)
        l = loss_fn(GT, P, t, xin)
        if (global_step % 100 == 0):
            print(l)
        epoch_losses.append(l.item())

        # 6) backward + clip + step
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # 7) warm-up step (only during the first warmup_steps)
        if global_step < warmup_steps:
            warmup_scheduler.step()
        global_step += 1

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch:>2} | avg loss: {avg_loss:0.1f} | LR: {opt.param_groups[0]['lr']:.2E}")

    # once warm-up is done, let the plateau scheduler take over
    if global_step >= warmup_steps:
        plateau_scheduler.step(avg_loss)
>>>>>>> 5ef3fe9be88c49d2ae7bd10b46754207eb21c73c
