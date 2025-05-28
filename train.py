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