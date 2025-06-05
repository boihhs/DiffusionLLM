import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from dataload import CharDataset, collate_fn, tokenizer
from BPETokenizer import tok
from model import Network
from DMDMLoss import DMDMLoss

vocab_size = tok.vocab_size
batch_size = 32

dataset = CharDataset("TinyStoriesV2-GPT4-train.txt", tok, max_chars=100)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Network(N=vocab_size).to(device)
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

opt = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
steps_per_epoch = len(loader)
warmup_steps = int(0.1 * steps_per_epoch)
warmup_sched = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
cosine_sched = CosineAnnealingLR(opt, T_max=max(1, steps_per_epoch - warmup_steps), eta_min=1e-7)

loss_fn = DMDMLoss()
global_step = 0
for epoch in range(1, 3):
    epoch_losses = []
    for batch in loader:
        batch = batch.to(device)
        B, L = batch.shape
        interval = 1.0 / float(B)
        u = torch.rand(B, device=device) * interval
        i = torch.arange(B, device=device, dtype=torch.float)
        t = (i * interval + u).clamp_min(1e-3)
        a_t = (1 - t).view(B, 1, 1)
        GT = F.one_hot(batch, num_classes=vocab_size).float()
        probs = GT * a_t
        probs[..., 0] = 1 - a_t.squeeze(-1)
        flat = probs.view(-1, vocab_size)
        idx = torch.multinomial(flat, num_samples=1).view(B, L)
        xin = F.one_hot(idx, num_classes=vocab_size).float()
        logits = model(xin, t)
        loss = loss_fn(GT, logits, t, xin)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if global_step < warmup_steps:
            warmup_sched.step()
        else:
            cosine_sched.step()
        if global_step % 100 == 0:
            print(f"Step {global_step:5d} | loss {loss.item():.4f}")
        global_step += 1
        epoch_losses.append(loss.item())
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch:2d} | avg loss {avg_loss:.4f}")
    torch.save(model.state_dict(), 'model_weights.pth')
