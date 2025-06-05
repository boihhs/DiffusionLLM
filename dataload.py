import torch
from torch.utils.data import Dataset
from WordTokenizer import tok


tokenizer = tok


class TextDataset(Dataset):
    """Dataset reading text segments and encoding them with a tokenizer."""
    def __init__(self, file_path, tokenizer, max_tokens=100, delimiter="<|endoftext|>"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        raw = [s.strip() for s in text.split(delimiter) if s.strip()]
        self.samples   = raw
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.samples[idx])
        if len(ids) > self.max_tokens:
            ids = ids[:self.max_tokens]
        return torch.tensor(ids, dtype=torch.long)

def make_collate_fn(tok):
    """Create a collate_fn using the given tokenizer for padding."""
    def collate_fn(batch):
        lengths = [seq.size(0) for seq in batch]
        max_len = max(lengths)
        b = len(batch)
        padded = torch.full((b, max_len), tok.pad_id, dtype=torch.long)
        for i, seq in enumerate(batch):
            padded[i, :lengths[i]] = seq
        return padded
    return collate_fn

# backwards compatibility
collate_fn = make_collate_fn(tokenizer)
