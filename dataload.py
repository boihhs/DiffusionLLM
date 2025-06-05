import torch
from torch.utils.data import Dataset, DataLoader

class CharTokenizer:
    def __init__(self, chars_list, unk_token="<unk>", mask_token="<mask>", pad_token="<pad>"):
        self.chars      = list(chars_list)
        self.unk_token  = unk_token
        self.mask_token = mask_token
        self.pad_token  = pad_token

        self.mask_id = 0
        self.char2id = {ch: i+1 for i, ch in enumerate(self.chars)}
        self.char2id[self.unk_token]  = len(self.chars) + 1
        self.char2id[self.mask_token] = self.mask_id
        self.pad_id = len(self.chars) + 2

        self.id2char = {i: ch for ch, i in self.char2id.items()}
        self.id2char[self.pad_id] = self.pad_token

    def encode(self, text: str):
        tokens = []
        i = 0
        m = len(self.mask_token)
        while i < len(text):
            if text[i:i+m] == self.mask_token:
                tokens.append(self.mask_id)
                i += m
            else:
                ch = text[i]
                tokens.append(self.char2id.get(ch, self.char2id[self.unk_token]))
                i += 1
        return tokens

    def decode(self, ids):
        return ''.join(self.id2char.get(i, '') for i in ids)

    @property
    def vocab_size(self):
        return self.pad_id + 1


chars_by_freq = [
    ' ', 'e', 'a', 't', 'o', 'h', 'n', 'd', 'i', 's', 'r', 'l', 'y', 'm',
    '.', 'w', 'u', 'p', 'g', 'c', 'f', 'b', ',', 'k', 'T', '\n', '"', 'v',
    'S', 'H', 'x', '|', 'I', 'O', 'L', "'", '!', 'B', 'M', '<', '>', 'A',
    'W', 'j', '?', 'Y', 'z', 'J', 'F', 'D', 'C', 'q', 'N', 'E', 'K', 'P',
    'G', 'R', '-', '”', '“', ':', '’', 'Z', 'V', 'U', '3', ';', 'Q', '–',
    '1', 'X', '—', '0', '2', '‘', '5', '4', '…', '/', '9', '`', '8', '6',
    '7'
]

# default tokenizer instance (character level)
tokenizer = CharTokenizer(chars_by_freq)


class CharDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_chars=100, delimiter="<|endoftext|>"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        raw = [s.strip() for s in text.split(delimiter) if s.strip()]
        self.samples   = raw
        self.tokenizer = tokenizer
        self.max_chars = max_chars

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.samples[idx])
        if len(ids) > self.max_chars:
            ids = ids[:self.max_chars]
        return torch.tensor(ids, dtype=torch.long)

def make_collate_fn(tok: CharTokenizer):
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
