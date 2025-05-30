<<<<<<< HEAD
import torch
from torch.utils.data import Dataset, DataLoader

class CharTokenizer:
    def __init__(self, chars_list, unk_token="<unk>", mask_token="<mask>", pad_token="<pad>"):
        self.chars      = list(chars_list)
        self.unk_token  = unk_token
        self.mask_token = mask_token
        self.pad_token  = pad_token

        # IDs:
        # mask_id = 0
        # chars    = 1..len(chars)
        # unk_id   = len(chars)+1
        # pad_id   = len(chars)+2
        self.mask_id = 0

        # build char→id
        self.char2id = { ch: i+1 for i, ch in enumerate(self.chars) }

        # add special tokens into char2id
        self.char2id[self.unk_token]  = len(self.chars) + 1
        self.char2id[self.mask_token] = self.mask_id
        self.pad_id                   = len(self.chars) + 2
        # (we usually don’t encode pad from text, so we don’t need to put pad_token→pad_id here)

        # build reverse map
        self.id2char = { i: ch for ch, i in self.char2id.items() }
        self.id2char[self.pad_id] = self.pad_token


    def encode(self, text):
        """
        Walk through `text`, pulling out either:
         - the literal "<mask>" → mask_id
         - any other single character → char2id[char] or unk_id
        """
        tokens = []
        i = 0
        L = len(text)
        M = len(self.mask_token)
        while i < L:
            # 1) if the next M characters are "<mask>", emit mask_id
            if text[i:i+M] == self.mask_token:
                tokens.append(self.mask_id)
                i += M
            else:
                # 2) otherwise, one character at a time
                ch = text[i]
                tokens.append(self.char2id.get(ch, self.char2id[self.unk_token]))
                i += 1

        return tokens

    def decode(self, ids):
        """
        Turn IDs back into a string.
        """
        return ''.join(self.id2char.get(i, "") for i in ids)

    @property
    def vocab_size(self):
        # highest ID is pad_id, so vocab size = pad_id + 1
        return self.pad_id + 1


# ─── build your tokenizer ───────────────────────────────────────────────────
chars_by_freq = [
    ' ', 'e', 'a', 't', 'o', 'h', 'n', 'd', 'i', 's', 'r', 'l', 'y', 'm',
    '.', 'w', 'u', 'p', 'g', 'c', 'f', 'b', ',', 'k', 'T', '\n', '"', 'v',
    'S', 'H', 'x', '|', 'I', 'O', 'L', "'", '!', 'B', 'M', '<', '>', 'A',
    'W', 'j', '?', 'Y', 'z', 'J', 'F', 'D', 'C', 'q', 'N', 'E', 'K', 'P',
    'G', 'R', '-', '”', '“', ':', '’', 'Z', 'V', 'U', '3', ';', 'Q', '–',
    '1', 'X', '—', '0', '2', '‘', '5', '4', '…', '/', '9', '`', '8', '6',
    '7'
]
tokenizer = CharTokenizer(chars_by_freq)

# print("mask_id:", tokenizer.mask_id)
# print("unk_id: ", tokenizer.unk_id)
# print("pad_id:", tokenizer.pad_id)
# print("vocab_size:", tokenizer.vocab_size)

# ─── Dataset and collate_fn ────────────────────────────────────────────────
class CharDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_chars, delimiter="<|endoftext|>"):
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

def collate_fn(batch):
    lengths = [seq.size(0) for seq in batch]
    max_len = max(lengths)
    B       = len(batch)

    # pad with pad_id (not 0)
    padded = torch.full((B, max_len),
                        tokenizer.pad_id,
                        dtype=torch.long)
    

    for i, seq in enumerate(batch):
        l = lengths[i]
        padded[i, :l] = seq

    return padded


=======
import torch
from torch.utils.data import Dataset, DataLoader

class CharTokenizer:
    def __init__(self, chars_list, unk_token="<unk>", mask_token="<mask>", pad_token="<pad>"):
        # 1) build the base vocab: chars + unk
        self.chars      = list(chars_list)
        self.unk_token  = unk_token
        self.mask_token = mask_token
        self.pad_token  = pad_token

        # IDs:
        # mask_id = 0
        # char IDs = 1..len(chars)
        # unk_id   = len(chars)+1
        # pad_id   = len(chars)+2

        # mask
        self.mask_id = 0

        # chars → 1..len(chars)
        self.char2id = {ch: i+1 for i, ch in enumerate(self.chars)}

        # unk
        self.unk_id = len(self.chars) + 1
        self.char2id[self.unk_token] = self.unk_id

        # pad
        self.pad_id = len(self.chars) + 2
        # we don't include pad in char2id since we won't encode it from text

        # build reverse map for decoding
        self.id2char = {i: ch for ch, i in self.char2id.items()}
        # you could also add:
        self.id2char[self.mask_id] = self.mask_token
        self.id2char[self.pad_id]  = self.pad_token

    def encode(self, text):
        """
        Map each character → its ID.
        Unknown chars → unk_id.
        Mask token not inserted here (used later if needed).
        """
        return [ self.char2id.get(ch, self.unk_id) for ch in text ]

    def decode(self, ids):
        """
        Turn IDs back into a string.
        """
        return ''.join(self.id2char.get(i, "") for i in ids)

    @property
    def vocab_size(self):
        # highest ID is pad_id, so vocab size = pad_id + 1
        return self.pad_id + 1


# ─── build your tokenizer ───────────────────────────────────────────────────
chars_by_freq = [
    ' ', 'e', 'a', 't', 'o', 'h', 'n', 'd', 'i', 's', 'r', 'l', 'y', 'm',
    '.', 'w', 'u', 'p', 'g', 'c', 'f', 'b', ',', 'k', 'T', '\n', '"', 'v',
    'S', 'H', 'x', '|', 'I', 'O', 'L', "'", '!', 'B', 'M', '<', '>', 'A',
    'W', 'j', '?', 'Y', 'z', 'J', 'F', 'D', 'C', 'q', 'N', 'E', 'K', 'P',
    'G', 'R', '-', '”', '“', ':', '’', 'Z', 'V', 'U', '3', ';', 'Q', '–',
    '1', 'X', '—', '0', '2', '‘', '5', '4', '…', '/', '9', '`', '8', '6',
    '7'
]
tokenizer = CharTokenizer(chars_by_freq)

# print("mask_id:", tokenizer.mask_id)
# print("unk_id: ", tokenizer.unk_id)
# print("pad_id:", tokenizer.pad_id)
# print("vocab_size:", tokenizer.vocab_size)

# ─── Dataset and collate_fn ────────────────────────────────────────────────
class CharDataset(Dataset):
    def __init__(self, file_path, tokenizer, delimiter="<|endoftext|>"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        raw = [s.strip() for s in text.split(delimiter) if s.strip()]
        self.samples   = raw
        self.tokenizer = tokenizer
        self.max_chars = 100

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.samples[idx])
        if len(ids) > self.max_chars:
            ids = ids[:self.max_chars]
        return torch.tensor(ids, dtype=torch.long)

def collate_fn(batch):
    lengths = [seq.size(0) for seq in batch]
    max_len = max(lengths)
    B       = len(batch)

    # pad with pad_id (not 0)
    padded = torch.full((B, max_len),
                        tokenizer.pad_id,
                        dtype=torch.long)
    

    for i, seq in enumerate(batch):
        l = lengths[i]
        padded[i, :l] = seq

    return padded


>>>>>>> 5ef3fe9be88c49d2ae7bd10b46754207eb21c73c
