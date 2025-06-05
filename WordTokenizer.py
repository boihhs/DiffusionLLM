import re
from pathlib import Path

class WordTokenizer:
    """Simple whitespace/punctuation tokenizer with special tokens."""
    def __init__(self, vocab_file: str, mask_token="<mask>", pad_token="<pad>", unk_token="<unk>"):
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]

        self.mask_id = 0
        self.word2id = {w: i+1 for i, w in enumerate(vocab)}
        self.unk_id = len(vocab) + 1
        self.word2id[self.unk_token] = self.unk_id
        self.word2id[self.mask_token] = self.mask_id
        self.pad_id = self.unk_id + 1
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.id2word[self.pad_id] = self.pad_token

    def tokenize(self, text: str):
        return re.findall(r'<mask>|\w+|[^\w\s]', text)

    def encode(self, text: str):
        ids = []
        for tok in self.tokenize(text):
            if tok == self.mask_token:
                ids.append(self.mask_id)
            else:
                ids.append(self.word2id.get(tok, self.word2id[self.unk_token]))
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i == self.pad_id:
                continue
            tokens.append(self.id2word.get(i, self.unk_token))
        return " ".join(tokens)

    @property
    def vocab_size(self):
        return self.pad_id + 1


def load_default_tokenizer():
    vocab_file = Path(__file__).parent / "word_vocab.txt"
    return WordTokenizer(str(vocab_file))

tok = load_default_tokenizer()
