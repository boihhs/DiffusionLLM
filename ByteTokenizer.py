class ByteTokenizer:
    """Byte-level tokenizer with explicit <mask> handling."""
    def __init__(self, mask_token="<mask>", pad_token="<pad>"):
        self.mask_tok = mask_token
        self.pad_tok = pad_token
        self.mask_id = 0
        self.pad_id = 257  # 256 byte values shifted by +1 plus mask

    def encode(self, text: str):
        tokens = []
        i = 0
        m = len(self.mask_tok)
        while i < len(text):
            if text[i:i+m] == self.mask_tok:
                tokens.append(self.mask_id)
                i += m
            else:
                j = text.find(self.mask_tok, i)
                slice_ = text[i:] if j == -1 else text[i:j]
                for b in slice_.encode("utf-8"):
                    tokens.append(b + 1)
                i += len(slice_)
        return tokens

    def decode(self, ids):
        out = []
        buf = bytearray()
        for idx in ids:
            if idx == self.mask_id:
                if buf:
                    out.append(buf.decode("utf-8", errors="ignore"))
                    buf = bytearray()
                out.append(self.mask_tok)
            elif idx == self.pad_id:
                if buf:
                    out.append(buf.decode("utf-8", errors="ignore"))
                    buf = bytearray()
                # skip pad
            else:
                buf.append(idx - 1)
        if buf:
            out.append(buf.decode("utf-8", errors="ignore"))
        return "".join(out)

    @property
    def vocab_size(self):
        return self.pad_id + 1

# default instance used by training/inference
tok = ByteTokenizer()
