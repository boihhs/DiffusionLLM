
import sentencepiece as spm
import torch
import torch.nn.functional as F
# TinyStories training corpus
# corpus_path = "TinyStoriesV2-GPT4-train.txt"

# spm.SentencePieceTrainer.Train(
#     input=corpus_path,
#     model_prefix="bpe1k",
#     vocab_size=1024 - 3,      # leave room for mask / pad / unk
#     model_type="bpe",
#     character_coverage=1.0,   # keep all utf-8 chars
#     pad_id    = 1,            # we will shift later so mask == 0
#     unk_id    = 2,
#     bos_id    = -1, eos_id = -1,  # we don’t need BOS/EOS pieces
#     user_defined_symbols = []     # we’ll inject <mask> ourselves
# )



class BPETokenizer:
    def __init__(self, model_file, mask_token="<mask>"):
        self.sp        = spm.SentencePieceProcessor(model_file=model_file)
        self.mask_tok  = mask_token
        self.mask_id   = 0
        self._shift    = 1            # we shift every sp-id by +1

    # ---------- encode --------------------------------------------------
    def encode(self, text):
        tokens, i, m = [], 0, len(self.mask_tok)
        while i < len(text):
            if text[i : i+m] == self.mask_tok:         # literal "<mask>"
                tokens.append(self.mask_id)
                i += m
            else:
                # grab the longest slice up to the next "<mask>" (or EOS)
                j = text.find(self.mask_tok, i)
                slice_ = text[i:] if j == -1 else text[i:j]
                # full SentencePiece encode on the slice
                piece_ids = self.sp.encode(slice_, out_type=int)
                tokens.extend([pid + self._shift for pid in piece_ids])
                i += len(slice_)
        return tokens

    # ---------- decode --------------------------------------------------
    def decode(self, ids):
        out, buf = [], []
        for idx in ids:
            if idx == self.mask_id:
                if buf:                                 # flush buffered pieces
                    out.append(self.sp.decode([b - self._shift for b in buf]))
                    buf = []
                out.append(self.mask_tok)
            else:
                buf.append(idx)
        if buf:                                         # flush tail
            out.append(self.sp.decode([b - self._shift for b in buf]))
        return "".join(out)

    # ---------- properties ----------------------------------------------
    @property
    def pad_id(self):  return self.sp.pad_id() + self._shift   # 2
    @property
    def unk_id(self):  return self.sp.unk_id() + self._shift   # 3
    @property
    def vocab_size(self):      # +1 for the mask id
        return self.sp.get_piece_size() + self._shift



tok = BPETokenizer("bpe1k.model")
# print(tok.vocab_size)
# ids   = tok.encode("<mask>Once upon a time...")
# print(ids)

# text  = tok.decode(ids)
# print(text)
# print("mask id =", tok.mask_id, "pad id =", tok.pad_id)
