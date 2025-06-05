import sys
import re
from collections import Counter

if len(sys.argv) < 3:
    print("Usage: python build_vocab.py <text_file> <vocab_out>")
    sys.exit(1)

text_path = sys.argv[1]
vocab_path = sys.argv[2]

with open(text_path, 'r', encoding='utf-8') as f:
    text = f.read()

# simple tokenizer: words and punctuation, with <mask> preserved
tokens = re.findall(r'<mask>|\w+|[^\w\s]', text)
counts = Counter(tokens)
# sort by frequency then alphabetically for determinism
sorted_tokens = [tok for tok, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

with open(vocab_path, 'w', encoding='utf-8') as f:
    for tok in sorted_tokens:
        f.write(tok + "\n")
