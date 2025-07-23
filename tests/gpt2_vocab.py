from common import FIXTURES_PATH, gpt2_bytes_to_unicode
import json

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"



gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
with open(VOCAB_PATH) as vocab_f:
    gpt2_vocab = json.load(vocab_f)
gpt2_bpe_merges = []
with open(MERGES_PATH) as f:
    for line in f:
        cleaned_line = line.rstrip()
        if cleaned_line and len(cleaned_line.split(" ")) == 2:
            gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
# The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
# just return the original bytes, so we don't force students to use
# any particular encoding scheme.
vocab = {
    gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
    for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
}

merges = [
    (
        bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
        bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
    )
    for merge_token_1, merge_token_2 in gpt2_bpe_merges
]

vocab_output = [f"{i}: {v}" for i, v in vocab.items()]
with open("vocab_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(vocab_output))

# merges 是 list[tuple[bytes, bytes]]
merges_output = [f"{a} + {b}" for a, b in merges]
with open("merges_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(merges_output))