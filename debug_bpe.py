#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cs336_basics.tokenizer import train_bpe

# 测试训练
input_path = "tests/fixtures/corpus.en"
vocab, merges = train_bpe(
    input_path=input_path,
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)

print("Generated merges:")
for i, merge in enumerate(merges[:50]):  # 只打印前50个
    print(f"{i}: {merge}")

print(f"\nTotal merges: {len(merges)}")
print(f"Vocab size: {len(vocab)}") 