import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

text1 = "\n\n"
text2 = "\n\ntexting!"

assert re.findall(PAT, text1) == ["\n\n"]
assert re.findall(PAT, text2) == ["\n", "\n", "texting", "!"]

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    # 构建正则匹配所有特殊符号（如 <|endoftext|>）
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    return re.split(f"({pattern})", text)

text1 = "<|endoftext|>\n\n"
text2 = "<|endoftext|>\n\ntexting!"

assert split_by_special_tokens(text1, ["<|endoftext|>"]) == ["", "<|endoftext|>", "\n\n"]
assert split_by_special_tokens(text2, ["<|endoftext|>"]) == ["", "<|endoftext|>", "\n\ntexting!"]


def split_by_special_tokens_2(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]

    # 按长度降序排列，保证正则优先匹配最长
    pattern = "|".join(sorted(map(re.escape, special_tokens), key=len, reverse=True))
    return re.split(f"({pattern})", text)

text3 = "<|endoftext|><|endoftext|><|endoftext|>"
text4 = "<|endoftext|><|endoftext|><|endoftext|><|endoftext|>"

special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
assert split_by_special_tokens_2(text3, special_tokens) == ["", "<|endoftext|><|endoftext|>", "", "<|endoftext|>", ""]
assert split_by_special_tokens_2(text4, special_tokens) == ["", "<|endoftext|><|endoftext|>", "", "<|endoftext|><|endoftext|>", ""]



