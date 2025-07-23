from abc import ABC
from dataclasses import dataclass
from typing import Union
from collections import defaultdict
import regex as re

# 实现一个分词器基类，让BPE继承它

# 分词器基类，包括解码和编码
class Tokenizer(ABC):
    """Abstract base class for tokenizers"""
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError

# 同时实现字符级和字节级分词器
class CharTokenizer(Tokenizer):
    """Character-level tokenizer"""
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

class ByteTokenizer(Tokenizer):
    """Byte-level tokenizer"""
    def encode(self, text: str) -> list[int]:
        string_bytes = text.encode("utf-8")
        return list(map(int, string_bytes))
    def decode(self, tokens: list[int]) -> str:
        string_bytes = bytes(tokens)
        return string_bytes.decode("utf-8") 

# 实现一个BPE分词器
# 首先我们要知道BPE的编码和解码过程
# 编码过程：根据合并规则遍历输入文本，将所有符合规则的相邻词元合并为新词元
# 解码过程：根据编码后的整数序号，将词元映射回原始文本

# 实现一个BPE独有的数据类型
@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[int, int], int]
    special_tokens: list[str]

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
        """Merge the pair of indices into a new index"""
        # 遍历当前indices所有相邻词元，如果是pair，那就合并为new_index
        new_indices = []
        i = 0
        while i < len(indices) - 1:
            if (indices[i], indices[i + 1]) == pair:
                new_indices.append(new_index)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1
        if i == len(indices) - 1: # 如果最后两个词元匹配了pair，i就会等于len(indices),不会进入if语句
            new_indices.append(indices[i])
        return new_indices

# 实现一个BPE分词器
class BPETokenizer(Tokenizer):
    """Byte-pair encoding tokenizer"""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
    
    def encode(self, text: str) -> list[int]:
        # 对单个字符的编码任然是uft-8编码
        # 在遍历merges规则时，维护一个int list
        indices = list(map(int, text.encode("utf-8")))
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices
    
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8")
        return string

def gpt2_pretokenize_to_freq_dict(text: Union[str, list[str]]) -> dict[tuple[int], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    freq_dict = defaultdict(int)
    if isinstance(text, str):
        segments = re.findall(PAT, text)
        for segment in segments:
            freq_dict[tuple(segment.encode("utf-8"))] += 1
    elif isinstance(text, list):
        for t in text:
            segments = re.findall(PAT, t)
            for segment in segments:
                freq_dict[tuple(segment.encode("utf-8"))] += 1
    return freq_dict

def train_bpe(data: Union[str, dict[tuple[int, ...], int]], num_merges: int, special_tokens: list[str]=None) -> BPETokenizerParams:
    # 改进版本是不再操作原文本，而是操作一个哈希表，key是被特殊符号处理分割过的单词（还没登记到词表），value是该单词的频率
    # 这种方法把时间复杂度从 O(total byte) 降为 O(unique tokens)，对于大规模的文本数据，只要做好管理存储，也很便于维护新数据
    # 0. 初始化词表
    vocab = {i: bytes([i]) for i in range(256)}
    merges = dict()

    # 1. 预处理特殊符号，得到哈希表，函数设计成可接受str或者哈希表
    if isinstance(data, str):
        data = gpt2_pretokenize_to_freq_dict(data)

    while len(merges) < num_merges:
        # 2. 计算词频
        freq_dict = defaultdict(int)
        for token, freq in data.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                freq_dict[pair] += freq
        
        # preferring the lexicographically greater pair
        max_pair = max(freq_dict.items(), key=lambda x: (x[1], x[0]))[0]
        # 更新merges
        merges[max_pair] = len(vocab)

        # 3. 更新词表和哈希表key
        new_data = defaultdict(int)
        for token, freq in data.items():
            new_token = merge(token, max_pair, len(vocab))
            new_data[tuple(new_token)] += freq
        data = new_data

        print(vocab[max_pair[0]], vocab[max_pair[1]])
        new_token_bytes = vocab[max_pair[0]] + vocab[max_pair[1]]
        vocab[len(vocab)] = new_token_bytes

    return BPETokenizerParams(vocab, merges, special_tokens)
    

if __name__ == "__main__":
    text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    data = gpt2_pretokenize_to_freq_dict(text)
    tokenizer = BPETokenizer(train_bpe(data, 15))
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))
    assert tokenizer.decode(tokenizer.encode(text)) == text

    # test 1 empty string
    test_string = ""
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string