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

# 实现一个BPE分词器
# 首先我们要知道BPE的编码和解码过程
# 编码过程：根据合并规则遍历输入文本，将所有符合规则的相邻词元合并为新词元
# 解码过程：根据编码后的整数序号，将词元映射回原始文本

# 实现一个BPE独有的数据类型
@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Merge the pair of indices into a new index"""
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

def merge_bytes(byte_list: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        """Merge the pair of indices into a new index"""
        # 遍历当前byte_list所有相邻词元，如果是pair，那就合并为新的bytes
        new_list = []
        i = 0
        while i < len(byte_list) - 1:
            if (byte_list[i], byte_list[i + 1]) == pair:
                new_list.append(byte_list[i] + byte_list[i + 1])
                i += 2
            else:
                new_list.append(byte_list[i])
                i += 1
        if i == len(byte_list) - 1: # 如果最后两个词元匹配了pair，i就会等于len(indices),不会进入if语句
            new_list.append(byte_list[i])
        return new_list

# 实现一个BPE分词器
class BPETokenizer(Tokenizer):
    """Byte-pair encoding tokenizer"""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
        self.vocab_lookup: dict[bytes, int] = {v: k for k, v in self.params.vocab.items()}

    def encode(self, text: str) -> list[int]:
        # 对单个字符的编码任然是uft-8编码
        # 在遍历merges规则时，维护一个int list
        tokens = []
        special_token_set = set(self.params.special_tokens or [])
        
        if special_token_set:
            # 构建正则分割 pattern
            # 先排序，让最长的特殊符号先被匹配 def test_overlapping_special_tokens()
            special_pattern = "|".join(
                re.escape(tok) for tok in sorted(special_token_set, key=len, reverse=True)
            )
            pattern = f"({special_pattern})"
            segments = re.split(pattern, text)
        else:
            segments = [text]

        for segment in segments:
            if not segment:
                continue  # 空字符串跳过
            if segment in special_token_set:
                byte_token = segment.encode("utf-8")
                token_id = self.vocab_lookup.get(byte_token)
                if token_id is None:
                    raise ValueError(f"Special token {segment} not in vocab")
                tokens.append(token_id)
            else:
                # 正常文本走 BPE 编码
                byte_seq = segment.encode("utf-8")
                byte_list = [bytes([b]) for b in byte_seq]
                for pair in self.params.merges:
                    byte_list = merge_bytes(byte_list, pair)
                tokens.extend(self.vocab_lookup[b] for b in byte_list)

        return tokens

        # byte_seq = text.encode("utf-8")
        # byte_list = [bytes([b]) for b in byte_seq]

        # for pair in self.params.merges:
        #     byte_list = merge_bytes(byte_list, pair)

        # # 将byte_list转换为int list
        # indices = [self.vocab_lookup[b] for b in byte_list]
        # return indices
    
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
    merges = list()

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
        merges.append([vocab[max_pair[0]], vocab[max_pair[1]]])

        # 3. 哈希表key
        new_data = defaultdict(int)
        for token, freq in data.items():
            new_token = merge(token, max_pair, len(vocab))
            new_data[tuple(new_token)] += freq
        data = new_data

        # 4. 更新词表
        vocab[len(vocab)] = vocab[max_pair[0]] + vocab[max_pair[1]]

    return BPETokenizerParams(vocab, merges, special_tokens)
    

if __name__ == "__main__":
    text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    data = gpt2_pretokenize_to_freq_dict(text)
    print(train_bpe(data, 15))
    tokenizer = BPETokenizer(train_bpe(data, 15))
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))
    assert tokenizer.decode(tokenizer.encode(text)) == text

    # test 1 empty string
    test_string = ""
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string