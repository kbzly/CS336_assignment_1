from abc import ABC
from ctypes import addressof
import os
from dataclasses import dataclass
from typing import Union
from collections import defaultdict
import regex as re
from .pretokenization import find_chunk_boundaries
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

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

def match_max_pair(token: tuple[bytes], max_pair: tuple[bytes, bytes]) -> bool:
    i = 0
    a, b = max_pair
    while i < len(token) - 1:
        if a == token[i] and b == token[i + 1]:
            return True
        i += 1
    return False

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

def merge_bytes(byte_tuple: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
        # 遍历当前byte_list所有相邻词元，如果是pair，那就合并为新的bytes
        new_tuple: list[bytes] = []
        i = 0
        a, b = pair
        while i < len(byte_tuple) - 1:
            cur = byte_tuple[i]
            nxt = byte_tuple[i + 1]
            if cur == a and nxt == b:
                new_tuple.append(cur + nxt)
                i += 2
            else:
                new_tuple.append(cur)
                i += 1
        if i == len(byte_tuple) - 1: # 如果最后两个词元匹配了pair，i就会等于len(indices),不会进入if语句
            new_tuple.append(byte_tuple[i])
        return tuple(new_tuple)

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
    
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8")
        return string
    
def lexicographically_greater_pair(freq_dict: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    max_pair = max(freq_dict.items(), key=lambda x: (x[1], x[0]))[0]
    return max_pair


def gpt2_pretokenize_to_freq_dict(text: Union[str, list[str]]) -> dict[bytes, int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    freq_dict = defaultdict(int)
    if isinstance(text, str):
        segments = re.findall(PAT, text)
        for segment in segments:
            freq_dict[segment.encode("utf-8")] += 1
    elif isinstance(text, list):
        for t in text:
            segments = re.findall(PAT, t)
            for segment in segments:
                freq_dict[segment.encode("utf-8")] += 1
    return freq_dict

def split_and_remove_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    从 text 中用 special_tokens 做分割，返回分割后的 list[str]，
    每一段都是原始文本的一部分，保留空格和格式，不包含特殊 token。
    """
    # 构建正则模式，匹配所有特殊 token
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    # 分割并返回（保留空格，去掉特殊 token 本身）
    return [part for part in re.split(pattern, text) if part != ""]


def process_chunk(args: tuple[str, list[str], int, int]) -> dict[bytes, int]:
    text, special_tokens, start, end = args
    with open(text, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore") # 用str做正则
    # 处理特殊符号
    if special_tokens:
        chunk = split_and_remove_special_tokens(chunk, special_tokens)
    return gpt2_pretokenize_to_freq_dict(chunk)

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], **kwargs,) -> BPETokenizerParams:
    # 改进版本是不再操作原文本，而是操作一个哈希表，key是被特殊符号处理分割过的单词（还没登记到词表），value是该单词的频率
    # 这种方法把时间复杂度从 O(total byte) 降为 O(unique tokens)，对于大规模的文本数据，只要做好管理存储，也很便于维护新数据
    # 0. 初始化词表
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = list()

    # 1.读取文件，处理边界
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks=8, split_special_token=b"<|endoftext|>" if special_tokens else None)
    
    # 2. 多进程处理
    chunk_args = [(input_path, special_tokens, start, end) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]  
    with Pool(processes=8) as pool:
        results = pool.map(process_chunk, chunk_args) # 返回一个list，每个元素是dict[bytes, int]
    
    # 3. 合并结果，之后只操作哈希表
    data: defaultdict[tuple[bytes], int] = defaultdict(int) # 哈希表的key是tuple[bytes]，value是频率
    for result in results:
        for token, freq in result.items():
            data[tuple([bytes([b]) for b in token])] += freq

    # 4. 训练BPE
    num_merges = vocab_size - len(vocab) - len(special_tokens) # 目标词表大小减去初始字符数量和特殊符号数量
    while len(merges) < num_merges:
        # 4.1 计算词频
        freq_dict: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
        for token, freq in data.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                freq_dict[pair] += freq
        # preferring the lexicographically greater pair
        max_pair = lexicographically_greater_pair(freq_dict)

        # 4.2 更新merges和词表
        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        
        # 4.3 更新哈希表key，只处理包含max_pair的token
        tokens_to_update = []
        for token, freq in data.items():
            if match_max_pair(token, max_pair):
                tokens_to_update.append(token)
        
        for token in tokens_to_update:
            freq = data[token]
            new_token = merge_bytes(token, max_pair)
            data[new_token] += freq
            del data[token]

    # 5. 补齐vocab里的特殊符号
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    return BPETokenizerParams(vocab, merges, special_tokens)
    

if __name__ == "__main__":
    address = "/home/wyx/my_project/cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt"
    tokenizer = BPETokenizer(train_bpe(address, 500, special_tokens=["<|endoftext|>"]))
    print(tokenizer.params.merges)
    text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    encode_ids = tokenizer.encode(text)
    assert text == tokenizer.decode(encode_ids)
