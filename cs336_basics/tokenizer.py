from abc import ABC
from ctypes import addressof
import heapq
import os
import json
from dataclasses import dataclass
from typing import Union, Iterable, Iterator
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

def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]

    # 按长度降序排列，保证正则优先匹配最长
    pattern = "|".join(sorted(map(re.escape, special_tokens), key=len, reverse=True))
    return re.split(f"({pattern})", text)


# 实现一个BPE分词器
class BPETokenizer(Tokenizer):
    """Byte-pair encoding tokenizer"""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
        self.vocab_lookup: dict[bytes, int] = {v: k for k, v in self.params.vocab.items()}
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens:list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = json.load(f)
        with open(merges_filepath, "rb") as f:
            merges = json.load(f)
        return cls(BPETokenizerParams(vocab, merges, special_tokens))

    def encode(self, text: str) -> list[int]:
        # 对单个字符的编码任然是uft-8编码
        # 在遍历merges规则时，维护一个int list
        tokens = []
        special_token_set = set(self.params.special_tokens or [])
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        segments = split_by_special_tokens(text, self.params.special_tokens)

        for segment in segments:
            if not segment:
                continue  # 空字符串跳过
            if segment in special_token_set:
                byte_token = segment.encode("utf-8")
                token_id = self.vocab_lookup.get(byte_token)
                tokens.append(token_id)
            else:
                # 正常文本走 BPE 编码
                # 这里需要再做一遍gpt2正则规则
                pre_tokens = re.findall(PAT, segment)
                for pre_token in pre_tokens:
                    byte_seq = pre_token.encode("utf-8")
                    byte_list = [bytes([b]) for b in byte_seq]
                    for pair in self.params.merges:
                        byte_list = merge_bytes(byte_list, pair)
                    tokens.extend(self.vocab_lookup[b] for b in byte_list)
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            # 逐行 encode
            token_ids = self.encode(chunk)
            for token_id in token_ids:
                yield token_id

    
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8", errors="replace")
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

class HeapItem:
    def __init__(self, freq: int, pair: tuple[bytes, bytes]):
        self.freq = freq
        self.pair = pair

    def __lt__(self, other: "HeapItem") -> bool:
        if self.freq != other.freq:
            return self.freq > other.freq
        return self.pair > other.pair

class FreqHeap:
    """
    A heap of pairs with their frequencies.
    """
    def __init__(self):
        self.heap: list[HeapItem] = []
        self.freq_dict: dict[tuple[bytes, bytes], int] = defaultdict(int)

    def push(self, pair: tuple[bytes, bytes]):
        freq = self.freq_dict.get(pair, 0)
        heapq.heappush(self.heap, HeapItem(freq, pair))

    def pop_max(self) -> tuple[bytes, bytes]:
        """
        返回value最大，key偏好lexicographically greater的pair，并从heap中删除。
        如果返回的pair的freq在freq_dict中已经为小于neg_freq，说明我们需要更新它同时更新heap，再重复取max
        """
        while self.heap:
            item = heapq.heappop(self.heap)
            current_freq = self.freq_dict.get(item.pair, 0)
            if item.freq == current_freq and current_freq > 0:
                return item.pair
            elif current_freq > 0:
                # 频率降低但仍有效，重新插入再取
                heapq.heappush(self.heap, HeapItem(current_freq, item.pair))
    # 否则当前_freq 为 0，表示已无效，跳过继续下一个
        raise IndexError("Heap is empty or all frequencies are stale.")

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
    
    data_with_id: dict[tuple[bytes], (int, int)] = {}
    id_to_token: dict[int, tuple[bytes]] = {}
    for i, (token, freq) in enumerate(data.items()):
        data_with_id[token] = (i, freq)
        id_to_token[i] = token
    data = data_with_id


    # 4. 训练BPE
    num_merges = vocab_size - len(vocab) - len(special_tokens) # 目标词表大小减去初始字符数量和特殊符号数量

    # 4.1 计算词频
    freq_heap: FreqHeap = FreqHeap()
    vocab_to_id: dict[bytes, int] = {}

    for pre_token, (id, freq) in data.items():
        exist_token: set[bytes] = set()
        for i in range(len(pre_token) - 1):
            exist_token.add(pre_token[i])
            pair = (pre_token[i], pre_token[i + 1])
            freq_heap.freq_dict[pair] += freq
        exist_token.add(pre_token[len(pre_token) - 1]) # 加上最后一个词元

        for token in exist_token:
            # 读取可能存在该token的pretoken，然后修改tuple再加回去
            # 每个pre_token都不一样，所以不用考虑重复
            if token in vocab_to_id:
                tmp = list(vocab_to_id[token])
                tmp.append(id)
                vocab_to_id[token] = tuple(tmp)
            else:
                vocab_to_id[token] = (id,)
    
    for pair, freq in freq_heap.freq_dict.items():
        freq_heap.push(pair)

    while len(merges) < num_merges:
        # 4.1 计算词频
        max_pair = freq_heap.pop_max()
        new_freq_dict: dict[tuple[bytes, bytes], int] = defaultdict(int)

        # 4.2 更新merges和词表
        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        
        # 4.3 更新哈希表key，只处理包含max_pair的token
        a, b = max_pair
        candidates = []
        # candidates应该在vocab_to_id同时出现过
        id_a = vocab_to_id[a]  # 获取键 a 对应的 int
        id_b = vocab_to_id[b]  # 获取键 b 对应的 int

        # 转换为集合，以便高效查找交集
        set_a = set(id_a)
        set_b = set(id_b)

        # 找出共有的 tuple[bytes]
        common_ids = set_a & set_b

        # 如果需要结果保持 tuple 类型
        candidates = tuple(common_ids)
        
        for candidate_id in candidates:
            candidate = id_to_token[candidate_id]
            freq = data[candidate][1]
            if match_max_pair(candidate, max_pair):
                # 尝试merge
                new_token = merge_bytes(candidate, max_pair)

                # 更新 freq_dict：先移除 candidate 中所有相邻 pair 的频率
                for i in range(len(candidate) - 1):
                    freq_heap.freq_dict[(candidate[i], candidate[i+1])] -= freq

                # 更新 freq_dict：再为 new_token 添加新      的频率
                for i in range(len(new_token) - 1):
                    freq_heap.freq_dict[(new_token[i], new_token[i+1])] += freq
                    if (new_token[i] == a + b or new_token[i+1] == a + b): 
                        new_freq_dict[(new_token[i], new_token[i+1])] += freq

                # 更新 vocab_to_id
                if a+b in vocab_to_id:
                    tmp = list(vocab_to_id[a+b])
                    tmp.append(candidate_id)
                    vocab_to_id[a+b] = tuple(tmp)
                else:
                    vocab_to_id[a+b] = (candidate_id,)
                
                # 更新 id_to_token
                id_to_token[candidate_id] = new_token

                # 更新 data
                data[new_token] = (candidate_id, freq)
                del data[candidate]
        # 现在我们有更新过后的data，freq_dict，vocab2pretoken
        # 可以开始下一轮了
        for pair, freq in new_freq_dict.items():
            freq_heap.push(pair)
        
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
