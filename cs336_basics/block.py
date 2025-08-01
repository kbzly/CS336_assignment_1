from re import S
import torch
from torch import nn
from torch.nn import init
import math
from einops import einsum, reduce, rearrange
from dataclasses import dataclass
from .utils import strip_prefix_from_state_dict

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        std = math.sqrt(2.0 / (in_features + out_features))
        init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out") # x@self.W.T

class Embedding(nn.Module):
    def __init__(  
        self,
        num_embeddings,
        embedding_dim,
        device=None, 
        dtype=None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        std = 1
        init.trunc_normal_(self.W, mean=0.0, std=std, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids] # Select rows of self.weight by token_ids

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int, 
        eps: float = 1e-5, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = (x.pow(2).mean(dim=-1, keepdim=True)+ self.eps).sqrt() 
        normalized_x = x / RMS
        
        results = normalized_x * self.weight # W will automatically broadcast to ..., d_model

        # Return the result in the original dtype
        return results.to(in_dtype)

    
def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1 = Linear(d_model, d_ff, **factory_kwargs)
        self.linear2 = Linear(d_model, d_ff, **factory_kwargs)
        self.linear3 = Linear(d_ff, d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.linear2(SiLU(self.linear1(x)) * self.linear3(x))

def RoPe(d_k: int, theta: float, max_seq_len: int, 
        in_query_or_key: torch.Tensor, 
        token_positions: torch.Tensor):
    # 得到even和add matrix
    even_matrix = in_query_or_key[..., 0::2]
    odd_matrix = in_query_or_key[..., 1::2]

    # 计算cos和sin
    d_idx = torch.arange(0, d_k // 2, device=in_query_or_key.device, dtype=in_query_or_key.dtype)
    theta_idx = 1 / (theta ** (2 * d_idx / d_k))
    # theta_matrix = token_positions.unsqueeze(-1) * theta_idx.unsqueeze(0)
    # 或者用下面这个
    theta_matrix = token_positions[..., None] * theta_idx[None, :]

    cos_matrix = torch.cos(theta_matrix)
    sin_matrix = torch.sin(theta_matrix)

    even_matrix_rotated = even_matrix * cos_matrix - odd_matrix * sin_matrix
    odd_matrix_rotated = even_matrix * sin_matrix + odd_matrix * cos_matrix

    results = torch.stack([even_matrix_rotated, odd_matrix_rotated], dim=-1)
    return results.flatten(-2)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        super().__init__()
        d_half = d_k // 2
        position = torch.arange(max_seq_len).unsqueeze(-1)
        d_idx = torch.arange(d_half).unsqueeze(0)
        theta_idx = 1.0 / (theta ** (2 * d_idx / d_k))

        angle = position * theta_idx  # (max_seq_len, d_half)
        self.register_buffer("cos_cached", torch.cos(angle), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angle), persistent=False)

    def forward(self, in_query_or_key: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        x_even = in_query_or_key[..., 0::2]
        x_odd = in_query_or_key[..., 1::2]

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        return torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp_x

def ScaledDotProductAttention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Q: (..., queries, d_k)
    K: (..., keys, d_k)
    V: (..., values, d_v)
    mask: (..., queries, keys) or None
    Returns: (..., queries, d_v)
    """
    #  keys和values是相等的
    attn_logits = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(Q.shape[-1])
    # 或者用下面这个
    # attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
    attn_weights = softmax(attn_logits, dim=-1)
    return einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    # 或者用下面这个
    # return torch.matmul(attn_weights, V)

def build_causal_mask(seq_len: int, device=None):
    # shape: (1, 1, seq_len, seq_len)
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


class MultiheadSelfAttention(nn.Module):
    """
    d_model: int,
    num_heads: int,
    assignment1 spec里说dk = dv = d_model/num_heads
    adapters.py里的注释: (q_proj_weight (Float[Tensor, "d_k d_in"]))， 这里的d_k其实是所以head合的维度
    统一标准：d_k是所有head合的维度，d_k_head是每个head的维度，有假设d_k_head = d_model/num_heads。d_v同理
    """
    def __init__(self,
        d_model: int,
        num_heads: int,):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model
        self.d_k_head = d_model // num_heads
        self.d_v = d_model
        self.d_v_head = d_model // num_heads
        self.Q_proj = Linear(d_model, self.d_k)
        self.K_proj = Linear(d_model, self.d_k)
        self.V_proj = Linear(d_model, self.d_v)
        self.O_proj = Linear(self.d_v, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 虽然每个头有不同的参数，也会分别计算attention，但是可以并行计算以加速
        Q = self.Q_proj(x) # (..., seq_len, d_k)
        K = self.K_proj(x) # (..., seq_len, d_k)
        V = self.V_proj(x) # (..., seq_len, d_v)

        # mask是在seqence意义的维度上，所以需要rearrange
        Q = rearrange(Q, "... seq_len (num_heads d_head_k) -> ... num_heads seq_len d_head_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_head_k) -> ... num_heads seq_len d_head_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_head_v) -> ... num_heads seq_len d_head_v", num_heads=self.num_heads)

        mask = build_causal_mask(Q.shape[-2], device=Q.device)
        
        attn_output = ScaledDotProductAttention(Q, K, V, mask) # (..., num_heads, seq_len, d_head_v)
        attn_output = rearrange(attn_output, "... num_heads seq_len d_head_v -> ... seq_len (num_heads d_head_v)") # (..., seq_len, d_v)
        return self.O_proj(attn_output)

    
class MultiheadSelfAttentionWithRoPE(nn.Module):
    """
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    token_positions: Int[Tensor, " ... sequence_length"] | None = None
    """
    def __init__(self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        RoPE: RotaryPositionalEmbedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_k = d_model
        self.d_k_head = d_model // num_heads
        self.d_v = d_model
        self.d_v_head = d_model // num_heads
        self.Q_proj = Linear(d_model, self.d_k)
        self.K_proj = Linear(d_model, self.d_k)
        self.V_proj = Linear(d_model, self.d_v)
        self.O_proj = Linear(self.d_v, d_model)
        self.RoPE = RoPE
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # print(f"[DEBUG] self.RoPE id: {id(self.RoPE)}")
        # print(f"[DEBUG] token_positions id: {id(token_positions)}")
        Q = self.Q_proj(x) # (..., seq_len, d_k)
        K = self.K_proj(x) # (..., seq_len, d_k)
        V = self.V_proj(x) # (..., seq_len, d_v)

        Q = rearrange(Q, "... seq_len (num_heads d_head_k) -> ... num_heads seq_len d_head_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_head_k) -> ... num_heads seq_len d_head_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_head_v) -> ... num_heads seq_len d_head_v", num_heads=self.num_heads)

        if token_positions is not None:
            Q = self.RoPE(Q, token_positions)
            K = self.RoPE(K, token_positions)

        mask = build_causal_mask(Q.shape[-2], device=Q.device)
        
        attn_output = ScaledDotProductAttention(Q, K, V, mask) # (..., num_heads, seq_len, d_v)
        attn_output = rearrange(attn_output, "... num_heads seq_len d_head_v -> ... seq_len (num_heads d_head_v)") # (..., seq_len, d_v)
        return self.O_proj(attn_output)

@dataclass
class TransformerBlockConfig:
    d_model: int
    num_heads: int
    d_ff: int
    max_seq_len: int
    theta: float
    RoPE: RotaryPositionalEmbedding | None = None

class TransformerBlock(nn.Module): # pre-norm
    """
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention.
    d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    max_seq_len: int Maximum sequence length.
    theta: float Theta value for RoPE.
    weights: dict[str, torch.Tensor] Dictionary of weights for the Transformer block.
    valid key of weights:
        attn.q_proj.weight: (d_k, d_in)
        attn.k_proj.weight: (d_k, d_in)
        attn.v_proj.weight: (d_v, d_in)
        attn.output_proj.weight: (d_model, d_v)
        ln1.weight: (d_model,)
        ln2.weight: (d_model,)
        ffn.w1.weight: (d_ff, d_model)
        ffn.w2.weight: (d_model, d_ff)
        ffn.w3.weight: (d_ff, d_model)
    """
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.config = config
        self.norm1 = RMSNorm(self.config.d_model)
        self.norm2 = RMSNorm(self.config.d_model)
        self.ffn = PositionwiseFeedForward(self.config.d_model, self.config.d_ff)
        self.mha = MultiheadSelfAttentionWithRoPE(
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            max_seq_len=self.config.max_seq_len,
            theta=self.config.theta,
            RoPE=self.config.RoPE
        )
    
    def load_weights_dict(self, weights: dict[str, torch.Tensor]):
        self.norm1.weight.data = weights["ln1.weight"]
        self.norm2.weight.data = weights["ln2.weight"]
        self.ffn.linear1.W.data = weights["ffn.w1.weight"]
        self.ffn.linear2.W.data = weights["ffn.w2.weight"]
        self.ffn.linear3.W.data = weights["ffn.w3.weight"]
        self.mha.Q_proj.W.data = weights["attn.q_proj.weight"]
        self.mha.K_proj.W.data = weights["attn.k_proj.weight"]
        self.mha.V_proj.W.data = weights["attn.v_proj.weight"]
        self.mha.O_proj.W.data = weights["attn.output_proj.weight"]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # token_positions: (..., seq_len)
        # x: (..., seq_len, d_model)
        # token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.long) # shape: (seq_len,)
        # 用expand广播，x: (..., seq_len, d_model), token_positions: (seq_len,)
        # token_positions = token_positions.expand(x.shape[:-1])  # (..., seq_len), (seq_len,)
        # 或者
        # token_positions = token_positions.unsqueeze(0).expand(x.shape[:-1]) # (..., seq_len), (1, seq_len)
        x = x + self.mha(self.norm1(x), token_positions)
        x = x + self.ffn(self.norm2(x))
        return x

@dataclass
class TransformerLMConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

class TransformerLM(nn.Module):
    def __init__(self, config: TransformerLMConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = Embedding(config.vocab_size, config.d_model)
        RoPE = RotaryPositionalEmbedding(config.d_model // config.num_heads, config.rope_theta, config.context_length)
        transformer_block_config = TransformerBlockConfig(config.d_model, config.num_heads, config.d_ff, config.context_length, config.rope_theta, RoPE)
        self.layers = nn.ModuleList([TransformerBlock(transformer_block_config) for _ in range(config.num_layers)])
        self.ln_final = RMSNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size)

    def load_weights_dict(self, weights: dict[str, torch.Tensor]):
        self.token_embeddings.W.data = weights["token_embeddings.weight"]
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            layer.load_weights_dict(strip_prefix_from_state_dict(weights, prefix))
        self.ln_final.weight.data = weights["ln_final.weight"]
        self.lm_head.W.data = weights["lm_head.weight"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.long) # shape: (seq_len,)
        token_positions = token_positions.expand(x.shape[:-1]) # (..., seq_len)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)