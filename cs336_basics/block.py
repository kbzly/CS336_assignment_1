import torch
from torch import nn
from torch.nn import init
import math
from einops import einsum, reduce, rearrange

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
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"]
    """
    def __init__(self,
        d_model: int,
        num_heads: int,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        v_proj_weight: torch.Tensor,
        o_proj_weight: torch.Tensor):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = q_proj_weight.shape[0]
        self.d_v = v_proj_weight.shape[0]
        self.Q_proj = Linear(d_model, self.d_k)
        self.Q_proj.W.data = q_proj_weight
        self.K_proj = Linear(d_model, self.d_k)
        self.K_proj.W.data = k_proj_weight
        self.V_proj = Linear(d_model, self.d_v)
        self.V_proj.W.data = v_proj_weight
        self.O_proj = Linear(self.d_v, d_model)
        self.O_proj.W.data = o_proj_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Q_proj(x) # (..., seq_len, d_k)
        K = self.K_proj(x) # (..., seq_len, d_k)
        V = self.V_proj(x) # (..., seq_len, d_v)

        Q = rearrange(Q, "... seq_len (h d_head_k) -> ... h seq_len d_head_k", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_head_k) -> ... h seq_len d_head_k", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_head_v) -> ... h seq_len d_head_v", h=self.num_heads)

        mask = build_causal_mask(Q.shape[-2], device=Q.device)
        
        attn_output = ScaledDotProductAttention(Q, K, V, mask) # (..., num_heads, seq_len, d_v)
        attn_output = rearrange(attn_output, "... h seq_len d_head_v -> ... seq_len (h d_head_v)") # (..., seq_len, d_v)
        return self.O_proj(attn_output)

    
class MultiheadSelfAttentionWithRoPE(nn.Module):
    """
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None
    """
    def __init__(self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        v_proj_weight: torch.Tensor,
        o_proj_weight: torch.Tensor,
        token_positions: torch.Tensor | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.token_positions = token_positions
        self.d_k = q_proj_weight.shape[0]
        self.d_v = v_proj_weight.shape[0]
        self.Q_proj = Linear(d_model, self.d_k)
        self.Q_proj.W.data = q_proj_weight
        self.K_proj = Linear(d_model, self.d_k)
        self.K_proj.W.data = k_proj_weight
        self.V_proj = Linear(d_model, self.d_v)
        self.V_proj.W.data = v_proj_weight
        self.O_proj = Linear(self.d_v, d_model)
        self.O_proj.W.data = o_proj_weight

        if self.token_positions is not None:
            self.RoPE = RotaryPositionalEmbedding(self.d_k // self.num_heads, theta, max_seq_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Q_proj(x) # (..., seq_len, d_k)
        K = self.K_proj(x) # (..., seq_len, d_k)
        V = self.V_proj(x) # (..., seq_len, d_v)

        Q = rearrange(Q, "... seq_len (h d_head_k) -> ... h seq_len d_head_k", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_head_k) -> ... h seq_len d_head_k", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_head_v) -> ... h seq_len d_head_v", h=self.num_heads)

        if self.token_positions is not None:
            Q = self.RoPE(Q, self.token_positions)
            K = self.RoPE(K, self.token_positions)

        mask = build_causal_mask(Q.shape[-2], device=Q.device)
        
        attn_output = ScaledDotProductAttention(Q, K, V, mask) # (..., num_heads, seq_len, d_v)
        attn_output = rearrange(attn_output, "... h seq_len d_head_v -> ... seq_len (h d_head_v)") # (..., seq_len, d_v)
        return self.O_proj(attn_output)
