import torch
import math
from typing import Optional, Callable, Iterable
import numpy as np
import os
import typing

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between logits and labels.
    logits: (batch_size, vocab_size)
    labels: (batch_size)
    """
    # 找到每个logits里最大值的索引和对应的值
    max_indices = torch.argmax(logits, dim=-1, keepdim=True) # (..., 1)
    # torch.gather(input, dim, index,) -> Tensor
    # output[i][j][k] = input[i][j][index[i][j][k]] if dim = 2
    max_values = torch.gather(logits, dim=-1, index=max_indices) # (..., 1) -> (..., 1)
    # 所有logits减去对应的最大值
    logits = logits - max_values # max_values广播到(..., vocab_size)
    # 计算每个组的logsumexp
    logsumexp = torch.logsumexp(logits, dim=-1) # (...,)
    true_logits = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (...,)
    # 计算 loss = logsumexp - true_logit
    loss = logsumexp - true_logits  # shape: (...,) 
    return loss.mean()



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0 or betas[0] > 1:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if betas[1] < 0 or betas[1] > 1:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                state["m"] = betas[0] * m + (1 - betas[0]) * p.grad
                v = state.get("v", torch.zeros_like(p))
                state["v"] = betas[1] * v + (1 - betas[1]) * p.grad**2
                t = state.get("t", 0)
                lr_t = lr * math.sqrt(1 - betas[1]**(t+1)) / (1 - betas[0]**(t+1))
                state["t"] = t + 1

                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                p.data *= (1 - lr * weight_decay)
        return loss
    

def leanring_rate_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), p=2)
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)

def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(x) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    input_batch = np.stack([x[s : s + context_length] for s in starts])
    target_batch = np.stack([x[s + 1 : s + context_length + 1] for s in starts])

    input_tensor = torch.tensor(input_batch, device=device)
    target_tensor = torch.tensor(target_batch, device=device)

    return [input_tensor, target_tensor]

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]