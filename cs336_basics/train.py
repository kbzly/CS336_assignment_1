import torch

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