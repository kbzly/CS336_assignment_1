import torch

def strip_prefix_from_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }