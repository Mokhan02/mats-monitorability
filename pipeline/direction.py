import torch


def diff_in_means(pos_acts: dict[int, torch.Tensor], neg_acts: dict[int, torch.Tensor], layer: int) -> torch.Tensor:
    d = pos_acts[layer].mean(0) - neg_acts[layer].mean(0)
    return d / d.norm()
