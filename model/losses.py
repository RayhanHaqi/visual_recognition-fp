import torch
import torch.nn as nn


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(nn.functional.mse_loss(pred, target) + 1e-8)
