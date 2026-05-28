import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.build import build_counter
from model.losses import rmse_loss


def test_build_counter_forward():
    model = build_counter("resnet18", pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 5)


def test_build_counter_hidden_head():
    model = build_counter("resnet18", pretrained=False, head_hidden=32, dropout=0.5)
    assert model.head_hidden == 32
    assert len(model.head) == 5
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 5)


def test_rmse_loss():
    pred = torch.tensor([[1.0, 2, 3, 4, 5]])
    tgt = torch.tensor([[1.0, 2, 3, 4, 5]])
    # Identical inputs: MSE=0, loss uses sqrt(mse + 1e-8) for stability
    assert rmse_loss(pred, tgt).item() == pytest.approx(1e-4, abs=1e-6)
    diff = torch.tensor([[1.0, 3, 3, 4, 5]])
    assert rmse_loss(pred, diff).item() > 0
