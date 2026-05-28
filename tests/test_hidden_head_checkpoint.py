"""Checkpoint reload must match head architecture (linear vs MLP)."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.build import build_counter, build_counter_from_checkpoint_args
from utils.io import load_checkpoint, save_checkpoint


def test_hidden_head_checkpoint_roundtrip(tmp_path):
    device = torch.device("cpu")
    model = build_counter("resnet18", pretrained=False, dropout=0.5, head_hidden=32)
    ckpt_path = tmp_path / "mlp.pth"
    args = {
        "backbone": "resnet18",
        "dropout": 0.5,
        "head_hidden": 32,
        "tile_size": 128,
    }
    save_checkpoint(ckpt_path, model, None, 1, 0.0, args)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    rebuilt = build_counter_from_checkpoint_args(ckpt["args"], pretrained=False)
    load_checkpoint(ckpt_path, rebuilt, device)
    rebuilt.eval()

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = rebuilt(x)
    assert out.shape == (1, 5)


def test_linear_head_default_from_checkpoint_args():
    model = build_counter_from_checkpoint_args({"backbone": "resnet18"}, pretrained=False)
    assert model.head_hidden == 0
    assert len(model.head) == 2
