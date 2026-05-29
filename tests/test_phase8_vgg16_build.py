"""Smoke test: Phase 8 backbone (vgg16) builds at 300x300 tile size."""

import torch

from model.build import build_counter


def test_vgg16_builds_at_300():
    model = build_counter(backbone="vgg16", pretrained=False, head_hidden=0)
    x = torch.randn(2, 3, 300, 300)
    y = model(x)
    assert y.shape == (2, 5)
