"""timm backbone with 5-d count regression head."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


NUM_OUTPUTS = 5


def _make_head(feat_dim: int, dropout: float, head_hidden: int) -> nn.Sequential:
    if head_hidden and head_hidden > 0:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, NUM_OUTPUTS),
        )
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(feat_dim, NUM_OUTPUTS),
    )


class CountRegressor(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        dropout: float = 0.2,
        head_hidden: int = 0,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.dropout_p = dropout
        self.head_hidden = head_hidden
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head = _make_head(feat_dim, dropout, head_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


def build_counter(
    backbone: str = "resnet50",
    pretrained: bool = True,
    dropout: float = 0.2,
    head_hidden: int = 0,
) -> CountRegressor:
    model = CountRegressor(backbone, pretrained=pretrained, dropout=dropout, head_hidden=head_hidden)
    n = count_parameters(model)
    head_desc = "linear" if not head_hidden else f"mlp({head_hidden})"
    print(f"Model {backbone} ({head_desc}): {n:,} trainable parameters")
    return model


def build_counter_from_checkpoint_args(ckpt_args: dict, pretrained: bool = False) -> CountRegressor:
    """Rebuild model architecture saved in a training checkpoint."""
    return build_counter(
        backbone=ckpt_args.get("backbone", "resnet50"),
        pretrained=pretrained,
        dropout=float(ckpt_args.get("dropout", 0.2)),
        head_hidden=int(ckpt_args.get("head_hidden", 0) or 0),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
