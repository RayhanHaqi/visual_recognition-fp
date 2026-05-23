"""timm backbone with 5-d count regression head."""

import torch
import torch.nn as nn
import timm


NUM_OUTPUTS = 5


class CountRegressor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, NUM_OUTPUTS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


def build_counter(backbone: str = "resnet50", pretrained: bool = True, dropout: float = 0.2) -> CountRegressor:
    model = CountRegressor(backbone, pretrained=pretrained, dropout=dropout)
    n = count_parameters(model)
    print(f"Model {backbone}: {n:,} trainable parameters")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
