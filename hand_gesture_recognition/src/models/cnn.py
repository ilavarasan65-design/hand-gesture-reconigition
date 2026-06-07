"""
cnn.py — CNN branch for extracting visual appearance features from hand images.

Uses pretrained MobileNetV3-Large or ResNet50 as a backbone,
with a projection head to match the GCN embedding dimension.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNBranch(nn.Module):
    """
    CNN visual feature extractor.

    Input:  image tensor [B, 3, 224, 224]
    Output: feature embedding [B, out_dim]

    Args:
        backbone: 'mobilenet_v3' | 'resnet50' | 'efficientnet_b0'
        out_dim:  output embedding size (should match GCN branch)
        pretrained: load ImageNet weights
        freeze_backbone: freeze all backbone layers (transfer learning)
    """

    def __init__(self, backbone='mobilenet_v3', out_dim=256,
                 pretrained=True, freeze_backbone=False):
        super().__init__()

        self.backbone_name = backbone
        weights = 'DEFAULT' if pretrained else None

        if backbone == 'mobilenet_v3':
            base = models.mobilenet_v3_large(weights=weights)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Identity()
            self.backbone = base

        elif backbone == 'resnet50':
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base

        elif backbone == 'efficientnet_b0':
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base

        else:
            raise ValueError(f'Unknown backbone: {backbone}')

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        feats = self.backbone(x)          # [B, in_features]
        return self.proj(feats)           # [B, out_dim]

    def unfreeze_backbone(self, layers_from_end=2):
        """Progressively unfreeze last N layer groups (for fine-tuning)."""
        children = list(self.backbone.children())
        for child in children[-layers_from_end:]:
            for param in child.parameters():
                param.requires_grad = True
