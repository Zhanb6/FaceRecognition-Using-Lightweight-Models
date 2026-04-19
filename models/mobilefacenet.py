"""
MobileFaceNet — Lightweight face recognition model with pretrained weights.

Paper:  MobileFaceNets: Efficient CNNs for Accurate Real-Time Face
        Verification on Mobile Devices (arXiv:1804.07573)

This implementation uses MobileNetV2 backbone from torchvision with
ImageNet pretrained weights, adapted for face recognition with a
128-dimensional embedding output.

Input:  112×112 RGB face crop
Output: 128-dimensional embedding vector
Params: ~2.3 M  (vs FaceNet's ~27.9 M)
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet for face recognition.

    Uses MobileNetV2 backbone (pretrained on ImageNet) with a custom
    embedding head for 128-d face embeddings.
    """

    def __init__(self, embedding_size=128, input_size=112, **kwargs):
        super().__init__()

        # Load MobileNetV2 with pretrained ImageNet weights
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Use only the feature extractor (drop classification head)
        self.features = backbone.features   # output: [B, 1280, H/32, W/32]

        # Adaptive pooling to handle any input size
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Embedding head: 1280 → embedding_size
        self.embedding = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        # Initialize embedding head
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        print(f"  MobileFaceNet: MobileNetV2 backbone (pretrained=ImageNet)")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Parameters: {total_params / 1e6:.2f}M")

    def forward(self, x):
        x = self.features(x)       # [B, 1280, H', W']
        x = self.pool(x)           # [B, 1280, 1, 1]
        x = x.flatten(1)           # [B, 1280]
        x = self.embedding(x)      # [B, embedding_size]
        return x
