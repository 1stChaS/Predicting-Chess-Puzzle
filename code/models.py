"""
Three model architectures for the chess puzzle project:

  1. MLPBaseline       — handcrafted features → MLP (Baseline 1)
  2. SingleTaskCNN     — CNN backbone → rating regression only (Baseline 2)
  3. MultiTaskCNN      — CNN backbone → rating head + theme head (Main model)

All CNNs accept input shape (B, 16, 8, 8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# shared CNN backbone
class ChessCNNBackbone(nn.Module):
    """
    Shared convolutional feature extractor.

    Architecture:
        Conv(16→64, 3×3) → BN → ReLU
        Conv(64→128, 3×3) → BN → ReLU
        Conv(128→256, 3×3) → BN → ReLU   [padding keeps 8×8]
        GlobalAvgPool → flatten → Linear(256→256) → ReLU → Dropout

    Output: feature vector of size 256.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 64,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        # Residual-style skip from conv1 to conv3 input (channel projection)
        self.skip = nn.Conv2d(64, 256, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)   # → (B, 256, 1, 1)
        self.fc      = nn.Linear(256, 256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 16, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))          # (B, 64, 8, 8)
        skip = self.skip(x)                           # (B, 256, 8, 8)

        x = F.relu(self.bn2(self.conv2(x)))           # (B, 128, 8, 8)
        x = F.relu(self.bn3(self.conv3(x)) + skip)    # (B, 256, 8, 8) residual

        x = self.global_pool(x).flatten(1)            # (B, 256)
        x = self.dropout(F.relu(self.fc(x)))          # (B, 256)
        return x


# Baseline 1: MLP on handcrafted features
class MLPBaseline(nn.Module):
    """
    Simple MLP that takes handcrafted scalar features as input.

    Features (computed externally, see utils/features.py):
        piece counts per type (10), material balance (1), mobility proxy (1) = 12

    Outputs:
        rating_pred : (B,) float   — regression
        theme_logits: (B, C) float — classification
    """

    def __init__(self, in_features: int = 12, num_themes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),          nn.ReLU(),
        )
        self.rating_head = nn.Linear(64, 1)
        self.theme_head  = nn.Linear(64, num_themes)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.rating_head(h).squeeze(-1), self.theme_head(h)


# Baseline 2: Single-Task CNN
class SingleTaskCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.backbone    = ChessCNNBackbone(dropout)
        self.rating_head = nn.Sequential(
            nn.Linear(257, 64), nn.ReLU(),   # 256 + 1 move_count
            nn.Linear(64, 1)
        )

    def forward(self, x, move_count):         # add move_count arg
        features = self.backbone(x)
        combined = torch.cat([features, move_count.unsqueeze(1)], dim=1)
        return self.rating_head(combined).squeeze(-1)


# Main Model: Multi-Task CNN
class MultiTaskCNN(nn.Module):
    def __init__(self, num_themes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.backbone    = ChessCNNBackbone(dropout)
        self.rating_head = nn.Sequential(
            nn.Linear(257, 64), nn.ReLU(),   # 256 + 1 move_count
            nn.Linear(64, 1)
        )
        self.theme_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),   # 256 only, no move_count
            nn.Dropout(dropout),
            nn.Linear(64, num_themes)
        )

    def forward(self, x, move_count):         # add move_count arg
        features     = self.backbone(x)
        combined     = torch.cat([features, move_count.unsqueeze(1)], dim=1)
        rating_pred  = self.rating_head(combined).squeeze(-1)
        theme_logits = self.theme_head(features)
        return rating_pred, theme_logits


# model factory
def build_model(model_type: str, num_themes: int = 4, dropout: float = 0.3):
    """
    Convenience factory.
    model_type: 'mlp' | 'single' | 'multi'
    """
    if model_type == 'mlp':
        return MLPBaseline(num_themes=num_themes)
    elif model_type == 'single':
        return SingleTaskCNN(dropout=dropout)
    elif model_type == 'multi':
        return MultiTaskCNN(num_themes=num_themes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
