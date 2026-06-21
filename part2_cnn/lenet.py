"""
Part II — LeNet-5 Architecture
Supports MNIST, Fashion-MNIST (1-channel 28×28)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """Classic LeNet-5 adapted for 1-channel 28×28 images (MNIST / Fashion-MNIST).

    Architecture:
      Conv1 (1→6, k=5, p=2) → AvgPool(2,2) → Tanh
      Conv2 (6→16, k=5)     → AvgPool(2,2) → Tanh
      Flatten
      FC1(400→120) → Tanh
      FC2(120→84)  → Tanh
      FC3(84→n_classes)
    """

    def __init__(self, n_classes: int = 10, n_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.n_classes  = n_classes
        self.n_channels = n_channels
        self.img_size   = img_size

        # ── Convolutional feature extractor ───────────────────────────────────
        self.conv1 = nn.Conv2d(n_channels, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Compute flattened dimension dynamically
        self._flat_dim = self._get_flat_dim()

        # ── Fully connected classifier ─────────────────────────────────────────
        self.fc1 = nn.Linear(self._flat_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def _get_flat_dim(self) -> int:
        """Run a dummy forward pass to get the flattened feature size."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_channels, self.img_size, self.img_size)
            x = self.pool2(torch.tanh(self.conv2(
                self.pool1(torch.tanh(self.conv1(dummy))))))
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        # Block 2
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        # Classifier
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
class ImprovedLeNet(nn.Module):
    """An improved LeNet variant with ReLU, BatchNorm, and MaxPool."""

    def __init__(self, n_classes: int = 10, n_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.n_channels = n_channels
        self.img_size   = img_size

        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self._flat_dim = self._get_flat_dim()
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self._flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def _get_flat_dim(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_channels, self.img_size, self.img_size)
            return self.features(dummy).numel()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lenet = LeNet5(n_classes=10)
    improved = ImprovedLeNet(n_classes=10)
    dummy = torch.randn(4, 1, 28, 28)
    print("LeNet-5:")
    print(lenet)
    print(f"  Params: {lenet.count_parameters():,}")
    print(f"  Output: {lenet(dummy).shape}")
    print("\nImproved LeNet:")
    print(f"  Params: {improved.count_parameters():,}")
    print(f"  Output: {improved(dummy).shape}")
