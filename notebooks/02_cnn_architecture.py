"""
02 — Custom CNN Architecture (Baseline)

Defines a lightweight 2-conv-layer CNN for binary deepfake detection.
Used as a baseline before switching to the ResNet-18 transfer-learning
approach in 04_transfer_learning.py.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Hardware — prefer AMD DirectML, fall back to CPU gracefully
# ---------------------------------------------------------------------------
print("Checking hardware connection...")
try:
    import torch_directml

    device = torch_directml.device()
    print(f"Success! Model connected to AMD GPU: {device}")
except ImportError:
    device = torch.device("cpu")
    print(f"DirectML not found. Falling back to: {device}")


class DeepGuardCNN(nn.Module):
    """Minimal 2-layer CNN for real-vs-fake binary classification.

    Input:  (batch, 3, 128, 128)
    Output: (batch, 1) — sigmoid probability
    """

    def __init__(self):
        super(DeepGuardCNN, self).__init__()

        # feature extraction layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=0.5)

        # classifier — 128×128 pooled twice → 32×32, so 64 * 32 * 32 = 65536
        self.fc1 = nn.Linear(65536, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1, 65536)  # flatten feature maps

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


if __name__ == "__main__":
    model = DeepGuardCNN().to(device)
    print("\n--- Model Architecture ---")
    print(model)

    # smoke test — push a dummy tensor through to verify dims
    print("\nTesting data flow through the model...")
    dummy_image = torch.randn(1, 3, 128, 128).to(device)
    output = model(dummy_image)
    print(f"Success! Dummy image passed through. Output shape: {output.shape}")