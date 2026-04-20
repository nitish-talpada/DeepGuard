"""
03 — Train Custom CNN (Baseline)

Trains the DeepGuardCNN from 02_cnn_architecture.py on the raw dataset.
This is the baseline experiment — the transfer-learning approach in step 04
supersedes this for production.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# directml had issues during early dev — train on CPU for the baseline
device = torch.device("cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = r"D:\DeepGuard\data\raw"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # pixels [0-255] → floats [0.0-1.0]
])

print("Loading datasets...")
# ImageFolder assigns labels alphabetically: Fake=0, Real=1
try:
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"Found {len(dataset)} images in {len(dataset.classes)} classes: {dataset.classes}")
except FileNotFoundError:
    print(f"Error: Could not find images in {DATA_DIR}. "
          "Make sure Real and Fake folders exist!")
    exit()


# ---------------------------------------------------------------------------
# Architecture (duplicated here so the script is self-contained)
# ---------------------------------------------------------------------------
class DeepGuardCNN(nn.Module):
    def __init__(self):
        super(DeepGuardCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(65536, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 65536)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
model = DeepGuardCNN().to(device)
criterion = nn.BCELoss()  # binary cross-entropy — standard for real/fake
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(epochs=5):
    """Run the baseline training loop.

    Args:
        epochs: Number of full passes over the dataset.
    """
    print("\n--- Starting Training ---")
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            # BCELoss expects float labels shaped (batch, 1)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(dataloader):.4f}")

    os.makedirs(r"D:\DeepGuard\models", exist_ok=True)
    save_path = r"D:\DeepGuard\models\custom_cnn.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete! Model saved to {save_path}")


if __name__ == "__main__":
    train_model()