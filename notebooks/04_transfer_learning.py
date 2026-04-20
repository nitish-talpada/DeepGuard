"""
04 — Transfer Learning with ResNet-18 (Production Model)

Fine-tunes a pre-trained ResNet-18 on the merged dataset with
augmentation.  Only the custom classification head is trained;
backbone weights stay frozen.  This is the model that ships in app.py.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
device = torch_directml.device()
print(f"Training device: {device} (AMD DirectML)")

# ---------------------------------------------------------------------------
# Data — augmentations force the model to generalise beyond memorisation
# ---------------------------------------------------------------------------
TRAIN_DIR = r"D:\DeepGuard\data\train"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),                # simulate mirror selfies
    transforms.RandomRotation(degrees=15),                  # slight head tilts
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # handle lighting variance
    transforms.GaussianBlur(kernel_size=3),                 # blur vs. sharp fakes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("Loading training dataset...")
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    pin_memory=True,
)
print(f"Dataset ready. Total images: {len(train_dataset)} | Batch size: 128")

# ---------------------------------------------------------------------------
# Model — freeze backbone, replace head with a binary classifier
# ---------------------------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# freeze all pre-trained conv layers
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),  # regularise the small head to prevent overfit
    nn.Linear(256, 1),
    nn.Sigmoid(),
)

model = model.to(device)

# ---------------------------------------------------------------------------
# Optimiser & loss — only train the new head parameters
# ---------------------------------------------------------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_resnet(epochs=15):
    """Fine-tune the ResNet-18 classification head.

    Args:
        epochs: Number of full passes over the training set.
    """
    print(f"\n--- Starting training ({epochs} epochs) ---")
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Step [{i + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}\n")

    save_path = r"D:\DeepGuard\models\resnet_deepguard.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training finished. Model saved to {save_path}")


if __name__ == "__main__":
    train_resnet()