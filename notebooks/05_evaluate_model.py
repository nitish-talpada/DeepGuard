"""
05 — Model Evaluation

Loads the trained ResNet-18 and evaluates it against the held-out test
split.  Generates a classification report (precision / recall / F1) and
a confusion matrix heatmap.
"""

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch_directml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
device = torch_directml.device()
print(f"Evaluation device: {device}")

# ---------------------------------------------------------------------------
# Test data — NO augmentation, just resize + normalise (same as inference)
# ---------------------------------------------------------------------------
TEST_DIR = r"D:\DeepGuard\data\test"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("Loading test dataset...")
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ---------------------------------------------------------------------------
# Rebuild architecture — must mirror 04_transfer_learning.py exactly
# ---------------------------------------------------------------------------
print("Rebuilding model architecture...")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid(),
)

MODEL_PATH = r"D:\DeepGuard\models\resnet_deepguard.pt"
model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
model = model.to(device)
model.eval()

# ---------------------------------------------------------------------------
# Inference on test set
# ---------------------------------------------------------------------------
all_preds = []
all_labels = []

print("Running evaluation on test images...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        preds = (outputs >= 0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 30)
print("DEEPGUARD EVALUATION RESULTS")
print("=" * 30)

target_names = ["Fake", "Real"]
print(classification_report(all_labels, all_preds, target_names=target_names))

# confusion matrix heatmap
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix — DeepGuard ResNet-18")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()