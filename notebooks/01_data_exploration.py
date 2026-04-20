"""
01 — Data Exploration

Quick sanity checks before training: class balance and preprocessing
pipeline verification.  Run this first on a fresh dataset download.
"""

import os

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DATA_PATH = r"D:\DeepGuard\data\raw"
REAL_PATH = os.path.join(RAW_DATA_PATH, "Real")
FAKE_PATH = os.path.join(RAW_DATA_PATH, "Fake")


def check_data_balance():
    """Print real/fake counts and plot the class distribution."""
    print("Checking data distribution...")
    try:
        real_count = len(os.listdir(REAL_PATH))
        fake_count = len(os.listdir(FAKE_PATH))
    except FileNotFoundError:
        print(f"Error: Could not find Real/Fake folders in {RAW_DATA_PATH}")
        return

    print(f"Real Images: {real_count}")
    print(f"Fake Images: {fake_count}")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=["Real", "Fake"], y=[real_count, fake_count], palette="mako")
    plt.title("Dataset Class Distribution (Real vs. Fake)")
    plt.ylabel("Number of Images")
    plt.show()


def process_and_show_sample():
    """Load one real image, resize + normalise it, and display both stages."""
    print("Testing image preprocessing pipeline...")
    try:
        sample_file = os.listdir(REAL_PATH)[0]
        sample_path = os.path.join(REAL_PATH, sample_file)
    except (FileNotFoundError, IndexError):
        print("No images found in the 'Real' folder to process.")
        return

    img = cv2.imread(sample_path)
    # opencv loads BGR by default — flip to RGB for display & training
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # standardize to 128×128 — same dims used by the model
    img_resized = cv2.resize(img_rgb, (128, 128))

    # scale pixels to [0, 1] — keeps gradient updates stable
    img_normalized = img_resized / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original Image\nShape: {img_rgb.shape}")
    axes[0].axis("off")

    axes[1].imshow(img_normalized)
    axes[1].set_title(f"Processed for AI\nShape: {img_normalized.shape}")
    axes[1].axis("off")

    plt.suptitle("Phase 1: DeepGuard Preprocessing Test")
    plt.show()


if __name__ == "__main__":
    check_data_balance()
    process_and_show_sample()