"""
06 — Data Merger

Flattens and moves images from multiple source datasets (FaceForensics++,
Celeb-DF, FairFace) into a single train/Real and train/Fake directory
structure that ImageFolder expects.

NOTE: This script moves (not copies) files to save disk space.  The
source directories will be empty after a successful run.  Sections that
have already been executed are commented out to prevent accidental re-runs.
"""

import os
import shutil
from pathlib import Path

MASTER_REAL = r"D:\DeepGuard\data\train\Real"
MASTER_FAKE = r"D:\DeepGuard\data\train\Fake"

os.makedirs(MASTER_REAL, exist_ok=True)
os.makedirs(MASTER_FAKE, exist_ok=True)


def flatten_and_move(source_dir, target_dir, prefix):
    """Recursively find images in source_dir and move them to target_dir.

    Files are renamed with a sequential counter so names never collide
    across datasets.

    Args:
        source_dir: Root directory to scan (searched recursively).
        target_dir: Flat output directory.
        prefix: String prepended to each filename for traceability.
    """
    print(f"\nScanning {source_dir}...")
    count = 0

    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for filepath in Path(source_dir).rglob(ext):
            new_filename = f"{prefix}_{count}{filepath.suffix}"
            target_path = os.path.join(target_dir, new_filename)

            shutil.move(str(filepath), target_path)
            count += 1

            if count % 5000 == 0:
                print(f"  Moved {count} images...")

    print(f"Finished. Total moved from {prefix}: {count}")


if __name__ == "__main__":
    print("Starting Data Merger...")

    # --- Already executed (kept for reference) ---
    # ff_fake_src = r"D:\DeepGuard\data\kaggle\FaceForensics++\fake"
    # flatten_and_move(ff_fake_src, MASTER_FAKE, "FF_Fake")
    # ff_real_src = r"D:\DeepGuard\data\kaggle\FaceForensics++\real"
    # flatten_and_move(ff_real_src, MASTER_REAL, "FF_Real")
    # fairface_src = r"D:\DeepGuard\data\kaggle\FairFace\FairFace\train"
    # flatten_and_move(fairface_src, MASTER_REAL, "FairFace")

    # --- Celeb-DF ---
    print("Moving Celeb-DF Fakes...")
    celeb_fake_src = r"D:\DeepGuard\data\kaggle\Celeb-DF v2\Celeb_V2\Train\fake"
    flatten_and_move(celeb_fake_src, MASTER_FAKE, "CelebDF_Fake")

    print("Moving Celeb-DF Reals...")
    celeb_real_src = r"D:\DeepGuard\data\kaggle\Celeb-DF v2\Celeb_V2\Train\real"
    flatten_and_move(celeb_real_src, MASTER_REAL, "CelebDF_Real")

    print("\nData flattening complete. Ready for training.")