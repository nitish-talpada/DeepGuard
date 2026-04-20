# DeepGuard: AI-Generated Media Detector

A binary image classifier that distinguishes authentic human portraits from AI-generated deepfakes. Built with a fine-tuned **ResNet-18** backbone, served through a **Streamlit** web interface, and gated by an **OpenCV** Haar Cascade face detector that rejects non-portrait inputs before inference.

Developed as a university capstone project exploring transfer learning for media forensics.

---

## Demo

Upload any face image through the Streamlit UI. DeepGuard will:

1. **Gate** — Verify a human face is present (OpenCV Haar Cascade).
2. **Classify** — Run the image through the fine-tuned ResNet-18.
3. **Report** — Display a REAL / FAKE verdict with a confidence score.

---

## Tech Stack

| Layer | Tool |
| --- | --- |
| Deep Learning | PyTorch, torchvision (ResNet-18) |
| Face Detection | OpenCV (Haar Cascade pre-filter) |
| Web UI | Streamlit |
| GPU Accel. | torch-directml (AMD GPU support) |
| Language | Python 3.10+ |

---

## Project Structure

```text
DeepGuard/
├── app.py                          # Streamlit inference app
├── models/
│   ├── resnet_deepguard.pt         # Production weights (ResNet-18)
│   └── custom_cnn.pt              # Baseline CNN weights
├── notebooks/
│   ├── 01_data_exploration.py     # Class balance and preprocessing checks
│   ├── 02_cnn_architecture.py     # Baseline CNN definition
│   ├── 03_train_cnn.py            # Baseline CNN training loop
│   ├── 04_transfer_learning.py    # ResNet-18 fine-tuning (production)
│   ├── 05_evaluate_model.py       # Test-set evaluation and metrics
│   └── 06_data_merger.py          # Multi-dataset merge utility
├── data/
│   ├── train/                     # Training split (real/ and fake/)
│   ├── test/                      # Test split (real/ and fake/)
│   └── valid/                     # Validation split (real/ and fake/)
├── requirements.txt
└── README.md
```

---

## Dataset Strategy

The training data is a **hybrid dataset** assembled from three public sources to improve generalisation and reduce demographic bias:

| Dataset | Role | Why |
| --- | --- | --- |
| **FaceForensics++** | Fake and Real faces | Industry-standard benchmark covering face-swap and reenactment methods |
| **Celeb-DF (v2)** | Fake and Real faces | Higher-quality synthesis that challenges simpler detectors |
| **FairFace** | Real faces (diversity) | Adds age, gender, and ethnic diversity to the real class to prevent bias |

Images were flattened into a single `train/Real` and `train/Fake` directory using the `06_data_merger.py` utility script. Data augmentation (random flips, rotation, colour jitter, Gaussian blur) is applied at training time to further reduce overfitting.

---

## Model Architecture

- **Backbone**: ResNet-18 pre-trained on ImageNet (frozen convolutional layers).
- **Head**: Custom fully-connected classifier:

```text
Linear(512 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 1) -> Sigmoid
```

- **Loss**: Binary Cross-Entropy (BCELoss).
- **Optimiser**: Adam (lr = 0.001), applied only to the classification head.
- **Training**: 15 epochs, batch size 128, on AMD RX 6800M via DirectML.

A baseline 2-layer CNN (`02_cnn_architecture.py` / `03_train_cnn.py`) was trained first to establish a performance floor before moving to transfer learning.

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/nitish-talpada/DeepGuard.git
cd DeepGuard
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:
```bash
venv\Scripts\activate
```

On macOS / Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `torch-directml` is only needed for AMD GPU acceleration. On NVIDIA hardware, replace it with the appropriate CUDA-enabled PyTorch build. On CPU-only machines, remove `torch-directml` and edit `app.py` to use `torch.device("cpu")`.

### 4. Download or place model weights

Place `resnet_deepguard.pt` inside the `models/` directory. If you are training from scratch, run the pipeline scripts in order:

```bash
python notebooks/01_data_exploration.py
python notebooks/03_train_cnn.py
python notebooks/04_transfer_learning.py
python notebooks/05_evaluate_model.py
```

### 5. Run the app

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`.

---

## Evaluation Results

Evaluation is performed on a held-out test split using `05_evaluate_model.py`. The script produces:

- **Classification Report** — per-class precision, recall, and F1-score.
- **Confusion Matrix** — visual heatmap of true/false positives and negatives.

Target performance: **>= 80% accuracy** on the test set.

---

## Limitations

- **Face-only**: The Haar Cascade gate rejects images without a clearly visible frontal face. Profile shots and heavy occlusion may be filtered out.
- **Training distribution**: Performance on generation methods not represented in the training data (e.g., very recent diffusion models) may be lower.
- **Hardware dependency**: The current `app.py` is configured for AMD DirectML. Switching to CUDA or CPU requires a one-line device change.

---

## License

This project was developed for academic purposes. See individual dataset licenses (FaceForensics++, Celeb-DF, FairFace) for data usage terms.
