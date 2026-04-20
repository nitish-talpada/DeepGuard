"""
DeepGuard AI — Streamlit inference app.

Loads a fine-tuned ResNet-18 binary classifier and exposes a simple
upload-and-predict UI for deepfake detection.  An OpenCV Haar Cascade
pre-filter rejects images that contain no detectable face, saving GPU
cycles on irrelevant inputs.
"""

import os
import warnings

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch_directml
from PIL import Image
from torchvision import models, transforms

# keep the terminal clean during demo sessions
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_RESNET_WEIGHTS = os.path.join(_MODEL_DIR, "resnet_deepguard.pt")


# ---------------------------------------------------------------------------
# Face gate — reject non-portrait uploads before hitting the model
# ---------------------------------------------------------------------------
def detect_face(image):
    """Run a Haar Cascade face check on the uploaded image.

    Args:
        image: PIL Image in any colour mode.

    Returns:
        True if at least one face is found, False otherwise.
    """
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # haar cascade is fast enough for a single-image gate
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    return len(faces) > 0


# ---------------------------------------------------------------------------
# Streamlit page config (must be first st.* call)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DeepGuard AI", page_icon="🛡️", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stHeader { color: #00d4ff; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loader — cached so it only runs once per session
# ---------------------------------------------------------------------------
@st.cache_resource
def load_deepguard_model():
    """Rebuild the ResNet-18 head and load trained weights.

    Returns:
        Tuple of (model, device) ready for inference.
    """
    device = torch_directml.device()

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features

    # custom classification head — must match 04_transfer_learning.py exactly
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

    # weights_only=False needed for DirectML-serialised tensors
    model.load_state_dict(torch.load(_RESNET_WEIGHTS, weights_only=False))
    model = model.to(device)
    model.eval()
    return model, device


model, device = load_deepguard_model()


# ---------------------------------------------------------------------------
# Image pre-processing — match training transforms
# ---------------------------------------------------------------------------
def preprocess_image(image):
    """Resize, tensor-ise, and normalise a PIL image for ResNet.

    Args:
        image: PIL Image (RGB).

    Returns:
        Tensor of shape (1, 3, 128, 128) on the active device.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # imagenet stats — same as training pipeline
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(image).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🛡️ DeepGuard AI")
st.subheader("High-Performance Deepfake Detection")
st.write("Upload an image to verify its authenticity using our ResNet-18 Neural Engine.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if not detect_face(image):
        st.error("❌ No human face detected. DeepGuard only analyzes human portraits.")
    else:
        st.success("✅ Human face detected. Proceeding with Deepfake Analysis...")

        with st.spinner("Analyzing pixel artifacts..."):
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)
                confidence = output.item()

            # sigmoid output: >0.5 → real (class 1), ≤0.5 → fake (class 0)
            is_real = confidence > 0.5
            score = confidence if is_real else (1 - confidence)

        st.divider()

        if is_real:
            st.success("### VERDICT: REAL (Authentic)")
            st.info(f"Confidence Score: {score:.2%}")
        else:
            st.error("### VERDICT: FAKE (AI Generated)")
            st.warning(f"Detection Confidence: {score:.2%}")

        st.progress(score)

st.sidebar.title("System Stats")
st.sidebar.info("Backend: AMD RX 6800M\nModel: ResNet-18 Turbo\nStatus: Online")