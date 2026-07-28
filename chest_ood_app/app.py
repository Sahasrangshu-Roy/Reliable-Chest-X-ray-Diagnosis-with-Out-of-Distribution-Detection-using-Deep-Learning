import os
import sys
import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Ensure root folder is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import ALL_LABELS, get_transforms
from src.model import get_densenet_model

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="Chest X-ray OOD Detector",
    page_icon="🫁",
    layout="wide"
)

# ── OOD Threshold Constants (Energy Score stats from Val set) ──
ENERGY_MEAN  = -2.316
ENERGY_STD   = 0.751
ENERGY_LOWER = -2.9461  # 20th percentile
ENERGY_UPPER = -1.6833  # 80th percentile

# ── Load Model Resource ────────────────────────────────────
@st.cache_resource
def load_cached_model(model_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        if os.path.exists(os.path.join(os.path.dirname(__file__), "best_model.pth")):
            model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        else:
            return None, device
    model = get_densenet_model(num_classes=14, pretrained=False, checkpoint_path=model_path, device=device)
    model.eval()
    return model, device

# ── Inference Pipeline ─────────────────────────────────────
def predict_image(image, model, device):
    _, eval_transform = get_transforms()
    tensor = eval_transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    
    with torch.no_grad():
        logits = model(tensor)                              # [1, 14]
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
        energy = -torch.logsumexp(logits, dim=1).item()
        
    return probs, energy

def get_ood_status(energy):
    if energy < ENERGY_LOWER:
        return "OOD (Suspicious / Foreign Pattern)", "red", \
               "⚠️ This image appears suspicious or out-of-distribution. Prediction may be unreliable."
    elif energy > ENERGY_UPPER:
        return "OOD — High Uncertainty", "orange", \
               "⚠️ The model is uncertain about this image. Prediction may be unreliable."
    else:
        return "In Distribution", "green", \
               "✅ Image pattern is familiar (In-Distribution). Prediction is likely reliable."

# ══════════════════════════════════════════════════════════
# STREAMLIT USER INTERFACE
# ══════════════════════════════════════════════════════════

st.title("🫁 Chest X-ray Pathology Classifier with OOD Safety Filter")
st.markdown("""
This web application classifies chest X-rays across **14 common pathologies** using a **DenseNet-121** deep neural network, 
integrated with an **Energy Score Out-Of-Distribution (OOD)** filter to detect foreign or corrupted inputs before clinical reporting.
""")

st.divider()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("📌 System Overview")
    st.markdown("""
    - **Architecture:** DenseNet-121 (ImageNet Pretrained)
    - **Dataset:** NIH ChestX-ray14 (112k images)
    - **OOD Evaluator:** CheXpert
    - **OOD Method:** Energy Score (-logsumexp)
    - **Pathology Classes:** 14 Diseases
    """)
    st.divider()
    st.header("💡 How It Works")
    st.markdown("""
    1. Upload a chest X-ray (`.png`, `.jpg`, `.jpeg`).
    2. DenseNet-121 computes disease logit outputs.
    3. Energy Score detects if image is Out-Of-Distribution.
    4. If OOD, automated prediction is suppressed to protect patient safety.
    """)

# ── Main Layout ───────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Chest X-ray")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image file",
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file:
        model, device = load_cached_model("best_model.pth")
        
        if model is None:
            st.error("❌ Model checkpoint `best_model.pth` was not found!")
            st.info("""
            **First-time Setup Required:**
            The model weights file (`best_model.pth`) is ~0.98 GB and is not included directly in GitHub.
            
            Please run the training script in your terminal to generate the model checkpoint:
            ```bash
            python train.py --data_dir /path/to/nih_dataset
            ```
            """)
        else:
            probs, energy = predict_image(image, model, device)
            ood_status, ood_color, ood_msg = get_ood_status(energy)
            
            st.subheader("OOD Safety Status")
            if ood_color == "green":
                st.success(f"**Status: {ood_status}** (Energy: {energy:.4f})")
                st.markdown(ood_msg)
            elif ood_color == "orange":
                st.warning(f"**Status: {ood_status}** (Energy: {energy:.4f})")
                st.markdown(ood_msg)
            else:
                st.error(f"**Status: {ood_status}** (Energy: {energy:.4f})")
                st.markdown(ood_msg)
                
            st.divider()
            st.subheader("Pathology Predictions")
            
            if ood_color == "red":
                st.error("🚨 Prediction suppressed — Image flagged as Out-of-Distribution. Please consult a qualified radiologist.")
            else:
                # Probability bar plot
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#d9534f' if p > 0.5 else '#4682b4' for p in probs]
                bars = ax.barh(ALL_LABELS, probs, color=colors)
                ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Threshold (0.5)')
                ax.set_xlabel("Probability")
                ax.set_title("Disease Probability Distribution")
                ax.set_xlim(0, 1)
                ax.legend(loc='lower right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Findings list
                findings = [(ALL_LABELS[i], probs[i]) for i in range(14) if probs[i] > 0.5]
                st.subheader("Clinical Summary")
                if findings:
                    for label, prob in sorted(findings, key=lambda x: -x[1]):
                        st.markdown(f"🔴 **{label}** — {prob:.1%} confidence")
                else:
                    st.markdown("✅ **No Finding** — No pathology detected above 50% probability threshold.")
    else:
        st.info("👈 Upload a chest X-ray image from the sidebar or left panel to get started.")
