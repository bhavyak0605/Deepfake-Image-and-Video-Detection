import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import cv2
import time
import json


# PAGE CONFIG

st.set_page_config(
    page_title="Explainable Deepfake Detection System",
    layout="wide",
    page_icon="🛡️"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

.stApp {
    background: linear-gradient(135deg,#0f172a,#020617);
    color:white;
}

[data-testid="stSidebar"] {
    background:#020617;
}

.stButton>button {
    border-radius:10px;
    background:linear-gradient(90deg,#3b82f6,#06b6d4);
    color:white;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL 1 (EfficientNet-B0)

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 2)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()


# LOAD MODEL 2 (ResNet50) - runs in parallel with Model 1, independent opinion

@st.cache_resource
def load_model2():
    model2 = models.resnet50(weights=None)
    model2.fc = nn.Linear(2048, 2)
    model2.load_state_dict(torch.load("best_model2.pth", map_location=device))
    model2.to(device)
    model2.eval()
    return model2

try:
    model2 = load_model2()
    model2_available = True
except FileNotFoundError:
    model2 = None
    model2_available = False


# TRANSFORM (same preprocessing for both models)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# a second, blurred view of the image used for test-time augmentation (see predict_robust)
from PIL import ImageFilter
transform_blur = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda im: im.filter(ImageFilter.GaussianBlur(1.0))),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# CALIBRATION (temperature scaling) - makes confidence scores honest.
# Fit once by calibrate.py; does NOT change which class is predicted, only how
# extreme the confidence number is allowed to be. Defaults to 1.0 (no change)
# if calibration.json is missing, so the app still works without it.
try:
    with open("calibration.json") as _f:
        TEMPERATURE = json.load(_f)
except FileNotFoundError:
    TEMPERATURE = {"model1": 1.0, "model2": 1.0}


def predict_robust(model, image, temperature):
    """Test-time augmentation + calibrated confidence.
    Averages softmax probabilities from the image and a slightly blurred copy
    (two independent 'looks' at the same image), which smooths out spurious,
    overconfident errors a single forward pass can make - then applies the
    fitted temperature so the final confidence number is realistic rather
    than a false 100%. Returns (fake_prob, real_prob, image_tensor_for_gradcam).
    """
    x_sharp = transform(image).unsqueeze(0).to(device)
    x_blur = transform_blur(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_sharp = model(x_sharp)
        logits_blur = model(x_blur)
        p_sharp = torch.softmax(logits_sharp / temperature, dim=1)
        p_blur = torch.softmax(logits_blur / temperature, dim=1)
        probs = (p_sharp + p_blur) / 2
    return probs[0][0].item(), probs[0][1].item(), x_sharp


# GRAD CAM (Model 1 only)

def generate_gradcam(model, image_tensor):

    gradients = []
    activations = []

    def forward_hook(module,input,output):
        activations.append(output)

    def backward_hook(module,grad_in,grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    class_idx = output.argmax()

    model.zero_grad()
    output[0,class_idx].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads,axis=(1,2))
    cam = np.zeros(acts.shape[1:],dtype=np.float32)

    for i,w in enumerate(weights):
        cam += w*acts[i]

    cam = np.maximum(cam,0)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    h1.remove()
    h2.remove()

    return cam

# SIDEBAR
st.sidebar.title("⚙ Model Dashboard")
st.sidebar.write(f"Device: {device}")

total_params = sum(p.numel() for p in model.parameters())
st.sidebar.metric("Model 1 Parameters (EfficientNet-B0)",f"{total_params:,}")

if model2_available:
    total_params2 = sum(p.numel() for p in model2.parameters())
    st.sidebar.metric("Model 2 Parameters (ResNet50)",f"{total_params2:,}")
else:
    st.sidebar.warning("Model 2 (best_model2.pth) not found - showing Model 1 only")

# TITLE
st.title("🛡️ Explainable Deepfake Detection System")
st.markdown("Dual-Model Deepfake Detection with Visual & Textual Explainability")

# FILE UPLOAD
uploaded = st.file_uploader("Upload Image",type=["jpg","jpeg","png","webp"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        st.image(image,caption="Uploaded Image",use_container_width=True)

    st.markdown("---")

    col1,col2,col3 = st.columns([2,1,2])
    with col2:
        analyze = st.button("🔍 Analyze Image",use_container_width=True)

    if analyze:

        start_time = time.time()

        with st.spinner("Analyzing image using both models..."):

            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)

            # MODEL 1 - EfficientNet-B0 (test-time augmentation + calibrated confidence)
            fake_prob1, real_prob1, image_tensor = predict_robust(model, image, TEMPERATURE.get("model1",1.0))

            # MODEL 2 - ResNet50 (runs independently, in parallel, on the SAME image)
            if model2_available:
                fake_prob2, real_prob2, _ = predict_robust(model2, image, TEMPERATURE.get("model2",1.0))

        end_time = time.time()
        processing_time = end_time - start_time

        prediction1 = "REAL IMAGE" if real_prob1>fake_prob1 else "FAKE IMAGE"
        confidence1 = max(real_prob1,fake_prob1)*100

        if model2_available:
            prediction2 = "REAL IMAGE" if real_prob2>fake_prob2 else "FAKE IMAGE"
            confidence2 = max(real_prob2,fake_prob2)*100

        # ---- SIDE-BY-SIDE MODEL COMPARISON ----
        st.markdown("---")
        st.subheader("Prediction Comparison")
        st.caption(f"Both models analyzed the same image independently | Total processing time: {processing_time:.2f} seconds")

        if model2_available:
            colA, colB = st.columns(2)
        else:
            colA = st.container()

        with colA:
            st.markdown("### Model 1: EfficientNet-B0")
            if prediction1=="REAL IMAGE":
                st.success(prediction1)
            else:
                st.error(prediction1)
            st.write(f"Confidence: {confidence1:.2f}%")

            fig1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence1,
                title={'text':"Confidence"},
                gauge={'axis':{'range':[0,100]}}
            ))
            st.plotly_chart(fig1,use_container_width=True,key="gauge1")

            st.progress(int(real_prob1*100))
            st.write(f"Real: {real_prob1*100:.2f}%")
            st.progress(int(fake_prob1*100))
            st.write(f"Fake: {fake_prob1*100:.2f}%")

        if model2_available:
            with colB:
                st.markdown("### Model 2: ResNet50")
                if prediction2=="REAL IMAGE":
                    st.success(prediction2)
                else:
                    st.error(prediction2)
                st.write(f"Confidence: {confidence2:.2f}%")

                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence2,
                    title={'text':"Confidence"},
                    gauge={'axis':{'range':[0,100]}}
                ))
                st.plotly_chart(fig2,use_container_width=True,key="gauge2")

                st.progress(int(real_prob2*100))
                st.write(f"Real: {real_prob2*100:.2f}%")
                st.progress(int(fake_prob2*100))
                st.write(f"Fake: {fake_prob2*100:.2f}%")

            # Agreement check
            st.markdown("---")
            if prediction1 == prediction2:
                st.info(f"✅ Both models agree: **{prediction1}**")
            else:
                st.warning(f"⚠️ Models disagree — Model 1 says **{prediction1}**, Model 2 says **{prediction2}**. Treat this image with extra caution.")

        # HEATMAP (Model 1 only)
        st.markdown("---")
        st.subheader("Model Attention Analysis (Model 1 - EfficientNet-B0)")
        st.caption("Grad-CAM is currently implemented for Model 1 only.")

        cam = generate_gradcam(model,image_tensor)
        original_np = np.array(image)

        cam_resized = cv2.resize(cam,(original_np.shape[1],original_np.shape[0]))

        heatmap = cv2.applyColorMap(np.uint8(255*cam_resized),cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)

        overlay = heatmap*0.4 + original_np*0.6

        col1,col2 = st.columns(2)

        with col1:
            st.image(image,caption="Original Image",use_container_width=True)

        with col2:
            st.image(np.uint8(overlay),
                     caption="Grad-CAM Heatmap (Model 1)",
                     use_container_width=True)

        st.info("""
🔍 Grad-CAM Interpretation Guide

🔴 Red / 🟡 Yellow → High importance regions
🔵 Blue → Low importance regions
""")

        # MODEL INFO
        st.markdown("---")
        st.subheader("Model Information")

        col1,col2 = st.columns(2)

        with col1:
            st.write("Model 1 Architecture: EfficientNet-B0")
            st.write("Model 2 Architecture: ResNet50" if model2_available else "Model 2: not loaded")
            st.write("Framework: PyTorch")

        with col2:
            st.write("Dataset: 140k Real and Fake Faces (FFHQ + StyleGAN)")
            st.write("Explainability Method: Grad-CAM (Model 1)")

        # DOWNLOAD REPORT
        st.markdown("---")

        if model2_available:
            report = f"""
Deepfake Detection Report (Dual-Model)

Model 1 - EfficientNet-B0
Prediction: {prediction1}
Confidence: {confidence1:.2f}%
Real Probability: {real_prob1*100:.2f}%
Fake Probability: {fake_prob1*100:.2f}%

Model 2 - ResNet50
Prediction: {prediction2}
Confidence: {confidence2:.2f}%
Real Probability: {real_prob2*100:.2f}%
Fake Probability: {fake_prob2*100:.2f}%

Agreement: {"Yes" if prediction1 == prediction2 else "No - models disagree"}
Processing Time: {processing_time:.2f} seconds
"""
        else:
            report = f"""
Deepfake Detection Report

Model 1 - EfficientNet-B0
Prediction: {prediction1}
Confidence: {confidence1:.2f}%
Real Probability: {real_prob1*100:.2f}%
Fake Probability: {fake_prob1*100:.2f}%
Processing Time: {processing_time:.2f} seconds

(Model 2 was not available for this run)
"""

        st.download_button("Download Report",report,"prediction_report.txt")

st.markdown("---")
st.markdown("Final Year Deep Learning Project | PyTorch + Streamlit | Dual-Model (EfficientNet-B0 + ResNet50)")