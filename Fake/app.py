import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image 
import numpy as np
import plotly.graph_objects as go
import cv2
import time


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


# LOAD MODEL

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 2)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()


# TRANSFORM

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])


# GRAD CAM

def generate_gradcam(model, image_tensor):

    gradients = []
    activations = []

    def forward_hook(module,input,output):
        activations.append(output)

    def backward_hook(module,grad_in,grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

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

    return cam

# SIDEBAR
st.sidebar.title("⚙ Model Dashboard")
st.sidebar.write(f"Device: {device}")

total_params = sum(p.numel() for p in model.parameters())
st.sidebar.metric("Total Parameters",f"{total_params:,}")

# TITLE
st.title("🛡️ Explainable Deepfake Detection System")
st.markdown("Deepfake Detection with Visual & Textual Explainability")

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

        with st.spinner("Analyzing image using deep learning model..."):

            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)

            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                probs = torch.softmax(output,dim=1)

        end_time = time.time()
        processing_time = end_time - start_time

        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()

        prediction = "REAL IMAGE" if real_prob>fake_prob else "FAKE IMAGE"
        confidence = max(real_prob,fake_prob)*100

        # RESULT
        col1,col2,col3 = st.columns([1,2,1])
        with col2:

            st.subheader("Prediction Result")

            if prediction=="REAL IMAGE":
                st.success(prediction)
            else:
                st.error(prediction)

            st.write(f"Confidence Score: {confidence:.2f}%")
            st.write(f"Processing Time: {processing_time:.2f} seconds")

        # CONFIDENCE GAUGE
        col1,col2,col3 = st.columns([1,2,1])
        with col2:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text':"Confidence Level"},
                gauge={'axis':{'range':[0,100]}}
            ))

            st.plotly_chart(fig,use_container_width=True)

            # CONFIDENCE INTERPRETATION
            if confidence >= 85:
                st.success("Reliability: High Confidence Prediction")
            elif confidence >= 65:
                st.warning("Reliability: Moderate Confidence – image may contain filters or lighting variations")
            else:
                st.error("Reliability: Low Confidence – prediction uncertain")
                

        # PROBABILITY
        col1,col2,col3 = st.columns([1,2,1])
        with col2:

            st.subheader("Probability Distribution")

            st.progress(int(real_prob*100))
            st.write(f"Real: {real_prob*100:.2f}%")

            st.progress(int(fake_prob*100))
            st.write(f"Fake: {fake_prob*100:.2f}%")

        # HEATMAP COMPARISON
        st.markdown("---")
        st.subheader("Model Attention Analysis")

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
                     caption="Grad-CAM Heatmap",
                     use_container_width=True)

        st.info("""
🔍 Grad-CAM Interpretation Guide

🔴 Red / 🟡 Yellow → High importance regions  
🔵 Blue → Low importance regions
""")

        # AI VERDICT
        st.markdown("---")
        st.subheader("Model Verdict Summary")

        if prediction=="FAKE IMAGE":
            st.warning("""
The model detected irregular visual patterns and texture inconsistencies
which are commonly associated with synthetic or manipulated images.
""")
        else:
            st.success("""
The facial structure and texture distribution appear consistent,
indicating a natural and authentic image.
""")

        # MODEL INFO
        st.markdown("---")
        st.subheader("Model Information")

        col1,col2 = st.columns(2)

        with col1:
            st.write("Architecture: EfficientNet-B0")
            st.write("Framework: PyTorch")

        with col2:
            st.write("Dataset Size: 120,000 images")
            st.write("Explainability Method: Grad-CAM")

        # DOWNLOAD REPORT
        st.markdown("---")

        report = f"""
Deepfake Detection Report
Prediction: {prediction}
Confidence: {confidence:.2f}%
Real Probability: {real_prob*100:.2f}%
Fake Probability: {fake_prob*100:.2f}%
Processing Time: {processing_time:.2f} seconds
"""

        st.download_button("Download Report",report,"prediction_report.txt")

st.markdown("---")
st.markdown("Final Year Deep Learning Project | PyTorch + Streamlit | RTX Optimized")