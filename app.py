
import streamlit as st
from helpers.functions import load_model, predict_image, preprocess_image
from helpers.metrics import plot_confusion_matrix, get_metrics
from utils.label_map import CLASS_NAMES
import os
from PIL import Image

st.set_page_config(page_title='Eye Disease Detector', layout='wide')
st.title("👁️ Eye Disease Detection App using EyeNet & Swin Transformer")

model_type = st.sidebar.selectbox("Select Model", ["EyeNet", "Swin Transformer", "Hybrid"])
uploaded_image = st.file_uploader("Upload an Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Eye Image", use_column_width=True)

    st.subheader("Prediction Result")
    model = load_model(model_type)
    img_tensor = preprocess_image(image)
    pred_class, prob = predict_image(model, img_tensor)

    st.success(f"Predicted Class: **{CLASS_NAMES[pred_class]}** with confidence **{prob*100:.2f}%**")

    st.subheader("Model Performance")
    acc, cm = get_metrics(model)
    st.write(f"Model Accuracy: {acc*100:.2f}%")
    plot_confusion_matrix(cm, CLASS_NAMES)
