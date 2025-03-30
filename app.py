import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet")

def preprocess_image(img):
    """Preprocess the uploaded medical image for model prediction."""
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_disease(img):
    """Predict the medical condition from the uploaded X-ray/MRI."""
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Streamlit Web Interface
st.title("AI-Powered Medical Imaging Analysis")
st.write("Upload an X-ray, CT, or MRI scan to get a prediction.")

# File uploader
uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(img, caption="Uploaded Medical Image", use_column_width=True)

    # Predict
    predictions = predict_disease(img)

    # Display predictions
    st.subheader("AI Predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i + 1}. **{label}**: {score * 100:.2f}%")
