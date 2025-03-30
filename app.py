import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("medical_imaging_model.keras")

st.title("AI-Powered Medical Imaging System")
st.write("Upload a medical image (X-ray, MRI) for analysis.")

uploaded_file = st.file_uploader("Choose a PNG/JPG image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using the model
    prediction = model.predict(img_array)
    class_names = ["Normal", "Abnormal"]
    result = class_names[np.argmax(prediction)]

    st.image(image, caption=f"Prediction: {result}", use_column_width=True)
