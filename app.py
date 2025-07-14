import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "model/plant_disease_model.h5"

st.title("🌿 Plant Disease Detector")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Train the model or place weights at model/plant_disease_model.h5")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
class_names = [
    "Tomato_Healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Leaf_Curl_Virus",
    "Tomato_Bacterial_spot"
]

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)

    img = img.resize((128, 128))
    arr = np.expand_dims(np.array(img) / 255.0, 0)
    preds = model.predict(arr)[0]
    pred_class = class_names[int(np.argmax(preds))]
    confidence = np.max(preds) * 100

    st.success(f"**{pred_class}**  ({confidence:.2f} %)")
