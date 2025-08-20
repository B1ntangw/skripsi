import streamlit as st
import numpy as np
import tensorflow as tf
import json
import pandas as pd
from PIL import Image

# -------------------------
# 1. Load Model
# -------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("models/model_tomat.keras")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# -------------------------
# 2. Load Class Labels dari JSON
# -------------------------
def load_class_labels():
    try:
        with open("models/class_labels.json", "r") as f:
            labels = json.load(f)
        return labels
    except Exception as e:
        st.error(f"Gagal memuat class labels: {e}")
        return []

class_labels = load_class_labels()

# -------------------------
# 3. Preprocess Image
# -------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 256, 256, 3)
    return img_array

# -------------------------
# 4. Predict Function
# -------------------------
def predict_image(model, img_array, class_labels):
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    label = class_labels[idx]
    confidence = float(preds[0][idx])
    return label, confidence, preds[0]

# -------------------------
# 5. Streamlit UI
# -------------------------
st.title("🌱 Deteksi Penyakit Daun Tomat")

uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None and class_labels:
    # Preprocess
    img_array = preprocess_image(uploaded_file)

    # Predict
    label, confidence, preds = predict_image(model, img_array, class_labels)

    # Tampilkan hasil
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)
    st.success(f"Prediksi: **{label}** dengan confidence **{confidence:.2f}**")

    # Visualisasi semua probabilitas kelas
    df_preds = pd.DataFrame([preds], columns=class_labels)
    st.bar_chart(df_preds)
else:
    st.info("Silakan upload gambar untuk memulai deteksi.")
