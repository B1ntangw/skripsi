import streamlit as st
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
import tensorflow as tf

# ======================== CONFIG ==========================
st.set_page_config(page_title="Tomato Leaf Disease Classifier", page_icon="🍅", layout="wide")

MODEL_PATH = Path("models/model_tomat.h5")
LABEL_PATH = Path("models/class_labels.json")
IMG_SIZE = (256, 256)  # Sesuaikan dengan input model

# ====================== LOAD MODEL =========================
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# ===================== LOAD LABELS =========================
def load_labels():
    if LABEL_PATH.exists():
        with open(LABEL_PATH, "r") as f:
            data = json.load(f)
        return data.get("classes", [])
    return []

class_labels = load_labels()

# ================== PREDICT FUNCTION =======================
def preprocess_image(image: Image.Image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:  # remove alpha channel
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

def predict_image(img: Image.Image):
    if model is None:
        return None, None
    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)[0]
    return preds, class_labels[np.argmax(preds)] if class_labels else str(np.argmax(preds))

# ======================= SIDEBAR ===========================
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih Menu", ["Beranda", "Prediksi Tunggal", "Prediksi Batch", "Tentang"])

# ====================== MAIN PAGE ==========================
if menu == "Beranda":
    st.title("🍅 Tomato Leaf Disease Classifier")
    st.write("Aplikasi ini menggunakan model *Deep Learning* untuk mengklasifikasi penyakit pada daun tomat.")
    if model is not None:
        st.success("Model berhasil dimuat!")
    else:
        st.error("Model belum tersedia. Pastikan file `model_tomat.h5` ada di folder models.")
    if class_labels:
        st.write("**Daftar Kelas:**")
        for idx, lbl in enumerate(class_labels):
            st.write(f"{idx}. {lbl}")

elif menu == "Prediksi Tunggal":
    st.title("Prediksi Gambar Tunggal")
    uploaded_file = st.file_uploader("Upload Gambar Daun Tomat", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diupload", use_column_width=True)
        if st.button("Prediksi"):
            preds, label = predict_image(img)
            if preds is not None:
                st.subheader(f"Prediksi: {label}")
                st.bar_chart(preds)

elif menu == "Prediksi Batch":
    st.title("Prediksi Batch")
    files = st.file_uploader("Upload Beberapa Gambar", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files:
        results = []
        for file in files:
            img = Image.open(file).convert("RGB")
            preds, label = predict_image(img)
            results.append({"Nama File": file.name, "Prediksi": label})
        st.write("Hasil Prediksi:")
        st.dataframe(results)

elif menu == "Tentang":
    st.title("Tentang Aplikasi")
    st.write("Aplikasi ini dibangun menggunakan Streamlit dan TensorFlow/Keras.")
    st.write("Model `model_tomat.h5` berasal dari training pada dataset penyakit daun tomat (PlantVillage).")