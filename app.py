import streamlit as st
import numpy as np
from PIL import Image
import json
from pathlib import Path
import tensorflow as tf
from streamlit_option_menu import option_menu
import base64
import pandas as pd
from typing import Optional, Tuple, List

# ======================== CONFIG ==========================
st.set_page_config(
    page_title="🍅 Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="wide",
)

MODEL_PATH = Path("models/model_tomat.h5")  # bisa .h5 atau .keras
LABEL_PATH = Path("models/class_labels.json")
# fallback jika tidak bisa baca dari model
DEFAULT_IMG_SIZE: Tuple[int, int] = (256, 256)

# ====================== UTILITIES =========================
def _infer_input_size_from_model(m: tf.keras.Model) -> Tuple[int, int]:
    """Coba ambil ukuran input (height, width) dari model; jika gagal pakai default."""
    try:
        shape = getattr(m, "input_shape", None)
        # Contoh shape: (None, 256, 256, 3)
        if shape is None:
            return DEFAULT_IMG_SIZE
        # Beberapa model punya list of shapes (multi-input)
        if isinstance(shape, (list, tuple)) and isinstance(shape[0], (list, tuple)):
            # ambil input pertama
            shape = shape[0]
        if len(shape) >= 3:
            h, w = int(shape[1]), int(shape[2])
            if h > 0 and w > 0:
                return (h, w)
    except Exception:
        pass
    return DEFAULT_IMG_SIZE

# ====================== LOAD MODEL =========================
@st.cache_resource(show_spinner=True)
def load_model() -> Optional[tf.keras.Model]:
    """Load model Keras. compile=False untuk hindari ketidakcocokan optimizer/loss."""
    try:
        # Coba load apa adanya
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e1:
        # Jika gagal dan ekstensi .h5, coba opsi penyelamatan: SavedModel folder (tanpa ekstensi)
        st.warning(f"Gagal memuat {MODEL_PATH.name} dengan pesan: {e1}. Mencoba format alternatif…")
        try:
            alt_path = MODEL_PATH.with_suffix("")
            if alt_path.exists():
                model = tf.keras.models.load_model(str(alt_path), compile=False)
                return model
        except Exception as e2:
            st.error(f"Gagal memuat model (format alternatif): {e2}")
    return None

model = load_model()

# Tentukan IMG_SIZE dari model jika memungkinkan
IMG_SIZE: Tuple[int, int] = _infer_input_size_from_model(model) if model is not None else DEFAULT_IMG_SIZE

# ===================== LOAD LABELS =========================
def load_labels() -> List[str]:
    """Membaca label dari JSON. Mendukung dua format:
    1) {"classes": ["Tomato___Bacterial_spot", ...]}
    2) ["Tomato___Bacterial_spot", ...]
    """
    if LABEL_PATH.exists():
        try:
            with open(LABEL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                labels = data.get("classes", [])
                if isinstance(labels, list):
                    return labels
                else:
                    st.warning("Key 'classes' ada tapi bukan list. Mengabaikan.")
                    return []
            elif isinstance(data, list):
                return data
            else:
                st.warning("Format file label tidak dikenali. Gunakan list atau dict dengan key 'classes'.")
        except Exception as e:
            st.error(f"Gagal membaca label: {e}")
    return []

class_labels: List[str] = load_labels()

# ================= LABEL CLEANER ===========================
def clean_label(lbl: str) -> str:
    lbl = lbl.replace("Tomato", "").replace("___", " ").strip()
    lbl = lbl.replace("_", " ")
    return " ".join(part.capitalize() for part in lbl.split())

# ================== CLASS IMAGE MAPPING ====================
CLASS_IMAGES = {
    "Bacterial Spot": "img/bacterial_spot.JPG",
    "Early Blight": "img/early_blight.JPG",
    "Late Blight": "img/late_blight.JPG",
    "Leaf Mold": "img/leaf_mold.JPG",
    "Septoria Leaf Spot": "img/septoria_leaf_spot.JPG",
    "Spider Mites": "img/spider_mites.JPG",
    "Target Spot": "img/target_spot.JPG",
    "Yellow Leaf Curl Virus": "img/yellow_leaf_curl_virus.JPG",
    "Mosaic Virus": "img/mosaic_virus.JPG",
    "Healthy": "img/healthy.JPG",
}

CLASS_DESCRIPTIONS = {
    "Bacterial Spot": (
        "Penyakit ini disebabkan oleh bakteri *Xanthomonas campestris pv. vesicatoria*. "
        "Gejalanya berupa bercak kecil berwarna cokelat kehitaman pada daun yang terkadang dikelilingi halo kuning. "
        "Infeksi juga dapat menyebar ke buah. Dampaknya adalah daun mudah rontok sehingga proses fotosintesis terganggu dan hasil panen berkurang."
    ),
    "Early Blight": (
        "Penyakit bercak daun awal ini disebabkan oleh jamur *Alternaria solani*. "
        "Gejalanya berupa bercak cokelat dengan pola lingkaran konsentris menyerupai cincin pada daun tua. "
        "Seiring waktu, daun menguning dan rontok. Dampaknya membuat luas daun hijau berkurang, buah lebih kecil, dan tanaman menjadi lemah."
    ),
    "Late Blight": (
        "Penyakit hawar daun lanjut disebabkan oleh jamur semu *Phytophthora infestans*. "
        "Gejalanya berupa bercak hijau gelap atau cokelat berair pada daun yang cepat meluas. "
        "Pada kondisi lembab sering muncul lapisan jamur putih di tepi bercak. "
        "Penyakit ini sangat merusak karena dapat membunuh tanaman hanya dalam beberapa hari."
    ),
    "Leaf Mold": (
        "Penyakit bercak jamur ini disebabkan oleh *Passalora fulva*. "
        "Gejalanya berupa bercak kuning di permukaan atas daun, sementara di bagian bawah daun muncul lapisan beludru berwarna hijau keabu-abuan. "
        "Akibatnya, daun mengering dan tanaman kekurangan energi untuk tumbuh optimal."
    ),
    "Septoria Leaf Spot": (
        "Penyakit ini disebabkan oleh jamur *Septoria lycopersici*. "
        "Gejalanya berupa bercak kecil berbentuk bulat berwarna cokelat dengan pusat abu-abu pucat, biasanya muncul pada daun tua. "
        "Penyakit ini sering mempercepat kerontokan daun, terutama pada lingkungan dengan kelembaban tinggi."
    ),
    "Spider Mites": (
        "Gangguan ini disebabkan oleh serangan hama *Tetranychus urticae*. "
        "Gejalanya berupa daun yang menguning dengan bercak kecil, serta adanya jaring halus di bagian bawah daun. "
        "Dampaknya adalah penurunan fotosintesis, tanaman melemah, dan dalam kondisi parah daun bisa kering serta mati."
    ),
    "Target Spot": (
        "Penyakit bercak sasaran disebabkan oleh jamur *Corynespora cassiicola*. "
        "Gejalanya adalah bercak cokelat dengan lingkaran konsentris yang mirip sasaran tembak. "
        "Penyakit ini dapat menyebabkan kerontokan daun yang berat, terutama ketika kondisi lingkungan lembab."
    ),
    "Yellow Leaf Curl Virus": (
        "Penyakit ini disebabkan oleh virus yang ditularkan oleh kutu kebul *Bemisia tabaci*. "
        "Gejalanya berupa daun yang menguning, menggulung ke atas, serta pertumbuhan tanaman yang terhambat sehingga menjadi kerdil. "
        "Dampaknya adalah tanaman sulit berbuah atau menghasilkan buah yang kecil sehingga merugikan petani."
    ),
    "Mosaic Virus": (
        "Penyakit ini disebabkan oleh *Tomato mosaic virus (ToMV)* atau *Tobacco mosaic virus (TMV)*. "
        "Gejalanya berupa daun belang dengan pola hijau tua dan muda (mosaik), keriting, serta pertumbuhan yang tidak normal. "
        "Akibatnya, tanaman menjadi lemah dan hasil panen menurun."
    ),
    "Healthy": (
        "Tanaman tomat yang sehat memiliki daun hijau segar tanpa bercak, tidak mengalami penggulungan ataupun perubahan warna. "
        "Pertumbuhan tanaman berjalan normal sehingga mampu menghasilkan buah dengan baik."
    ),
}

def resolve_image_path(p: str) -> Optional[Path]:
    base = Path(p)
    if base.exists():
        return base
    exts = [".JPG", ".jpg", ".jpeg", ".png", ".PNG"]
    stem = base.with_suffix("")
    for ext in exts:
        cand = stem.with_suffix(ext)
        if cand.exists():
            return cand
    return None

# ================== PREDICT FUNCTION =======================
def preprocess_image(image: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    img = image.resize(size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Pastikan 3 channel (RGB)
    if img_array.ndim == 2:
        # grayscale -> RGB
        img_array = np.stack([img_array]*3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

@st.cache_resource(show_spinner=False)
def get_placeholder_labels(num_classes: int) -> List[str]:
    return [f"Class {i}" for i in range(num_classes)]

def predict_image(img: Image.Image):
    if model is None:
        return None, None, None
    x = preprocess_image(img, IMG_SIZE)
    preds = model.predict(x, verbose=0)
    # pastikan bentuknya (num_classes,)
    preds = np.squeeze(preds)
    if preds.ndim == 0:
        preds = np.array([preds])
    # tentukan label
    if class_labels and len(class_labels) == preds.shape[-1]:
        labels = class_labels
    else:
        labels = get_placeholder_labels(preds.shape[-1])
    top_idx = int(np.argmax(preds))
    return preds, labels, labels[top_idx]

# ======================= NAVBAR ============================
with st.container():
    selected = option_menu(
        menu_title="",
        options=["Beranda", "Deteksi Tanaman"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#1e1e1e"},
            "icon": {"display": "none"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#333333",
            },
            "nav-link-selected": {"background-color": "#ff6f61", "color": "white"},
        },
    )

st.markdown("""<style>.nav-link::before { display: none !important; }</style>""", unsafe_allow_html=True)

# ====================== MAIN PAGE ==========================
if selected == "Beranda":
    st.title(" Tomato Leaf Disease Classifier")
    st.markdown(
        """
        <div style="padding:20px; background-color:#2c2c2c; border-radius:10px; margin-bottom:20px; color:#f1f1f1;">
        <h3>Selamat Datang Di Website</h3>
        <p>Aplikasi ini menggunakan model <b>Convolutional Neural Network (CNN)</b> 
        untuk mendeteksi penyakit pada daun tomat secara otomatis</p>
        <p> Pada halaman ini terdapat 9 jenis penyakit tanaman tomat beserta deskripsi penyakitnya</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if class_labels:
        st.subheader(" Daftar Kelas")
        st.markdown("<br>", unsafe_allow_html=True)

        cols = st.columns(3)
        for idx, raw_lbl in enumerate(class_labels):
            clean_lbl = clean_label(raw_lbl)
            mapped = CLASS_IMAGES.get(clean_lbl)
            img_path = resolve_image_path(mapped) if mapped else None
            desc = CLASS_DESCRIPTIONS.get(clean_lbl, "Deskripsi belum tersedia.")

            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:center; align-items:center;'>
                        <div style='font-size:18px; font-weight:bold; margin-bottom:8px; text-align:center;'>
                            {clean_lbl}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if img_path:
                    try:
                        encoded = base64.b64encode(open(img_path, "rb").read()).decode()
                        st.markdown(
                            f"""
                            <div style='display:flex; justify-content:center; margin-bottom:15px;'>
                                <img src="data:image/png;base64,{encoded}" width="200">
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        st.markdown(
                            f"<div style='padding:15px; background:#333; border-radius:8px; text-align:center; margin-bottom:15px;'>❌ (Gambar tidak dapat dibaca)</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        f"<div style='padding:15px; background:#333; border-radius:8px; text-align:center; margin-bottom:15px;'>❌ (Gambar tidak ditemukan)</div>",
                        unsafe_allow_html=True,
                    )

                with st.expander("Deskripsi Tanaman"):
                    st.markdown(
                        f"<div style='text-align:justify; line-height:1.6;'>{desc}</div>",
                        unsafe_allow_html=True,
                    )

elif selected == "Deteksi Tanaman":
    st.title(" Deteksi Penyakit Daun Tomat")
    uploaded_file = st.file_uploader("📤 Upload Gambar Daun Tomat", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📷 Gambar yang diupload", use_container_width=True)
        if st.button("Jalankan Prediksi"):
            preds, labels, top_label = predict_image(img)
            if preds is not None:
                st.success(f"**Hasil Prediksi: {clean_label(top_label)}**")
                # tampilkan probabilitas sebagai bar chart dengan label rapi
                df = pd.DataFrame([preds], columns=[clean_label(x) for x in labels])
                st.bar_chart(df)
            else:
                st.error("Model belum siap atau terjadi kesalahan saat prediksi.")
