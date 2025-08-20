import streamlit as st
import numpy as np
from PIL import Image
import json
from pathlib import Path
import tensorflow as tf
from streamlit_option_menu import option_menu
import base64
<<<<<<< HEAD
=======
import pandas as pd
from typing import Optional, Tuple, List
>>>>>>> 2e4a2763af99f92a9bf259d63558cbff5aeef162

# ======================== CONFIG ==========================
st.set_page_config(
    page_title="🍅 Tomato Leaf Disease Classifier",
    page_icon="🍅",
<<<<<<< HEAD
    layout="wide"
=======
    layout="wide",
>>>>>>> 2e4a2763af99f92a9bf259d63558cbff5aeef162
)

MODEL_PATH = Path("models/model_tomat.h5")  # bisa .h5 atau .keras
LABEL_PATH = Path("models/class_labels.json")
<<<<<<< HEAD
IMG_SIZE = (256, 256)
=======
IMG_SIZE: Tuple[int, int] = (256, 256)  # fallback

# ====================== UTILITIES =========================
def _infer_input_size_from_model(m: Optional[tf.keras.Model]) -> Tuple[int, int]:
    """Coba ambil ukuran input (height, width) dari model; jika gagal pakai default."""
    try:
        if m is None:
            return IMG_SIZE
        shape = getattr(m, "input_shape", None)
        if shape is None:
            return IMG_SIZE
        if isinstance(shape, (list, tuple)) and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if len(shape) >= 3:
            h, w = int(shape[1]), int(shape[2])
            if h > 0 and w > 0:
                return (h, w)
    except Exception:
        pass
    return IMG_SIZE

def clean_label(lbl: str) -> str:
    lbl = lbl.replace("Tomato", "").replace("___", " ").strip()
    lbl = lbl.replace("_", " ")
    return " ".join(part.capitalize() for part in lbl.split())

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
>>>>>>> 2e4a2763af99f92a9bf259d63558cbff5aeef162

# ====================== LOAD MODEL =========================
@st.cache_resource(show_spinner=True)
def load_model() -> Optional[tf.keras.Model]:
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    except Exception as e1:
        st.warning(f"Gagal memuat {MODEL_PATH.name}: {e1}")
        if MODEL_PATH.suffix == ".h5":
            try:
                temp_model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
                converted_path = MODEL_PATH.with_suffix(".keras")
                temp_model.save(converted_path, save_format="keras")
                return tf.keras.models.load_model(converted_path, compile=False, safe_mode=False)
            except Exception as e2:
                st.error(f"Konversi gagal: {e2}")
        alt_path = MODEL_PATH.with_suffix("")
        if alt_path.exists():
            try:
                return tf.keras.models.load_model(str(alt_path), compile=False, safe_mode=False)
            except Exception as e3:
                st.error(f"Gagal memuat model (format alternatif): {e3}")
    return None

model = load_model()
IMG_SIZE = _infer_input_size_from_model(model)

# ====================== LOAD LABELS =========================
def load_labels() -> List[str]:
    if LABEL_PATH.exists():
        try:
            with open(LABEL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                labels = data.get("classes", [])
                if isinstance(labels, list):
                    return labels
            elif isinstance(data, list):
                return data
        except Exception as e:
            st.error(f"Gagal membaca label: {e}")
    return []

class_labels: List[str] = load_labels()

<<<<<<< HEAD
# ================= LABEL CLEANER ===========================
def clean_label(lbl: str) -> str:
    lbl = lbl.replace("Tomato", "").replace("___", " ").strip()
    lbl = lbl.replace("_", " ")
    return " ".join(part.capitalize() for part in lbl.split())

# ================== CLASS IMAGE MAPPING ====================
=======
# ================== CLASS IMAGE & DESCRIPTION ===============
>>>>>>> 2e4a2763af99f92a9bf259d63558cbff5aeef162
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
<<<<<<< HEAD
    "Bacterial Spot": "Penyakit ini disebabkan oleh bakteri *Xanthomonas campestris pv. vesicatoria*. "
    "Gejalanya berupa bercak kecil berwarna cokelat kehitaman pada daun yang terkadang dikelilingi halo kuning. "
    "Infeksi juga dapat menyebar ke buah. Dampaknya adalah daun mudah rontok sehingga proses fotosintesis terganggu dan hasil panen berkurang.",

    "Early Blight": "Penyakit bercak daun awal ini disebabkan oleh jamur *Alternaria solani*. "
    "Gejalanya berupa bercak cokelat dengan pola lingkaran konsentris menyerupai cincin pada daun tua. "
    "Seiring waktu, daun menguning dan rontok. Dampaknya membuat luas daun hijau berkurang, buah lebih kecil, dan tanaman menjadi lemah.",

    "Late Blight": "Penyakit hawar daun lanjut disebabkan oleh jamur semu *Phytophthora infestans*. "
    "Gejalanya berupa bercak hijau gelap atau cokelat berair pada daun yang cepat meluas. "
    "Pada kondisi lembab sering muncul lapisan jamur putih di tepi bercak. "
    "Penyakit ini sangat merusak karena dapat membunuh tanaman hanya dalam beberapa hari.",

    "Leaf Mold": "Penyakit bercak jamur ini disebabkan oleh *Passalora fulva*. "
    "Gejalanya berupa bercak kuning di permukaan atas daun, sementara di bagian bawah daun muncul lapisan beludru berwarna hijau keabu-abuan. "
    "Akibatnya, daun mengering dan tanaman kekurangan energi untuk tumbuh optimal.",

    "Septoria Leaf Spot": "Penyakit ini disebabkan oleh jamur *Septoria lycopersici*. "
    "Gejalanya berupa bercak kecil berbentuk bulat berwarna cokelat dengan pusat abu-abu pucat, biasanya muncul pada daun tua. "
    "Penyakit ini sering mempercepat kerontokan daun, terutama pada lingkungan dengan kelembaban tinggi.",

    "Spider Mites": "Gangguan ini disebabkan oleh serangan hama *Tetranychus urticae*. "
    "Gejalanya berupa daun yang menguning dengan bercak kecil, serta adanya jaring halus di bagian bawah daun. "
    "Dampaknya adalah penurunan fotosintesis, tanaman melemah, dan dalam kondisi parah daun bisa kering serta mati.",

    "Target Spot": "Penyakit bercak sasaran disebabkan oleh jamur *Corynespora cassiicola*. "
    "Gejalanya adalah bercak cokelat dengan lingkaran konsentris yang mirip sasaran tembak. "
    "Penyakit ini dapat menyebabkan kerontokan daun yang berat, terutama ketika kondisi lingkungan lembab.",

    "Yellow Leaf Curl Virus": "Penyakit ini disebabkan oleh virus yang ditularkan oleh kutu kebul *Bemisia tabaci*. "
    "Gejalanya berupa daun yang menguning, menggulung ke atas, serta pertumbuhan tanaman yang terhambat sehingga menjadi kerdil."
    " Dampaknya adalah tanaman sulit berbuah atau menghasilkan buah yang kecil sehingga merugikan petani.",

    "Mosaic Virus": "Penyakit ini disebabkan oleh *Tomato mosaic virus (ToMV)* atau *Tobacco mosaic virus (TMV)*. "
    "Gejalanya berupa daun belang dengan pola hijau tua dan muda (mosaik), keriting, serta pertumbuhan yang tidak normal. "
    "Akibatnya, tanaman menjadi lemah dan hasil panen menurun.",

    "Healthy": "Tanaman tomat yang sehat memiliki daun hijau segar tanpa bercak, tidak mengalami penggulungan ataupun perubahan warna. "
    "Pertumbuhan tanaman berjalan normal sehingga mampu menghasilkan buah dengan baik."
}

def resolve_image_path(p: str) -> Path | None:
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
def preprocess_image(image: Image.Image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
=======
    "Bacterial Spot": "Penyakit ini disebabkan oleh bakteri Xanthomonas campestris ...",
    "Early Blight": "Penyakit bercak daun awal ini disebabkan oleh jamur Alternaria solani ...",
    "Late Blight": "Penyakit hawar daun lanjut disebabkan oleh jamur semu Phytophthora infestans ...",
    "Leaf Mold": "Penyakit bercak jamur ini disebabkan oleh Passalora fulva ...",
    "Septoria Leaf Spot": "Penyakit ini disebabkan oleh jamur Septoria lycopersici ...",
    "Spider Mites": "Gangguan ini disebabkan oleh serangan hama Tetranychus urticae ...",
    "Target Spot": "Penyakit bercak sasaran disebabkan oleh jamur Corynespora cassiicola ...",
    "Yellow Leaf Curl Virus": "Penyakit ini disebabkan oleh virus yang ditularkan oleh kutu kebul Bemisia tabaci ...",
    "Mosaic Virus": "Penyakit ini disebabkan oleh Tomato mosaic virus atau Tobacco mosaic virus ...",
    "Healthy": "Tanaman tomat yang sehat memiliki daun hijau segar tanpa bercak ...",
}

# ================== PREDICTION FUNCTION ====================
def preprocess_image(image: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    img = image.resize(size)
    img_array = np.array(img, dtype=np.float32)/255.0
    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
>>>>>>> 2e4a2763af99f92a9bf259d63558cbff5aeef162
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
    preds = np.squeeze(model.predict(x, verbose=0))
    if preds.ndim == 0:
        preds = np.array([preds])
    labels = class_labels if class_labels and len(class_labels) == preds.shape[-1] else get_placeholder_labels(preds.shape[-1])
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
<<<<<<< HEAD
            "icon": {"display":"none"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#333333",
            },
            "nav-link-selected": {"background-color": "#ff6f61", "color": "white"},
        },
    )

=======
            "icon": {"display": "none"},
            "nav-link": {"font-size":"16px","text-align":"center","margin":"0px","--hover-color":"#333333"},
            "nav-link-selected": {"background-color":"#ff6f61","color":"white"},
        },
    )
>>>>>>> 2e4a2763af99f92a9bf259d63558cbff5aeef162
st.markdown("""<style>.nav-link::before { display: none !important; }</style>""", unsafe_allow_html=True)

# ====================== MAIN PAGE ==========================
if selected == "Beranda":
<<<<<<< HEAD
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
                # Judul center
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:center; align-items:center;'>
                        <div style='font-size:18px; font-weight:bold; margin-bottom:8px; text-align:center;'>
                            {clean_lbl}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if img_path:
                    # Gambar center + kasih jarak bawah
                    st.markdown(
                        f"""
                        <div style='display:flex; justify-content:center; margin-bottom:15px;'>
                            <img src="data:image/png;base64,{base64.b64encode(open(img_path, "rb").read()).decode()}" width="200">
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='padding:15px; background:#333; border-radius:8px; text-align:center; margin-bottom:15px;'>❌ (Gambar tidak ditemukan)</div>",
                        unsafe_allow_html=True
                    )

                # Deskripsi justify biar rapi
                with st.expander("Deskripsi Tanaman"):
                    st.markdown(f"<div style='text-align:justify; line-height:1.6;'>{desc}</div>", unsafe_allow_html=True)


elif selected == "Deteksi Tanaman":
    st.title(" Deteksi Penyakit Daun Tomat")
    uploaded_file = st.file_uploader("📤 Upload Gambar Daun Tomat", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="📷 Gambar yang diupload", use_container_width=True)
        if st.button("Jalankan Prediksi"):
            preds, label = predict_image(img)
            if preds is not None:
                st.success(f"**Hasil Prediksi: {clean_label(label)}**")
                st.bar_chart(preds)
=======
    st.title("🍅 Tomato Leaf Disease Classifier")
    st.markdown("""
        <div style="padding:20px; background-color:#2c2c2c; border-radius:10px; margin-bottom:20px; color:#f1f1f1;">
        <h3>Selamat Datang Di Website</h3>
        <p>Aplikasi ini menggunakan model <b>CNN</b> untuk mendeteksi penyakit pada daun tomat secara otomatis.</p>
        </div>
    """, unsafe_allow_html=True)

    if class_labels:
        st.subheader("Daftar Kelas")
        cols = st.columns(3)
        for idx, lbl in enumerate(class_labels):
            clean_lbl = clean_label(lbl)
            img_path = resolve_image_path(CLASS_IMAGES.get(clean_lbl, ""))
            desc = CLASS_DESCRIPTIONS.get(clean_lbl, "Deskripsi belum tersedia.")
            with cols[idx % 3]:
                st.markdown(f"**{clean_lbl}**", unsafe_allow_html=True)
                if img_path:
                    st.image(img_path, width=200)
                else:
                    st.warning("Gambar tidak ditemukan")
                with st.expander("Deskripsi"):
                    st.write(desc)

elif selected == "Deteksi Tanaman":
    st.title("Deteksi Penyakit Daun Tomat")
    uploaded_file = st.file_uploader("📤 Upload Gambar Daun Tomat", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diupload", use_container_width=True)
        if st.button("Jalankan Prediksi"):
            preds, labels, top_label = predict_image(img)
            if preds is not None:
                conf = float(np.max(preds))*100
                st.success(f"**Hasil Prediksi: {clean_label(top_label)} ({conf:.2f}%)**")
                df = pd.DataFrame([preds], columns=[clean_label(x) for x in labels])
                st.bar_chart(df)
            else:
                st.error("Model belum siap atau terjadi kesalahan saat prediksi.")
>>>>>>> 2e4a2763af99f92a9bf259d63558cbff5aeef162
