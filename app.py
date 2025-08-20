import streamlit as st
import numpy as np
from PIL import Image
import json
from pathlib import Path
import tensorflow as tf
from streamlit_option_menu import option_menu
import base64
import gdown  # pip install gdown

# ================= CONFIG =================
st.set_page_config(
    page_title="Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="wide"
)

# ================= PATH MODEL =================
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_H5 = MODEL_DIR / "models/model_tomat.keras"
MODEL_SM = MODEL_DIR / "model_tomat_savedmodel"
LABEL_PATH = MODEL_DIR / "class_labels.json"

# ================= DOWNLOAD MODEL DARI DRIVE =================
DRIVE_LINK = "https://drive.google.com/file/d/1lYga4myscy7wDcaKwlS3IiaQDKUBqkRl/view?usp=drive_link"
if not MODEL_H5.exists() and not MODEL_SM.exists():
    with st.spinner("Mengunduh model dari Google Drive..."):
        gdown.download(DRIVE_LINK, str(MODEL_H5), quiet=False)

# ================= LOAD MODEL =================
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        if MODEL_SM.exists():
            return tf.keras.models.load_model(MODEL_SM)
        elif MODEL_H5.exists():
            model = tf.keras.models.load_model(MODEL_H5)
            model.save(MODEL_SM)  # simpan ke SavedModel untuk menghindari dtype error
            return model
        else:
            st.error("Model tidak ditemukan!")
            return None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# ================= LOAD LABELS =================
def load_labels():
    if LABEL_PATH.exists():
        with open(LABEL_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("classes", [])
        elif isinstance(data, list):
            return data
    return []

class_labels = load_labels()

# ================= LABEL CLEANER =================
def clean_label(lbl: str) -> str:
    lbl = lbl.replace("Tomato", "").replace("___", " ").replace("_", " ").strip()
    return " ".join(part.capitalize() for part in lbl.split())

# ================= CLASS IMAGE & DESCRIPTION =================
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

# ================= PREDICT FUNCTION =================
IMG_SIZE = (256, 256)
def preprocess_image(image: Image.Image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)/255.0
    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

def predict_image(img: Image.Image):
    if model is None:
        return None, None
    x = preprocess_image(img)
    preds = np.squeeze(model.predict(x, verbose=0))
    if preds.ndim == 0:
        preds = np.array([preds])
    labels = class_labels if class_labels and len(class_labels)==preds.shape[-1] else [f"Class {i}" for i in range(len(preds))]
    top_idx = int(np.argmax(preds))
    return preds, labels[top_idx]
    
# ======================= NAVBAR ============================
with st.container():
    selected = option_menu(
        menu_title="",
        options=["Beranda", "Deteksi Tanaman"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#1e1e1e"},
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

st.markdown("""<style>.nav-link::before { display: none !important; }</style>""", unsafe_allow_html=True)

# ================= MAIN PAGE =================
if selected == "Beranda":
    st.title("🍅 Tomato Leaf Disease Classifier")
    st.markdown(
        """
        <div style="padding:20px; background-color:#2c2c2c; border-radius:10px; margin-bottom:20px; color:#f1f1f1;">
        <h3>Selamat Datang Di Website</h3>
        <p>Aplikasi ini menggunakan model <b>CNN</b> untuk mendeteksi penyakit pada daun tomat.</p>
        </div>
        """, unsafe_allow_html=True
    )

    if class_labels:
        st.subheader("Daftar Kelas")
        cols = st.columns(3)
        for idx, lbl in enumerate(class_labels):
            clean_lbl = clean_label(lbl)
            img_path = resolve_image_path(CLASS_IMAGES.get(clean_lbl, ""))
            desc = CLASS_DESCRIPTIONS.get(clean_lbl, "Deskripsi belum tersedia.")
            with cols[idx % 3]:
                st.markdown(f"**{clean_lbl}**")
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
            preds, label = predict_image(img)
            if preds is not None:
                st.success(f"**Hasil Prediksi: {clean_label(label)}**")
                st.bar_chart(preds)
            else:
                st.error("Model belum siap atau terjadi kesalahan saat prediksi.")
