import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# CONFIG & STYLE
# ==========================
st.set_page_config(page_title="🦙 Alpaca Vision", page_icon="🧠", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #FBEAFF 0%, #E3FDFD 100%);
        padding: 1rem 2rem;
    }
    h1 {
        color: #7A1CAC;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] {
        background-color: #F6EFFF;
        color: black !important;
    }
    .stImage {
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(122, 28, 172, 0.2);
    }
    div.stButton > button {
        background-color: #7A1CAC;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #9C27B0;
        color: #fff;
    }
    .result-box {
        background-color: #ffffffcc;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        color: #4B0082;
        box-shadow: 0 0 10px rgba(100, 0, 150, 0.1);
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Deteksi objek Alpaca
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Klasifikasi furniture
    return yolo_model, classifier

yolo_model, classifier = load_models()

# Cek input shape dari model klasifikasi
input_shape = classifier.input_shape[1:3]  # (height, width)

# ==========================
# UI
# ==========================
st.title("🦙 Alpaca & Non-Alpaca Vision Dashboard")

with st.sidebar:
    st.header("✨ Pengaturan Mode")
    menu = st.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.markdown("---")
    st.markdown(
        """
        <div style="color:black;">
            💡 <i>Unggah gambar Alpaca / Non-Alpaca untuk deteksi, atau furniture untuk klasifikasi!</i>
        </div>
        """,
        unsafe_allow_html=True
    )

uploaded_file = st.file_uploader("Klik! Unggah Gambar Disini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Gambar yang Diupload")
        st.image(img, width='stretch')

    with col2:
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek")
            with st.spinner("Sedang mendeteksi objek... ⏳"):
                results = yolo_model(img)
                result_img = results[0].plot()
            st.image(result_img, caption="Output Deteksi", width='stretch')

        elif menu == "Klasifikasi Gambar":
            st.subheader("📊 Hasil Klasifikasi")
            with st.spinner("Sedang menganalisis gambar... 🧠"):
                # Resize sesuai ukuran input model
                img_resized = img.resize(input_shape)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                probability = np.max(prediction)

            # Label sesuai dataset furniture
            labels = ["Chair", "Table", "Nightstand", "Sofa", "Bed"]

            st.markdown(f"""
            <div class="result-box">
                <p><b>Prediksi:</b> {labels[class_index]}</p>
                <p><b>Probabilitas:</b> {probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
st.markdown("""
<hr>
<div style="text-align:center; font-size:14px; color:gray;">
by <b>@naylarhmdn</b> | Alpaca Vision Project 🦙
</div>
""", unsafe_allow_html=True)
