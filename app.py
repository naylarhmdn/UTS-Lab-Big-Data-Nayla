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
st.set_page_config(page_title="Smart Vision AI", page_icon="🧠", layout="wide")

# Custom CSS (tema terang dengan teks coklat tua elegan)
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #FAF3E0 0%, #F5E6CA 100%);
        padding: 1rem 2rem;
        font-family: 'Poppins', sans-serif;
        color: #3B2F2F;
    }

    h1 {
        color: #4B3621;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #EBD9C7;
        color: #3B2F2F !important;
        border-right: 3px solid #CBB89D;
    }

    h2, h3, .stMarkdown {
        color: #4E342E;
    }

    .stImage {
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(90, 62, 43, 0.15);
    }

    div.stButton > button {
        background-color: #8B5E3C;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #A47148;
        color: #fff;
    }

    .result-box {
        background-color: #FFF7EC;
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        color: #4B3621;
        box-shadow: 0 0 15px rgba(139, 94, 60, 0.1);
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .float {
        animation: float 3s ease-in-out infinite;
    }

    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi Alpaca
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi furniture
    return yolo_model, classifier

yolo_model, classifier = load_models()
input_shape = classifier.input_shape[1:3]  # (height, width)

# ==========================
# UI
# ==========================
st.title("🦙 Smart Vision: Alpaca Detection & Furniture Classification")

with st.sidebar:
    st.header("⚙️ Pengaturan Mode")
    menu = st.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.markdown("---")
    st.markdown(
        """
        <div style="color:#3B2F2F;">
            💡 <i>Unggah gambar Alpaca/Non-Alpaca untuk deteksi objek, atau furniture untuk klasifikasi.</i>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.write("📋 Deskripsi Singkat:")
    st.markdown(
        """
        <p style="font-size:14px; color:#3B2F2F;">
        Aplikasi ini dapat mendeteksi keberadaan <b>Alpaca</b> dalam gambar menggunakan model <i>YOLO</i>, 
        dan juga mengklasifikasikan jenis <b>furniture</b> (chair, table, nightstand, sofa, bed) menggunakan model deep learning berbasis CNN.
        </p>
        """, unsafe_allow_html=True
    )

uploaded_file = st.file_uploader("📂 Klik atau drag file ke sini", type=["jpg", "jpeg", "png"])

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
                img_resized = img.resize(input_shape)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                probability = np.max(prediction)

            labels = ["Chair", "Table", "Nightstand", "Sofa", "Bed"]

            st.markdown(f"""
            <div class="result-box float">
                <p>🪑 <b>Prediksi:</b> {labels[class_index]}</p>
                <p>📈 <b>Probabilitas:</b> {probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
st.markdown("""
<hr>
<div style="text-align:center; font-size:14px; color:#4B3621;">
by <b>@naylarhmdn</b> | Smart Vision Project ☕🦙
</div>
""", unsafe_allow_html=True)
