import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="Smart Vision AI", page_icon="🧠", layout="wide")

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.header("⚙️ Pengaturan Mode")
    menu = st.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

    st.markdown("---")
    theme = st.radio("Pilih Tampilan:", ["Terang", "Gelap"])
    st.markdown("---")

    st.markdown(
        """
        💡 *Unggah gambar Alpaca/Non-Alpaca untuk deteksi objek, atau furniture untuk klasifikasi.*
        """
    )

    st.markdown(
        """
        **Deskripsi Singkat:**  
        Aplikasi ini dapat mendeteksi keberadaan **Alpaca** dalam gambar menggunakan model *YOLO*,  
        serta mengklasifikasikan jenis **furniture** (chair, table, nightstand, sofa, bed)  
        menggunakan model deep learning berbasis CNN.
        """
    )

# ==========================
# CSS DYNAMIC (berdasarkan tema)
# ==========================
if theme == "Terang":
    main_grad = "linear-gradient(135deg, #FFF7EC 0%, #F5E6CA 100%)"
    text_color = "#3B2F2F"          # teks utama gelap
    sidebar_bg = "linear-gradient(180deg, #5C4033 0%, #4E3B31 100%)"
    sidebar_text = "#FAF3E0"
    button_bg = "#8B5E3C"
    hover_bg = "#A47148"
    result_bg = "#FFF2E0"
    result_text = "#4B3621"
else:
    main_grad = "linear-gradient(135deg, #2C1810 0%, #3E2723 100%)"
    text_color = "#FAF3E0"          # teks utama terang
    sidebar_bg = "linear-gradient(180deg, #1B0F0A 0%, #2E1A12 100%)"
    sidebar_text = "#EEDDC2"
    button_bg = "#C49A6C"
    hover_bg = "#D7B48B"
    result_bg = "#4E342E"
    result_text = "#FAF3E0"

st.markdown(f"""
    <style>
    /* MAIN AREA */
    .main {{
        background: {main_grad};
        padding: 1rem 2rem;
        font-family: 'Poppins', sans-serif;
        color: {text_color};
    }}
    h1, h2, h3, p, label, div, span {{
        color: {text_color} !important;
    }}
    h1 {{
        text-align: center;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }}

    /* SIDEBAR */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        color: {sidebar_text} !important;
        border-right: 3px solid #CBB89D;
    }}
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p {{
        color: {sidebar_text} !important;
    }}

    /* Upload area */
    .stFileUploader label {{
        color: {text_color} !important;
        font-weight: 500;
    }}

    /* Tombol */
    div.stButton > button {{
        background-color: {button_bg};
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        border: none;
    }}
    div.stButton > button:hover {{
        background-color: {hover_bg};
        color: #fff;
    }}

    /* Box hasil */
    .result-box {{
        background-color: {result_bg};
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        color: {result_text};
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
    }}

    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0px); }}
    }}
    .float {{
        animation: float 3s ease-in-out infinite;
    }}

    footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()
input_shape = classifier.input_shape[1:3]

# ==========================
# MAIN CONTENT
# ==========================
st.title("🦙 Smart Vision: Alpaca Detection & Furniture Classification")

uploaded_file = st.file_uploader("📂 Klik atau drag file ke sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Gambar yang Diupload")
        st.image(img, use_container_width=True)

    with col2:
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🔍 Hasil Deteksi Objek")
            with st.spinner("Sedang mendeteksi objek... ⏳"):
                results = yolo_model(img)
                result_img = results[0].plot()
            st.image(result_img, caption="Output Deteksi", use_container_width=True)

        elif menu == "Klasifikasi Gambar":
            st.subheader("📊 Hasil Klasifikasi")
            with st.spinner("Sedang menganalisis gambar... 🧠"):
                img_resized = img.resize(input_shape)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                probability = np.max(prediction)

            labels = ["Chair", "Table", "Nightstand", "Sofa", "Bed"]

            st.markdown(f"""
            <div class="result-box float">
                <p><b>Prediksi:</b> {labels[class_index]}</p>
                <p><b>Probabilitas:</b> {probability:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
st.markdown(f"""
<hr>
<div style="text-align:center; font-size:14px; color:{text_color};">
by <b>@naylarhmdn</b> | Smart Vision Project ☕🦙
</div>
""", unsafe_allow_html=True)
