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
    theme = st.radio("🎨 Pilih Tampilan:", ["Terang", "Gelap"])
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
    main_bg = "#FAF3E0"
    main_grad = "linear-gradient(135deg, #FAF3E0 0%, #F5E6CA 100%)"
    text_color = "#3B2F2F"
    sidebar_bg = "linear-gradient(180deg, #3B2F2F 0%, #4E3B31 100%)"
    button_bg = "#8B5E3C"
    hover_bg = "#A47148"
else:
    main_bg = "#2C1810"
    main_grad = "linear-gradient(135deg, #2C1810 0%, #3E2723 100%)"
    text_color = "#F5E6CA"
    sidebar_bg = "linear-gradient(180deg, #1B0F0A 0%, #2E1A12 100%)"
    button_bg = "#C49A6C"
    hover_bg = "#D7B48B"

st.markdown(f"""
    <style>
    /* MAIN AREA */
    .main {{
        background: {main_grad};
        padding: 1rem 2rem;
        font-family: 'Poppins', sans-serif;
        color: {text_color};
    }}
    h1 {{
        color: {text_color};
        text-align: center;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }}
    h2, h3, .stMarkdown {{
        color: {text_color};
    }}
    /* SIDEBAR */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        color: #FAF3E0 !important;
        border-right: 3px solid #CBB89D;
    }}
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p {{
        color: #FAF3E0 !important;
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
    .result-box {{
        background-color: #FFF7EC;
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        color: #4B3621;
        box-shadow: 0 0 15px rgba(139, 94, 60, 0.1);
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
                <p>🪑 <b>Prediksi:</b> {labels[class_index]}</p>
                <p>📈 <b>Probabilitas:</b> {probability:.2%}</p>
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
