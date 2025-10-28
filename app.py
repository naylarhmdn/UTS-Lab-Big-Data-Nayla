import streamlit as st
from PIL import Image

# ==========================
# CONFIG & STYLE
# ==========================
st.set_page_config(page_title="Smart Vision: Alpaca Detection", page_icon="🐪", layout="wide")

# Custom CSS (warna & style)
st.markdown("""
    <style>
    /* --- Warna keseluruhan halaman --- */
    .main {
        background-color: #f7e7d4; /* coklat muda / krem */
        color: #2b1d12; /* teks gelap tua */
        font-family: 'Poppins', sans-serif;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: #5c4033; /* coklat tua */
        color: #f3e5d8;
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #f3e5d8 !important;
    }

    /* --- Upload area --- */
    .stFileUploader label {
        color: #2b1d12 !important;
        font-weight: 500;
    }

    /* --- Heading utama --- */
    h1, h2, h3 {
        color: #2b1d12;
    }

    /* --- Box konten utama --- */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* --- Tombol --- */
    .stButton>button {
        background-color: #5c4033;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
    }

    .stButton>button:hover {
        background-color: #7b5544;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("⚙️ Pengaturan Mode")
mode = st.sidebar.radio("Pilih mode:", ["Deteksi Objek", "Klasifikasi Gambar"])

st.sidebar.markdown("🧩 **Unggah gambar Alpaca/Non-Alpaca** untuk deteksi objek, atau furniture untuk klasifikasi.")
theme = st.sidebar.radio("Pilih tampilan:", ["Terang", "Gelap"])

st.sidebar.markdown("""
**Deskripsi Singkat**  
Aplikasi ini dapat mendeteksi keberadaan *Alpaca* dalam gambar menggunakan model YOLO,  
dan mengklasifikasikan jenis furniture (*chair, table, lamp, bed*)  
menggunakan model berbasis CNN.
""")


# ==========================
# KONTEN UTAMA
# ==========================
st.title("🐪 Smart Vision: Alpaca Detection & Furniture Classification")

st.markdown("### 📂 Klik atau drag file ke sini")
uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
else:
    st.info("Silakan unggah gambar untuk memulai analisis.")

st.markdown("---")
st.markdown("by **@naylarhmdn** | Smart Vision Project ☕🐫")
