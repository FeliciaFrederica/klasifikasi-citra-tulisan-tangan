import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import cv2
import numpy as np
from datetime import datetime, timedelta, timezone

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="ScanAja - Klasifikasi Tulisan Tangan", 
    page_icon="✍️", 
    layout="centered"
)

# Kustomisasi CSS untuk UI/UX yang modern dan bersih tanpa ikon
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    /* Banner Header */
    .header-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.15);
    }
    .header-title {
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-top: 8px;
        margin-bottom: 0;
    }
    /* Styling Tombol Utama */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        background-color: #2563eb;
        color: white;
        padding: 0.6rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    /* Kartu Hasil Utama */
    .result-card {
        padding: 24px;
        border-radius: 12px;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.2);
        margin-bottom: 20px;
    }
    /* Kustomisasi warna latar header Expander Riwayat */
    [data-testid="stExpander"] {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    [data-testid=stFileUploaderDropzoneInstructions] > * { display: none; }
    [data-testid=stFileUploaderDropzoneInstructions]::after { content: '1MB per file \\2022 JPG, JPEG, PNG'; font-size: 14px; opacity: 0.6; }
    </style>
""", unsafe_allow_html=True)

# Inisialisasi Session State
if "step" not in st.session_state:
    st.session_state.step = "input"
if "history" not in st.session_state:
    st.session_state.history = []
if "current_preds" not in st.session_state:
    st.session_state.current_preds = None

# --- HEADER BANNER ---
st.markdown("""
    <div class="header-banner">
        <h1 class="header-title">ScanAja</h1>
        <p class="header-subtitle">Sistem Klasifikasi Citra Tulisan Tangan</p>
    </div>
""", unsafe_allow_html=True)

# 2. Load Model Keras (Cached)
@st.cache_resource
def load_my_model():
    return load_model("model_efficientnet_terbaik.keras")

with st.spinner("Memuat model ke memori..."):
    model = load_my_model()

def scan_otomatis(image):
    rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    gray_asli = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    skala = 1000 / rgb.shape[1]
    tinggi = int(rgb.shape[0] * skala)
    kecil = cv2.resize(rgb, (1000, tinggi))
    sat = cv2.cvtColor(kecil, cv2.COLOR_RGB2HSV)[..., 1]
    gray = cv2.cvtColor(kecil, cv2.COLOR_RGB2GRAY)
    bersih = cv2.divide(gray, cv2.GaussianBlur(gray, (0, 0), 15), scale=255)

    tinta = (bersih < 170).astype(np.uint8)
    n, label, stats, _ = cv2.connectedComponentsWithStats(tinta)
    simpan = []
    for i in range(1, n):
        x, y, w, h, luas = stats[i]
        if luas < 8 or h > 0.3 * tinggi or w > 500:
            continue
        if x == 0 or y == 0 or x + w >= 1000 or y + h >= tinggi:
            continue
        x0, y0 = max(x - 8, 0), max(y - 8, 0)
        x1, y1 = min(x + w + 8, 1000), min(y + h + 8, tinggi)
        komp = (label[y0:y1, x0:x1] == i).astype(np.uint8)
        cincin = cv2.dilate(komp, np.ones((13, 13), np.uint8)) - cv2.dilate(komp, np.ones((5, 5), np.uint8))
        if (sat[y0:y1, x0:x1][cincin > 0] < 45).mean() > 0.9:
            simpan.append(i)
    pilih = np.isin(label, simpan).astype(np.uint8)

    hasil = cv2.divide(gray_asli, cv2.GaussianBlur(gray_asli, (0, 0), 15 / skala), scale=255)
    baris = cv2.dilate(pilih, np.ones((9, 61), np.uint8))
    n, label, stats, _ = cv2.connectedComponentsWithStats(baris)
    if n > 1:
        jumlah = np.bincount(label[pilih > 0], minlength=n)
        x, y, w, h, _ = stats[1 + int(np.argmax(jumlah[1:]))]
        x, y, w, h = [int(v / skala) for v in (x, y, w, h)]
        hasil = hasil[y:y + h, x:x + w]
    return Image.fromarray(hasil)

# TAMPILAN 1: KARTU METODE INPUT

if st.session_state.step == "input":
    st.markdown("#### Pilih Metode Masukan Citra")
    
    col_card1, col_card2 = st.columns(2)
    with col_card1:
        method_choice = st.radio("Metode", ("Upload File Gambar", "Ambil Foto dengan Kamera"), label_visibility="collapsed")
    
    st.markdown("")
    uploaded_file = None
    if method_choice == "Upload File Gambar":
        uploaded_file = st.file_uploader("Pilih file citra tulisan tangan (Format: JPG/JPEG, PNG, maks. 1 MB)", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Ambil foto tulisan tangan langsung dari kamera")

    if uploaded_file is not None:
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            image = Image.open(uploaded_file)
            if method_choice == "Ambil Foto dengan Kamera":
                image = scan_otomatis(image)
            st.image(image, caption='Citra Tulisan Tangan yang Diuji', width=300)
        
        st.markdown("")
        if st.button("Jalankan Analisis"):
            with st.spinner('Model sedang menganalisis pola goresan tulisan...'):
                try:
                    # Preprocessing sesuai model (Grayscale & tile 64x256)
                    gray = np.array(image.convert('L'))
                    tinta = np.where(gray < 200)
                    if len(tinta[0]) > 0:
                        gray = gray[tinta[0].min():tinta[0].max() + 1, tinta[1].min():tinta[1].max() + 1]
                    lebar = max(1, int(gray.shape[1] * 64 / gray.shape[0]))
                    gray = cv2.resize(gray, (lebar, 64), interpolation=cv2.INTER_AREA)
                    posisi = list(range(0, max(lebar - 256, 0) + 1, 128))
                    if lebar > 256 and posisi[-1] != lebar - 256:
                        posisi.append(lebar - 256)
                    tiles = []
                    for x in posisi:
                        tile = np.full((64, 256), 255, dtype=np.uint8)
                        potong = gray[:, x:x + 256]
                        tile[:, :potong.shape[1]] = potong
                        if (tile < 200).mean() > 0.01 or len(tiles) == 0:
                            tiles.append(tile)
                    img_array = np.array(tiles, dtype=np.float32)[..., np.newaxis] / 255.0
                    
                    # Prediksi Model
                    predictions = model.predict(img_array).mean(axis=0)
                    top_3_indices = np.argsort(predictions)[::-1][:3]
                    
                    top_3_raw = predictions[top_3_indices]
                    sum_top_3 = np.sum(top_3_raw)
                    top_3_normalized = (top_3_raw / sum_top_3) * 100 if sum_top_3 > 0 else top_3_raw * 100

                    with open("class_names.txt", "r") as f:
                        class_labels = [line.strip() for line in f.readlines()]
                    
                    top_prediction = class_labels[top_3_indices[0]]
                    top_confidence = top_3_normalized[0]
                    
                    timestamp = datetime.now(timezone(timedelta(hours=7))).strftime("%d %b %Y, %H:%M")
                    
                    current_data = {
                        "image": image,
                        "top_pred": top_prediction,
                        "top_conf": top_confidence,
                        "indices": top_3_indices,
                        "normalized": top_3_normalized,
                        "labels": class_labels,
                        "time": timestamp
                    }
                    
                    st.session_state.current_preds = current_data
                    st.session_state.history.append(current_data)
                    
                    st.session_state.step = "result"
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

# TAMPILAN 2: HASIL PREDIKSI

elif st.session_state.step == "result":
    data = st.session_state.current_preds
    
    st.markdown("### Hasil Prediksi")
    
    st.markdown(f"""
        <div class="result-card">
            <p style="margin:0; font-size:14px; text-transform:uppercase; letter-spacing:1px; opacity:0.9;">Prediksi Penulis Utama</p>
            <h1 style="margin:8px 0; font-size:36px; font-weight:800; letter-spacing:1px;">{data['top_pred']}</h1>
            <p style="margin:0; font-size:16px;">Tingkat Kepercayaan: <b>{data['top_conf']:.2f}%</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Top 3 Kandidat Teratas:")
    for i, idx in enumerate(data['indices']):
        label = data['labels'][idx]
        conf = float(data['normalized'][i])
        
        col_c1, col_c2, col_c3 = st.columns([1, 5, 1.5])
        with col_c1:
            st.markdown(f"**#{i+1}**")
        with col_c2:
            st.markdown(f"ID: `{label}`")
            st.progress(conf / 100.0)
        with col_c3:
            st.markdown(f"**{conf:.1f}%**")
            
    report_text = f"""- LAPORAN HASIL KLASIFIKASI SCANAJA -
Waktu Pengujian : {data['time']}
Prediksi Utama  : {data['top_pred']} ({data['top_conf']:.2f}%)

Top 3 Kandidat Alternatif:
1. ID: {data['labels'][data['indices'][0]]} - {data['normalized'][0]:.1f}%
2. ID: {data['labels'][data['indices'][1]]} - {data['normalized'][1]:.1f}%
3. ID: {data['labels'][data['indices'][2]]} - {data['normalized'][2]:.1f}%
-
"""
    st.markdown("")
    st.download_button(
        label="Unduh Laporan Hasil (.txt)",
        data=report_text,
        file_name=f"Laporan_{data['top_pred']}.txt",
        mime="text/plain"
    )
    
    st.markdown("---")
    if st.button("Uji Gambar Lain"):
        st.session_state.step = "input"
        st.rerun()

#BAGIAN: RIWAYAT KLASIFIKASI CITRA

if len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown("### Riwayat Klasifikasi Citra")
    st.markdown("Daftar riwayat pengujian sebelumnya dapat dilihat pada rincian di bawah ini:")
    
    # Gunakan list b menampung indeks riwayat yang akan dihapus
    indices_to_delete = []
    
    for idx, item in enumerate(st.session_state.history):
        history_num = idx + 1
        expander_label = f"Riwayat #{history_num} ({item['time']}) — Prediksi: {item['top_pred']} ({item['top_conf']:.1f}%)"
        
        with st.expander(expander_label):
            st.markdown(f"**Waktu Pengujian:** `{item['time']}`")
            st.markdown("**Citra yang Diuji:**")
            st.image(item['image'], width="stretch")
            
            st.markdown("---")
            st.markdown(f"**Hasil Utama:** `{item['top_pred']}` &nbsp;|&nbsp; **Kepercayaan:** `{item['top_conf']:.2f}%`")
            
            st.markdown("**Top 3 Kandidat:**")
            for i, c_idx in enumerate(item['indices']):
                c_label = item['labels'][c_idx]
                c_conf = float(item['normalized'][i])
                st.markdown(f"- **#{i+1}** ID: `{c_label}` — **{c_conf:.1f}%**")
            
            st.markdown("")
            # Tombol Hapus Spesifik per Riwayat
            if st.button("Hapus Riwayat Ini", key=f"del_{idx}"):
                indices_to_delete.append(idx)

    # Buat Eksekusi penghapusan item riwayat yang dipilih
    if indices_to_delete:
        for i in sorted(indices_to_delete, reverse=True):
            del st.session_state.history[i]
        st.rerun()
