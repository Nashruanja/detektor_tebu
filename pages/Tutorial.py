"""
Tutorial - Deteksi Penyakit Daun Tebu
BACKWARD COMPATIBLE - Works with old Streamlit versions
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Tutorial",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Hide sidebar */
    [data-testid="collapsedControl"],
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    [data-testid="stSidebar"] > div {
        display: none !important;
    }
    
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #f8faf7 0%, #ffffff 100%);
    }
    
    /* BIGGER FONT */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #1a1a1a !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }
    
    /* Headings */
    h1 {
        color: #2d5016 !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }
    
    h2 {
        color: #2d5016 !important;
        font-weight: 800 !important;
        font-size: 1.7rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #4a7c2e !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    
    /* Floating Menu */
    .floating-menu-btn {
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #2d5016 0%, #4a7c2e 100%);
        color: white !important;
        width: 50px;
        height: 50px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(45, 80, 22, 0.3);
        z-index: 1000;
        font-size: 24px;
        transition: transform 0.2s ease;
    }
    
    .floating-menu-btn:hover {
        transform: scale(1.05);
    }
    
    .floating-menu {
        position: fixed;
        top: 80px;
        right: 20px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        padding: 0.8rem;
        display: none;
        z-index: 999;
        min-width: 180px;
    }
    
    .floating-menu.show {
        display: block;
    }
    
    .floating-menu a {
        display: block;
        padding: 0.7rem 0.9rem;
        color: #2d5016 !important;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.95rem !important;
        border-radius: 8px;
        transition: all 0.2s ease;
        margin: 0.2rem 0;
    }
    
    .floating-menu a:hover {
        background: #f0f7ed;
    }
    
    .floating-menu a.active {
        background: linear-gradient(135deg, #2d5016 0%, #4a7c2e 100%);
        color: white !important;
    }
    
    /* Header - SMALLER */
    .tutorial-header {
        background: linear-gradient(135deg, #2d5016 0%, #4a7c2e 100%);
        padding: 2rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(45, 80, 22, 0.15);
    }
    
    .tutorial-header h1 {
        color: white !important;
        font-size: 2.2rem !important;
        margin: 0 0 0.4rem 0 !important;
    }
    
    .tutorial-header p {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.05rem !important;
        margin: 0 !important;
    }
    
    /* DO/DON'T CARDS - SMALLER */
    .do-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f4 100%);
        border-left: 4px solid #4caf50;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
    
    .do-card h3 {
        color: #2e7d32 !important;
        margin: 0 0 0.8rem 0 !important;
        font-size: 1.25rem !important;
    }
    
    .do-card ul {
        margin: 0;
        padding-left: 1.1rem;
    }
    
    .do-card li {
        color: #1a1a1a !important;
        margin: 0.5rem 0;
        font-size: 0.98rem !important;
    }
    
    .dont-card {
        background: linear-gradient(135deg, #ffebee 0%, #fef5f5 100%);
        border-left: 4px solid #f44336;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
    
    .dont-card h3 {
        color: #c62828 !important;
        margin: 0 0 0.8rem 0 !important;
        font-size: 1.25rem !important;
    }
    
    .dont-card ul {
        margin: 0;
        padding-left: 1.1rem;
    }
    
    .dont-card li {
        color: #1a1a1a !important;
        margin: 0.5rem 0;
        font-size: 0.98rem !important;
    }
    
    /* FAQ CARDS - SMALLER */
    .faq-card {
        background: linear-gradient(135deg, #fff8e1 0%, #ffffff 100%);
        border: 2px solid #ffe0b2;
        border-left: 4px solid #ff9800;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 6px rgba(255, 152, 0, 0.06);
        transition: all 0.2s ease;
    }
    
    .faq-card:hover {
        border-color: #ff9800;
        box-shadow: 0 3px 10px rgba(255, 152, 0, 0.12);
    }
    
    .faq-question {
        color: #e65100 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        margin: 0 0 0.7rem 0 !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .faq-answer {
        color: #5d4037 !important;
        font-size: 0.98rem !important;
        line-height: 1.6 !important;
        margin: 0 !important;
    }
    
    /* Step boxes - SMALLER */
    .step-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffffff 100%);
        border: 2px solid #ffe0b2;
        border-left: 4px solid #ff9800;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1.2rem 0;
        box-shadow: 0 2px 6px rgba(255, 152, 0, 0.06);
    }
    
    .step-box h3 {
        color: #e65100 !important;
        margin-top: 0 !important;
        margin-bottom: 0.8rem !important;
        font-size: 1.2rem !important;
    }
    
    .step-box p, .step-box ul {
        font-size: 0.98rem !important;
    }
    
    hr {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 2rem 0;
    }
    
    /* ========== MOBILE RESPONSIVE ========== */
    
    @media (max-width: 768px) {
        .tutorial-header {
            padding: 1.5rem 1.2rem !important;
        }
        
        .tutorial-header h1 {
            font-size: 1.6rem !important;
        }
        
        .tutorial-header p {
            font-size: 0.95rem !important;
        }
        
        h2 {
            font-size: 1.4rem !important;
        }
        
        h3 {
            font-size: 1.15rem !important;
        }
        
        .do-card, .dont-card {
            padding: 1rem !important;
        }
        
        .do-card h3, .dont-card h3 {
            font-size: 1.1rem !important;
        }
        
        .do-card li, .dont-card li {
            font-size: 0.9rem !important;
        }
        
        .step-box {
            padding: 1rem !important;
        }
        
        .step-box h3 {
            font-size: 1.1rem !important;
        }
        
        .faq-card {
            padding: 0.9rem 1rem !important;
        }
        
        .faq-question {
            font-size: 0.95rem !important;
        }
        
        .faq-answer {
            font-size: 0.88rem !important;
        }
        
        [data-testid="column"] {
            width: 100% !important;
            margin-bottom: 0.5rem !important;
        }
    }
    
    @media (max-width: 480px) {
        .tutorial-header h1 {
            font-size: 1.4rem !important;
        }
        
        h2 {
            font-size: 1.2rem !important;
            margin-top: 1.5rem !important;
        }
        
        .floating-menu-btn {
            width: 45px !important;
            height: 45px !important;
            font-size: 20px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Floating Menu
st.markdown("""
<div class="floating-menu-btn" onclick="toggleFloatingMenu()">☰</div>

<div class="floating-menu" id="floatingMenu">
    <a href="/">🏠 Beranda</a>
    <a href="/Tutorial" class="active">📚 Tutorial</a>
    <a href="/Penyakit">🌿 Penyakit</a>
</div>

<script>
function toggleFloatingMenu() {
    var menu = document.getElementById('floatingMenu');
    menu.classList.toggle('show');
}

document.addEventListener('click', function(event) {
    var menu = document.getElementById('floatingMenu');
    var btn = document.querySelector('.floating-menu-btn');
    if (!menu.contains(event.target) && !btn.contains(event.target)) {
        menu.classList.remove('show');
    }
});
</script>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="tutorial-header">
    <h1>📚 Tutorial Penggunaan</h1>
    <p>Panduan lengkap menggunakan sistem deteksi penyakit daun tebu</p>
</div>
""", unsafe_allow_html=True)

# Content
st.markdown("## 📸 Cara Foto yang Benar")
st.write("Kualitas foto sangat mempengaruhi akurasi hasil analisis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="do-card">
        <h3>✅ Yang Harus Dilakukan</h3>
        <ul>
            <li><strong>Fokus pada 1 daun</strong> yang jelas terlihat</li>
            <li><strong>Jarak 20-50 cm</strong> dari daun</li>
            <li><strong>Daun mengisi 60-70%</strong> area foto</li>
            <li><strong>Foto di siang hari</strong> dengan cahaya cukup</li>
            <li><strong>Pastikan tajam</strong> dan tidak blur</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="dont-card">
        <h3>❌ Yang Harus Dihindari</h3>
        <ul>
            <li><strong>Foto dari jarak terlalu jauh</strong></li>
            <li><strong>Banyak daun lain</strong> dalam frame</li>
            <li><strong>Ada tangan atau orang</strong> dalam foto</li>
            <li><strong>Foto blur</strong> atau tidak fokus</li>
            <li><strong>Pencahayaan terlalu gelap</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# IMAGE LOADING - BACKWARD COMPATIBLE
st.write("")  # Spacing

image_path = Path("assets") / "Benar_Salah.png"

try:
    # Try new API first (Streamlit >= 1.10.0)
    st.image(str(image_path), use_column_width=True, 
             caption="Contoh perbandingan foto yang benar dan salah")
except TypeError:
    # Fallback for very old Streamlit (< 0.62.0) - no caption parameter
    try:
        st.image(str(image_path), use_column_width=True)
        st.caption("Contoh perbandingan foto yang benar dan salah")
    except:
        # Last resort - just image
        st.image(str(image_path))
        st.text("Contoh perbandingan foto yang benar dan salah")
except Exception as e:
    st.error(f"""
    ⚠️ **Gambar tidak dapat dimuat!**
    
    **Error:** `{str(e)}`
    
    **Solusi:**
    1. Pastikan file `Benar_Salah.png` ada di folder `assets/` (di root project)
    2. Update Streamlit: `pip install --upgrade streamlit`
    3. Restart Streamlit setelah update
    """)

st.markdown("---")

st.markdown("## 📱 Langkah-Langkah Penggunaan")

st.markdown("""
<div class="step-box">
    <h3>1️⃣ Upload Foto Daun</h3>
    <p><strong>Dari HP:</strong> Klik tombol 'Browse files', kemudian pilih foto dari galeri atau ambil foto langsung</p>
    <p><strong>Dari Laptop:</strong> Klik tombol 'Browse files' atau langsung drag & drop foto ke area upload</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="step-box">
    <h3>2️⃣ Lakukan Analisis</h3>
    <ul>
        <li>Preview foto akan muncul di layar</li>
        <li>Klik tombol hijau <strong>'Analisis Gambar'</strong></li>
        <li>Tunggu 5-15 detik untuk proses analisis</li>
        <li>Jangan tutup atau refresh aplikasi selama proses</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="step-box">
    <h3>3️⃣ Lihat dan Pahami Hasil</h3>
    <p>Setelah analisis selesai, sistem akan menampilkan:</p>
    <ul>
        <li><strong>Nama penyakit</strong> yang terdeteksi</li>
        <li><strong>Tingkat keyakinan</strong> sistem dalam persentase</li>
        <li><strong>3 visualisasi XAI</strong> untuk menjelaskan keputusan AI</li>
        <li><strong>Rekomendasi penanganan</strong> yang dapat dilakukan</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("## ❓ Pertanyaan yang Sering Diajukan")

st.markdown("""
<div class="faq-card">
    <div class="faq-question">❓ Kamera HP tidak bisa diakses?</div>
    <div class="faq-answer">
        Gunakan opsi 'Browse files' untuk memilih foto dari galeri. Fitur kamera langsung membutuhkan koneksi HTTPS untuk alasan keamanan.
    </div>
</div>

<div class="faq-card">
    <div class="faq-question">❓ Apakah foto blur bisa dianalisis?</div>
    <div class="faq-answer">
        Bisa, namun hasilnya kemungkinan kurang akurat. Sebaiknya ambil foto ulang yang lebih fokus dan tajam untuk hasil terbaik.
    </div>
</div>

<div class="faq-card">
    <div class="faq-question">❓ Berapa lama waktu analisis?</div>
    <div class="faq-answer">
        Normalnya memakan waktu 5-15 detik, tergantung kecepatan koneksi internet dan ukuran file foto.
    </div>
</div>

<div class="faq-card">
    <div class="faq-question">❓ Apakah hasil analisis bisa salah?</div>
    <div class="faq-answer">
        Tidak ada sistem AI yang 100% sempurna. Untuk hasil terbaik, gunakan foto berkualitas tinggi dan perhatikan tingkat keyakinan sistem (>80% = hasil akurat).
    </div>
</div>

<div class="faq-card">
    <div class="faq-question">❓ Bisa analisis banyak foto sekaligus?</div>
    <div class="faq-answer">
        Saat ini sistem hanya dapat menganalisis satu foto per waktu. Untuk menganalisis foto berikutnya, silakan upload foto baru dan klik 'Analisis Gambar' lagi.
    </div>
</div>

<div class="faq-card">
    <div class="faq-question">❓ Bagaimana cara mendapat hasil paling akurat?</div>
    <div class="faq-answer">
        Ambil foto di siang hari dengan cahaya alami, fokuskan pada 1 daun yang menunjukkan gejala jelas, pastikan jarak 20-50 cm, dan foto harus tajam (tidak blur).
    </div>
</div>
""", unsafe_allow_html=True)
