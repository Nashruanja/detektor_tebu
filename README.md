# 🌾 Sistem Deteksi Penyakit Daun Tebu

Aplikasi web berbasis Streamlit untuk klasifikasi penyakit daun tebu menggunakan Support Vector Machine (SVM) dengan Explainable AI (SHAP).

## 📋 Fitur Utama

- ✅ **Klasifikasi Otomatis**: Deteksi 5 kondisi daun tebu (Healthy, Mosaic, RedRot, Rust, Yellow)
- 🔬 **Explainable AI (XAI)**: Visualisasi SHAP untuk interpretasi model
  - Force Plot
  - Waterfall Plot
  - Feature Importance
- 🎨 **Desain Tema Tebu**: UI dengan gradient hijau-kuning yang menarik
- 📊 **Akurasi Tinggi**: Model dengan akurasi >90%
- 🚀 **Real-time Processing**: Analisis instan setelah upload gambar

## 🛠️ Teknologi yang Digunakan

- **Framework**: Streamlit
- **Machine Learning**: scikit-learn (SVM)
- **Computer Vision**: OpenCV
- **Explainable AI**: SHAP
- **Visualisasi**: Matplotlib
- **Image Processing**: PIL/Pillow

## 📦 Persyaratan Sistem

- Python 3.8 atau lebih baru
- Visual Studio Code (recommended)
- RAM minimum 4GB
- Storage ~500MB untuk dependencies

## 🚀 Cara Instalasi & Menjalankan

### 1️⃣ Persiapan Awal

```bash
# Clone atau download project ini
# Pastikan Anda berada di folder project
cd path/to/sugarcane-disease-detection
```

### 2️⃣ Buat Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Catatan:** Proses ini mungkin memakan waktu 5-10 menit tergantung kecepatan internet.

### 4️⃣ Persiapan Model

Pastikan file model berada di folder yang sama dengan `app.py`:
```
sugarcane-disease-detection/
├── app.py
├── feature_extraction.py
├── requirements.txt
├── README.md
└── sugarcane_disease_classifier_full.pkl  ← FILE MODEL HARUS ADA DI SINI
```

### 5️⃣ Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan otomatis membuka di browser pada alamat:
```
http://localhost:8501
```

## 💻 Cara Menggunakan di Visual Studio Code

### 1. Buka Project di VSCode

```bash
code .
```

### 2. Install Python Extension

- Buka Extensions (Ctrl+Shift+X)
- Cari "Python"
- Install extension dari Microsoft

### 3. Pilih Python Interpreter

- Tekan `Ctrl+Shift+P`
- Ketik "Python: Select Interpreter"
- Pilih interpreter dari virtual environment (`venv`)

### 4. Buka Terminal di VSCode

- Menu: Terminal → New Terminal
- Atau tekan `` Ctrl+` ``

### 5. Jalankan Aplikasi

```bash
# Pastikan virtual environment aktif (lihat (venv) di terminal)
streamlit run app.py
```

## 📸 Cara Menggunakan Aplikasi

1. **Upload Gambar**
   - Klik tombol "Browse files"
   - Pilih gambar daun tebu (format: JPG, JPEG, PNG)
   - Gambar akan otomatis diproses

2. **Lihat Hasil Klasifikasi**
   - Nama penyakit terdeteksi
   - Tingkat keyakinan (confidence)
   - Deskripsi dan karakteristik penyakit
   - Rekomendasi penanganan

3. **Analisis XAI (Explainable AI)**
   - **Tab Force Plot**: Lihat kontribusi fitur dari base value ke prediksi akhir
   - **Tab Waterfall Plot**: Visualisasi kumulatif top 15 fitur
   - **Tab Feature Importance**: Ranking 20 fitur paling berpengaruh

## 📁 Struktur File

```
sugarcane-disease-detection/
│
├── app.py                              # Main aplikasi Streamlit
├── feature_extraction.py               # Modul ekstraksi fitur
├── requirements.txt                    # Dependencies Python
├── README.md                           # Dokumentasi (file ini)
├── sugarcane_disease_classifier_full.pkl  # Model terlatih
│
└── venv/                               # Virtual environment (dibuat saat instalasi)
```

## 🎨 Penjelasan Kelas Penyakit

| Kelas | Deskripsi | Warna Label |
|-------|-----------|-------------|
| **Healthy** | Daun sehat tanpa gejala penyakit | Hijau |
| **Mosaic** | Penyakit viral dengan pola mosaik | Orange |
| **RedRot** | Busuk merah pada batang dan daun | Merah |
| **Rust** | Infeksi jamur seperti karat | Coklat |
| **Yellow** | Daun menguning (defisiensi/stress) | Kuning |

## 🔧 Troubleshooting

### Error: "Module not found"
```bash
# Pastikan virtual environment aktif
# Install ulang dependencies
pip install -r requirements.txt
```

### Error: "Model file not found"
```bash
# Pastikan file sugarcane_disease_classifier_full.pkl ada di folder yang sama dengan app.py
```

### Error: Port already in use
```bash
# Gunakan port berbeda
streamlit run app.py --server.port 8502
```

### Aplikasi lambat saat loading
- SHAP computation membutuhkan waktu (normal 10-30 detik)
- Pastikan RAM cukup (minimum 4GB)

## 📊 Informasi Model

- **Model Type**: Support Vector Machine (SVM)
- **Kernel**: RBF/Polynomial
- **Features**: ~1000+ visual features
  - Color Histograms (RGB, HSV, LAB, YCrCb)
  - Statistical Moments
  - GLCM Texture
  - LBP Texture
  - Gabor Wavelets
  - HOG Features
  - Edge Features
- **Accuracy**: >90%
- **Classes**: 5 (Healthy, Mosaic, RedRot, Rust, Yellow)

## ⚙️ Konfigurasi Lanjutan

### Mengubah Port Default

Edit `app.py` atau jalankan dengan parameter:
```bash
streamlit run app.py --server.port 8080
```

### Disable File Watcher (untuk development)

```bash
streamlit run app.py --server.fileWatcherType none
```

### Mengubah Theme

Buat file `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#2e7d32"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## 📝 Catatan Penting

1. **File Model**: Pastikan file `sugarcane_disease_classifier_full.pkl` berada di folder yang sama dengan `app.py`

2. **Virtual Environment**: Selalu aktifkan virtual environment sebelum menjalankan aplikasi:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **First Run**: Saat pertama kali load model dan compute SHAP, aplikasi akan lebih lambat (normal)

4. **Browser**: Aplikasi berjalan di browser. Gunakan Chrome/Firefox untuk performa terbaik

## 🤝 Support

Jika mengalami masalah:
1. Pastikan semua dependencies terinstall dengan benar
2. Periksa versi Python (3.8+)
3. Pastikan file model ada dan tidak corrupt
4. Cek error message di terminal

## 📜 License

Project ini dikembangkan untuk keperluan edukasi dan penelitian.

---

**Dikembangkan dengan ❤️ untuk membantu petani tebu Indonesia**

🌾 Happy Farming! 🌾
