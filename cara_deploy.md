# 🚀 Panduan Deploy Aplikasi Deteksi Penyakit Daun Tebu

Panduan lengkap untuk mempublikasikan aplikasi ke internet dengan **API Key Gemini tetap aman**.

---

## 📋 Daftar Isi

1. [Persiapan Sebelum Deploy](#1-persiapan-sebelum-deploy)
2. [Pilihan Platform Deploy](#2-pilihan-platform-deploy)
3. [Deploy ke Streamlit Cloud (Recommended)](#3-deploy-ke-streamlit-cloud-recommended)
4. [Deploy ke Hugging Face Spaces](#4-deploy-ke-hugging-face-spaces-alternatif)
5. [Konfigurasi API Key yang Aman](#5-konfigurasi-api-key-yang-aman)
6. [Troubleshooting](#6-troubleshooting)
7. [Tips Keamanan](#7-tips-keamanan)

---

## 1. Persiapan Sebelum Deploy

### ✅ Checklist Persiapan

- [ ] **Akun GitHub** - Daftar gratis di [github.com](https://github.com)
- [ ] **API Key Gemini** - Dapatkan dari [makersuite.google.com](https://makersuite.google.com)
- [ ] **Model file** - Pastikan `sugarcane_disease_classifier_full.pkl` ada
- [ ] **File aplikasi** lengkap:
  - `app.py` (file utama)
  - `feature_extraction.py`
  - `requirements.txt`
  - `pages/Tutorial.py`
  - `assets/Benar_Salah.png` (gambar tutorial)

### 📁 Struktur Folder yang Benar

```
sugarcane-disease-detection/
│
├── app.py                                    # File utama aplikasi
├── feature_extraction.py                     # Module ekstraksi fitur
├── requirements.txt                          # Dependencies Python
├── sugarcane_disease_classifier_full.pkl     # Model ML (PENTING!)
│
├── pages/                                    # Halaman tambahan
│   └── Tutorial.py
│
├── assets/                                   # Gambar/resource
│   └── Benar_Salah.png
│
└── .streamlit/                               # Konfigurasi (akan dibuat)
    └── secrets.toml                          # API KEY DI SINI (LOKAL)
```

---

## 2. Pilihan Platform Deploy

### 🏆 Streamlit Cloud (GRATIS & MUDAH) ⭐ **RECOMMENDED**

**Kelebihan:**
- ✅ Gratis selamanya untuk public apps
- ✅ Integrasi langsung dengan GitHub
- ✅ Auto-deploy setiap kali update code
- ✅ Secret management built-in (API key aman!)
- ✅ Custom domain gratis (.streamlit.app)

**Kekurangan:**
- ⚠️ Resource terbatas (1GB RAM per app)
- ⚠️ Cold start ~10-20 detik jika tidak diakses lama

### 🤗 Hugging Face Spaces (ALTERNATIF)

**Kelebihan:**
- ✅ Gratis untuk public apps
- ✅ Resource lebih besar (2GB RAM)
- ✅ Community machine learning yang besar

**Kekurangan:**
- ⚠️ Setup sedikit lebih kompleks
- ⚠️ Interface kurang user-friendly

---

## 3. Deploy ke Streamlit Cloud (RECOMMENDED)

### 📝 Langkah 1: Siapkan Repository GitHub

#### A. Buat Repository Baru

1. Login ke [github.com](https://github.com)
2. Klik tombol **"New"** (hijau) atau **"+"** → **"New repository"**
3. Isi detail:
   - **Repository name:** `sugarcane-disease-detection`
   - **Description:** "Aplikasi deteksi penyakit daun tebu menggunakan AI"
   - **Visibility:** 🔓 **Public** (WAJIB untuk Streamlit Cloud gratis)
   - ✅ **Add README file**
4. Klik **"Create repository"**

#### B. Upload File ke GitHub

**Cara 1: Upload via Web (Paling Mudah)**

1. Di halaman repository, klik **"Add file"** → **"Upload files"**
2. Drag & drop semua file project Anda:
   ```
   app.py
   feature_extraction.py
   requirements.txt
   sugarcane_disease_classifier_full.pkl  ← PENTING!
   ```
3. **Buat folder** dengan cara:
   - Upload file di dalam folder dengan format: `pages/Tutorial.py`
   - Upload file di dalam folder: `assets/Benar_Salah.png`
4. Scroll ke bawah, tulis commit message: "Initial commit"
5. Klik **"Commit changes"**

**Cara 2: Upload via Git (Untuk yang terbiasa terminal)**

```bash
# Di folder project Anda
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/USERNAME/sugarcane-disease-detection.git
git push -u origin main
```

#### C. ⚠️ JANGAN Upload File `.streamlit/secrets.toml`

**PENTING!** File `secrets.toml` berisi API key yang rahasia. **JANGAN** di-upload ke GitHub!

Tambahkan file `.gitignore` dengan isi:
```
.streamlit/secrets.toml
*.pkl
__pycache__/
*.pyc
.env
```

**CATATAN:** Jika file `.pkl` sudah ter-upload sebelumnya, tidak masalah. Yang penting `secrets.toml` tidak boleh di-upload!

---

### 🚀 Langkah 2: Deploy ke Streamlit Cloud

#### A. Login Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Klik **"Sign up"** atau **"Continue with GitHub"**
3. Authorize Streamlit untuk akses GitHub Anda

#### B. Deploy Aplikasi

1. Klik tombol **"New app"** (atau **"Create app"**)
2. Isi form deployment:
   
   **Repository:**
   - Repository: `username/sugarcane-disease-detection`
   - Branch: `main`
   - Main file path: `app.py`
   
   **App URL (custom):**
   - `sugarcane-detector` (atau nama lain yang Anda mau)
   - URL final: `https://sugarcane-detector.streamlit.app`

3. Klik **"Deploy!"** (atau **"Deploy app"**)

#### C. Tunggu Proses Deploy

- **Progress bar** akan muncul
- Proses biasanya **5-10 menit** pertama kali
- Jika ada error, lihat di **"Manage app"** → **"Logs"**

---

### 🔐 Langkah 3: Tambahkan API Key Gemini (AMAN!)

#### A. Buka Settings Aplikasi

1. Setelah deploy selesai, klik **ikon gear ⚙️** atau **"..."** → **"Settings"**
2. Pilih tab **"Secrets"**

#### B. Tambahkan Secret

Di bagian **"Edit secrets"**, masukkan:

```toml
GEMINI_API_KEY = "AIzaSy...your-actual-api-key-here"
```

**⚠️ PERHATIAN:**
- **Ganti** `AIzaSy...` dengan API key ASLI Anda
- **Format HARUS TEPAT:** `GEMINI_API_KEY = "..."`
- **Tidak ada spasi** sebelum/sesudah tanda `=`
- API key harus dalam **tanda kutip ganda** `"..."`

**Contoh BENAR:**
```toml
GEMINI_API_KEY = "AIzaSyC1234abcd5678efgh_actual-key-example"
```

**Contoh SALAH:**
```toml
GEMINI_API_KEY= AIzaSyC1234...    ❌ (tidak ada spasi, tidak ada tanda kutip)
GEMINI_API_KEY ="AIzaSyC1234..."  ❌ (spasi salah tempat)
```

#### C. Save & Reboot

1. Klik **"Save"**
2. Aplikasi akan **restart otomatis** (~30 detik)
3. API key sekarang tersimpan aman di server Streamlit!

---

### ✅ Langkah 4: Verifikasi Deploy Berhasil

1. **Buka URL aplikasi:** `https://sugarcane-detector.streamlit.app`
2. **Test upload gambar** daun tebu
3. **Test analisis** → Pastikan hasil muncul
4. **Test rekomendasi AI** → Pastikan tidak ada error "API Key not found"

**Jika muncul error:** Lihat bagian [Troubleshooting](#6-troubleshooting)

---

## 4. Deploy ke Hugging Face Spaces (Alternatif)

### 📝 Langkah 1: Siapkan Akun & Space

1. Login ke [huggingface.co](https://huggingface.co)
2. Klik profil (kanan atas) → **"New Space"**
3. Isi detail:
   - **Space name:** `sugarcane-disease-detector`
   - **License:** Apache 2.0
   - **SDK:** **Streamlit**
   - **Visibility:** 🔓 Public
4. Klik **"Create Space"**

### 📤 Langkah 2: Upload Files

#### A. Upload via Web Interface

1. Di halaman Space, klik **"Files"** → **"Add file"** → **"Upload files"**
2. Upload semua file seperti di GitHub:
   ```
   app.py
   feature_extraction.py
   requirements.txt
   sugarcane_disease_classifier_full.pkl
   pages/Tutorial.py
   assets/Benar_Salah.png
   ```
3. **Commit changes**

#### B. Buat File `.streamlit/config.toml` (Optional)

Upload file baru `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 7860

[browser]
gatherUsageStats = false
```

### 🔐 Langkah 3: Tambahkan Secret (API Key)

1. Di Space, klik **"Settings"** (tab atas)
2. Scroll ke **"Repository secrets"**
3. Klik **"New secret"**:
   - **Name:** `GEMINI_API_KEY`
   - **Value:** `AIzaSy...your-actual-key`
4. Klik **"Add secret"**

### 📝 Langkah 4: Edit `app.py` untuk Hugging Face

Tambahkan di bagian `get_gemini_recommendation()`, setelah baris Streamlit secrets:

```python
# Di fungsi get_gemini_recommendation()

# OPTION 1: Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY", None)

# OPTION 2: Environment Variable (untuk Hugging Face)
import os
if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")
```

### ✅ Langkah 5: Verifikasi

1. Space akan auto-build (~3-5 menit)
2. Buka URL: `https://huggingface.co/spaces/USERNAME/sugarcane-disease-detector`
3. Test aplikasi seperti biasa

---

## 5. Konfigurasi API Key yang Aman

### 🔐 3 Cara Menyimpan API Key (Ranking Keamanan)

#### 1️⃣ **Streamlit Secrets** ⭐ PALING AMAN & MUDAH

**File lokal:** `.streamlit/secrets.toml` (JANGAN di-upload ke GitHub!)

```toml
GEMINI_API_KEY = "AIzaSy...your-key-here"
```

**Akses di code:**
```python
api_key = st.secrets.get("GEMINI_API_KEY", None)
```

**Untuk production (Streamlit Cloud):**
- Secrets disimpan di dashboard → Settings → Secrets
- Tidak pernah ter-expose di code atau Git history

---

#### 2️⃣ **Environment Variables** (Alternatif)

**Linux/Mac:**
```bash
export GEMINI_API_KEY="AIzaSy...your-key"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="AIzaSy...your-key"
```

**Akses di code:**
```python
import os
api_key = os.getenv("GEMINI_API_KEY")
```

---

#### 3️⃣ **Hardcode** ❌ TIDAK AMAN (hanya untuk testing lokal)

```python
# ⚠️ JANGAN LAKUKAN INI UNTUK PRODUCTION!
api_key = "AIzaSy...your-key"  # Akan ter-expose di GitHub!
```

**BAHAYA:**
- Siapapun yang akses GitHub bisa lihat API key Anda
- API key bisa disalahgunakan
- Google bisa suspend/revoke key jika terdeteksi di public repo

---

### 🛡️ Best Practice Keamanan

1. **SELALU gunakan Secrets** untuk production
2. **Tambahkan `.gitignore`** untuk file sensitif
3. **Rotate API key** secara berkala (6 bulan sekali)
4. **Monitor usage** API key di Google Cloud Console
5. **Set quota limits** untuk menghindari penyalahgunaan

---

## 6. Troubleshooting

### ❌ Error: "API Key not found"

**Penyebab:**
- Secret belum ditambahkan atau nama salah
- Format secret tidak tepat

**Solusi:**
1. Cek di Settings → Secrets, pastikan ada `GEMINI_API_KEY`
2. Pastikan format: `GEMINI_API_KEY = "..."`
3. Tidak ada typo di nama variable
4. Reboot aplikasi setelah save secrets

---

### ❌ Error: "Model file not found"

**Penyebab:**
- File `.pkl` tidak ter-upload ke GitHub

**Solusi:**
1. Cek repository, pastikan `sugarcane_disease_classifier_full.pkl` ada
2. Jika tidak ada, upload manual:
   - GitHub → Add file → Upload files
   - Pilih file `.pkl` dari komputer Anda
3. Commit & push
4. Reboot app di Streamlit Cloud

---

### ❌ Error: "Module not found" / Import Error

**Penyebab:**
- `requirements.txt` tidak lengkap atau salah

**Solusi:**
1. Cek `requirements.txt` sudah di-upload
2. Pastikan isinya lengkap:
```
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.26.0
joblib>=1.3.0
matplotlib>=3.7.0
shap>=0.42.0
Pillow>=10.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
pandas>=2.0.0
google-generativeai
```
3. Commit perubahan
4. Reboot app

---

### ❌ App Stuck di "Booting"

**Penyebab:**
- File `.pkl` terlalu besar
- Memory exceeded

**Solusi:**
1. **Compress model** (jika >100MB):
   ```python
   import joblib
   import gzip
   
   # Save compressed
   with gzip.open('model.pkl.gz', 'wb') as f:
       joblib.dump(model, f)
   
   # Load compressed
   with gzip.open('model.pkl.gz', 'rb') as f:
       model = joblib.load(f)
   ```

2. **Upgrade ke Streamlit Cloud Team** (paid) untuk resource lebih besar

---

### ❌ Error 429: Quota Exceeded (Gemini)

**Penyebab:**
- Quota API Gemini habis (free tier: 60 requests/menit)

**Solusi:**
1. **Tunggu 1 menit** untuk quota reset
2. **Gunakan fallback rule-based** (sudah ada di code):
   ```python
   # Code sudah handle ini otomatis
   # Jika Gemini error → fallback ke rule-based recommendation
   ```
3. **Upgrade API key** ke paid tier (jika usage tinggi)

---

### ❌ Aplikasi Lambat/Cold Start

**Penyebab:**
- Streamlit Cloud sleep mode setelah inactivity

**Solusi:**
1. **Acceptable:** First load 10-20 detik (normal)
2. **Keep-alive service** (advanced):
   - Buat cron job ping app setiap 10 menit
   - Gunakan service seperti [UptimeRobot](https://uptimerobot.com) (gratis)

---

### 🔍 Debug: Lihat Logs

**Streamlit Cloud:**
1. Dashboard → Your app → **"Manage app"**
2. Klik **"Logs"**
3. Lihat error message real-time

**Hugging Face:**
1. Space → **"Logs"** tab
2. Scroll ke error message

---

## 7. Tips Keamanan

### 🛡️ Checklist Keamanan

- [ ] ✅ **JANGAN** commit file `.streamlit/secrets.toml` ke Git
- [ ] ✅ Tambahkan `.gitignore` dengan:
  ```
  .streamlit/secrets.toml
  .env
  *.env
  ```
- [ ] ✅ Gunakan **Secrets** di platform deploy (Streamlit Cloud/Hugging Face)
- [ ] ✅ **Rotate API key** secara berkala
- [ ] ✅ **Monitor usage** API di Google Cloud Console
- [ ] ✅ Set **quota limits** untuk API key
- [ ] ✅ Jika key ter-expose: **REVOKE immediately** di Google Cloud Console

---

### 🔄 Cara Rotate API Key (Disarankan 6 Bulan Sekali)

1. **Buat API key baru** di [makersuite.google.com](https://makersuite.google.com)
2. **Update secret** di Streamlit Cloud:
   - Settings → Secrets → Edit
   - Ganti dengan key baru
   - Save
3. **Revoke key lama** di Google Cloud Console
4. **Test aplikasi** dengan key baru

---

### 🚨 Jika API Key Ter-Expose (Kebocoran)

**TINDAKAN SEGERA:**

1. **Login ke Google Cloud Console**
2. **API & Services** → **Credentials**
3. **Find your API key** → **Klik 3 titik** → **Delete** atau **Regenerate**
4. **Buat key baru** dan update di Streamlit Cloud
5. **Monitor usage** untuk aktivitas mencurigakan

---

## 📚 Resources Tambahan

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gemini API Docs](https://ai.google.dev/docs)
- [Git & GitHub Tutorial](https://docs.github.com/en/get-started)

---

## 🎉 Selamat!

Aplikasi Anda sekarang sudah **live** dan bisa diakses dari mana saja!

**URL Aplikasi:**
- Streamlit Cloud: `https://sugarcane-detector.streamlit.app`
- Hugging Face: `https://huggingface.co/spaces/USERNAME/sugarcane-disease-detector`

**Share dengan:**
- 🌾 Petani tebu
- 📚 Penyuluh pertanian
- 🎓 Mahasiswa & peneliti
- 🌐 Social media

---

## 💬 Butuh Bantuan?

Jika ada pertanyaan atau masalah:

1. **Cek Troubleshooting** di atas
2. **Lihat Logs** aplikasi untuk error message
3. **Google error message** untuk solusi cepat
4. **Streamlit Forum:** [discuss.streamlit.io](https://discuss.streamlit.io)

---

**Dibuat dengan ❤️ untuk membantu petani tebu Indonesia**