# 🎙️ Notulen Rapat Lokal Otomatis

Aplikasi web berbasis **Gradio** untuk transkripsi audio rapat berbahasa Indonesia dan penyusunan notulen rapat formal secara otomatis. Berjalan **100% lokal di komputer Anda**, gratis tanpa langganan API berbayar (OpenAI/Cloud), menjaga privasi dan kerahasiaan data rapat.

---

## ⚡ Teknologi yang Digunakan

- **Transkripsi (STT):** OpenAI Whisper Large-v3-Turbo (via Hugging Face `transformers` & PyTorch)
- **Pemrosesan Audio:** FFmpeg
- **Penyusun Notulen (LLM):** Qwen 2.5 (7B / 3B) via Ollama lokal
- **Antarmuka Web:** Gradio
- **Ekspor Dokumen:** Python-docx (Word `.docx`), Markdown (`.md`), dan Teks (`.txt`)

---

## 📋 Prasyarat Sistem

1. **Sistem Operasi:** Windows 10/11 64-bit (atau Linux/macOS)
2. **Python:** Python 3.10 – 3.12 (disarankan 64-bit)
3. **RAM:** Minimal 8 GB (disarankan 16 GB)
4. **GPU (Opsional tapi disarankan):** NVIDIA GPU dengan VRAM minimal 4 GB (misal GTX 1650 / RTX 3050 ke atas) untuk pemrosesan cepat. Jika tanpa GPU, aplikasi otomatis berjalan di CPU.
5. **Perangkat Lunak Tambahan:**
   - **FFmpeg:** Wajib terinstal dan terdaftar di System PATH Windows.
   - **Ollama:** Unduh dari [ollama.com](https://ollama.com).

---

## 🚀 Panduan Instalasi & Setup

### 1. Clone Repository

Buka terminal (Git Bash, Command Prompt, atau PowerShell), lalu jalankan:

```bash
git clone https://github.com/nfldffa/audio-notulen.git
cd "Audio Notulen"
```

*(Ganti `<username>` dan `<nama-repo>` dengan URL repository Anda).*

---

### 2. Siapkan Python Virtual Environment

Buat virtual environment agar dependensi terisolasi rapi:

**Menggunakan Command Prompt / PowerShell:**
```bash
python -m venv env
```

**Aktifkan Virtual Environment:**
- **Windows (Command Prompt):**
  ```cmd
  env\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .\env\Scripts\Activate.ps1
  ```
- **Git Bash / Linux:**
  ```bash
  source env/Scripts/activate
  ```

---

### 3. Install Dependensi

Perbarui `pip` dan pasang seluruh pustaka yang dibutuhkan:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Catatan Pengguna GPU NVIDIA:**  
> Untuk mengaktifkan akselerasi GPU CUDA pada PyTorch, pasang PyTorch versi CUDA (jika belum terpasang otomatis):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

### 4. Siapkan Ollama & Model LLM

1. Pastikan aplikasi **Ollama** sudah terpasang dan berjalan di background (ikon di taskbar).
2. Unduh model **Qwen 2.5 7B** (rekomendasi untuk akurasi terbaik):
   ```bash
   ollama pull qwen2.5:7b
   ```
   *Alternatif untuk komputer dengan RAM/VRAM terbatas (< 8 GB):*
   ```bash
   ollama pull qwen2.5:3b
   ```

---

## ▶️ Menjalankan Aplikasi

Pastikan virtual environment telah aktif, lalu jalankan:

```bash
python app_gradio.py
```

Atau cukup **klik dua kali** file:
```
Jalankan_Notulen.bat
```

Aplikasi akan otomatis terbuka di browser pada alamat:
👉 **`http://127.0.0.1:7860`**

---

## 📖 Cara Penggunaan

1. **Input Audio:**
   - Unggah berkas audio/video rekaman rapat (`.wav`, `.mp3`, `.m4a`, `.mp4`, `.webm`, dll.), **atau**
   - Rekam suara langsung menggunakan mikrofon.
2. **Informasi Rapat (Opsional):**
   - Buka menu *Informasi Rapat* untuk mengisi Judul, Tanggal, atau Catatan Tambahan (peserta, istilah teknis).
3. **Mulai Proses:**
   - Klik tombol **⚡ Buat Notulen**.
   - Sistem akan melakukan konversi audio → transkripsi Whisper → generasi notulen secara streaming.
4. **Unduh Berkas:**
   - Unduh hasil notulen dalam format **Word (.docx)**, **Markdown (.md)**, atau transkrip teks mentah **(.txt)**.

---

## 🔒 Privasi Data

Semua rekaman audio dan teks diproses secara lokal di mesin Anda. Tidak ada data yang dikirim ke server luar atau pihak ketiga. Berkas temporer otomatis dibersihkan setelah selesai diproses.

