# Notulen Rapat Lokal Otomatis

Aplikasi web berbasis **Gradio** untuk transkripsi audio/video rapat berbahasa Indonesia dan penyusunan notulen rapat formal secara otomatis. Berjalan **100% lokal di komputer Anda**, gratis tanpa langganan API berbayar, serta menjaga privasi dan kerahasiaan data rapat secara penuh.

## Teknologi yang Digunakan

* **Transkripsi (STT):** OpenAI Whisper Large-v3-Turbo (via Hugging Face `transformers` & PyTorch)

* **Pemrosesan Media:** FFmpeg (Mendukung ekstraksi otomatis dari format Video ke Audio)

* **Penyusun Notulen (LLM):** Qwen 2.5 (7B / 3B) via Ollama lokal

* **Antarmuka Web:** Gradio 6.0+

* **Ekspor Dokumen:** Python-docx (Word `.docx`), Markdown (`.md`), dan Teks (`.txt`)

## Prasyarat Sistem

1. **Sistem Operasi:** Windows 10/11 64-bit (atau Linux/macOS)

2. **Python:** Python 3.10 – 3.12 (disarankan 64-bit)

3. **RAM:** Minimal 8 GB (disarankan 16 GB)

4. **GPU (Opsional tapi direkomendasikan):** NVIDIA GPU dengan VRAM minimal 4 GB (misal GTX 1650 / RTX 3050 ke atas) untuk pemrosesan cepat. *Jika sistem tidak memiliki GPU, aplikasi akan otomatis beralih ke mode CPU.*

5. **Perangkat Lunak Tambahan:**

   * **FFmpeg:** Wajib terinstal dan terdaftar di System PATH OS Anda.

   * **Ollama:** Unduh dari [ollama.com](https://ollama.com).

## Panduan Instalasi & Setup

### 1. Clone Repository

Buka terminal (Git Bash, Command Prompt, atau PowerShell), lalu jalankan:

```
git clone https://github.com/nfldffa/audio-notulen.git
cd "Audio Notulen"

```

### 2. Siapkan Python Virtual Environment

Buat *virtual environment* agar dependensi terisolasi dengan rapi:

**Menggunakan Command Prompt / PowerShell:**

```
python -m venv env

```

**Aktifkan Virtual Environment:**

* **Windows (Command Prompt):**

  ```
  env\Scripts\activate
  
  ```

* **Windows (PowerShell):**

  ```
  .\env\Scripts\Activate.ps1
  
  ```

* **Git Bash / Linux:**

  ```
  source env/Scripts/activate
  
  ```

### 3. Install Dependensi

Perbarui `pip` dan pasang seluruh pustaka yang dibutuhkan:

```
python -m pip install --upgrade pip
pip install -r requirements.txt

```

> **Catatan Pengguna GPU NVIDIA:**
>
> Untuk mengaktifkan akselerasi GPU CUDA pada PyTorch, pasang PyTorch versi CUDA (jika belum terpasang otomatis):
>
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> 
> ```

### 4. Siapkan Ollama & Model LLM

1. Pastikan aplikasi **Ollama** sudah terpasang dan berjalan di latar belakang (cek ikon di taskbar).

2. Unduh model **Qwen 2.5 7B** (rekomendasi untuk akurasi terbaik):

   ```
   ollama pull qwen2.5:7b
   
   ```

   *Alternatif untuk komputer dengan RAM/VRAM sangat terbatas:*

   ```
   ollama pull qwen2.5:3b
   
   ```

   *(Catatan: Jika menggunakan model selain `qwen2.5:7b`, sesuaikan variabel `DEFAULT_OLLAMA_MODEL` di dalam file `app_gradio.py`).*

## Menjalankan Aplikasi

Pastikan *virtual environment* telah aktif, lalu jalankan perintah berikut di terminal:

```
python app_gradio.py

```

Atau cukup **klik dua kali** file batch Anda (jika ada):

```
Jalankan_Notulen.bat

```

Aplikasi akan otomatis terbuka di browser bawaan pada alamat:
👉 **`http://127.0.0.1:7860`**

## Cara Penggunaan

1. **Unggah Berkas Media:**

   * Seret dan lepas (drag & drop) atau klik untuk mengunggah berkas audio maupun video (`.wav`, `.mp3`, `.m4a`, `.mp4`, `.mkv`, `.webm`, dll.). Sistem akan secara otomatis mengekstrak audio dari video yang diunggah.

2. **Informasi Rapat (Opsional tapi disarankan):**

   * Isi **Judul Rapat**, **Tanggal**, dan **Konteks Tambahan** (seperti nama-nama peserta, akronim khusus, atau topik spesifik) agar hasil notulen lebih akurat.

   * Centang opsi *timestamp* jika Anda membutuhkan penanda waktu pada transkripsi.

3. **Mulai Proses:**

   * Klik tombol **Buat Notulen**.

   * Sistem akan melakukan konversi media → transkripsi Whisper → pembersihan memori otomatis → generasi notulen oleh Ollama secara *streaming*.

4. **Unduh Berkas:**

   * Setelah proses selesai, unduh hasil notulen dalam format **Word (.docx)**, **Markdown (.md)**, atau transkrip teks mentah **(.txt)**.

## Catatan Teknis (Troubleshooting)

* **Aplikasi Crash / Error 500 di akhir proses:** Jika Anda menemui Error 500 saat transkripsi selesai dan Ollama mulai menyusun notulen, ini menandakan keterbatasan *System RAM* atau *VRAM*. Sistem telah dilengkapi pembersih memori otomatis (`gc.collect()` & `torch.cuda.empty_cache()`), namun jika masih terjadi, pertimbangkan untuk menggunakan model LLM yang lebih kecil (misalnya `qwen2.5:3b` atau `llama3.2`).

## Privasi Data

Semua rekaman media dan teks diproses sepenuhnya secara lokal di dalam mesin Anda. Tidak ada data, suara, maupun teks yang dikirim ke server luar atau pihak ketiga. Berkas *temporary* (sementara) yang dibuat selama proses akan otomatis dihapus oleh sistem setelah notulen selesai di-generate.