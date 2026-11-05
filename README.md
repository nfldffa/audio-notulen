# Sistem Notulensi Terpadu

Aplikasi web berbasis **Gradio 6.0+** dengan desain sistem modern (Bento Grid) dan mode Gelap/Terang untuk transkripsi audio/video rapat berbahasa Indonesia serta penyusunan notulen rapat formal eksekutif secara otomatis. Berjalan **100% lokal di komputer**, tanpa langganan API berbayar, serta menjaga privasi dan kerahasiaan data rapat secara penuh.

---

## Fitur Utama

1. **Antarmuka Modern & Responsif:** Desain Bento Grid dengan tombol sakelar Mode Gelap / Terang (Dark/Light Mode).
2. **Ekstraksi & Transkripsi Otomatis (STT):** Menggunakan **OpenAI Faster Whisper (Large-v3)** dengan deteksi bahasa Indonesia, filter VAD, dan dukungan GPU CUDA (opsional) atau CPU.
3. **Pemrosesan Teks Panjang Cerdas (LLM):** Menggunakan **Qwen 2.5 (32B)** via Ollama lokal. Otomatis melakukan pemisahan _chunking_ (>3500 kata) dan ringkasan bertahap untuk transkrip panjang.
4. **Generator Notulen Berstandar Eksekutif:** Menghasilkan dokumen dengan struktur baku:
   - Informasi Rapat (Judul, Tanggal, Referensi/Konteks)
   - Ringkasan Eksekutif
   - Topik dan Pembahasan
   - Keputusan
   - Tindak Lanjut (Action Items) dalam bentuk Tabel (Tugas | Penanggung Jawab | Tenggat Waktu | Status)
   - Risiko & Hal yang Perlu Dikonfirmasi
   - Agenda Berikutnya
5. **Multi-Format Export:** Unduh hasil akhir dalam format **Word (`.docx`)** dengan styling profesional (Font Aptos, Table Grid), **Markdown (`.md`)**, serta **Transkrip Mentah (`.txt`)** dengan opsi penanda waktu (_timestamp_).
6. **Pelacakan Status Real-Time:** Kartu status interaktif dengan indikator progres dan persentase proses.

---

## Teknologi yang Digunakan

- **Frontend / UI:** Gradio 6.0+ (Custom Soft Theme + Custom CSS)
- **Transkripsi (STT):** `faster-whisper` (Large-v3 model)
- **Pemrosesan Media:** FFmpeg (Ekstraksi audio otomatis dari video)
- **Penyusun Notulen (LLM):** Qwen 2.5 (32B) via Ollama (`http://127.0.0.1:11434`)
- **Ekspor Dokumen:** `python-docx`, Markdown, Teks

---

## Prasyarat Sistem

1. **Sistem Operasi:** Windows 10/11 64-bit (atau Linux/macOS)
2. **Python:** Python 3.10 – 3.12 (disarankan 64-bit)
3. **RAM:** Minimal 8 GB (disarankan 16 GB+)
4. **GPU (Direkomendasikan):** NVIDIA GPU dengan VRAM minimal 4 GB (misal GTX 1650 / RTX 3050 ke atas) untuk akselerasi CUDA. Jika tidak tersedia, otomatis berjalan di CPU.
5. **Perangkat Lunak Tambahan:**
   - **FFmpeg:** Wajib terinstal dan terdaftar di System PATH.
   - **Ollama:** Terinstal dan berjalan (`ollama.com`).

---

## Panduan Instalasi & Setup

### 1. Clone & Direktori Kerja
Buka terminal dan pastikan berada di folder proyek.

### 2. Buat & Aktifkan Virtual Environment
```bash
python -m venv env
```
- **Windows (Command Prompt):** `env\Scripts\activate`
- **Windows (PowerShell):** `.\env\Scripts\Activate.ps1`
- **Git Bash / Linux:** `source env/Scripts/activate`

### 3. Install Dependensi
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install faster-whisper
```

> **Catatan GPU NVIDIA:**
> Untuk mengaktifkan CUDA pada PyTorch:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Siapkan Ollama & Model
1. Pastikan Ollama aktif di latar belakang.
2. Unduh model Qwen 2.5 32B:
   ```bash
   ollama pull qwen2.5:32b
   ```
   *(Atau sesuaikan variabel `DEFAULT_OLLAMA_MODEL` di `app_gradio.py` jika menggunakan model lain).*

---

## Menjalankan Aplikasi

Jalankan skrip utama Gradio:
```bash
python app_gradio.py
```

Aplikasi akan otomatis berjalan pada server lokal di port 80:
👉 **`http://127.0.0.1`** (atau `http://localhost`)

---

## Cara Penggunaan

1. **01. Unggah Berkas Media:** Seret dan lepas berkas audio/video (`.wav`, `.mp3`, `.m4a`, `.mp4`, `.mkv`, `.webm`, dll.).
2. **02. Metadata Rapat:** Masukkan **Judul Agenda Rapat**, **Tanggal Pelaksanaan**, serta **Konteks & Glosarium Khusus** (akronim, nama narasumber, istilah teknis) untuk meningkatkan akurasi transkripsi. Anda juga dapat memilih opsi penanda waktu (_timestamp_).
3. **Mulai Pemrosesan:** Klik tombol **"Mulai Pemrosesan Data"**.
4. **03 & 04. Pelacakan & Pratinjau:** Pantau status progres secara real-time. Setelah selesai, tinjau hasil notulen pada tab **"Dokumen Notulensi Resmi"** atau transkrip pada tab **"Log Transkripsi Mentah"**.
5. **05. Unduh Berkas:** Unduh hasil akhir dalam format **Word (.docx)**, **Markdown (.md)**, atau **Transkrip (.txt)**.

---

## Catatan Teknis & Pembersihan Memori

- **Manajemen Memori:** Sistem dilengkapi dengan pembersihan cache otomatis (`gc.collect()` & `torch.cuda.empty_cache()`) setelah transkripsi selesai untuk mencegah kehabisan VRAM/RAM saat model LLM besar (`qwen2.5:32b`) dijalankan.
- **Privasi:** Seluruh proses berjalan secara lokal tanpa koneksi ke server eksternal. Berkas sementara (_temporary files_) otomatis dibersihkan setelah proses selesai.
