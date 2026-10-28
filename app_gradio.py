import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import gc

import gradio as gr
import torch
import requests
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# KONFIGURASI Latar Belakang
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://127.0.0.1:11434"

# PEMERIKSAAN KOMPONEN LOKAL
def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")

def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Ollama" / "ollama.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None

def get_ollama_models() -> list[str]:
    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    resp.raise_for_status()
    return [item.get("name", "") for item in resp.json().get("models", [])]

def model_is_installed(requested: str, installed: list[str]) -> bool:
    req = requested.strip()
    for name in installed:
        if name == req or name.split(":")[0] == req.split(":")[0]:
            return True
    return False

# WHISPER ENGINE
_asr_cache: dict = {}

def load_whisper(use_gpu: bool):
    key = "gpu" if use_gpu else "cpu"
    if key in _asr_cache:
        return _asr_cache[key]

    device = "cuda:0" if use_gpu else "cpu"
    dtype = torch.float16 if use_gpu else torch.float32

    if use_gpu:
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    batch_size = 8 if use_gpu else 4

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=batch_size,
    )

    _asr_cache[key] = asr
    return asr

# AUDIO CONVERSION & TRANSCRIPTION
def convert_to_wav(input_path: str, output_path: str):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg tidak ditemukan.")

    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            output_path,
        ],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=None, check=False,
    )

    if result.returncode != 0:
        raise RuntimeError("Gagal memproses media:\n" + result.stderr[-2000:])

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("File hasil konversi kosong.")

def transcribe_audio(asr, wav_path: str, with_timestamps: bool) -> tuple[str, str]:
    with torch.inference_mode():
        result = asr(
            wav_path,
            return_timestamps=with_timestamps,
            generate_kwargs={
                "language": "indonesian",
                "task": "transcribe",
                "num_beams": 5,
                "condition_on_prev_tokens": True,
                "compression_ratio_threshold": 2.4,
                "temperature": 0.0,
            },
        )

    transcript = result.get("text", "").strip()
    chunks = result.get("chunks", [])

    if with_timestamps and chunks:
        lines = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            ts = chunk.get("timestamp", (None, None))
            if not text:
                continue
            start = _fmt_ts(ts[0] if ts else None)
            end = _fmt_ts(ts[1] if ts else None)
            lines.append(f"[{start} - {end}] {text}")
        timestamped = "\n".join(lines) if lines else transcript
    else:
        timestamped = transcript

    return transcript, timestamped

def _fmt_ts(seconds) -> str:
    if seconds is None:
        return "--:--:--"
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

# OLLAMA STREAMING & PROMPT
def ollama_stream_generator(model: str, prompt: str):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,
                "num_predict": 1024,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        },
        stream=True,
        timeout=1800,
    )
    response.raise_for_status()

    collected = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        try:
            data = json.loads(raw_line)
        except ValueError:
            continue

        token = data.get("response", "")
        if token:
            collected.append(token)
            yield "".join(collected)

        if data.get("done", False):
            break

    if not collected:
        raise RuntimeError("Ollama tidak mengembalikan respons.")

def make_prompt(transcript: str, title: str, date: str, context: str) -> str:
    ctx = context.strip() or "Tidak ada konteks tambahan."
    title = title.strip() or "Tidak disebutkan"
    date = date.strip() or "Tidak disebutkan"

    return f"""Anda adalah sekretaris rapat profesional yang menulis dalam bahasa Indonesia.
Buat notulen rapat yang faktual, rapi, dan mudah ditindaklanjuti.

ATURAN WAJIB:
- Jangan mengarang nama, angka, keputusan, peserta, atau tenggat.
- Jika informasi tidak tersedia, tulis "Tidak disebutkan dalam transkripsi".
- PENTING — Bedakan dengan ketat antara:
  * USULAN/SARAN: kalimat dengan kata "sebaiknya", "mungkin", "bagaimana kalau", "rencana".
    → Catat di bagian Topik dan Pembahasan, BUKAN di Keputusan.
  * KEPUTUSAN FINAL: kalimat kesepakatan ("sepakat", "diputuskan", "setuju", "fix").
    → Catat HANYA di bagian Keputusan.
- Jika tidak ada keputusan formal: "Tidak ada keputusan formal yang dinyatakan dalam rapat ini."
- Perbaiki filler kata tanpa mengubah maksud kalimat.
- Format ringkas, jelas, dan profesional.

INFORMASI RAPAT:
Judul: {title}
Tanggal: {date}
Konteks: {ctx}

BAGIAN NOTULEN:
# NOTULEN RAPAT
## 1. Informasi Rapat
## 2. Ringkasan Eksekutif
## 3. Topik dan Pembahasan
## 4. Keputusan
## 5. Tindak Lanjut (Action Items) (Gunakan format tabel: Tugas | Penanggung Jawab | Tenggat | Status)
## 6. Risiko & Hal yang Perlu Dikonfirmasi
## 7. Agenda Berikutnya

TRANSKRIPSI:
\"\"\"
{transcript}
\"\"\""""

# EXPORT DOCX
def export_docx(markdown_text: str, title: str) -> bytes:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(0.7)
    sec.left_margin = sec.right_margin = Inches(0.8)
    doc.core_properties.title = title or "Notulen Rapat"
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)

    # Fungsi internal untuk membaca sintaks **bold**
    def add_formatted_text(paragraph, text: str):
        parts = text.split("**")
        for i, part in enumerate(parts):
            if not part: continue
            run = paragraph.add_run(part.replace("`", ""))
            # Index ganjil adalah teks yang berada di dalam **...**
            if i % 2 != 0:
                run.bold = True

    in_table = False
    table_data = []

    def build_word_table():
        if not table_data: return
        cols = max((len(r) for r in table_data), default=0)
        if cols == 0: return
        table = doc.add_table(rows=len(table_data), cols=cols)
        table.style = 'Table Grid'
        for r_idx, row in enumerate(table_data):
            cells = table.rows[r_idx].cells
            for c_idx, cell_text in enumerate(row):
                if c_idx < len(cells):
                    p = cells[c_idx].paragraphs[0]
                    add_formatted_text(p, cell_text) # Terapkan format bold di dalam tabel
                    if r_idx == 0:
                        for run in p.runs:
                            run.bold = True # Header tabel otomatis tebal
        table_data.clear()

    lines = markdown_text.splitlines()
    for raw in lines:
        line = raw.strip()
        
        # Penanganan baris kosong
        if not line:
            if in_table:
                build_word_table()
                in_table = False
            continue

        # Penanganan Tabel
        if line.startswith("|") or (in_table and "|" in line):
            in_table = True
            # Abaikan garis pemisah tabel markdown (misal: |---|---|)
            if re.match(r"^[\|\-\s:]+$", line):
                continue
            # Pisahkan sel dan buang bagian tepi yang kosong
            cells = [c.strip() for c in line.split("|")]
            if cells and not cells[0]: cells = cells[1:]
            if cells and not cells[-1]: cells = cells[:-1]
            if cells:
                table_data.append(cells)
            continue
        else:
            if in_table:
                build_word_table()
                in_table = False

        # Penanganan Teks, Heading, dan List
        if line.startswith("# "):
            p = doc.add_heading(level=0)
            add_formatted_text(p, line[2:].strip())
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith("## "):
            p = doc.add_heading(level=1)
            add_formatted_text(p, line[3:].strip())
        elif line.startswith("### "):
            p = doc.add_heading(level=2)
            add_formatted_text(p, line[4:].strip())
        elif re.match(r"^[-*]\s+", line):
            p = doc.add_paragraph(style="List Bullet")
            add_formatted_text(p, re.sub(r"^[-*]\s+", "", line))
        elif re.match(r"^\d+[.)]\s+", line):
            p = doc.add_paragraph(style="List Number")
            add_formatted_text(p, re.sub(r"^\d+[.)]\s+", "", line))
        else:
            p = doc.add_paragraph()
            add_formatted_text(p, line)

    # Render sisa tabel jika berada di akhir file
    if in_table:
        build_word_table()

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()

def export_word_file(notulen_text: str, meeting_title: str) -> str | None:
    if not notulen_text or not notulen_text.strip():
        raise gr.Error("Belum ada notulen untuk diekspor.")
    try:
        docx_bytes = export_docx(notulen_text, meeting_title)
        safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", meeting_title or "").strip("_")
        if not safe_title:
            safe_title = "Notulen_Rapat"
            
        path = tempfile.mktemp(suffix=".docx", prefix=f"{safe_title}_")
        Path(path).write_bytes(docx_bytes)
        
        # Tambahkan alert pop-up
        gr.Info("Dokumen Word berhasil disiapkan! Silakan simpan berkas Anda.")
        
        return path
    except ImportError:
        raise gr.Error("python-docx belum terinstal.")

# PIPELINE UTAMA
def run_pipeline(
    media_path: str | None,
    meeting_title: str,
    meeting_date: str,
    extra_context: str,
    with_timestamps: bool,
):
    if not media_path:
        yield "Masukkan berkas media terlebih dahulu.", "", "", None, None
        return

    if not find_ffmpeg():
        yield "Sistem membutuhkan FFmpeg. Pastikan FFmpeg sudah terinstal.", "", "", None, None
        return

    if not find_ollama():
        yield "Sistem membutuhkan Ollama. Pastikan Ollama sudah terinstal dan berjalan.", "", "", None, None
        return

    try:
        installed = get_ollama_models()
    except Exception:
        yield "Layanan Ollama tidak aktif.", "", "", None, None
        return

    if not model_is_installed(DEFAULT_OLLAMA_MODEL, installed):
        yield f"Model Ollama '{DEFAULT_OLLAMA_MODEL}' belum terinstal.", "", "", None, None
        return

    use_gpu = torch.cuda.is_available()
    input_path = None
    wav_path = None
    started = time.time()

    try:
        yield "Menyiapkan berkas media...", "", "", None, None
        suffix = Path(media_path).suffix.lower() or ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(Path(media_path).read_bytes())
            input_path = tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        convert_to_wav(input_path, wav_path)

        yield "Mentranskripsi rekaman...", "", "", None, None
        asr = load_whisper(use_gpu)
        transcript, timestamped = transcribe_audio(asr, wav_path, with_timestamps)

        if not transcript:
            yield "Suara tidak terdeteksi pada media yang diberikan.", "", "", None, None
            return

        yield "Menyusun notulen rapat...", timestamped, "", None, None

        # Hapus Whisper dari memori agar Ollama bisa masuk
        global _asr_cache
        _asr_cache.clear()
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

        prompt = make_prompt(transcript, meeting_title, meeting_date, extra_context)
        minutes = ""
        for partial in ollama_stream_generator(DEFAULT_OLLAMA_MODEL, prompt):
            minutes = partial
            yield "Menyusun notulen rapat...", timestamped, minutes, None, None

        elapsed = round(time.time() - started, 1)

        source_base = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(media_path).stem).strip("_") or "rapat"
        notulen_path = tempfile.mktemp(suffix=f"_Notulen_{source_base}.md")
        Path(notulen_path).write_text(minutes, encoding="utf-8")

        transkrip_path = tempfile.mktemp(suffix=f"_Transkripsi_{source_base}.txt")
        Path(transkrip_path).write_text(timestamped, encoding="utf-8")

        yield f"Proses selesai dalam {elapsed} detik.", timestamped, minutes, notulen_path, transkrip_path

    except Exception as exc:
        yield f"Terjadi kesalahan: {str(exc)}", "", "", None, None

    finally:
        for p in (input_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

# ANTARMUKA GRADIO

# 1. Konfigurasi Tema Gradio
modern_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

# 2. Injeksi CSS Kustom (Lebar Dimaksimalkan & Layout Grid)
custom_css = """
.gradio-container { max-width: 1450px !important; width: 95% !important; margin: auto !important; padding-top: 2rem !important; position: relative; }
.header-wrapper { text-align: center; margin-bottom: 1.5rem; }
.header-title { font-size: 2.2rem !important; font-weight: 700; margin-bottom: 0.5rem; }
.header-subtitle { font-size: 1.1rem; font-weight: 400; opacity: 0.8; }

/* Tombol Toggle Tema (Pojok Kanan Atas) */
.theme-toggle-btn { position: absolute !important; top: 15px !important; right: 15px !important; width: auto !important; background: var(--block-background-fill) !important; border: 1px solid var(--border-color-primary) !important; z-index: 1000; box-shadow: var(--block-shadow-sm) !important; }

/* Layout Kartu Modern */
.modern-card { border: 1px solid var(--border-color-primary) !important; border-radius: 12px !important; padding: 1.2rem !important; background: var(--block-background-fill); margin-bottom: 1rem; box-shadow: var(--block-shadow); }

/* Status Bar */
#status-card { background-color: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 1rem; box-shadow: var(--block-shadow); }
#status-text { font-size: 1.05rem; font-weight: 600; margin: 0; color: var(--color-accent); }

/* Tombol Utama */
.primary-btn { margin-top: 0 !important; border-radius: 10px !important; font-weight: bold !important; font-size: 1.1rem !important; letter-spacing: 0.3px; }
"""

# JS Paksa Light Mode Saat Halaman Dimuat
force_light_js = """
function() {
    document.body.classList.remove('dark');
}
"""

# 3. Struktur Layout Block
with gr.Blocks(title="Notulen Rapat Otomatis", theme=modern_theme, css=custom_css, js=force_light_js) as demo:
    
    # Tombol Ganti Tema
    theme_toggle = gr.Button("🌓 Terang / Gelap", size="sm", elem_classes="theme-toggle-btn")
    theme_toggle.click(None, None, None, js="() => { document.body.classList.toggle('dark'); }")
    
    # Bagian Header
    with gr.Column(elem_classes="header-wrapper"):
        gr.Markdown("<h1 class='header-title'>Sistem Notulen Otomatis</h1>")
        gr.Markdown("<p class='header-subtitle'>Transkripsi audio & video menjadi dokumen notulen.</p>")

    with gr.Row(equal_height=False):
        
        # KOLOM KIRI: Input & Pengaturan (Lebih Lebar dan Padat)
        with gr.Column(scale=5):
            
            # Kartu 1: Unggah Media
            with gr.Column(elem_classes="modern-card"):
                media_input = gr.File(
                    label="Unggah Rekaman (Audio / Video)",
                    file_types=["audio", "video"],
                    type="filepath",
                )
            
            # Kartu 2: Formulir Informasi (Dimampatkan menjadi 2 baris saja)
            with gr.Column(elem_classes="modern-card"):
                gr.Markdown("### 📝 Informasi Rapat")
                
                # Baris pertama formulir
                with gr.Row():
                    meeting_title = gr.Textbox(label="Judul Rapat", placeholder="Contoh: Evaluasi Q3", scale=1)
                    meeting_date = gr.Textbox(label="Tanggal", placeholder="Contoh: 28 Okt 2026", scale=1)
                
                # Baris kedua formulir (Konteks & Checkbox kini bersebelahan)
                with gr.Row():
                    extra_context = gr.Textbox(
                        label="Konteks Tambahan",
                        placeholder="Nama peserta, istilah teknis...",
                        lines=1,
                        scale=3
                    )
                    with_timestamps = gr.Checkbox(label="Sertakan Timestamp", value=True, scale=2)

            btn_process = gr.Button("Buat Notulen", variant="primary", size="lg", elem_classes="primary-btn")

        # KOLOM KANAN: Hasil & Output
        with gr.Column(scale=7):
            
            with gr.Column(elem_id="status-card"):
                status_out = gr.Markdown("Silakan unggah berkas untuk memulai.", elem_id="status-text")

            with gr.Column(elem_classes="modern-card"):
                with gr.Tabs():
                    with gr.TabItem("Notulen Rapat"):
                        notulen_out = gr.Markdown("*Hasil notulen yang telah disusun AI akan ditampilkan di sini.*")

                    with gr.TabItem("Transkrip Mentah"):
                        transcript_out = gr.Textbox(
                            label="Teks Transkripsi",
                            lines=15,
                            max_lines=20,
                            interactive=False,
                            show_label=False,
                            placeholder="Transkripsi lengkap media akan ditampilkan di sini.",
                        )

            with gr.Column(elem_classes="modern-card"):
                gr.Markdown("### 💾 Unduh Berkas")
                with gr.Row():
                    btn_export_word = gr.Button("1. Siapkan Word (.docx)", variant="secondary")
                    file_word = gr.DownloadButton("2. Unduh Word (.docx)", variant="primary")
                with gr.Row():
                    file_notulen = gr.DownloadButton("Unduh Markdown (.md)")
                    file_transkripsi = gr.DownloadButton("Unduh Transkrip (.txt)")

    notulen_state = gr.State("")
    
    # Alert Upload Berkas
    def on_upload_success():
        gr.Info("Berkas media berhasil diunggah! Sistem siap memproses.")
    
    media_input.upload(fn=on_upload_success, inputs=None, outputs=None)

    # Proses Utama
    btn_process.click(
        fn=run_pipeline,
        inputs=[
            media_input,
            meeting_title,
            meeting_date,
            extra_context,
            with_timestamps,
        ],
        outputs=[
            status_out,
            transcript_out,
            notulen_out,
            file_notulen,
            file_transkripsi,
        ],
    ).then(
        # Alert selesai diproses
        fn=lambda: gr.Info("Notulen dan transkripsi berhasil diselesaikan!"), 
        inputs=None, outputs=None
    )

    notulen_out.change(fn=lambda x: x, inputs=notulen_out, outputs=notulen_state)

    btn_export_word.click(
        fn=export_word_file,
        inputs=[notulen_state, meeting_title],
        outputs=[file_word],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
    )