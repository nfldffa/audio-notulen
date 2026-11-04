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
from faster_whisper import WhisperModel

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISBLE_SYMLINKS_WARNING"] = "1"

WHISPER_MODEL_SIZE = "large-v3"
DEFAULT_OLLAMA_MODEL = "qwen2.5:32b"
OLLAMA_URL = "http://127.0.0.1:11434"

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

_asr_cache = {}

def load_whisper(use_gpu: bool):
    key = "gpu" if use_gpu else "cpu"
    if key in _asr_cache:
        return _asr_cache[key]
    device = "cuda" if use_gpu else "cpu"
    compute_type = "float16" if use_gpu else "float32"
    if use_gpu:
        torch.cuda.empty_cache()
    model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    _asr_cache[key] = model
    return model

def convert_to_wav(input_path: str, output_path: str):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg tidak ditemukan.")
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            output_path,
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Gagal memproses media:\n" + result.stderr[-2000:])
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("File hasil konversi kosong.")

def transcribe_audio(model, wav_path: str, with_timestamps: bool, extra_context: str) -> tuple[str, str]:
    segments, info = model.transcribe(
        wav_path,
        language="id",
        beam_size=5,
        condition_on_previous_text=False,
        initial_prompt=extra_context.strip() if extra_context else None,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vad_filter=True,
    )
    lines, full_text = [], []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        full_text.append(text)
        if with_timestamps:
            start = _fmt_ts(segment.start)
            end = _fmt_ts(segment.end)
            lines.append(f"[{start} - {end}] {text}")
    transcript = " ".join(full_text)
    timestamped = "\n".join(lines) if (with_timestamps and lines) else transcript
    return transcript, timestamped

def _fmt_ts(seconds) -> str:
    if seconds is None:
        return "--:--:--"
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def ollama_generate_sync(model: str, prompt: str, context_window: int) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": context_window,
                "num_predict": 2048,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        },
        timeout=1800,
    )
    response.raise_for_status()
    return response.json().get("response", "")

def split_text_into_chunks(text: str, chunk_size: int = 2500, overlap: int = 200) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def ollama_stream_generator(model: str, prompt: str, transcript_length: int):
    estimated_tokens = transcript_length // 4
    context_window = max(8192, min(32768, estimated_tokens + 2048))
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.0,
                "num_ctx": context_window,
                "num_predict": 3072,
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
    return f"""Anda adalah Sekretaris Eksekutif yang ditugaskan menyusun Notulen Rapat dari transkrip yang disediakan. Gunakan bahasa Indonesia yang formal, rapi, dan mudah ditindaklanjuti.

ATURAN WAJIB:
1. FAKTA SAJA: Jangan mengarang nama, angka, keputusan, peserta, atau tenggat waktu. Jika informasi tidak ada di transkrip, tulis "Tidak disebutkan".
2. KEPUTUSAN VS USULAN:
   - USULAN: Rencana, opini, saran ("sebaiknya", "bagaimana kalau"). -> Masukkan ke Topik Pembahasan.
   - KEPUTUSAN: Kesepakatan final ("sepakat", "diputuskan", "fix"). -> Masukkan HANYA ke Keputusan.
3. KOREKSI BAHASA: Perbaiki kalimat terbata-bata atau kata pengisi karena audio samar tanpa mengubah esensi informasi.
4. FORMAT: Gunakan struktur Markdown persis seperti di bawah ini tanpa menambahkan bagian lain.

# INFORMASI RAPAT
* Judul: {title}
* Tanggal: {date}
* Konteks: {ctx}

# STRUKTUR NOTULEN
## 1. Informasi Rapat
## 2. Ringkasan Eksekutif
## 3. Topik dan Pembahasan
## 4. Keputusan
## 5. Tindak Lanjut (Action Items)
(Harus berupa tabel Markdown dengan kolom: Tugas | Penanggung Jawab | Tenggat Waktu | Status)
## 6. Risiko & Hal yang Perlu Dikonfirmasi
## 7. Agenda Berikutnya

Transkrip:
<transkrip>
{transcript}
</transkrip>

Buat notulen sekarang:"""

def export_docx(markdown_text: str, title: str) -> bytes:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(0.7)
    sec.left_margin = sec.right_margin = Inches(0.8)
    doc.core_properties.title = title or "Notulen"
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)

    def add_formatted_text(paragraph, text: str):
        parts = text.split("**")
        for i, part in enumerate(parts):
            if not part:
                continue
            run = paragraph.add_run(part.replace("`", ""))
            if i % 2 != 0:
                run.bold = True

    in_table = False
    table_data = []

    def build_word_table():
        if not table_data:
            return
        cols = max((len(r) for r in table_data), default=0)
        if cols == 0:
            return
        table = doc.add_table(rows=len(table_data), cols=cols)
        table.style = 'Table Grid'
        for r_idx, row in enumerate(table_data):
            cells = table.rows[r_idx].cells
            for c_idx, cell_text in enumerate(row):
                if c_idx < len(cells):
                    p = cells[c_idx].paragraphs[0]
                    add_formatted_text(p, cell_text)
                    if r_idx == 0:
                        for run in p.runs:
                            run.bold = True
        table_data.clear()

    for raw in markdown_text.splitlines():
        line = raw.strip()
        if not line:
            if in_table:
                build_word_table()
                in_table = False
            continue
        if line.startswith("|") or (in_table and "|" in line):
            in_table = True
            if re.match(r"^[\|\-\s:]+$", line):
                continue
            cells = [c.strip() for c in line.split("|")]
            if cells and not cells[0]:
                cells = cells[1:]
            if cells and not cells[-1]:
                cells = cells[:-1]
            if cells:
                table_data.append(cells)
            continue
        else:
            if in_table:
                build_word_table()
                in_table = False

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
            safe_title = "Notulen"
        path = tempfile.mktemp(suffix=".docx", prefix=f"{safe_title}_")
        Path(path).write_bytes(docx_bytes)
        gr.Info("Dokumen Word berhasil disiapkan.")
        return path
    except ImportError:
        raise gr.Error("python-docx belum terinstal.")

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
        yield "FFmpeg tidak ditemukan.", "", "", None, None
        return
    if not find_ollama():
        yield "Ollama tidak aktif.", "", "", None, None
        return
    try:
        installed = get_ollama_models()
    except Exception:
        yield "Gagal terhubung ke Ollama.", "", "", None, None
        return
    if not model_is_installed(DEFAULT_OLLAMA_MODEL, installed):
        yield f"Model {DEFAULT_OLLAMA_MODEL} belum terinstal.", "", "", None, None
        return

    use_gpu = torch.cuda.is_available()
    input_path = None
    wav_path = None
    started = time.time()

    try:
        yield "Menyiapkan media...", "", "", None, None
        suffix = Path(media_path).suffix.lower() or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(Path(media_path).read_bytes())
            input_path = tmp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        convert_to_wav(input_path, wav_path)

        yield "Mentranskripsi...", "", "", None, None
        asr = load_whisper(use_gpu)
        transcript, timestamped = transcribe_audio(asr, wav_path, with_timestamps, extra_context)

        if not transcript:
            yield "Suara tidak terdeteksi.", "", "", None, None
            return

        global _asr_cache
        _asr_cache.clear()
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

        words_count = len(transcript.split())
        minutes = ""

        if words_count > 3500:
            yield f"Menganalisis rapat panjang ({words_count} kata)...", timestamped, "Memecah transkrip...", None, None
            chunks = split_text_into_chunks(transcript, chunk_size=2500, overlap=200)
            intermediate_summaries = []

            for i, chunk in enumerate(chunks):
                yield f"Mengekstrak bagian {i+1} dari {len(chunks)}...", timestamped, f"Membaca bagian {i+1}...", None, None
                chunk_prompt = f"Ekstrak topik, fakta, keputusan, dan tugas dari potongan transkrip berikut:\n\n{chunk}"
                chunk_ctx = max(4096, min(16384, (len(chunk) // 4) + 1024))
                summary = ollama_generate_sync(DEFAULT_OLLAMA_MODEL, chunk_prompt, chunk_ctx)
                intermediate_summaries.append(f"--- Ringkasan Bagian {i+1} ---\n{summary}")

            combined_text = "\n\n".join(intermediate_summaries)
            prompt = make_prompt(combined_text, meeting_title, meeting_date, extra_context)

            yield "Menyusun notulen final...", timestamped, "Merangkum hasil...", None, None
            for partial in ollama_stream_generator(DEFAULT_OLLAMA_MODEL, prompt, len(combined_text)):
                minutes = partial
                yield "Menyusun notulen final...", timestamped, minutes, None, None
        else:
            prompt = make_prompt(transcript, meeting_title, meeting_date, extra_context)
            yield "Menyusun notulen...", timestamped, "", None, None
            for partial in ollama_stream_generator(DEFAULT_OLLAMA_MODEL, prompt, len(transcript)):
                minutes = partial
                yield "Menyusun notulen...", timestamped, minutes, None, None

        elapsed = round(time.time() - started, 1)
        source_base = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(media_path).stem).strip("_") or "rapat"
        notulen_path = tempfile.mktemp(suffix=f"_Notulen_{source_base}.md")
        Path(notulen_path).write_text(minutes, encoding="utf-8")
        transkrip_path = tempfile.mktemp(suffix=f"_Transkripsi_{source_base}.txt")
        Path(transkrip_path).write_text(timestamped, encoding="utf-8")

        yield f"Selesai dalam {elapsed} detik.", timestamped, minutes, notulen_path, transkrip_path

    except Exception as exc:
        yield f"Terjadi kesalahan: {str(exc)}", "", "", None, None

    finally:
        for p in (input_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

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

custom_css = """
.gradio-container { max-width: 1450px !important; width: 95% !important; margin: auto !important; padding-top: 2rem !important; position: relative; }
.header-wrapper { text-align: center; margin-bottom: 1.5rem; }
.header-title { font-size: 2.2rem !important; font-weight: 700; margin-bottom: 0.5rem; }
.header-subtitle { font-size: 1.1rem; font-weight: 400; opacity: 0.8; }
.theme-toggle-btn { position: absolute !important; top: 15px !important; right: 15px !important; width: auto !important; background: var(--block-background-fill) !important; border: 1px solid var(--border-color-primary) !important; z-index: 1000; box-shadow: var(--block-shadow-sm) !important; }
.modern-card { border: 1px solid var(--border-color-primary) !important; border-radius: 12px !important; padding: 1.2rem !important; background: var(--block-background-fill); margin-bottom: 1rem; box-shadow: var(--block-shadow); }
#status-card { background-color: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 1rem; box-shadow: var(--block-shadow); }
#status-text { font-size: 1.05rem; font-weight: 600; margin: 0; color: var(--color-accent); }
.primary-btn { margin-top: 0 !important; border-radius: 10px !important; font-weight: bold !important; font-size: 1.1rem !important; letter-spacing: 0.3px; }
"""

force_light_js = "function() { document.body.classList.remove('dark'); }"

with gr.Blocks(title="Notulen Otomatis") as demo:
    theme_toggle = gr.Button("Terang / Gelap", size="sm", elem_classes="theme-toggle-btn")
    theme_toggle.click(None, None, None, js="() => { document.body.classList.toggle('dark'); }")
    
    with gr.Column(elem_classes="header-wrapper"):
        gr.Markdown("<h1 class='header-title'>Sistem Notulen Otomatis</h1>")
        gr.Markdown("<p class='header-subtitle'>Transkripsi audio & video menjadi dokumen notulen.</p>")

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            with gr.Column(elem_classes="modern-card"):
                media_input = gr.File(
                    label="Unggah Rekaman (Audio / Video)",
                    file_types=["audio", "video"],
                    type="filepath",
                )
            with gr.Column(elem_classes="modern-card"):
                gr.Markdown("### Informasi")
                with gr.Row():
                    meeting_title = gr.Textbox(label="Judul", placeholder="Contoh: Evaluasi Q3", scale=1)
                    meeting_date = gr.Textbox(label="Tanggal", placeholder="Contoh: 28 Okt 2026", scale=1)
                with gr.Row():
                    extra_context = gr.Textbox(
                        label="Konteks Tambahan",
                        placeholder="Nama peserta, istilah teknis...",
                        lines=1,
                        scale=3
                    )
                    with_timestamps = gr.Checkbox(label="Sertakan Timestamp", value=True, scale=2)

            btn_process = gr.Button("Buat Notulen", variant="primary", size="lg", elem_classes="primary-btn")

        with gr.Column(scale=7):
            with gr.Column(elem_id="status-card"):
                status_out = gr.Markdown("Silakan unggah berkas untuk memulai.", elem_id="status-text")

            with gr.Column(elem_classes="modern-card"):
                with gr.Tabs():
                    with gr.TabItem("Notulen"):
                        notulen_out = gr.Markdown("*Hasil notulen akan ditampilkan di sini.*")
                    with gr.TabItem("Transkrip Mentah"):
                        transcript_out = gr.Textbox(
                            label="Teks Transkripsi",
                            lines=15,
                            max_lines=20,
                            interactive=False,
                            show_label=False,
                            placeholder="Transkripsi lengkap akan ditampilkan di sini.",
                        )

            with gr.Column(elem_classes="modern-card"):
                gr.Markdown("### Unduh Berkas")
                with gr.Row():
                    btn_export_word = gr.Button("1. Siapkan Word (.docx)", variant="secondary")
                    file_word = gr.DownloadButton("2. Unduh Word (.docx)", variant="primary")
                with gr.Row():
                    file_notulen = gr.DownloadButton("Unduh Markdown (.md)")
                    file_transkripsi = gr.DownloadButton("Unduh Transkrip (.txt)")

    notulen_state = gr.State("")
    
    media_input.upload(fn=lambda: gr.Info("Berkas berhasil diunggah."), inputs=None, outputs=None)

    btn_process.click(
        fn=run_pipeline,
        inputs=[media_input, meeting_title, meeting_date, extra_context, with_timestamps],
        outputs=[status_out, transcript_out, notulen_out, file_notulen, file_transkripsi],
    ).then(
        fn=lambda: gr.Info("Proses selesai."), 
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
        server_name="0.0.0.0",
        server_port=80,
        inbrowser=True,
        show_error=True,
        theme=modern_theme,
        css=custom_css,
        js=force_light_js,
    )