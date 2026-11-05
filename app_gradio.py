import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import base64
from pathlib import Path
import gc

import av
_old_av_open = av.open
av.open = lambda *args, metadata_errors=None, **kwargs: _old_av_open(*args, **kwargs)

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import gradio as gr
import torch
import requests
from faster_whisper import WhisperModel

WHISPER_MODEL_SIZE = "large-v3"
DEFAULT_OLLAMA_MODEL = "qwen2.5:32b"
OLLAMA_URL = "http://127.0.0.1:11434"

logo_path = "logo_bakom.svg"
if os.path.exists(logo_path):
    try:
        with open(logo_path, "rb") as f:
            b64_logo = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/svg+xml;base64,{b64_logo}" class="app-logo" alt="Logo">'
    except Exception:
        logo_html = '<div class="app-logo-fallback">BKP</div>'
else:
    logo_html = '<div class="app-logo-fallback">BKP</div>'

def render_status(percent: int, text: str, state: str = "loading") -> str:
    if state == "idle":
        return f'<div class="status-card"><div class="status-icon info">ℹ</div><div class="status-content"><div class="status-title">Status Sistem</div><div class="status-desc">{text}</div></div></div>'
    if state == "error":
        return f'<div class="status-card error"><div class="status-icon err">✕</div><div class="status-content"><div class="status-title">Kesalahan Proses</div><div class="status-desc">{text}</div></div></div>'
    if state == "success":
        return f'<div class="status-card success"><div class="status-icon succ">✓</div><div class="status-content"><div class="status-title">Berhasil</div><div class="status-desc">{text}</div></div></div>'
        
    return f"""
    <div class="status-card processing">
        <div class="status-flex">
            <div class="spinner-modern"></div>
            <div style="flex: 1;">
                <div class="status-title">Memproses Data ({percent}%)</div>
                <div class="status-desc">{text}</div>
            </div>
        </div>
        <div class="progress-bar-wrap">
            <div class="progress-bar-bar" style="width: {percent}%;"></div>
        </div>
    </div>"""

def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")

def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found: return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Ollama" / "ollama.exe",
    ]
    for c in candidates:
        if c.is_file(): return str(c)
    return None

_asr_cache = {}

def load_whisper(use_gpu: bool):
    key = "gpu" if use_gpu else "cpu"
    if key in _asr_cache: return _asr_cache[key]
    device = "cuda" if use_gpu else "cpu"
    compute_type = "float16" if use_gpu else "float32"
    if use_gpu: torch.cuda.empty_cache()
    model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    _asr_cache[key] = model
    return model

def convert_to_wav(input_path: str, output_path: str):
    ffmpeg = find_ffmpeg()
    if not ffmpeg: raise RuntimeError("FFmpeg tidak ditemukan.")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0: raise RuntimeError("Gagal mengekstrak media.")
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0: raise RuntimeError("Berkas konversi kosong.")

def _fmt_ts(seconds) -> str:
    if seconds is None: return "--:--:--"
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def ollama_generate_sync(model: str, prompt: str, context_window: int) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "num_ctx": context_window, "num_predict": 2048}},
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
        if i + chunk_size >= len(words): break
    return chunks

def ollama_stream_generator(model: str, prompt: str, transcript_length: int):
    estimated_tokens = transcript_length // 4
    context_window = max(8192, min(32768, estimated_tokens + 2048))
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True, "options": {"temperature": 0.0, "num_ctx": context_window, "num_predict": 3072}},
        stream=True, timeout=1800,
    )
    response.raise_for_status()
    collected = []
    for raw_line in response.iter_lines():
        if not raw_line: continue
        try:
            data = json.loads(raw_line)
            token = data.get("response", "")
            if token:
                collected.append(token)
                yield "".join(collected)
            if data.get("done", False): break
        except ValueError: continue

def make_prompt(transcript: str, title: str, date: str, context: str) -> str:
    ctx = str(context).strip() or "Tidak ada konteks spesifik."
    title = str(title).strip() or "Tidak disebutkan"
    date_str = str(date).strip() if date else "Tidak disebutkan"
    return f"""Anda adalah Sekretaris Eksekutif. Buat Notulen Rapat formal berdasarkan transkrip berikut.

ATURAN:
1. FAKTA SAJA. Jika informasi tidak ada, tulis "Tidak disebutkan".
2. USULAN masuk ke Topik Pembahasan. KEPUTUSAN FINAL masuk ke Keputusan.
3. Perbaiki struktur tata bahasa tanpa mengubah makna asli.
4. Gunakan struktur Markdown persis seperti di bawah ini.

# INFORMASI RAPAT
* Judul: {title}
* Tanggal: {date_str}
* Referensi: {ctx}

# STRUKTUR NOTULEN
## 1. Informasi Rapat
## 2. Ringkasan Eksekutif
## 3. Topik dan Pembahasan
## 4. Keputusan
## 5. Tindak Lanjut (Action Items)
(Format tabel: Tugas | Penanggung Jawab | Tenggat Waktu | Status)
## 6. Risiko & Hal yang Perlu Dikonfirmasi
## 7. Agenda Berikutnya

Transkrip:
{transcript}

Susun dokumen sekarang:"""

def export_docx(markdown_text: str, title: str) -> bytes:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(0.7)
    sec.left_margin = sec.right_margin = Inches(0.8)
    doc.core_properties.title = str(title) or "Dokumen Notulensi"
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)

    def add_formatted_text(paragraph, text: str):
        parts = text.split("**")
        for i, part in enumerate(parts):
            if not part: continue
            run = paragraph.add_run(part.replace("`", ""))
            if i % 2 != 0: run.bold = True

    in_table, table_data = False, []

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
                    add_formatted_text(p, cell_text)
                    if r_idx == 0:
                        for run in p.runs: run.bold = True
        table_data.clear()

    for raw in markdown_text.splitlines():
        line = raw.strip()
        if not line:
            if in_table: build_word_table(); in_table = False
            continue
        if line.startswith("|") or (in_table and "|" in line):
            in_table = True
            if re.match(r"^[\|\-\s:]+$", line): continue
            cells = [c.strip() for c in line.split("|")]
            if cells and not cells[0]: cells = cells[1:]
            if cells and not cells[-1]: cells = cells[:-1]
            if cells: table_data.append(cells)
            continue
        else:
            if in_table: build_word_table(); in_table = False

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

    if in_table: build_word_table()
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()

def run_pipeline(media_path, meeting_title, meeting_date, extra_context, with_timestamps):
    if not media_path:
        yield render_status(0, "Harap unggah berkas media terlebih dahulu.", "error"), "", "", None, None, None
        return
    if not find_ffmpeg() or not find_ollama():
        yield render_status(0, "Dependensi sistem (FFmpeg/Ollama) tidak ditemukan.", "error"), "", "", None, None, None
        return

    use_gpu = torch.cuda.is_available()
    input_path, wav_path = None, None
    started = time.time()

    try:
        yield render_status(1, "Mempersiapkan ekstraksi audio...", "loading"), "", "", None, None, None
        suffix = Path(media_path).suffix.lower() or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(Path(media_path).read_bytes())
            input_path = tmp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        convert_to_wav(input_path, wav_path)
        yield render_status(5, "Memuat modul pengenalan suara...", "loading"), "", "", None, None, None

        asr = load_whisper(use_gpu)
        segments, info = asr.transcribe(
            wav_path,
            language="id",
            beam_size=5,
            condition_on_previous_text=False,
            initial_prompt=extra_context.strip() if extra_context else None,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vad_filter=True,
        )
        
        duration = info.duration
        lines, full_text = [], []
        last_yield_time = time.time()

        for segment in segments:
            text = segment.text.strip()
            if text:
                full_text.append(text)
                if with_timestamps:
                    start = _fmt_ts(segment.start)
                    end = _fmt_ts(segment.end)
                    lines.append(f"[{start} - {end}] {text}")
            
            current_progress = min(50, 5 + int((segment.end / duration) * 45))
            if time.time() - last_yield_time > 1.5:
                yield render_status(current_progress, "Mentranskripsi percakapan...", "loading"), "", "", None, None, None
                last_yield_time = time.time()

        transcript = " ".join(full_text)
        timestamped = "\n".join(lines) if (with_timestamps and lines) else transcript

        if not transcript:
            yield render_status(0, "Berkas audio kosong atau suara tidak terdeteksi.", "error"), "", "", None, None, None
            return

        global _asr_cache
        _asr_cache.clear()
        gc.collect()
        if use_gpu: torch.cuda.empty_cache()

        words_count = len(transcript.split())
        minutes = ""

        if words_count > 3500:
            yield render_status(52, f"Menganalisis dokumen panjang ({words_count} kata)...", "loading"), timestamped, "", None, None, None
            chunks = split_text_into_chunks(transcript, chunk_size=2500, overlap=200)
            intermediate_summaries = []

            for i, chunk in enumerate(chunks):
                chunk_prog = 52 + int((i / len(chunks)) * 38)
                yield render_status(chunk_prog, f"Mengekstraksi segmen {i+1} dari {len(chunks)}...", "loading"), timestamped, "", None, None, None
                
                chunk_prompt = f"Ekstrak poin penting:\n\n{chunk}"
                chunk_ctx = max(4096, min(16384, (len(chunk) // 4) + 1024))
                summary = ollama_generate_sync(DEFAULT_OLLAMA_MODEL, chunk_prompt, chunk_ctx)
                intermediate_summaries.append(f"--- Segmen {i+1} ---\n{summary}")

            combined_text = "\n\n".join(intermediate_summaries)
            prompt = make_prompt(combined_text, meeting_title, meeting_date, extra_context)

            for partial in ollama_stream_generator(DEFAULT_OLLAMA_MODEL, prompt, len(combined_text)):
                minutes = partial
                if time.time() - last_yield_time > 1.0:
                    yield render_status(95, "Menyusun format akhir dokumen...", "loading"), timestamped, minutes, None, None, None
                    last_yield_time = time.time()
        else:
            prompt = make_prompt(transcript, meeting_title, meeting_date, extra_context)
            for partial in ollama_stream_generator(DEFAULT_OLLAMA_MODEL, prompt, len(transcript)):
                minutes = partial
                if time.time() - last_yield_time > 1.0:
                    yield render_status(80, "Menyusun struktur dokumen...", "loading"), timestamped, minutes, None, None, None
                    last_yield_time = time.time()

        source_base = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(media_path).stem).strip("_") or "rapat"
        notulen_path = tempfile.mktemp(suffix=f"_Notulen_{source_base}.md")
        Path(notulen_path).write_text(minutes, encoding="utf-8")
        transkrip_path = tempfile.mktemp(suffix=f"_Transkripsi_{source_base}.txt")
        Path(transkrip_path).write_text(timestamped, encoding="utf-8")
        
        docx_path = None
        try:
            docx_bytes = export_docx(minutes, meeting_title)
            docx_path = tempfile.mktemp(suffix=f"_Notulen_{source_base}.docx")
            Path(docx_path).write_bytes(docx_bytes)
        except Exception:
            pass 

        elapsed = round((time.time() - started) / 60, 2)
        yield render_status(100, f"Diselesaikan dalam waktu {elapsed} menit. Berkas siap diunduh.", "success"), timestamped, minutes, notulen_path, transkrip_path, docx_path

    except Exception as exc:
        yield render_status(0, str(exc), "error"), "", "", None, None, None
    finally:
        for p in (input_path, wav_path):
            if p and os.path.exists(p):
                try: os.remove(p)
                except OSError: pass

# Custom Design System Theme & Styles
modern_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), gr.themes.GoogleFont("Inter"), "sans-serif"],
)

modern_css = """
/* Global Reset & Base Layout */
footer { display: none !important; }
body, html { 
    background-color: var(--neutral-50, #f8fafc) !important;
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    color: var(--body-text-color, #1e293b);
    overflow-x: hidden !important;
}

.gradio-container { 
    max-width: 1440px !important; 
    width: 100% !important; 
    margin: 0 auto !important; 
    padding: 24px 28px !important; 
    box-sizing: border-box !important; 
}

/* Header Navbar Style */
.app-header {
    background: var(--block-background-fill, #ffffff);
    border: 1px solid var(--border-color-primary, #e2e8f0);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01);
}

.brand-wrapper {
    display: flex;
    align-items: center;
    gap: 20px;
}

.app-logo { 
    height: 52px !important; 
    width: auto !important; 
    object-fit: contain; 
}

.app-logo-fallback { 
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%); 
    color: white; 
    border-radius: 12px; 
    width: 52px; 
    height: 52px; 
    display: flex; 
    justify-content: center; 
    align-items: center; 
    font-size: 18px; 
    font-weight: 800;
    box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.2);
}

.app-title-group h1 {
    margin: 0 !important; 
    font-size: 22px !important; 
    font-weight: 800 !important; 
    color: var(--body-text-color, #0f172a);
    letter-spacing: -0.025em;
}

.app-title-group p {
    margin: 2px 0 0 0 !important; 
    font-size: 13px !important; 
    color: var(--body-text-color-subdued, #64748b); 
    font-weight: 500; 
}

/* Theme Toggle Button (Using Gradio Built-in Dark Mode classes & JS) */
.theme-toggle-btn {
    background: var(--background-fill-secondary, #f1f5f9) !important;
    border: 1px solid var(--border-color-primary, #cbd5e1) !important;
    color: var(--body-text-color, #334155) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

.theme-toggle-btn:hover {
    background: var(--border-color-primary, #e2e8f0) !important;
    border-color: var(--primary-500, #6366f1) !important;
    transform: translateY(-1px);
}

/* Layout Cards (Bento Modern Style) */
.panel-card {
    background: var(--block-background-fill, #ffffff) !important;
    border: 1px solid var(--border-color-primary, #e2e8f0) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.03), 0 8px 10px -6px rgba(0, 0, 0, 0.03) !important;
    margin-bottom: 0 !important;
}

.section-header {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: var(--primary-600, #4f46e5) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-header::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border-color-primary, #f1f5f9);
    margin-left: 12px;
}

/* Form Elements Styling & Margin Tightening */
.custom-input-label {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: var(--body-text-color, #1e293b) !important;
    margin-bottom: 2px !important;
    margin-top: 14px !important;
    display: block !important;
}

/* Mengurangi padding/margin wrapper bawaan Gradio di bawah input DateTime & Textbox */
#meeting-date-picker, div:has(> #meeting-date-picker), 
.gr-box, .block {
    margin-bottom: 0px !important;
}

input[type="text"], textarea, .gr-input, .gr-box {
    border-radius: 10px !important;
    border-color: var(--border-color-primary, #cbd5e1) !important;
    background-color: var(--background-fill-primary, #ffffff) !important;
    font-size: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

input[type="text"]:focus, textarea:focus {
    border-color: var(--primary-500, #6366f1) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* File Upload Component Tuning */
#file-upload-dropzone {
    border: 2px dashed var(--border-color-primary, #cbd5e1) !important;
    border-radius: 14px !important;
    background: var(--background-fill-secondary, #f8fafc) !important;
    padding: 24px !important;
    transition: all 0.2s ease !important;
}

#file-upload-dropzone:hover {
    border-color: var(--primary-500, #6366f1) !important;
    background: var(--primary-50, rgba(99, 102, 241, 0.02)) !important;
}

/* FIX SKALA KALENDER: Bebas dari potongan & dipusatkan */
.picker-container, .flatpickr-calendar {
    transform: scale(0.85) !important;
    transform-origin: top left !important;
}

/* Action CTA Button */
.cta-process-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 14px 24px !important;
    color: white !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
    transition: all 0.2s ease !important;
    margin-top: 16px !important;
}

.cta-process-btn:hover {
    background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%) !important;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35) !important;
    transform: translateY(-1px);
}

/* Status Cards Component */
.status-card {
    background: var(--background-fill-secondary, #f8fafc);
    border: 1px solid var(--border-color-primary, #e2e8f0);
    border-radius: 14px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.01);
}

.status-card.success {
    background: rgba(16, 185, 129, 0.05);
    border-color: #10b981;
}

.status-card.error {
    background: rgba(239, 68, 68, 0.05);
    border-color: #ef4444;
}

.status-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 16px;
    flex-shrink: 0;
}

.status-icon.info { background: #e0f2fe; color: #0284c7; }
.status-icon.succ { background: #d1fae5; color: #059669; }
.status-icon.err { background: #fee2e2; color: #dc2626; }

.status-flex {
    display: flex;
    align-items: center;
    gap: 14px;
    width: 100%;
}

.status-title {
    font-weight: 700;
    font-size: 13px;
    color: var(--body-text-color, #1e293b);
}

.status-desc {
    font-size: 12px;
    color: var(--body-text-color-subdued, #64748b);
    margin-top: 2px;
}

.spinner-modern {
    border: 3px solid rgba(79, 70, 229, 0.15);
    border-top: 3px solid #4f46e5;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
}

.progress-bar-wrap {
    background: var(--border-color-primary, #e2e8f0);
    border-radius: 6px;
    width: 100%;
    height: 6px;
    margin-top: 12px;
    overflow: hidden;
}

.progress-bar-bar {
    background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s ease;
}

@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

/* Tabs Styling */
.tabs {
    border: none !important;
    background: transparent !important;
}

.tab-nav {
    border-bottom: 2px solid var(--border-color-primary, #e2e8f0) !important;
    margin-bottom: 16px !important;
    gap: 24px !important;
}

.tab-nav button {
    border: none !important;
    background: transparent !important;
    color: var(--body-text-color-subdued, #64748b) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 8px 4px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    transition: color 0.2s ease !important;
}

.tab-nav button:hover {
    color: var(--body-text-color, #0f172a) !important;
}

.tab-nav button.selected {
    color: var(--primary-600, #4f46e5) !important;
    border-bottom: 2px solid var(--primary-600, #4f46e5) !important;
}

/* Download Grid Styling */
.download-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    width: 100%;
    margin-top: 16px;
}

.dl-button {
    background: var(--background-fill-secondary, #f8fafc) !important;
    border: 1px solid var(--border-color-primary, #cbd5e1) !important;
    color: var(--body-text-color, #334155) !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    min-height: 48px !important;
    padding: 10px 14px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    font-size: 12px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
}

.dl-button:hover {
    background: var(--block-background-fill, #ffffff) !important;
    border-color: var(--primary-500, #6366f1) !important;
    color: var(--primary-600, #4f46e5) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

@media (max-width: 1024px) {
    .download-grid {
        grid-template-columns: 1fr;
    }
}
"""

with gr.Blocks(title="Sistem Notulensi Terpadu | Bakom RI") as demo:
    
    # Header Navbar Modern dengan Integrasi Tombol Mode Gelap/Terang JS Bawaan Gradio
    gr.HTML(f"""
    <div class="app-header">
        <div class="brand-wrapper">
            {logo_html}
            <div class="app-title-group">
                <h1>Sistem Notulensi Terpadu</h1>
                <p>Badan Komunikasi Pemerintah Republik Indonesia</p>
            </div>
        </div>
        <div>
            <button id="theme-toggler" class="theme-toggle-btn" onclick="document.body.classList.toggle('dark');">
                <svg style="width:16px;height:16px" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path></svg>
                Mode Gelap / Terang
            </button>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False, variant="compact"):
        # Kolom Kiri: Panel Input & Konfigurasi (Bento Grid Card)
        with gr.Column(scale=4, elem_classes="panel-card"):
            gr.HTML("<div class='section-header'>01. Unggah Berkas Media</div>")
            
            gr.HTML("<div class='custom-input-label' style='margin-top:0;'>Sumber Rekaman (Audio / Video)</div>")
            media_input = gr.File(show_label=False, file_types=["audio", "video"], type="filepath", elem_id="file-upload-dropzone")
            
            gr.HTML("<div class='section-header' style='margin-top: 24px;'>02. Metadata Rapat</div>")
            
            gr.HTML("<div class='custom-input-label'>Judul Agenda Rapat</div>")
            meeting_title = gr.Textbox(show_label=False, placeholder="Contoh: Rapat Koordinasi Nasional...", lines=1)
            
            gr.HTML("<div class='custom-input-label'>Tanggal Pelaksanaan</div>")
            meeting_date = gr.DateTime(show_label=False, include_time=False, elem_id="meeting-date-picker")
            
            with gr.Accordion("Pengaturan Lanjutan & Konteks", open=False):
                gr.HTML("<div class='custom-input-label'>Konteks & Glosarium Khusus</div>")
                extra_context = gr.Textbox(show_label=False, placeholder="Masukkan singkatan, istilah teknis, atau nama narasumber...", lines=3)
                
                with_timestamps = gr.Checkbox(label="Sertakan Penanda Waktu (Timestamp)", value=True)
            
            btn_process = gr.Button("Mulai Pemrosesan Data", elem_classes="cta-process-btn")

        # Kolom Kanan: Panel Proses & Hasil Dokumen (Bento Grid Card)
        with gr.Column(scale=8, elem_classes="panel-card"):
            gr.HTML("<div class='section-header'>03. Status & Pelacakan Sistem</div>")
            status_out = gr.HTML(render_status(0, "Menunggu instruksi. Sistem dalam keadaan siaga.", "idle"))
            
            gr.HTML("<div class='section-header' style='margin-top: 24px;'>04. Pratinjau Dokumen Hasil</div>")
            with gr.Tabs(elem_classes="tabs"):
                with gr.TabItem("Dokumen Notulensi Resmi"):
                    notulen_out = gr.Markdown("*Dokumen terstruktur berstandar eksekutif akan dirender otomatis pada area ini setelah proses selesai.*")
                with gr.TabItem("Log Transkripsi Mentah"):
                    transcript_out = gr.Textbox(lines=15, max_lines=20, interactive=False, show_label=False, placeholder="Log transkripsi lengkap akan ditampilkan di sini...")
            
            gr.HTML("<div class='section-header' style='margin-top: 24px;'>05. Unduh Berkas Akhir</div>")
            with gr.Row(elem_classes="download-grid"):
                file_word = gr.DownloadButton("Unduh Word (.docx)", elem_classes="dl-button")
                file_notulen = gr.DownloadButton("Unduh Markdown (.md)", elem_classes="dl-button")
                file_transkripsi = gr.DownloadButton("Unduh Transkrip (.txt)", elem_classes="dl-button")

    btn_process.click(
        fn=run_pipeline,
        inputs=[media_input, meeting_title, meeting_date, extra_context, with_timestamps],
        outputs=[status_out, transcript_out, notulen_out, file_notulen, file_transkripsi, file_word],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=80,
        inbrowser=True,
        theme=modern_theme,
        css=modern_css
    )