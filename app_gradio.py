import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from functools import lru_cache

import gradio as gr
import torch
import requests
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# KONFIGURASI

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


def system_status() -> str:
    lines = []
    lines.append(f"FFmpeg: {'✅ Siap' if find_ffmpeg() else '❌ Tidak ditemukan'}")
    lines.append(f"GPU/CUDA: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '⚠️ CPU Mode'}")
    lines.append(f"Ollama: {'✅ Siap' if find_ollama() else '❌ Tidak terdeteksi'}")
    try:
        models = get_ollama_models()
        lines.append(f"Model: {', '.join(models) if models else 'Kosong'}")
    except Exception:
        lines.append("Model: Gagal terhubung ke Ollama")
    return "\n".join(lines)


# WHISPER ENGINE

_asr_cache: dict = {}


def load_whisper(use_gpu: bool):
    key = "gpu" if use_gpu else "cpu"
    if key in _asr_cache:
        return _asr_cache[key]

    device = "cuda:0" if use_gpu else "cpu"
    dtype = torch.float16 if use_gpu else torch.float32

    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia. Silakan matikan opsi GPU.")

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
        raise RuntimeError("Gagal memproses audio:\n" + result.stderr[-2000:])

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("File audio hasil konversi kosong.")


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
                "num_ctx": 8192,
                "num_predict": 4096,
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

    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# "):
            p = doc.add_heading(line[2:].strip(), level=0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
        elif re.match(r"^\s*[-*]\s+", line):
            doc.add_paragraph(re.sub(r"^\s*[-*]\s+", "", line), style="List Bullet")
        elif re.match(r"^\s*\d+[.)]\s+", line):
            doc.add_paragraph(re.sub(r"^\s*\d+[.)]\s+", "", line), style="List Number")
        elif line.startswith("|"):
            doc.add_paragraph(line.replace("|", "  |  "))
        else:
            doc.add_paragraph(line.replace("**", "").replace("`", ""))

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def export_word_file(notulen_text: str, meeting_title: str) -> str | None:
    if not notulen_text or not notulen_text.strip():
        raise gr.Error("Belum ada notulen untuk diekspor.")
    try:
        docx_bytes = export_docx(notulen_text, meeting_title)
        path = tempfile.mktemp(suffix="_Notulen.docx")
        Path(path).write_bytes(docx_bytes)
        return path
    except ImportError:
        raise gr.Error("python-docx belum terinstal. Jalankan: pip install python-docx")


# PIPELINE UTAMA

def run_pipeline(
    audio_path: str | None,
    use_gpu: bool,
    llm_model: str,
    meeting_title: str,
    meeting_date: str,
    extra_context: str,
    with_timestamps: bool,
):
    if not audio_path:
        yield "❌ Masukkan rekaman atau file audio terlebih dahulu.", "", "", None, None
        return

    if not find_ffmpeg():
        yield "❌ FFmpeg tidak ditemukan di sistem.", "", "", None, None
        return

    if not find_ollama():
        yield "❌ Ollama tidak ditemukan.", "", "", None, None
        return

    try:
        installed = get_ollama_models()
    except Exception:
        yield "❌ Layanan Ollama tidak aktif.", "", "", None, None
        return

    if not model_is_installed(llm_model.strip(), installed):
        yield f"❌ Model Ollama '{llm_model}' belum terinstal.", "", "", None, None
        return

    input_path = None
    wav_path = None
    started = time.time()

    try:
        yield "⏳ Menyiapkan audio...", "", "", None, None
        suffix = Path(audio_path).suffix.lower() or ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(Path(audio_path).read_bytes())
            input_path = tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        convert_to_wav(input_path, wav_path)

        yield "⏳ Mentranskripsi rekaman audio...", "", "", None, None
        asr = load_whisper(use_gpu)
        transcript, timestamped = transcribe_audio(asr, wav_path, with_timestamps)

        if not transcript:
            yield "⚠️ Suara tidak terdeteksi pada audio.", "", "", None, None
            return

        yield "⏳ Menyusun notulen rapat...", timestamped, "", None, None

        prompt = make_prompt(transcript, meeting_title, meeting_date, extra_context)
        minutes = ""
        for partial in ollama_stream_generator(llm_model.strip(), prompt):
            minutes = partial
            yield "⏳ Menyusun notulen rapat...", timestamped, minutes, None, None

        elapsed = round(time.time() - started, 1)

        source_base = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(audio_path).stem).strip("_") or "rapat"
        notulen_path = tempfile.mktemp(suffix=f"_Notulen_{source_base}.md")
        Path(notulen_path).write_text(minutes, encoding="utf-8")

        transkrip_path = tempfile.mktemp(suffix=f"_Transkripsi_{source_base}.txt")
        Path(transkrip_path).write_text(timestamped, encoding="utf-8")

        yield f"✅ Selesai dalam {elapsed} detik.", timestamped, minutes, notulen_path, transkrip_path

    except Exception as exc:
        yield f"❌ Terjadi kesalahan: {str(exc)}", "", "", None, None

    finally:
        for p in (input_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ANTARMUKA GRADIO (MINIMALIS & SIMPEL)

custom_css = """
.gradio-container { max-width: 1000px !important; margin: auto !important; }
#status-text { font-size: 0.95rem; font-weight: 500; }
"""

with gr.Blocks(title="Notulen Rapat Otomatis", css=custom_css) as demo:
    gr.Markdown(
        """
        # 🎙️ Notulen Rapat Otomatis
        Unggah audio atau rekam suara untuk membuat notulen rapat secara otomatis.
        """
    )

    with gr.Row():
        # Kolom Kiri: Input Sederhana
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Audio / Rekaman",
                sources=["upload", "microphone"],
                type="filepath",
                format="wav",
            )

            btn_process = gr.Button("⚡ Buat Notulen", variant="primary", size="lg")

            # Detail tambahan disembunyikan dalam Accordion agar rapi
            with gr.Accordion("📝 Informasi Rapat (Opsional)", open=False):
                meeting_title = gr.Textbox(label="Judul Rapat", placeholder="Contoh: Rapat Mingguan Proyek")
                meeting_date = gr.Textbox(label="Tanggal", placeholder="Contoh: 27 Oktober 2026")
                extra_context = gr.Textbox(
                    label="Catatan / Peserta",
                    placeholder="Nama peserta, topik bahasan, atau istilah penting...",
                    lines=2,
                )

            with gr.Accordion("⚙️ Pengaturan Lanjutan", open=False):
                use_gpu = gr.Checkbox(label="Gunakan GPU (NVIDIA CUDA)", value=torch.cuda.is_available())
                llm_model = gr.Textbox(label="Model Ollama", value=DEFAULT_OLLAMA_MODEL)
                with_timestamps = gr.Checkbox(label="Sertakan timestamp di transkrip", value=True)
                sys_status = gr.Textbox(value=system_status, label="Status Sistem", lines=4, interactive=False)

        # Kolom Kanan: Hasil & Tab
        with gr.Column(scale=2):
            status_out = gr.Markdown("Siap memproses audio.", elem_id="status-text")

            with gr.Tabs():
                with gr.TabItem("📄 Notulen Rapat"):
                    notulen_out = gr.Markdown("*Hasil notulen akan muncul di sini...*")

                with gr.TabItem("📝 Transkrip Lengkap"):
                    transcript_out = gr.Textbox(
                        label="Teks Transkripsi",
                        lines=12,
                        max_lines=20,
                        interactive=False,
                        placeholder="Transkripsi audio akan ditampilkan di sini...",
                    )

            # Bagian Unduhan yang Ringkas
            with gr.Accordion("💾 Unduh Berkas", open=True):
                with gr.Row():
                    btn_export_word = gr.Button("📄 Unduh Word (.docx)", size="sm")
                    file_word = gr.File(label="Berkas Word", interactive=False)
                with gr.Row():
                    file_notulen = gr.File(label="Notulen (.md)", interactive=False)
                    file_transkripsi = gr.File(label="Transkrip (.txt)", interactive=False)

    # State untuk menyimpan notulen teks murni
    notulen_state = gr.State("")

    # Event Handlers
    btn_process.click(
        fn=run_pipeline,
        inputs=[
            audio_input,
            use_gpu,
            llm_model,
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
        theme=gr.themes.Soft(),
    )