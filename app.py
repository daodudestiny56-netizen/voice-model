"""
Voice Clone Studio - a local web app that clones a voice from a short
recording and speaks any text you type, using the open-source XTTS-v2 model.

Run:  run.bat   (or:  .venv\\Scripts\\python app.py)
Then open http://127.0.0.1:7860
"""

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

# XTTS-v2 is released under the Coqui Public Model License (non-commercial).
# Setting this accepts that license so the model can download without a prompt.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

import gradio as gr
import imageio_ffmpeg
import torch

ROOT = Path(__file__).parent
VOICES_DIR = ROOT / "voices"
OUTPUTS_DIR = ROOT / "outputs"
VOICES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MAX_REF_SECONDS = 30

LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Polish": "pl", "Turkish": "tr",
    "Russian": "ru", "Dutch": "nl", "Czech": "cs", "Arabic": "ar",
    "Chinese": "zh-cn", "Japanese": "ja", "Hungarian": "hu", "Korean": "ko",
    "Hindi": "hi",
}

_tts = None
_tts_lock = threading.Lock()


def get_tts():
    """Load the model once, on first use (first run downloads ~1.8 GB)."""
    global _tts
    with _tts_lock:
        if _tts is None:
            from TTS.api import TTS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            _tts = TTS(MODEL_NAME).to(device)
        return _tts


def prepare_reference(src: str, dest: Path) -> Path:
    """Convert any audio/video file to clean mono 22.05 kHz WAV, max 30 s,
    with gentle noise filtering and loudness normalisation."""
    cmd = [
        FFMPEG, "-y", "-i", src,
        "-t", str(MAX_REF_SECONDS),
        "-ac", "1", "-ar", "22050",
        "-af", "highpass=f=70,lowpass=f=10000,"
               "silenceremove=start_periods=1:start_threshold=-45dB,"
               "loudnorm",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dest.exists():
        raise gr.Error("Could not read that audio file. Try a WAV or MP3.")
    return dest


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9 _-]", "", name or "").strip()
    return name.replace(" ", "_")[:50]


def list_voices():
    return sorted(p.stem for p in VOICES_DIR.glob("*.wav"))


def save_voice(audio_path, name):
    if not audio_path:
        raise gr.Error("Upload or record a voice sample first.")
    name = safe_name(name)
    if not name:
        raise gr.Error("Give the voice a name.")
    prepare_reference(audio_path, VOICES_DIR / f"{name}.wav")
    return (
        gr.update(choices=list_voices(), value=name),
        f"Saved voice **{name}**. You can now pick it from the dropdown.",
    )


def delete_voice(name):
    if name:
        (VOICES_DIR / f"{name}.wav").unlink(missing_ok=True)
    return gr.update(choices=list_voices(), value=None), f"Deleted **{name}**." if name else ""


def generate(audio_path, saved_voice, text, language, speed, temperature,
             progress=gr.Progress()):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type some text for the voice to say.")

    # Pick the reference: a fresh upload/recording wins over a saved voice.
    if audio_path:
        ref = prepare_reference(audio_path, OUTPUTS_DIR / "_last_reference.wav")
    elif saved_voice:
        ref = VOICES_DIR / f"{saved_voice}.wav"
        if not ref.exists():
            raise gr.Error("That saved voice no longer exists.")
    else:
        raise gr.Error("Upload/record a voice sample or pick a saved voice.")

    progress(0.1, desc="Loading model (first time downloads ~1.8 GB)...")
    tts = get_tts()

    progress(0.3, desc="Cloning voice and speaking... (CPU: ~10-60 s per sentence)")
    out = OUTPUTS_DIR / f"speech_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    start = time.time()
    tts.tts_to_file(
        text=text,
        speaker_wav=str(ref),
        language=LANGUAGES[language],
        file_path=str(out),
        split_sentences=True,
        speed=float(speed),
        temperature=float(temperature),
    )
    secs = time.time() - start
    return str(out), f"Done in {secs:.0f} s. Saved to `{out.relative_to(ROOT)}`."


with gr.Blocks(title="Voice Clone Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎙️ Voice Clone Studio\n"
        "Upload a short clip of someone speaking, type any text, and hear it in their voice. "
        "Everything runs locally on your computer.\n\n"
        "**Only clone voices you have permission to use.**"
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Voice sample")
            ref_audio = gr.Audio(
                label="Upload or record (6-30 s of clean speech, one speaker, no music)",
                sources=["upload", "microphone"],
                type="filepath",
            )
            with gr.Accordion("Save / reuse voices", open=False):
                with gr.Row():
                    voice_name = gr.Textbox(label="Name this voice", scale=3)
                    save_btn = gr.Button("Save voice", scale=1)
                with gr.Row():
                    saved_voice = gr.Dropdown(
                        label="...or use a saved voice", choices=list_voices(),
                        value=None, scale=3,
                    )
                    del_btn = gr.Button("Delete", variant="stop", scale=1)
                voice_msg = gr.Markdown()

        with gr.Column():
            gr.Markdown("### 2. What should it say?")
            text = gr.Textbox(
                label="Text", lines=7,
                placeholder="Type anything here. Long text is split into sentences automatically.",
            )
            language = gr.Dropdown(list(LANGUAGES), value="English", label="Language")
            with gr.Accordion("Advanced", open=False):
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                temperature = gr.Slider(
                    0.1, 1.0, value=0.7, step=0.05,
                    label="Temperature (lower = steadier, higher = more expressive)",
                )
            go = gr.Button("🔊 Generate speech", variant="primary", size="lg")

    gr.Markdown("### 3. Result")
    output_audio = gr.Audio(label="Cloned speech", type="filepath", interactive=False)
    status = gr.Markdown()

    save_btn.click(save_voice, [ref_audio, voice_name], [saved_voice, voice_msg])
    del_btn.click(delete_voice, [saved_voice], [saved_voice, voice_msg])
    go.click(
        generate,
        [ref_audio, saved_voice, text, language, speed, temperature],
        [output_audio, status],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
