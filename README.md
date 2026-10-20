# Voice Clone Studio

A local web app that clones a voice from a short recording and speaks any text you give it.
It uses the open-source **XTTS-v2** model and runs entirely on your PC, with no account and no upload.

## Start it
Double-click **`run.bat`**. Your browser opens at http://127.0.0.1:7860.

The first time you generate speech, the model downloads (about 1.8 GB, one time only).

## Use it
1. **Voice sample:** upload a recording (WAV, MP3, M4A, even MP4) or record one with your mic.
   - Best results: 6–30 seconds of clear speech, one speaker, no music or background noise.
   - Only the first 30 seconds are used. The app cleans and normalises the audio automatically.
   - Optional: name the voice and click **Save voice** so you can reuse it from the dropdown later.
2. **Text:** type what the voice should say and pick the language.
3. Click **Generate speech**. Play the result in the browser or find it in the `outputs/` folder.

## Tips
- Generating on CPU takes roughly 10–60 seconds per sentence. An NVIDIA GPU is used automatically if one is present.
- If the voice sounds off, try a cleaner or longer sample, or lower the **Temperature** setting.
- To get natural pauses, use punctuation.

## Folders
- `voices/`: saved voice samples
- `outputs/`: generated speech files
- `.venv/`: the Python 3.11 environment

## Reinstall from scratch
```
python -m pip install uv
python -m uv venv .venv --python 3.11
python -m uv pip install --python .venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m uv pip install --python .venv\Scripts\python.exe coqui-tts gradio soundfile imageio-ffmpeg
```

## License and ethics
XTTS-v2 is under the Coqui Public Model License, which allows **non-commercial use only**.
Only clone voices of people who have given permission. Don't use cloned audio to deceive anyone.
