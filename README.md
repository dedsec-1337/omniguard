---
title: OMNIGUARD
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12"
app_file: app.py
pinned: false
---

# 🛡️ OMNIGUARD — Protect your voice, protect your people.

> A two-tool web app that fights AI voice scams (vishing) — and makes the answer usable for people who cannot rely on reading a screen.

[![GitHub](https://img.shields.io/badge/GitHub-dedsec--1337%2Fomniguard-181717?logo=github)](https://github.com/dedsec-1337/omniguard)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/gradio-6.17.3-orange)](https://gradio.app/)

---

## Table of Contents

- [Why OMNIGUARD](#why-omniguard)
- [Two features](#two-features)
- [Run locally (CPU only, no GPU needed)](#run-locally-cpu-only-no-gpu-needed)
- [Deploy on Hugging Face Spaces (ZeroGPU)](#deploy-on-hugging-face-spaces-zerogpu)
- [Models used](#models-used)
- [Limitations](#limitations)

---

## Why OMNIGUARD

AI voice cloning needs only seconds of audio — and most people cannot tell a cloned voice from a real one.

In McAfee's 2023 *Beware the Artificial Impostor* survey:

- **69%** of Indian respondents could not tell an AI voice from a real one.
- **83%** of Indian victims reported losing money.

Seniors, and blind or low-vision users who cannot cross-check a call with visual cues, are especially exposed. OMNIGUARD gives everyone a second opinion — and a spoken safety message they can actually act on.

## Two features

| Tab | What it does |
|---|---|
| 🎙️ **Deepfake Detector** | Upload or record a clip (`.wav`, `.mp3`, `.m4a`, `.ogg`). A `wav2vec2` model scores the chance it is AI-generated and shows a traffic-light verdict: 🟢 under 30% fake · 🟡 30–70% uncertain · 🔴 over 70% fake. Also returns a plain-language explanation and a safety tip. |
| ♿ **Accessibility Assistant** | Transcribes the clip with `faster-whisper`, applies the same verdict, writes a short rule-based safety message (no LLM), and reads it aloud with Piper TTS via a large **Play Voice Alert** button. |

## Run locally (CPU only, no GPU needed)

```bash
git clone https://github.com/dedsec-1337/omniguard.git
cd omniguard
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                     # then open http://127.0.0.1:7860
```

### Notes

- **First run needs internet.** Models are downloaded into the usual Hugging Face cache (~1.2 GB for the detector, a few hundred MB for Whisper and the voice).
- **Piper voice.** The app downloads `en_US-lessac-medium.onnx` **and** `en_US-lessac-medium.onnx.json` on first run into `~/.cache/omniguard/piper_voices` (falls back to a temp folder), and checks both files exist before loading. **No voice file is stored in this repo.**
- **Offline backup (exhibition).** After one successful run, start with `HF_HUB_OFFLINE=1 python app.py` — everything works without internet.
- **`.m4a` files** need `ffmpeg` installed on your computer (already present on Hugging Face Spaces).
- **Graceful degradation.** If a model fails to download, the app stays up: the Accessibility Assistant still transcribes and speaks a cautious message even without the detector.

## Deploy on Hugging Face Spaces (ZeroGPU)

The YAML block at the top of this file is the Space configuration — Hugging Face reads it automatically. In the Space settings, choose the **ZeroGPU** hardware.

Only the small `wav2vec2` scoring function is decorated with `@spaces.GPU(duration=10)`; locally, that decorator is a no-op.

## Models used

| Purpose | Model |
|---|---|
| Deepfake detection | [`garystafford/wav2vec2-deepfake-voice-detector`](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector) |
| Speech-to-text | `faster-whisper` (`base`) |
| Text-to-speech | Piper voice `en_US-lessac-medium` |

**Nothing is trained here** — these are off-the-shelf models used as-is.

## Limitations

No detector is 100% accurate. This one was trained on a small set of text-to-speech services, so it may miss newer voice-cloning systems, and noisy phone audio can push results into the yellow zone.

Treat the result as a **warning sign, not proof**. Always verify by calling back on a known number.
