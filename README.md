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

**GitHub:** https://github.com/dedsec-1337/omniguard.git

OMNIGUARD is a two-tool web app that fights AI voice scams (vishing) and
makes the answer usable for people who cannot rely on reading a screen.

## Real-world impact

AI voice cloning needs only seconds of audio, and most people cannot tell a
cloned voice from a real one. In McAfee's 2023 "Beware the Artificial Impostor"
survey, 69% of Indian respondents said they could not tell an AI voice from a
real one, and 83% of the Indian victims reported losing money. Seniors, and
blind or low-vision users who cannot cross-check a call with visual cues, are
especially exposed.

## Two features

| Tab | What it does |
|---|---|
| 🎙️ **Deepfake Detector** | Upload or record a clip (.wav, .mp3, .m4a, .ogg). A wav2vec2 model scores the chance it is AI-generated and shows a traffic light: 🟢 under 30% fake, 🟡 30-70% uncertain, 🔴 over 70% fake, plus a plain-language explanation and a safety tip. |
| ♿ **Accessibility Assistant** | Transcribes the clip with faster-whisper, applies the same verdict, writes a short rule-based safety message (no LLM) and reads it aloud with Piper TTS via a large **Play Voice Alert** button. |

Models: `garystafford/wav2vec2-deepfake-voice-detector`, faster-whisper `base`,
Piper voice `en_US-lessac-medium`. Nothing is trained here.

## Run locally (CPU only, no GPU needed)

```bash
git clone https://github.com/dedsec-1337/omniguard.git
cd omniguard
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                     # then open http://127.0.0.1:7860
```

Notes:

- **First run needs internet.** It downloads the models (roughly 1.2 GB for the
  detector, a few hundred MB for Whisper and the voice) into the usual Hugging Face cache.
- **Piper voice:** the app downloads `en_US-lessac-medium.onnx` **and**
  `en_US-lessac-medium.onnx.json` by itself on first run, into
  `~/.cache/omniguard/piper_voices` (temp folder as a fallback), and checks that
  both files exist before loading. No voice file is stored in this repo.
- **Offline backup (exhibition):** after one successful run, start with
  `HF_HUB_OFFLINE=1 python app.py` and everything works without internet.
- `.m4a` files need `ffmpeg` installed on your computer (already present on Hugging Face Spaces).
- If a model fails to download, the app stays up: the Accessibility Assistant
  still transcribes and speaks a cautious message even without the detector.

## Deploy on Hugging Face Spaces (ZeroGPU)

The YAML block at the top of this file is the Space configuration. Select the
**ZeroGPU** hardware in the Space settings. Only the small wav2vec2 scoring
function uses `@spaces.GPU(duration=10)`; locally that decorator does nothing.

## Limitations

No detector is 100% accurate. This one was trained on a small set of text-to-speech
services and may miss newer voice-cloning systems, and noisy phone audio can push
results into the yellow zone. Treat the result as a warning sign, not proof.
Always verify by calling back on a known number.
