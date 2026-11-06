---
title: OMNIGUARD
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
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
| 🎙️ **Deepfake Detector** | Upload or record a clip (.wav, .mp3, .m4a, .ogg). Two detectors score it across several windows and vote. A ring shows the chance it is AI-generated, with a traffic light: 🟢 under 30%, 🟡 30-70% uncertain, 🔴 over 70%, plus a plain-language explanation and a safety tip. |
| ♿ **Accessibility Assistant** | Transcribes the clip with faster-whisper, applies the same verdict, writes a short rule-based safety message (no LLM) and reads it aloud with Piper TTS via a large **Play Voice Alert** button. |

## How detection works

No single open model catches every AI voice engine: on the Deepfake-Eval-2024
benchmark of real deepfakes circulated in 2024, open-source audio detectors saw
their AUC fall by about 48% compared with older academic benchmarks. OMNIGUARD therefore:

1. **Ensembles two detectors** trained on different data:
   `Speech-Arena-2025/DF_Arena_500M_V_1` (trained on ASVspoof 2019/2024, Codecfake,
   MLAAD, SpoofCeleb and more; its model card reports 1.76% EER on In-the-Wild) and
   `garystafford/wav2vec2-deepfake-voice-detector`. An optional third,
   `MelodyMachine/Deepfake-audio-detection-V2` (small and fast, independent lineage),
   can be switched on.
2. **Scores up to four ~4 s windows** spread across the clip (after trimming silence),
   which matches how the models were trained.
3. **Never says GREEN when the detectors disagree.**
4. **Calibrates on your own clips** (below), so the percentages mean something for your audio.

Nothing is trained here: only pre-trained models are used.

## Run locally (CPU only, no GPU needed)

```bash
git clone https://github.com/dedsec-1337/omniguard.git
cd omniguard
python3.12 -m venv venv
source venv/bin/activate          # fish shell: source venv/bin/activate.fish
pip install -r requirements.txt
python app.py                     # then open http://127.0.0.1:7860
```

Notes:

- **First run needs internet** and downloads roughly 3 GB of models (two detectors,
  Whisper, one voice) into the usual Hugging Face cache.
- **Piper voice:** the app downloads `en_US-lessac-medium.onnx` **and**
  `en_US-lessac-medium.onnx.json` by itself on first run, into
  `~/.cache/omniguard/piper_voices` (temp folder as a fallback), and checks that
  both files exist before loading. No voice file is stored in this repo.
- **Offline backup (exhibition):** after one successful run, start with
  `HF_HUB_OFFLINE=1 python app.py` and everything works without internet.
- **Faster on a slow laptop:** `OMNIGUARD_DETECTORS=gary python app.py` loads one detector.
  Add the optional third with `OMNIGUARD_DETECTORS=arena,gary,melody`.
- `.m4a` files need `ffmpeg` installed on your computer (already present on Hugging Face Spaces).
- If a model fails to download, the app stays up with whatever loaded; the
  Accessibility Assistant still transcribes and speaks a cautious message even
  with no detector.

## Calibrate on your own audio (recommended before any demo)

```bash
mkdir -p calib/real calib/fake
# 20+ recordings of real people in calib/real/, 20+ AI voices in calib/fake/ (any audio format)
python calibrate.py
```

It prints each clip's score per detector, the accuracy before and after, and saves
`calibration.json`. Commit that file; it is small. The `calib/` folder is git-ignored
because the repo may only contain the two example clips.

## Deploy on Hugging Face Spaces (ZeroGPU)

The YAML block at the top of this file is the Space configuration. Select the
**ZeroGPU** hardware in the Space settings. Only the scoring function uses
`@spaces.GPU(duration=15)`; locally that decorator does nothing. Commit
`calibration.json` so the Space uses your calibration.

## Licences and trust

- DF Arena 500M is released for **non-commercial** use and ships its own model code,
  which Transformers runs because of `trust_remote_code=True`. That is fine for an
  exhibition; do not use it in a paid product without the authors' permission.
- The other detectors are Apache-2.0.

## Limitations

No detector is 100% accurate. Newer voice-cloning systems, phone-quality audio and
background noise can all fool them, and noisy audio can push results into the yellow zone.
Treat the result as a warning sign, not proof. Always verify by calling back on a known number.
