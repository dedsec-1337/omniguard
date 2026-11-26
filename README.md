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

<div align="center">

# OMNIGUARD

**Protect your voice. Protect your people.**

Local deepfake voice detection · spoken safety alerts · offline-first

[![Python 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.x-FF7C00?style=flat-square)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Runs offline](https://img.shields.io/badge/Offline-ready-2dd4bf?style=flat-square)](#exhibition--low--vram-laptops)

[Features](#features) · [Install](#install) · [Exhibition](#exhibition--low--vram-laptops) · [Hardware](#hardware--gpu) · [Configure](#configuration)

</div>

---

## Why it exists

AI voice cloning needs only a few seconds of audio. Most people cannot tell a clone from the real person. In McAfee’s 2023 survey, **69%** of Indian respondents said they could not tell an AI voice from a real one, and **83%** of victims lost money.

Seniors and blind or low-vision users are especially exposed — they cannot cross-check a call with a screen.

OMNIGUARD is a **local** web app for that problem: detect synthetic speech, explain the result in plain language, and **speak** the safety message so the answer is usable without reading.

> Nothing is sent to the cloud after models are downloaded. Analysis stays on your machine.

---

## Features

| | Module | What you get |
|:--:|:--|:--|
| **1** | **Deepfake Detector** | Upload or record a clip. An ensemble of detectors scores short windows and votes. A probability ring and traffic light: green &lt; 30%, yellow 30–70%, red &gt; 70%. Plain-language tip included. |
| **2** | **Accessibility Assistant** | Same verdict, plus transcription and a rule-based safety message read aloud with a large **Play Voice Alert** control. No LLM in the safety path. |
| **3** | **Voice Assistant** | Speak or type. Guided responses for “I think I’m being scammed” and “I already sent money”, regional helplines, optional local LLM for open questions. |

**Detection approach**

1. Ensemble of models trained on different data (DF Arena, garystafford wav2vec2, MelodyMachine; Pellav2 optional).
2. Up to several ~4 s windows across the clip after silence trim.
3. Disagreement between detectors is never labelled green.
4. Optional calibration on *your* audio so scores match real conditions.

---

## Install

**Requirements:** Python **3.11 or 3.12**, ~**8 GB** free disk for models, internet on first run only.

```bash
git clone https://github.com/dedsec-1337/omniguard.git
cd omniguard

python3 -m venv venv

# Linux / macOS
source venv/bin/activate
# Windows CMD:        venv\Scripts\activate.bat
# Windows PowerShell: venv\Scripts\Activate.ps1

pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

First launch downloads roughly **3 GB** (detectors, Whisper, one Piper voice) into the Hugging Face cache. Later runs can be fully offline.

<details>
<summary><strong>Windows one-liner (after install)</strong></summary>

```bat
set HF_HUB_OFFLINE=1
set OMNIGUARD_PROFILE=fast
python app.py
```

</details>

---

## Exhibition & low‑VRAM laptops

Tuned for **GTX 1650‑class** machines (≈4 GB VRAM, 16–32 GB system RAM), including ASUS ROG Strix G531GT.

| Behaviour | Detail |
|:--|:--|
| GPU sharing | Models stay in system RAM; the GPU runs **one detector at a time** (peak VRAM ≈ largest model). |
| Auto profile | On ≤6 GB VRAM, default is **`fast`** unless you set `OMNIGUARD_PROFILE`. |
| Soft budget | Models above `OMNIGUARD_VRAM_GB` (default **3.2**) fall back to CPU instead of crashing. |
| Offline | `HF_HUB_OFFLINE=1` so venue Wi‑Fi cannot break the demo. |

**Day before**

1. Run once **online** so models download.
2. (Recommended) Calibrate on your own clips — see [Calibration](#calibration).
3. Confirm **Real voice** → green, **AI voice** → red.
4. Rehearse [`demo_script.md`](demo_script.md).

**On the day**

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
bash run_exhibition.sh
```

Equivalent:

```bash
HF_HUB_OFFLINE=1 OMNIGUARD_PROFILE=fast OMNIGUARD_SWAP=on python app.py
```

If VRAM is still tight:

```bash
OMNIGUARD_DETECTORS=gary HF_HUB_OFFLINE=1 OMNIGUARD_PROFILE=fast python app.py
```

---

## Hardware & GPU

| Component | Device |
|:--|:--|
| Detectors | **CUDA** when available (shared on ≤8 GB cards) |
| Whisper | **CPU** by default (keeps VRAM free) |
| Piper TTS | **CPU** |

You should see on startup:

```text
device=cuda swap=True detectors=['arena', 'gary', 'melody'] mode=fast
```

| Goal | Command |
|:--|:--|
| Force GPU sharing | `OMNIGUARD_SWAP=on python app.py` |
| Whisper on GPU (only if detectors are light) | `OMNIGUARD_WHISPER_DEVICE=cuda OMNIGUARD_DETECTORS=gary python app.py` |
| Check CUDA | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"` |

---

## Calibration

```bash
mkdir -p calib/real calib/fake
# 15–30 real recordings → calib/real/
# 15–30 AI / TTS clips  → calib/fake/
python calibrate.py
```

Writes `calibration.json` (safe to commit). The `calib/` folder is git‑ignored so private audio never leaves your machine.

Replace `examples/real_sample.wav` and `examples/fake_sample.wav` with clips that score green / red after calibration before any public demo.

---

## Configuration

| Variable | Default | Purpose |
|:--|:--|:--|
| `OMNIGUARD_PROFILE` | `fast` on ≤6 GB VRAM, else `balanced` | `fast` · `balanced` · `thorough` |
| `OMNIGUARD_DETECTORS` | `arena,gary,melody` | Comma list; e.g. `gary` for one model |
| `OMNIGUARD_SWAP` | `auto` | GPU residency sharing (on under 8 GB) |
| `OMNIGUARD_VRAM_GB` | `3.2` | Soft VRAM budget (GB) |
| `OMNIGUARD_WHISPER` | profile‑dependent | `tiny` · `base` · `small` · … |
| `OMNIGUARD_WHISPER_DEVICE` | `cpu` | Set `cuda` to put Whisper on GPU |
| `OMNIGUARD_REGION` | `IN` | Helpline text (e.g. India 1930) |
| `OMNIGUARD_LLM` | `qwen2.5:1.5b` | Ollama model, or `off` |
| `HF_HUB_OFFLINE` | unset | `1` = no network |

**Profiles**

| Profile | Detectors | Windows | Speech | Notes |
|:--|:--|:--|:--|:--|
| Fast | 2 quickest | 2 | tiny | Best for 4 GB / live demos |
| Balanced | all loaded | 4 | base | Everyday default on larger GPUs |
| Thorough | all loaded | 6 | small | Slowest, most careful |

---

## Deploy (Hugging Face Spaces)

The YAML front matter at the top of this file is the Space config. Select **ZeroGPU** hardware. Commit `calibration.json` if you use calibration.

---

## Optional extras

```bash
pip install -r requirements-optional.txt   # Pellav2 detector (~2.3 GB; opt-in via OMNIGUARD_DETECTORS)
```

```bash
# Optional local LLM for open questions in the Voice Assistant
ollama pull qwen2.5:1.5b
# Disable: OMNIGUARD_LLM=off
```

---

## Project layout

```text
omniguard/
├── app.py                 # Gradio UI
├── engine.py              # Detectors, scoring, Whisper, Piper
├── guardian.py            # Safety copy, intents, helplines
├── listening.py           # Continuous microphone
├── omni_runtime.py        # Profiles, status, GPU residency
├── channel_log.py         # Activity log
├── calibrate.py           # Fit thresholds / weights on your clips
├── run_exhibition.sh      # Offline + fast profile launcher
├── assets/                # CSS, JS, font
├── examples/              # Demo clips (replace before exhibition)
├── demo_script.md         # ~2 minute live script
├── requirements.txt
└── LICENSE
```

---

## Licences

| Component | Terms |
|:--|:--|
| OMNIGUARD (this repo) | [MIT](LICENSE) |
| DF Arena 500M | **Non‑commercial**; loads with `trust_remote_code=True` — fine for school / exhibition, not for paid products without permission |
| Other detectors | Apache‑2.0 / MIT |

---

## Limitations

No detector is perfect. New cloning systems, phone compression, and noise can produce yellow scores or false confidence.

**Treat every result as a warning, never as proof.** Always verify by calling back on a number you already know.

---

<div align="center">

**OMNIGUARD** · local · offline‑capable · built for real people

[GitHub](https://github.com/dedsec-1337/omniguard) · [Demo script](demo_script.md) · [License](LICENSE)

</div>
