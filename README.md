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

OMNIGUARD is a local web app that fights AI voice scams (vishing) and makes the result usable for people who cannot rely on reading a screen.

| Tab | What it does |
|---|---|
| 🎙️ **Deepfake Detector** | Upload or record a clip. Several detectors score short windows and vote. A ring shows the chance it is AI-generated: 🟢 under 30%, 🟡 30–70% uncertain, 🔴 over 70%, plus a plain-language tip. |
| ♿ **Accessibility Assistant** | Transcribes the clip, applies the same verdict, writes a short rule-based safety message (no LLM) and reads it aloud with a large **Play Voice Alert** button. |
| 🗣️ **Voice Assistant** | Speak or type a question. It answers out loud with safety guidance, helplines, and optional local LLM for open questions. |

Everything runs on your laptop. No cloud required after the first model download.

---

## Quick install (Windows / Linux / macOS)

You need **Python 3.11 or 3.12** and about **8 GB free disk** (models download on first run).

```bash
# 1. Get the code
git clone https://github.com/dedsec-1337/omniguard.git
cd omniguard

# 2. Create a virtual environment (keeps your system Python clean)
python3 -m venv venv

# 3. Activate it
#    Linux / macOS:
source venv/bin/activate
#    Windows (Command Prompt):
venv\Scripts\activate.bat
#    Windows (PowerShell):
venv\Scripts\Activate.ps1

# 4. Install the packages (takes a few minutes)
pip install -r requirements.txt

# 5. Start the app
python app.py
```

Open the address it prints (usually **http://127.0.0.1:7860**) in your browser.

**First run needs internet.** It downloads roughly 3 GB of models (detectors + Whisper + one Piper voice) into the Hugging Face cache. After that you can run offline.

### Optional: one-command exhibition start (after first successful run)

```bash
# Linux / macOS
HF_HUB_OFFLINE=1 OMNIGUARD_PROFILE=fast python app.py

# Windows (Command Prompt)
set HF_HUB_OFFLINE=1
set OMNIGUARD_PROFILE=fast
python app.py
```

Or use the helper script:

```bash
bash run_exhibition.sh          # Linux / macOS
```

---

## School exhibition — GTX 1650 (4 GB VRAM) / ROG Strix class laptop

This project is built for exactly this class of machine (GTX 1650 Mobile 4 GB, 32 GB system RAM).

**What the app does automatically on a 4 GB card**

- Detects ≤ 8 GB VRAM → turns on **GPU sharing** (models wait in system RAM, borrow the GPU one at a time).
- Peak VRAM ≈ the largest single model, not all of them at once.
- If no profile is set, defaults to **`fast`** (two quickest detectors, 2 windows, tiny Whisper).
- A model that will not fit the 3.2 GB budget runs on the CPU instead of crashing.

You can still force Balanced/Thorough; they are just slower on this GPU.

**Recommended live-demo settings**

| Setting | Value | Why |
|---|---|---|
| Profile | `fast` (auto) | Lowest VRAM, quick answers |
| Offline | `HF_HUB_OFFLINE=1` | Venue Wi-Fi is often bad |
| Swap | `on` (auto under 8 GB) | One detector on the GPU at a time |
| Detectors | default, or `gary` alone if still tight | Three models are fine with sharing |

**The day before**

1. Run once **with internet** so the ~3 GB of models download.
2. (Strongly recommended) Put real + AI clips in `calib/real/` and `calib/fake/`, then:
   ```bash
   python calibrate.py
   ```
3. Test **Real voice** → green, **AI voice** → red. Replace the example WAVs if not.
4. Practise `demo_script.md`.

**On the day**

```bash
source venv/bin/activate          # Windows: venv\Scripts\activate
bash run_exhibition.sh
# or:
# HF_HUB_OFFLINE=1 OMNIGUARD_PROFILE=fast OMNIGUARD_SWAP=on python app.py
```

If VRAM is still tight, run with one detector only:

```bash
OMNIGUARD_DETECTORS=gary HF_HUB_OFFLINE=1 OMNIGUARD_PROFILE=fast python app.py
```

Keep a phone hotspot only as emergency fallback.

---

## How detection works

No single open model catches every AI voice. On recent deepfake benchmarks, open detectors lose a lot of accuracy compared with older lab tests. OMNIGUARD therefore:

1. **Ensembles several detectors** trained on different data (DF Arena, garystafford wav2vec2, MelodyMachine, optional Pellav2).
2. **Scores up to four ~4 s windows** spread across the clip after trimming silence.
3. **Never says GREEN when the detectors disagree.**
4. **Can be calibrated** on your own clips so the percentages mean something for the voices you care about.

Nothing is trained here — only pre-trained models are used. All analysis stays on this computer.

---

## Calibrate (recommended before any serious demo)

```bash
mkdir -p calib/real calib/fake
# Drop 15–30 real human recordings into calib/real/
# Drop 15–30 AI / TTS voices into calib/fake/
python calibrate.py
```

It prints accuracy before and after, and saves `calibration.json`. Commit that small file. The `calib/` folder itself is git-ignored so private recordings never leave your machine.

---

## Environment variables (optional)

| Variable | Default | Effect |
|---|---|---|
| `OMNIGUARD_PROFILE` | `fast` on ≤6 GB VRAM, else `balanced` | `fast` / `balanced` / `thorough` |
| `OMNIGUARD_DETECTORS` | `arena,gary,melody` | Comma list of detector keys. Example: `gary` for one model only |
| `OMNIGUARD_SWAP` | `auto` | GPU sharing: models in RAM, one at a time on GPU (on under 8 GB) |
| `OMNIGUARD_VRAM_GB` | `3.2` | Soft budget; models larger than this run on CPU |
| `OMNIGUARD_WHISPER` | `base` | Whisper size (`tiny`, `base`, `small`…) |
| `OMNIGUARD_REGION` | `IN` | Helpline text (India 1930 / cybercrime.gov.in by default) |
| `OMNIGUARD_LLM` | `qwen2.5:1.5b` | Local Ollama model, or `off` |
| `HF_HUB_OFFLINE` | (unset) | `1` = never touch the network (exhibition mode) |

---

## Deploy on Hugging Face Spaces (ZeroGPU)

The YAML at the top of this file is the Space config. Choose **ZeroGPU** hardware. Only the scoring path uses the GPU decorator; locally that decorator does nothing. Commit `calibration.json` if you have one.

---

## Licences and trust

- DF Arena 500M is **non-commercial** and uses `trust_remote_code=True`. Fine for an exhibition or school project; do not use it in a commercial product without permission.
- Other detectors are Apache-2.0 or MIT.
- The app itself is MIT (see `LICENSE`).

---

## Limitations

No detector is 100 % accurate. Newer cloning systems, heavy phone compression and background noise can all push results into the yellow zone or produce false confidence. Treat every result as a **warning**, never as proof. Always verify by calling back on a number you already know.

---

## Optional extras

```bash
pip install -r requirements-optional.txt   # enables Pellav2 detector
```

Local LLM (optional open questions in the Voice Assistant):

```bash
# Install Ollama, then:
ollama pull qwen2.5:1.5b
# Leave OMNIGUARD_LLM at default, or set OMNIGUARD_LLM=off to disable
```

---

## Project layout

```
omniguard/
├── app.py              # Gradio UI
├── engine.py           # Detectors, scoring, Whisper, Piper
├── guardian.py         # Safety messages, intents, helplines
├── listening.py        # Continuous mic for the Voice Assistant
├── omni_runtime.py     # Profiles, status, hardware helpers
├── channel_log.py      # Activity log
├── calibrate.py        # Learn thresholds / weights on your clips
├── assets/             # CSS, JS, font
├── examples/           # Placeholder real / fake clips (replace before demo)
├── demo_script.md      # 2-minute live script
└── requirements.txt
```
