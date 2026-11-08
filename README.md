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


---

## Detector ensemble additions

The original detectors (`arena`, `gary`, `melody`) are unchanged and all three are active by default.
New entries were added next to them; each is shown in **System & models** with its status
(`loaded`, `failed` or `skipped`) and the reason.

| Key | Model | Status |
|---|---|---|
| `pellav2` | `Sadanie/pellav2-audio-deepfake-detector` (w2v-BERT 2.0, MIT, 2.3 GB checkpoint) | Added to the default set, but skipped until you run `pip install -r requirements-optional.txt` (needs torchaudio, scipy, pyloudnorm). No published benchmark: let `calibrate.py` decide how much it counts. |
| `anti_small` | `nii-yamagishilab/wav2vec-small-anti-deepfake` | Registered, **cannot load**: fairseq-format checkpoint (needs fairseq on Python 3.9), licence CC-BY-NC-SA 4.0. Shown as skipped. |
| `gustking` | `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` | Optional. It is the model `gary` was fine-tuned from, so it adds little independent evidence. |
| `aasist_lite` | `yinyiwen/AASIST-L` | Registered, **repo not found** on the Hugging Face Hub (checked November 2026). Shown as skipped. |

- `OMNIGUARD_DETECTORS=arena,gary,melody,gustking` pins an exact list (new detectors are only appended
  automatically when this variable is not set). `OMNIGUARD_DETECTORS=gary` loads one model.
- **Voting weights:** `WEIGHTS` in `app.py` (`melody` counts 0.6, `pellav2` 1.2, others 1.0). `python calibrate.py`
  also learns weights from each detector's AUC on your clips (stored under `_weights` in `calibration.json`)
  and merges them into the existing file, so detectors that were not loaded keep their old values.
- **Speed vs accuracy** (radio in the Detector tab, or `OMNIGUARD_PROFILE`): *Fast* = two quickest detectors and 2 windows;
  *Balanced* = every detector, 4 windows (the original behaviour); *Thorough* = Balanced plus the scam-wording check.

## Desktop layout

The original single-column layout (about 760 px) is unchanged on phones. Added on top:

| Screen | Layout |
|---|---|
| 1025 px and wider | two columns (input left, results right, sticky); page width grows with the window up to 1280 px |
| 768-1024 px | single column, up to about 960 px |
| 767 px and narrower | the original mobile layout; the three tabs share the width and wrap to two lines |

A top bar adds the wordmark, a live "detectors ready" pill, the local-LLM state and a light/dark toggle.

## A local AI ecosystem (all on your laptop)

```
voice clip -> [detectors: DF Arena + Voice detector + Wav2Vec2-base (+ Pellav2)]  -> how the voice sounds
           -> [Whisper -> words]  -> [scam-wording rules + small zero-shot model] -> what the voice says
                                        both -> Referee (noisy-OR) -> overall risk + plain-language advice
question   -> [Whisper] -> intent rules (safety answers) -> optional local LLM for open questions -> Piper voice
```

- **Cross-checking:** the voice and the words are independent evidence. A real voice reading a scam script still
  raises the risk (human scammers exist); an AI voice with innocent words tops out at "high".
- **The LLM never decides.** Verdicts come from calibrated detectors and transparent rules. The optional local LLM
  only answers open questions; its answers are screened (no markdown, short, and anything telling you to share a code
  or send money is thrown away in favour of the built-in answer).
- **Small-GPU plan (GTX 1650 class, about 4 GB VRAM):** detectors wait in RAM and borrow the GPU one at a time,
  so peak VRAM is the largest model, not the sum. Any CUDA problem falls back to the CPU for that model. This is on
  automatically for GPUs of 8 GB or less (`OMNIGUARD_SWAP=on|off|auto`). Whisper and the scam-wording model stay on
  the CPU; `OMNIGUARD_WHISPER=small` improves Hindi/Bengali, `OMNIGUARD_WHISPER_DEVICE=cuda` is opt-in (needs CUDA 12
  cuBLAS and cuDNN 9) and falls back to the CPU if a test run fails.

## Voice Assistant for people with disabilities

The **Voice Assistant** tab is voice-first: press the microphone, speak, press stop, and it answers out loud and in
large text. No button to find afterwards, no screen reading required.

- **Blind / low vision:** spoken answers (Piper), large text and high-contrast switches, screen-reader friendly live regions.
- **Deaf / hard of hearing / speech difficulty:** everything is also text; you can type instead of speaking.
- **Motor difficulty:** one big record control; the microphone clears itself for the next question.
- **Cognitive / seniors:** "Use simple words" is on by default; speaking speed slider; say "slower", "faster",
  "use simple words" or "repeat that" and it obeys.
- **What it can do:** "I think I'm being scammed" (what to do right now), "I already sent money" (a step-by-step plan:
  call your bank, call the cyber crime helpline **1930** or report at **cybercrime.gov.in** in India, keep evidence),
  check a message you read to it, or check an attached recording (voice **and** words). `OMNIGUARD_REGION` changes
  the helpline text for other countries.
- **Limits:** it answers in English (Whisper understands many languages and tells you which one it heard). Basic Hindi and
  Bengali scam keywords are built in, but coverage is small. High-contrast mode restyles this app's pages and most of
  Gradio's boxes, not the browser. It is a helper, not a substitute for your bank or the police.

### Optional: a local LLM with Ollama (free, no cloud)

Everything above works without it. To let Guardian answer open questions, install Ollama (on CachyOS/Arch):

```bash
sudo pacman -S ollama              # CPU build; keeps the GPU free for the detectors
sudo systemctl enable --now ollama
ollama pull qwen2.5:1.5b           # about 1 GB
```

Use a different model with `OMNIGUARD_LLM=qwen2.5:3b`, or turn it off with `OMNIGUARD_LLM=off`.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `OMNIGUARD_DETECTORS` | `arena,gary,melody` (+ new ones) | exact detector list |
| `OMNIGUARD_PROFILE` | `balanced` | `fast`, `balanced` or `thorough` |
| `OMNIGUARD_SWAP` | `auto` | GPU sharing for small GPUs |
| `OMNIGUARD_WHISPER` / `_DEVICE` | `base` / `cpu` | Whisper size and device |
| `OMNIGUARD_SEMANTIC` | `1` | `0` skips the zero-shot scam-wording model (about 0.7 GB) |
| `OMNIGUARD_LLM` / `OLLAMA_HOST` | `qwen2.5:1.5b` / `127.0.0.1:11434` | local LLM |
| `OMNIGUARD_REGION` | `IN` | helpline text |
