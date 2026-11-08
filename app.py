"""OMNIGUARD - Protect your voice, protect your people.

Tab 1: deepfake voice detector (ensemble of wav2vec2-family models).
Tab 2: accessibility assistant (faster-whisper transcript + rule-based safety
message + Piper spoken alert).
Runs on a plain laptop CPU (`python app.py`) and on Hugging Face ZeroGPU.
"""
import html
import json
import math
import os
import tempfile
import wave
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

# --- ADDED: orchestration layer (see omni_runtime.py and guardian.py) ---------------------------
import hashlib
import threading
import time

import guardian as gd
import omni_runtime as rt

# --- ZeroGPU shim -----------------------------------------------------------
# WHY: on Hugging Face the real `spaces.GPU` decorator borrows a GPU; on a laptop
# the package may be missing, so we swap in a decorator that does nothing.
try:
    import spaces
    _ZEROGPU = True
except ImportError:
    _ZEROGPU = False

    class _FakeSpaces:
        @staticmethod
        def GPU(fn=None, *, duration=60):
            return fn if fn is not None else (lambda f: f)

    spaces = _FakeSpaces()

# --- Settings ---------------------------------------------------------------
HERE = Path(__file__).parent
SAMPLE_RATE = 16_000            # every model here expects 16 kHz mono
WINDOW = 64_600                 # ~4.04 s: the clip length DF Arena was trained on
MAX_WINDOWS = 4                 # windows spread across the clip (keeps CPU fast)
MAX_SECONDS = 30                # never read more than this from a file
MIN_SECONDS = 0.5               # anything shorter is not a usable clip
SILENCE_RMS = 1e-3              # below this loudness we call the clip silent
PIPER_VOICE = "en_US-lessac-medium"
EXAMPLES = {"real": HERE / "examples" / "real_sample.wav",
            "fake": HERE / "examples" / "fake_sample.wav"}
# WHY: on ZeroGPU, torch reports CUDA as available outside @spaces.GPU too
# (emulation), so moving models once at startup is the documented pattern.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detector catalogue. WHY several: one small model trained on a few TTS engines
# misses newer voices (the field's known "generalisation gap"). Averaging models
# trained on different data is the cheapest robust fix.
#   kind "arena": custom-code model (needs trust_remote_code), fixed ~4 s window
#   kind "hf":    standard transformers audio-classification model
CATALOG = {
    "arena": {"name": "DF Arena 500M", "kind": "arena",
              "id": "Speech-Arena-2025/DF_Arena_500M_V_1"},
    "gary": {"name": "Voice detector", "kind": "hf",
             "id": "garystafford/wav2vec2-deepfake-voice-detector"},
    "melody": {"name": "Wav2Vec2-base detector", "kind": "hf",
               "id": "MelodyMachine/Deepfake-audio-detection-V2"},
}
# --- ADDED: extra detectors (additive; the entries above are untouched) ------------------------------
CATALOG.update({
    # Pellav2: w2v-BERT 2.0 backbone, custom .pt checkpoint, MIT. Needs torchaudio, scipy, pyloudnorm.
    # The repo id is Sadanie/..., not Sadenie/... .
    "pellav2": {"name": "Pellav2", "kind": "pellav", "id": "Sadanie/pellav2-audio-deepfake-detector",
                "requires": ("torchaudio", "scipy", "pyloudnorm")},
    # AntiDeepfake (NII): fairseq-format checkpoint (CC-BY-NC-SA 4.0). Registered so it shows up in the
    # status panel, but it cannot be loaded with Transformers.
    "anti_small": {"name": "AntiDeepfake Small", "kind": "fairseq",
                   "id": "nii-yamagishilab/wav2vec-small-anti-deepfake",
                   "unavailable": "fairseq-format checkpoint (needs fairseq on Python 3.9); not loadable with Transformers"},
    # gustking is the parent model gary was fine-tuned from, so it adds little independent evidence.
    "gustking": {"name": "XLSR detector", "kind": "hf", "optional": True,
                 "id": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"},
    "aasist_lite": {"name": "AASIST-lite", "kind": "hf", "optional": True, "id": "yinyiwen/AASIST-L",
                    "unavailable": "repo not found on the Hugging Face Hub (checked November 2026)"},
})
# Choose detectors without editing code:  OMNIGUARD_DETECTORS=arena,gary,melody
ACTIVE = [k.strip() for k in os.environ.get("OMNIGUARD_DETECTORS", "arena,gary").split(",")
          if k.strip() in CATALOG] or ["arena", "gary"]
# --- ADDED: melody is in the default set again; the new detectors are appended only when the user has
# not pinned a list (OMNIGUARD_DETECTORS=gary must still mean exactly one model).
_base = [k.strip() for k in os.environ.get("OMNIGUARD_DETECTORS", "arena,gary,melody").split(",")
         if k.strip() in CATALOG]
_new = [] if "OMNIGUARD_DETECTORS" in os.environ else [
    k for k in ("pellav2", "anti_small") if k in CATALOG and k not in _base]
ACTIVE = (_base + _new) or ["arena", "gary", "melody"]
# Voting weights (calibrate.py can replace them with weights learned from your own clips).
WEIGHTS = {"arena": 1.0, "gary": 1.0, "melody": 0.6, "pellav2": 1.2, "anti_small": 1.0,
           "gustking": 1.0, "aasist_lite": 1.1}
SPEED_RANK = ["melody", "gary", "arena", "gustking", "pellav2"]  # quickest first (profile "Fast")
rt.set_global_profile(os.environ.get("OMNIGUARD_PROFILE", "balanced").lower())


def read_calibration():
    """Per-model scaling {a, b} written by calibrate.py (optional)."""
    try:
        data = json.loads((HERE / "calibration.json").read_text())
        return {k: {"a": float(v["a"]), "b": float(v["b"])} for k, v in data.items() if k in CATALOG}
    except Exception:  # noqa: BLE001 - missing/bad file just means "use defaults"
        return {}


CALIBRATION = read_calibration()


def read_weights():
    """Optional voting weights learned by calibrate.py (stored under the "_weights" key)."""
    try:
        data = json.loads((HERE / "calibration.json").read_text()).get("_weights", {})
        return {k: float(v) for k, v in data.items() if k in CATALOG}
    except Exception:  # noqa: BLE001
        return {}


CAL_WEIGHTS = read_weights()


def effective_weight(key):
    return CAL_WEIGHTS.get(key, WEIGHTS.get(key, 1.0))


# --- Model loading (once, at startup) ---------------------------------------
# WHY: each loader returns (model, error_text) instead of raising, so one failed
# download never crashes the app - the other features keep working.
def _fake_index(config):
    """Find which output index means 'fake'/'spoof' from the model's own labels."""
    for idx, name in config.id2label.items():
        if any(w in str(name).lower() for w in ("fake", "spoof", "synthetic")):
            return int(idx)
    return 1


def load_detector(key):
    spec = CATALOG[key]
    try:
        from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification
        if spec["kind"] == "arena":
            model, extractor = AutoModel.from_pretrained(spec["id"], trust_remote_code=True), None
        else:
            extractor = AutoFeatureExtractor.from_pretrained(spec["id"])
            model = AutoModelForAudioClassification.from_pretrained(spec["id"])
        model = model.to(DEVICE).eval()
        return {"key": key, "kind": spec["kind"], "model": model, "extractor": extractor,
                "fake": _fake_index(model.config)}, None
    except Exception as exc:  # noqa: BLE001 - any failure must degrade gracefully
        return None, f"{type(exc).__name__}: {exc}"


def load_whisper():
    try:
        from faster_whisper import WhisperModel
        # WHY CPU/int8: CTranslate2 cannot use ZeroGPU's CUDA emulation, and the
        # base model is fast enough on CPU.
        return WhisperModel("base", device="cpu", compute_type="int8"), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _voice_folder():
    """Persistent cache first (survives reboots, needed for offline backup)."""
    for base in (Path.home() / ".cache", Path(tempfile.gettempdir())):
        folder = base / "omniguard" / "piper_voices"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        except OSError:
            continue
    return Path(tempfile.gettempdir())


def _present(path):
    return path.is_file() and path.stat().st_size > 0


def load_piper():
    try:
        from piper import PiperVoice
        folder = _voice_folder()
        onnx = folder / f"{PIPER_VOICE}.onnx"
        config = folder / f"{PIPER_VOICE}.onnx.json"
        if not (_present(onnx) and _present(config)):
            # piper-tts >= 1.3 ships this downloader (same as
            # `python -m piper.download_voices`); it fetches both files.
            from piper.download_voices import download_voice
            download_voice(PIPER_VOICE, folder)
        if not (_present(onnx) and _present(config)):  # verify BEFORE loading
            raise FileNotFoundError("voice files are missing after download")
        return PiperVoice.load(str(onnx)), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# --- ADDED: hardware plan, load timing and status tracking (wraps the loaders above) -------------------
# GTX 1650-class laptops have ~4 GB of VRAM: detectors wait in RAM and borrow the GPU one at a time.
SWAP_MODE = rt.want_swap(torch, DEVICE, _ZEROGPU, os.environ.get("OMNIGUARD_SWAP", "auto").lower())
HW = rt.hardware_summary(torch, DEVICE, SWAP_MODE)
RESIDENCY = rt.Residency(torch, SWAP_MODE)
_GPU_LOCK = threading.RLock()
_load_detector_base = load_detector
_load_whisper_base = load_whisper
_load_piper_base = load_piper


def load_detector(key):  # wraps the original loader (kept as _load_detector_base)
    """Skips unavailable/optional detectors with a reason, times every load, and parks models in RAM
    when the GPU is small."""
    spec = CATALOG[key]
    if spec.get("unavailable"):
        rt.set_status(key, "skipped", spec["unavailable"])
        return None, spec["unavailable"]
    missing = rt.missing_modules(spec.get("requires", ()))
    if missing:
        note = "optional packages missing: " + ", ".join(missing) + " (pip install -r requirements-optional.txt)"
        rt.set_status(key, "skipped", note)
        return None, note
    t0 = time.perf_counter()
    if spec["kind"] == "pellav":
        try:
            det, err = {"key": key, "kind": "pellav", "model": rt.load_pellav(torch, spec["id"]),
                        "extractor": None, "fake": 0}, None
        except Exception as exc:  # noqa: BLE001
            det, err = None, f"{type(exc).__name__}: {exc}"
        if det is not None and DEVICE == "cuda" and not SWAP_MODE:
            det["model"].to(DEVICE)
    else:
        det, err = _load_detector_base(key)
    rt.record_load(key, time.perf_counter() - t0)
    if det is None:
        rt.set_status(key, "failed", err or "")
    else:
        if SWAP_MODE:
            RESIDENCY.release(det["model"])
        rt.set_status(key, "loaded", "")
    return det, err


def load_whisper():  # wraps the original (kept as _load_whisper_base); adds OMNIGUARD_WHISPER[_DEVICE]
    """OMNIGUARD_WHISPER=small gives better Hindi/Bengali. OMNIGUARD_WHISPER_DEVICE=cuda is opt-in
    (needs CUDA 12 cuBLAS + cuDNN 9) and falls back to the CPU if a test run fails."""
    t0 = time.perf_counter()
    size = os.environ.get("OMNIGUARD_WHISPER", "").strip()
    want_cuda = os.environ.get("OMNIGUARD_WHISPER_DEVICE", "cpu").lower() == "cuda"
    if not size and not want_cuda:
        model, err = _load_whisper_base()
    else:
        model, err = None, None
        try:
            from faster_whisper import WhisperModel
            if want_cuda:
                try:
                    model = WhisperModel(size or "base", device="cuda", compute_type="int8_float16")
                    list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))[0])  # surfaces missing CUDA libs
                except Exception as exc:  # noqa: BLE001
                    print(f"[OMNIGUARD] Whisper on CUDA failed ({type(exc).__name__}); using the CPU")
                    model = None
            if model is None:
                model = WhisperModel(size or "base", device="cpu", compute_type="int8")
        except Exception as exc:  # noqa: BLE001
            model, err = None, f"{type(exc).__name__}: {exc}"
    rt.record_load("whisper", time.perf_counter() - t0)
    rt.set_status("whisper", "loaded" if model else "failed", err or "")
    return model, err


def load_piper():  # wraps the original (kept as _load_piper_base)
    t0 = time.perf_counter()
    voice, err = _load_piper_base()
    rt.record_load("piper", time.perf_counter() - t0)
    rt.set_status("piper", "loaded" if voice else "failed", err or "")
    return voice, err


DETECTORS = []
for _key in ACTIVE:
    _det, _err = load_detector(_key)
    if _det:
        DETECTORS.append(_det)
    else:  # technical detail goes to the console, not to the user's screen
        print(f"[OMNIGUARD] detector '{_key}' unavailable -> {_err}")
WHISPER, WHISPER_ERR = load_whisper()
VOICE, VOICE_ERR = load_piper()
for _name, _err in (("whisper", WHISPER_ERR), ("piper", VOICE_ERR)):
    if _err:
        print(f"[OMNIGUARD] {_name} unavailable -> {_err}")
if not CALIBRATION:
    print("[OMNIGUARD] no calibration.json: raw model scores in use "
          "(run `python calibrate.py` with your own clips to tune them)")


# --- ADDED: detectors not attempted, the "words" models, and a one-line startup summary -----------------
for _k, _spec in CATALOG.items():
    if _k not in rt.MODEL_STATUS:
        rt.set_status(_k, "skipped", _spec.get("unavailable") or (
            "optional: add it to OMNIGUARD_DETECTORS to load it" if _spec.get("optional") else "not in the active list"))
SEMANTIC = gd.Semantic()
_t0 = time.perf_counter()
_sem_ok = SEMANTIC.load()
rt.record_load("semantic", time.perf_counter() - _t0)
rt.set_status("semantic", "loaded" if _sem_ok else "skipped", "" if _sem_ok else (SEMANTIC.error or ""))
LLM = gd.Ollama()
print(f"[OMNIGUARD] device={DEVICE} swap={SWAP_MODE} detectors="
      f"{[d['key'] for d in DETECTORS]} semantic={'on' if _sem_ok else 'off'} "
      f"llm={'ready' if LLM.available() else 'off'} profile={rt.current_profile()}")

# --- Audio helpers ------------------------------------------------------------
class AudioProblem(Exception):
    """An error whose message is safe and friendly to show on screen."""


def load_audio(path):
    """Read any common format, resample to 16 kHz mono, trim silence, reject bad clips."""
    if not path:
        raise AudioProblem("No audio yet. Upload a clip or record one first.")
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
    except Exception:  # noqa: BLE001
        raise AudioProblem("That file could not be read as audio. "
                           "Try a .wav, .mp3, .m4a or .ogg clip.") from None
    if audio.size < SAMPLE_RATE * MIN_SECONDS or float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS:
        raise AudioProblem("The clip is silent or too short to analyse. Please try another one.")
    # WHY trim: DF Arena reads only a 4 s window; leading silence would waste it.
    trimmed, _ = librosa.effects.trim(audio, top_db=35)
    if trimmed.size >= SAMPLE_RATE * MIN_SECONDS:
        audio = trimmed
    return audio.astype(np.float32)


def make_windows(audio):
    """Up to MAX_WINDOWS windows spread evenly across the clip."""
    if audio.size <= WINDOW:
        return [audio]
    count = min(MAX_WINDOWS, math.ceil(audio.size / WINDOW))
    starts = np.linspace(0, audio.size - WINDOW, num=count).astype(int)
    return [audio[s:s + WINDOW] for s in starts]


def _fit(chunk):
    """Crop or tile to exactly WINDOW samples (what DF Arena's own extractor does)."""
    if chunk.size >= WINDOW:
        return chunk[:WINDOW]
    return np.tile(chunk, math.ceil(WINDOW / chunk.size))[:WINDOW]


def _score_window(det, chunk):
    """Probability (0-1) that one window is fake, according to one detector."""
    if det["kind"] == "arena":
        x = torch.from_numpy(_fit(chunk)).float().to(DEVICE)
        logits = det["model"](input_values=x)["logits"]
    else:
        inputs = det["extractor"](chunk, sampling_rate=SAMPLE_RATE,
                                  return_tensors="pt", padding=True)
        logits = det["model"](**{k: v.to(DEVICE) for k, v in inputs.items()}).logits
    return float(torch.softmax(logits, dim=-1)[0][det["fake"]])


@spaces.GPU(duration=15)
def _score_all(windows):
    """{detector key: [fake probability per window]}. Only this needs the GPU."""
    out = {}
    with torch.no_grad():
        for det in DETECTORS:
            try:
                out[det["key"]] = [_score_window(det, w) for w in windows]
            except Exception as exc:  # noqa: BLE001 - one bad model must not sink the rest
                print(f"[OMNIGUARD] {det['key']} failed on this clip -> {type(exc).__name__}: {exc}")
                out[det["key"]] = None
    return out


def raw_scores(audio):
    """{detector key: mean raw fake probability}. Also used by calibrate.py."""
    if not DETECTORS:
        raise AudioProblem("The deepfake models are not available right now. "
                           "Connect to the internet once so they can download, then restart.")
    per_window = _score_all(make_windows(audio))
    scores = {k: float(np.mean(v)) for k, v in per_window.items() if v}
    if not scores:
        raise AudioProblem("The deepfake models could not score this clip.")
    return scores


def calibrated(raw, key):
    """Turn a raw score into an honest probability using the scaling fitted on your own
    clips (Platt scaling). With no calibration.json this is the identity."""
    params = CALIBRATION.get(key)
    if not params:
        return raw
    r = min(max(raw, 1e-4), 1 - 1e-4)
    z = params["a"] * math.log(r / (1 - r)) + params["b"]
    return 1 / (1 + math.exp(-z))


def classify(p_fake):
    if p_fake < 0.30:
        return "green"
    return "yellow" if p_fake <= 0.70 else "red"


def analyse(audio):
    """Run every detector and combine them (safety first: disagreement is never green)."""
    models = [{"name": CATALOG[k]["name"], "p": calibrated(r, k)}
              for k, r in raw_scores(audio).items()]
    ps = [m["p"] for m in models]
    p, spread = float(np.mean(ps)), max(ps) - min(ps)
    key = classify(p)
    if key == "green" and spread > 0.5:
        key = "yellow"
    return {"p": p, "key": key, "models": models, "spread": spread}


def transcribe(audio):
    if WHISPER is None:
        return "(Transcription is unavailable right now.)"
    try:
        # WHY list(): faster-whisper returns a lazy generator; work starts here.
        segments, _ = WHISPER.transcribe(audio, beam_size=1, vad_filter=True)
        text = " ".join(s.text.strip() for s in list(segments)).strip()
        return text or "(No clear speech detected.)"
    except Exception:  # noqa: BLE001
        return "(The transcription failed for this clip.)"


def speak(text):
    """Return a path to a spoken WAV of `text`, or None if TTS is unavailable."""
    if VOICE is None:
        return None
    try:
        out = tempfile.NamedTemporaryFile(suffix=".wav", prefix="omniguard_", delete=False)
        out.close()
        with wave.open(out.name, "wb") as wav_file:
            VOICE.synthesize_wav(text, wav_file)
        return out.name
    except Exception:  # noqa: BLE001
        return None


# --- ADDED: multi-model orchestration (the original functions above remain as the fallback path) ----------
_WORK = threading.RLock()  # one analysis at a time: protects VRAM, lets a second UI listener reuse the result
_TL = threading.local()    # per-window scores of the latest scoring call (for the results table)
_CACHE = {"analyse": {}, "words": {}}


def _digest(audio):
    return hashlib.sha1(audio.tobytes()).hexdigest()


def _remember(store, key, value, limit=8):
    store[key] = value
    while len(store) > limit:
        store.pop(next(iter(store)))
    return value


def _score_window_on(det, chunk, device):
    """Probability (0-1) that one window is fake, with the model on `device` (all detector kinds)."""
    if det["kind"] == "pellav":
        x = torch.from_numpy(rt.pellav_preprocess(chunk))[None].to(device)
        return float(torch.sigmoid(det["model"](x)).reshape(-1)[0])
    if det["kind"] == "arena":
        x = torch.from_numpy(_fit(chunk)).float().to(device)
        logits = det["model"](input_values=x)["logits"]
    else:
        inputs = det["extractor"](chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        logits = det["model"](**{k: v.to(device) for k, v in inputs.items()}).logits
    return float(torch.softmax(logits, dim=-1)[0][det["fake"]])


def _run_detector(det, windows):
    """Score all windows with one detector; times it; one CUDA OOM retries on the CPU."""
    t0 = time.perf_counter()
    try:
        with RESIDENCY.borrow(det["model"]) as device:
            try:
                vals = [_score_window_on(det, w, device) for w in windows]
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or device == "cpu":
                    raise
                RESIDENCY.release(det["model"])
                device = "cpu"
                vals = [_score_window_on(det, w, "cpu") for w in windows]
        rt.record_latency(det["key"], (time.perf_counter() - t0) / max(len(windows), 1), device)
        return vals
    except Exception as exc:  # noqa: BLE001 - one bad model must not sink the rest
        print(f"[OMNIGUARD] {det['key']} failed on this clip -> {type(exc).__name__}: {exc}")
        return None


_score_all_base = _score_all  # the original ZeroGPU-decorated scorer, intact


def _score_all(windows):  # noqa: F811 - deliberate: profile-aware local scorer; ZeroGPU keeps the original
    if _ZEROGPU:
        return _score_all_base(windows)
    profile = rt.current_profile()
    chosen = rt.select_detectors(DETECTORS, profile, SPEED_RANK)
    windows = windows[: rt.window_limit(profile, len(windows))]
    out = {}
    with _GPU_LOCK, torch.no_grad():
        for det in chosen:
            out[det["key"]] = _run_detector(det, windows)
    _TL.last_windows = out
    return out


_analyse_base = analyse


def analyse(audio):  # noqa: F811 - weighted, cached, profile-aware ensemble (original kept as _analyse_base)
    key = (_digest(audio), rt.current_profile(), tuple(d["key"] for d in DETECTORS))
    with _WORK:
        hit = _CACHE["analyse"].get(key)
        if hit is not None:
            return hit
        _TL.last_windows = {}
        scores = raw_scores(audio)
        windows = getattr(_TL, "last_windows", {}) or {}
        models = [{"key": k, "name": CATALOG[k]["name"], "raw": raw, "p": calibrated(raw, k),
                   "weight": effective_weight(k),
                   "windows": [calibrated(w, k) for w in (windows.get(k) or [])]}
                  for k, raw in scores.items()]
        total = sum(m["weight"] for m in models) or 1.0
        p = sum(m["p"] * m["weight"] for m in models) / total
        ps = [m["p"] for m in models]
        spread = max(ps) - min(ps)
        verdict = classify(p)
        if verdict == "green" and spread > 0.5:
            verdict = "yellow"
        return _remember(_CACHE["analyse"], key, {"p": p, "key": verdict, "models": models,
                                                  "spread": spread, "profile": rt.current_profile()})


_transcribe_base = transcribe


def transcribe(audio):  # noqa: F811 - cached so a second listener never repeats Whisper's work
    key = _digest(audio)
    with _WORK:
        hit = _CACHE["words"].get(key)
        if hit is None:
            hit = _transcribe_base(audio)
            if not hit.startswith("("):
                _remember(_CACHE["words"], key, hit)
        return hit


def transcribe_ex(audio):
    """(text, detected language) for the voice assistant; ("", None) if Whisper is unavailable."""
    if WHISPER is None:
        return "", None
    try:
        segments, info = WHISPER.transcribe(audio, beam_size=1, vad_filter=True)
        return " ".join(s.text.strip() for s in list(segments)).strip(), getattr(info, "language", None)
    except Exception:  # noqa: BLE001
        return "", None


def speak_rate(text, speed=1.0):
    """Like speak() but with an adjustable speaking speed (Piper length_scale = 1 / speed)."""
    if VOICE is None:
        return None
    try:
        from piper import SynthesisConfig
        out = tempfile.NamedTemporaryFile(suffix=".wav", prefix="omniguard_", delete=False)
        out.close()
        cfg = SynthesisConfig(length_scale=1.0 / min(max(float(speed), 0.5), 1.6))
        with wave.open(out.name, "wb") as wav_file:
            VOICE.synthesize_wav(gd.speakable(text), wav_file, syn_config=cfg)
        return out.name
    except Exception:  # noqa: BLE001
        return None


def analyse_clip(path):
    """Voice + words for one recording: everything the cross-check card and the assistant need."""
    audio = load_audio(path)  # may raise AudioProblem
    transcript = transcribe(audio)
    heard = not transcript.startswith("(")
    words = gd.assess_text(transcript, SEMANTIC) if heard else {"score": 0.0, "signals": [], "semantic": None}
    try:
        res = analyse(audio)
    except AudioProblem:
        res = None  # words-only graceful path
    fused = gd.fuse(res["p"] if res else None, words["score"])
    voice = (f"The voice scores {res['p'] * 100:.0f} percent for AI-generated speech."
             if res else "I could not check the voice.")
    said = ("The words include: " + "; ".join(sg["label"] for sg in words["signals"][:3]) + "."
            if words["signals"] else "")
    return {"res": res, "words": words, "fused": fused, "transcript": transcript if heard else "",
            "summary": f"I checked the recording. {voice} {said} {fused['story']} Overall risk: {fused['level']}. {fused['advice']}",
            "summary_simple": f"Risk is {fused['level']}. {fused['story']} {fused['advice']}"}


# --- Rule-based verdict text (no LLM) ------------------------------------------
VERDICTS = {
    "green": {
        "color": "#16a34a", "label": "Likely real", "headline": "Sounds like a real person",
        "advice": "No strong sign of AI in this clip. Still, call back on a number you know before acting.",
        "why": "The detectors found no strong sign of synthetic speech in this clip.",
        "speech": "This voice sounds like a real person. But always verify by calling back on a known number.",
    },
    "yellow": {
        "color": "#d97706", "label": "Uncertain", "headline": "We can't be sure",
        "advice": "Do not send money or share codes. Hang up and call back on a number you know.",
        "why": "The score sits in the grey zone, or the detectors disagree. Noise, phone "
               "compression or a short clip can cause this.",
        "speech": "Be careful. I cannot tell if this voice is real or AI-generated. "
                  "Do not send money or share codes. Call back on a number you know.",
    },
    "red": {
        "color": "#dc2626", "label": "Likely AI-generated", "headline": "This voice is likely AI-generated",
        "advice": "Do not send money. Hang up and call a family member on a number you know.",
        "why": "AI voices often show unnatural pauses, overly smooth rhythm, odd spectral patterns "
               "and missing breath sounds. The detectors score these patterns; they do not point "
               "to exact moments.",
        "speech": "Warning. This voice is likely AI-generated. Do not send money. Call a family member first.",
    },
}
NO_VERDICT_SPEECH = ("I could not check whether this voice is real. Treat it with caution. "
                     "Do not send money. Call back on a number you know.")
TIP = "Never transfer money based on a single phone call. Call back on a known number to verify."


# --- HTML pieces (all inline: no external fonts or scripts, so offline-safe) ----
RING = 2 * math.pi * 52


def ring_svg(p, color):
    return (f'<svg viewBox="0 0 120 120" class="og-ring-svg" aria-hidden="true">'
            f'<circle cx="60" cy="60" r="52" class="og-ring-bg"/>'
            f'<circle cx="60" cy="60" r="52" class="og-ring-fg" transform="rotate(-90 60 60)" '
            f'style="stroke:{color};stroke-dasharray:{RING:.1f};--from:{RING:.1f};'
            f'stroke-dashoffset:{RING * (1 - p):.1f}"/>'
            f'<text x="60" y="64" class="og-ring-num">{p * 100:.0f}%</text>'
            f'<text x="60" y="83" class="og-ring-sub">AI-generated</text></svg>')


def result_card(res):
    v = VERDICTS[res["key"]]
    return (f'<div class="og-result" role="status" aria-live="polite" style="--accent:{v["color"]}">'
            f'<div class="og-ring">{ring_svg(res["p"], v["color"])}</div>'
            f'<div class="og-verdict"><span class="og-chip"><i class="og-dot"></i>{v["label"]}</span>'
            f'<h2>{v["headline"]}</h2><p>{v["advice"]}</p></div></div>')


def notice_card(message):
    return (f'<div class="og-result og-notice" role="status" aria-live="polite">'
            f'<div class="og-verdict"><span class="og-chip">Cannot analyse</span>'
            f'<p>{html.escape(message)}</p></div></div>')


EMPTY_CARD = ('<div class="og-empty">Your result will appear here.<br>'
              '<small>Add a clip above, then press the button.</small></div>')


def details_html(res):
    rows = "".join(
        f'<div class="og-row"><span>{html.escape(m["name"])}</span>'
        f'<div class="og-bar"><i style="width:{m["p"] * 100:.0f}%"></i></div>'
        f'<b>{m["p"] * 100:.0f}%</b></div>' for m in res["models"])
    split = ("<p><b>The detectors disagree,</b> so this is treated as uncertain.</p>"
             if res["spread"] > 0.5 and len(res["models"]) > 1 else "")
    return (f'<details class="og-more"><summary>How we decided</summary><div class="og-details">'
            f'<div class="og-rows">{rows}</div>{split}'
            f'<p>{VERDICTS[res["key"]]["why"]}</p><p><b>Safety tip:</b> {TIP}</p>'
            f'<p class="og-fine">Scores are the share of evidence for "AI-generated". '
            f'Only the first {MAX_SECONDS} seconds are analysed. No detector is 100% accurate.</p>'
            f'</div></details>')


# --- Tab 1: Deepfake Detector -------------------------------------------------
def detect(path):
    try:
        res = analyse(load_audio(path))
    except AudioProblem as problem:
        return notice_card(str(problem))
    except Exception:  # noqa: BLE001 - never show a stack trace
        return notice_card("Something went wrong while analysing this clip.")
    return result_card(res) + details_html(res)


# --- Tab 2: Accessibility Assistant -------------------------------------------
def assist(path):
    """Transcript + verdict + spoken alert. Works even if some models failed."""
    try:
        audio = load_audio(path)
    except AudioProblem as problem:
        return notice_card(str(problem)), "", str(problem), None
    transcript = transcribe(audio)
    try:
        res = analyse(audio)
        card, message = result_card(res), VERDICTS[res["key"]]["speech"]
    except Exception:  # noqa: BLE001 - graceful degradation without the detectors
        card, message = notice_card("Voice check unavailable. Be careful."), NO_VERDICT_SPEECH
    speech = speak(message)
    if speech is None:
        message += "\n\n⚠️ Voice alert unavailable - please read this message instead."
    return card, transcript, f"## {message}", speech


# --- Shared examples ------------------------------------------------------------
def load_example(kind):
    """Return an example clip path for one tab; fail politely if it is missing."""
    path = EXAMPLES[kind]
    try:
        sf.info(str(path))
    except Exception:  # noqa: BLE001
        gr.Warning("Example not available - please upload your own clip")
        return gr.update()
    return str(path)


# --- ADDED: results table, cross-check card, system panel, voice-assistant turn ----------------------------
LEVEL_COLOR = {"low": "#16a34a", "elevated": "#d97706", "high": "#dc2626", "critical": "#991b1b"}


def _spark(values):
    if not values:
        return ""
    bars = "".join(f'<rect x="{i * 9}" y="{16 - max(2, v * 16):.1f}" width="6" height="{max(2, v * 16):.1f}" rx="2"/>'
                   for i, v in enumerate(values[:4]))
    return f'<svg class="og-spark" viewBox="0 0 {len(values[:4]) * 9} 16" aria-hidden="true">{bars}</svg>'


def _mrow(i, key, name, p, weight, windows):
    status = rt.MODEL_STATUS.get(key, "skipped")
    score = (f'<div class="og-mbar"><i style="width:{p * 100:.0f}%"></i></div><b>{p * 100:.0f}%</b>'
             if p is not None else '<div class="og-mbar"></div><b>&mdash;</b>')
    info = ('<span class="og-info">Independent benchmark: variable</span>' if key == "melody" else "")
    return (f'<div class="og-mrow{"" if p is not None else " og-mute"}" style="--i:{i}">'
            f'<span class="og-mname">{html.escape(name)} <i class="og-w">&times;{weight:g}</i>{info}</span>'
            f'<span class="og-sparkcell">{_spark(windows)}</span>{score}'
            f'<span class="og-st og-st-{status}" title="{html.escape(rt.STATUS_NOTE.get(key, ""))}">{status}</span></div>')


def model_table_html(res):
    rows = [_mrow(i, m["key"], m["name"], m["p"], m["weight"], m["windows"]) for i, m in enumerate(res["models"])]
    shown = {m["key"] for m in res["models"]}
    rows += [_mrow(len(rows) + j, k, CATALOG[k]["name"], None, effective_weight(k), [])
             for j, k in enumerate(k for k in ACTIVE if k not in shown)]
    prof = rt.PROFILES.get(res.get("profile", "balanced"), rt.PROFILES["balanced"])["label"]
    return (f'<div class="og-mtable"><div class="og-mt-head"><b>Detectors</b>'
            f'<small>{prof} mode &middot; weighted vote</small></div>{"".join(rows)}</div>')


def crosscheck_card_html(info):
    f, words = info["fused"], info["words"]
    chips = "".join(f'<span class="og-sig" title="{html.escape(sg["evidence"])}">{html.escape(sg["label"])}</span>'
                    for sg in words["signals"][:5]) or '<span class="og-sig og-sig-none">no scam wording found</span>'
    said = (f'<details class="og-more"><summary>What was heard</summary><div class="og-details"><p>'
            f'{html.escape(info["transcript"])}</p></div></details>' if info["transcript"] else "")
    return (f'<div class="og-cross" style="--accent:{LEVEL_COLOR[f["level"]]}"><div class="og-cross-head">'
            f'<span class="og-chip"><i class="og-dot"></i>Overall risk: {f["level"]}</span>'
            f'<small>voice &times; words</small></div><p class="og-story">{html.escape(f["story"])}</p>'
            f'<div class="og-signals">{chips}</div><p class="og-advice">{html.escape(f["advice"])}</p>{said}</div>')


def detail_a(path):
    """Second listener on the Detector button: per-model table (+ the words check in Thorough mode).
    Reads the cached analysis, so the models do not run twice."""
    try:
        res = analyse(load_audio(path))
    except Exception:  # noqa: BLE001
        return "", ""
    cross = ""
    if rt.PROFILES[rt.current_profile()]["crosscheck"]:
        try:
            cross = crosscheck_card_html(analyse_clip(path))
        except Exception:  # noqa: BLE001
            cross = ""
    return model_table_html(res), cross


def crosscheck_b(path):
    try:
        return crosscheck_card_html(analyse_clip(path))
    except Exception:  # noqa: BLE001
        return ""


def set_profile(name):
    rt.set_global_profile(name)


def topbar_html():
    ready = len(DETECTORS)
    failed = sum(1 for k in ACTIVE if rt.MODEL_STATUS.get(k) == "failed")
    cls = "og-pill-warn" if (failed or not ready) else "og-pill-ok"
    label = f"{ready} detector{'s' if ready != 1 else ''} ready" + (f" &middot; {failed} failed" if failed else "")
    llm = (f'<span class="og-pill og-pill-ok"><i class="og-dot"></i>Local LLM {html.escape(LLM.model)}</span>'
           if LLM.available() else '<span class="og-pill og-pill-off"><i class="og-dot"></i>Local LLM off</span>')
    return ('<div class="og-topbar-in"><span class="og-wordmark">OMNIGUARD</span>'
            '<span class="og-tagline">Private &middot; Local &middot; Free</span>'
            f'<span class="og-pill {cls}"><i class="og-dot"></i>{label}</span>{llm}</div>')


def system_html():
    def chip(status):
        return f'<span class="og-st og-st-{status}">{status}</span>'

    def lat(key):
        v = rt.avg_latency(key)
        return f"{v:.1f} s/window on {rt.last_device(key)}" if v is not None else "not run yet"

    rows = "".join(
        f'<tr><td>{html.escape(spec["name"])}<br><small>{html.escape(spec["id"])}</small></td>'
        f'<td>{chip(rt.MODEL_STATUS.get(k, "skipped"))}</td>'
        f'<td>{rt.LOAD_SECONDS[k]:.1f} s</td><td>{lat(k)}</td>'
        f'<td><small>{html.escape(rt.STATUS_NOTE.get(k, ""))}</small></td></tr>' if k in rt.LOAD_SECONDS else
        f'<tr><td>{html.escape(spec["name"])}<br><small>{html.escape(spec["id"])}</small></td>'
        f'<td>{chip(rt.MODEL_STATUS.get(k, "skipped"))}</td><td>&mdash;</td><td>&mdash;</td>'
        f'<td><small>{html.escape(rt.STATUS_NOTE.get(k, ""))}</small></td></tr>'
        for k, spec in CATALOG.items())
    extras = "".join(
        f'<tr><td>{label}</td><td>{chip(rt.MODEL_STATUS.get(k, "skipped"))}</td>'
        f'<td>{rt.LOAD_SECONDS[k]:.1f} s</td><td>&mdash;</td><td><small>{html.escape(rt.STATUS_NOTE.get(k, ""))}</small></td></tr>'
        for k, label in (("whisper", "Speech-to-text (faster-whisper)"), ("piper", "Voice (Piper)"),
                         ("semantic", "Scam-wording model (zero-shot)")) if k in rt.LOAD_SECONDS)
    llm = "ready: " + LLM.model if LLM.available() else "off (optional: install Ollama, see README)"
    hw = (f'{HW["gpu"]} ({HW["vram_gb"]} GB)' if HW.get("gpu") else "no GPU in use (CPU)")
    warn = "".join(f"<p class='og-warn'>&#9888; {html.escape(w)}</p>" for w in rt.latency_warnings())
    return (f'<details class="og-more"><summary>System &amp; models</summary><div class="og-details">'
            f'<p><b>Hardware:</b> {hw} &middot; GPU sharing: {"on (one detector on the GPU at a time)" if SWAP_MODE else "off"}'
            f' &middot; Mode: {rt.PROFILES[rt.current_profile()]["label"]} &middot; Local LLM: {html.escape(llm)}</p>{warn}'
            f'<div class="og-scroll"><table class="og-sys"><thead><tr><th>Model</th><th>Status</th><th>Load time</th>'
            f'<th>Speed</th><th>Note</th></tr></thead><tbody>{rows}{extras}</tbody></table></div>'
            f'<p class="og-fine">Models run on this computer. Nothing is sent to a cloud service.</p></div></details>')


ASSIST_INTRO = ('<div class="og-assist-intro"><b>Guardian voice assistant.</b> Press the microphone, speak, '
                'then press stop. I answer out loud and in large text. You can also type.</div>')
ANSWER_EMPTY = ('<div class="og-empty">My answer will appear here, in large text, and I will read it aloud.<br>'
                '<small>Try: &ldquo;I think I am being scammed&rdquo;.</small></div>')
INTENT_LABEL = {"lost_money": "Safety plan", "live_call": "Right now", "analyze_text": "Message check",
                "explain_last": "Result", "repeat": "Again", "help": "Help", "howto": "How to",
                "problem": "Heads up"}


def answer_card(text, intent):
    return ('<div class="og-answer" role="status" aria-live="polite"><span class="og-chip">'
            f'<i class="og-dot"></i>{INTENT_LABEL.get(intent, "Guardian")}</span><p>{html.escape(text)}</p></div>')


def _history_html(items):
    if not items:
        return ""
    rows = "".join(f'<li><span class="og-q">{html.escape(q)}</span><span class="og-a">{html.escape(a)}</span></li>'
                   for q, a in reversed(items[-5:]))
    return f'<details class="og-more"><summary>Earlier answers</summary><ol class="og-hist">{rows}</ol></details>'


def assistant_turn(mic_path, text, attach_path, speed, simple, aloud, state):
    """One voice (or typed) turn: hear -> understand -> check the attached clip -> answer -> speak."""
    state = state or {"history": [], "analysis": None, "reply": None}
    speed = float(speed or 1.0)
    heard, lang, problem = (text or "").strip(), None, ""
    if not heard and mic_path:
        try:
            heard, lang = transcribe_ex(load_audio(mic_path))
        except AudioProblem as exc:
            problem = str(exc)
        if not heard and not problem:
            problem = ("Speech recognition is unavailable. Please type your question." if WHISPER is None
                       else "I could not make out any words. Please try again, or type your question.")
    info = None
    if attach_path:
        try:
            info = analyse_clip(attach_path)
            state["analysis"] = {"summary": info["summary"], "summary_simple": info["summary_simple"]}
        except AudioProblem as exc:
            problem = problem or str(exc)
        except Exception:  # noqa: BLE001
            problem = problem or "I could not check that recording."
    last = {**(state.get("analysis") or {}), "reply": state.get("reply")}
    if not heard and info is None:
        reply = {"text": problem or "Please speak or type a question.", "intent": "problem", "settings": {}}
    elif not heard:
        reply = {"text": info["summary_simple" if simple else "summary"], "intent": "explain_last", "settings": {}}
    else:
        facts = (state.get("analysis") or {}).get("summary", "")
        reply = gd.reply_to(heard, simple=bool(simple), last=last, llm=LLM, semantic=SEMANTIC, facts=facts)
    new_speed, new_simple = speed, bool(simple)
    change = reply.get("settings") or {}
    if "speed_delta" in change:
        new_speed = round(min(1.5, max(0.6, speed + change["speed_delta"])), 1)
    if "simple" in change:
        new_simple = bool(change["simple"])
    body = reply["text"]
    speech = speak_rate(body, new_speed) if aloud else None
    if aloud and speech is None:
        body += " (The voice is unavailable, please read this.)"
    if reply["intent"] not in ("repeat", "slower", "faster", "simpler", "normal", "empty", "problem"):
        state["reply"] = reply["text"]  # "repeat" should say the last real answer, not a settings confirmation
    state["history"] = (state.get("history", []) + [(heard or "(recording)", reply["text"])])[-8:]
    you = (f'<div class="og-you">You said: &ldquo;{html.escape(heard)}&rdquo;'
           + (f' <small>(language heard: {html.escape(str(lang))}; I answer in English for now)</small>'
              if lang and lang != "en" else "") + "</div>") if heard else ""
    return (answer_card(body, reply["intent"]), speech, crosscheck_card_html(info) if info else "", you,
            _history_html(state["history"][:-1]), state, None, "",
            gr.update(value=new_speed), gr.update(value=new_simple))


class _Wrap:
    """Open Gradio layout containers without re-indenting existing UI code (keeps the diff additive)."""

    def __init__(self):
        self.stack = []

    def open(self, cls, **kwargs):
        block = cls(**kwargs)
        block.__enter__()
        self.stack.append(block)
        return block

    def close(self, count=1):
        for _ in range(count):
            self.stack.pop().__exit__(None, None, None)


_w = _Wrap()


# --- UI -------------------------------------------------------------------------
HEADER = """
<div class="og-header">
  <svg class="og-logo" viewBox="0 0 48 48" aria-hidden="true">
    <defs><linearGradient id="ogg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#14b8a6"/></linearGradient></defs>
    <path d="M24 4 40 10v12c0 11-7 19-16 22C15 41 8 33 8 22V10z" fill="url(#ogg)"/>
    <path d="M15 24v-2m4.5 6V18m4.5 10V14m4.5 14V18m4.5 6v-2" stroke="#fff" stroke-width="2.6"
          stroke-linecap="round" fill="none"/>
  </svg>
  <div><div class="og-brand">OMNIGUARD</div>
  <div class="og-tag">Protect your voice, protect your people.</div></div>
</div>"""

FOOTER = """
<div class="og-footer">
  <p>🔒 Your audio is analysed only by the models inside this app and is not sent to any
  third-party service. Run it on your own laptop and it never leaves your device.</p>
  <p>No detector is 100% accurate. Always verify by calling back on a known number.</p>
</div>"""

CSS = """
:root{--og-bg:#f6f7fb;--og-card:#fff;--og-ink:#0f172a;--og-muted:#64748b;--og-line:#e6e8ef;
 --og-brand:#4f46e5;--og-shadow:0 1px 2px rgba(15,23,42,.04),0 10px 32px rgba(15,23,42,.07)}
.dark{--og-bg:#0b0f1a;--og-card:#131a2b;--og-ink:#e8ecf5;--og-muted:#97a3b8;--og-line:#232c42;
 --og-shadow:0 1px 2px rgba(0,0,0,.4),0 12px 40px rgba(0,0,0,.35)}
html,body,gradio-app,.gradio-container,.app{background:var(--og-bg)!important;color:var(--og-ink)}
.main.fillable{padding-left:clamp(6px,2.5vw,32px)!important;padding-right:clamp(6px,2.5vw,32px)!important}
.gradio-container{max-width:760px!important;margin:0 auto!important;padding:20px clamp(4px,2vw,16px) 40px!important;
 font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif!important}
footer{display:none!important}
.og-flush .html-container{padding:0!important}
.og-header{display:flex;align-items:center;gap:14px;padding:6px 0 10px}
.og-header,.og-footer{padding-left:0!important;padding-right:0!important}
.og-logo{width:46px;height:46px;flex:none}
.og-brand{font-size:1.45rem;font-weight:800;letter-spacing:.14em;line-height:1.1}
.og-tag{color:var(--og-muted);font-size:.98rem;margin-top:3px}
.og-samples{align-items:center;gap:8px!important;margin:0 0 8px!important;flex-wrap:nowrap!important;justify-content:flex-start!important}
.og-samples>*{flex:0 0 auto!important;width:auto!important;min-width:0!important}
.og-samples button{height:34px!important;padding:0 16px!important;font-size:.9rem!important}
.og-samples p{color:var(--og-muted);margin:0!important;font-size:.92rem}
.og-ghost{background:transparent!important;border:1px solid var(--og-line)!important;color:var(--og-ink)!important;
 border-radius:999px!important;box-shadow:none!important;min-width:0!important;font-weight:500!important}
.og-ghost:hover{border-color:var(--og-brand)!important;color:var(--og-brand)!important}
.og-cta{background:linear-gradient(135deg,#4f46e5,#7c3aed)!important;color:#fff!important;border:0!important;
 border-radius:16px!important;min-height:58px;font-size:1.08rem!important;font-weight:650!important;
 box-shadow:0 8px 24px rgba(79,70,229,.32)!important;transition:transform .12s,box-shadow .12s}
.og-cta:hover{transform:translateY(-1px);box-shadow:0 12px 30px rgba(79,70,229,.4)!important}
#play-alert{min-height:76px;font-size:1.3rem!important}
.og-result{display:flex;gap:22px;align-items:center;margin:0;background:color-mix(in srgb,var(--accent,#94a3b8) 6%,var(--og-card));border:1px solid color-mix(in srgb,var(--accent,#94a3b8) 30%,var(--og-line));
 border-radius:20px;padding:22px;box-shadow:var(--og-shadow);
 animation:og-in .35s ease-out}
.og-ring{flex:none;width:142px}
.og-ring-svg{width:100%;height:auto;display:block}
.og-ring-bg{fill:none;stroke:var(--og-line);stroke-width:9}
.og-ring-fg{fill:none;stroke-width:9;stroke-linecap:round;animation:og-draw 1s cubic-bezier(.2,.8,.2,1)}
.og-ring-num{text-anchor:middle;font-size:25px;font-weight:800;fill:var(--og-ink)}
.og-ring-sub{text-anchor:middle;font-size:8.5px;fill:var(--og-muted);letter-spacing:.04em}
.og-chip{display:inline-flex;align-items:center;gap:7px;font-size:.78rem;font-weight:700;letter-spacing:.06em;
 text-transform:uppercase;color:var(--accent,var(--og-muted));
 background:color-mix(in srgb,var(--accent,#64748b) 12%,transparent);padding:5px 11px;border-radius:999px}
.og-dot{width:8px;height:8px;border-radius:50%;background:var(--accent,var(--og-muted))}
.og-verdict h2{margin:10px 0 4px;font-size:1.5rem;line-height:1.2;font-weight:750;color:var(--og-ink)}
.og-verdict p{margin:0;color:var(--og-muted);font-size:1.02rem;line-height:1.5}
.og-notice .og-verdict p{margin-top:8px;color:var(--og-ink)}
.og-empty{border:1.5px dashed var(--og-line);border-radius:20px;padding:30px 16px;text-align:center;
 color:var(--og-muted);line-height:1.6}
.og-more{margin-top:12px;background:var(--og-card);border:1px solid var(--og-line);border-radius:16px;padding:2px 18px}
.og-more summary{cursor:pointer;padding:14px 0;font-weight:600;color:var(--og-ink);list-style:none;display:flex;justify-content:space-between}
.og-more summary::-webkit-details-marker{display:none}
.og-more summary::after{content:"+";color:var(--og-muted);font-weight:500;font-size:1.2rem;line-height:1}
.og-more[open] summary::after{content:"\\2212"}
.og-details{padding:0 0 14px;color:var(--og-ink);line-height:1.55}
.og-rows{display:grid;gap:10px;margin-bottom:12px}
.og-row{display:grid;grid-template-columns:150px 1fr 44px;gap:12px;align-items:center;font-size:.95rem}
.og-bar{height:8px;background:var(--og-line);border-radius:99px;overflow:hidden}
.og-bar i{display:block;height:100%;background:linear-gradient(90deg,#4f46e5,#14b8a6);border-radius:99px}
.og-row b{text-align:right}
.og-fine{color:var(--og-muted);font-size:.85rem}
#alert-text h2{font-size:1.45rem;line-height:1.4;color:var(--og-ink)}
.og-footer{margin-top:22px;color:var(--og-muted);font-size:.85rem;text-align:center;line-height:1.55}
.og-footer p{margin:6px 0}
.gradio-container .tab-container button,.gradio-container [role=tab]{font-weight:600!important}
@keyframes og-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@keyframes og-draw{from{stroke-dashoffset:var(--from)}}
@media(max-width:560px){
 .og-result{flex-direction:column;text-align:center;padding:20px 16px}
 .og-ring{width:150px}.og-row{grid-template-columns:104px 1fr 40px;font-size:.88rem}
 .og-brand{font-size:1.25rem}
 .tab-container button{padding:8px 6px!important;font-size:.82rem!important;white-space:nowrap}
 .og-header{gap:10px}.og-tag{font-size:.9rem}}
@media(prefers-reduced-motion:reduce){.og-ring-fg,.og-result{animation:none}}
"""

CSS = CSS + """
/* --- ADDED: desktop dashboard layer. The original 760px .gradio-container rule above is untouched. --- */
/* Gradio rewrites selectors inside @media blocks so they cannot reach ancestors of .contain; the page-width
   breakpoints are therefore done with :has() + clamp(): ~760px on phones, up to 960px on tablets, 1280px on desktop. */
.gradio-container:has(.og-shell--wide){max-width:clamp(760px,94vw,1280px)!important}
.og-shell{display:grid!important;grid-template-columns:minmax(0,1fr);gap:16px;width:100%}
.og-col-input,.og-col-result{min-width:0!important;width:100%!important;flex:none!important}
@media (min-width:1025px){
 .og-shell{grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:26px;align-items:start}
 .og-col-result{position:sticky;top:14px}
}
@media (min-width:768px) and (max-width:1024px){.og-shell{grid-template-columns:minmax(0,1fr)}}
@media (max-width:767px){.og-shell{grid-template-columns:minmax(0,1fr)}} /* phones: unchanged behaviour */

/* --- ADDED: top bar -------------------------------------------------------------------------------- */
.og-topbar{align-items:center!important;justify-content:space-between;gap:10px!important;margin-bottom:2px!important;flex-wrap:nowrap!important}
.og-topbar>*{flex:0 1 auto!important;min-width:0!important;width:auto!important}
.og-topbar-in{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.og-wordmark{font-weight:800;letter-spacing:.16em;font-size:.82rem;color:var(--og-ink)}
.og-tagline{color:var(--og-muted);font-size:.8rem}
.og-pill{display:inline-flex;align-items:center;gap:6px;font-size:.74rem;font-weight:650;padding:4px 10px;border-radius:999px;
 background:color-mix(in srgb,var(--p,#64748b) 13%,transparent);color:var(--p,#64748b)}
.og-pill .og-dot{background:var(--p,#64748b)}
.og-pill-ok{--p:#16a34a}.og-pill-warn{--p:#d97706}.og-pill-off{--p:#64748b}
.og-theme{min-width:0!important;padding:0 12px!important}

/* --- ADDED: per-model results table, cross-check card, system panel ----------------------------------- */
.og-mtable,.og-cross{margin-top:12px;background:var(--og-card);border:1px solid var(--og-line);border-radius:18px;padding:14px 16px}
.og-mt-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;color:var(--og-ink)}
.og-mt-head small,.og-cross-head small{color:var(--og-muted)}
.og-mrow{display:grid;grid-template-columns:minmax(0,1.5fr) 44px minmax(60px,1fr) 42px auto;gap:10px;align-items:center;padding:7px 0;
 border-top:1px solid var(--og-line);font-size:.9rem;color:var(--og-ink);animation:og-in-soft .4s both;animation-delay:calc(var(--i,0)*70ms)}
.og-mrow:first-of-type{border-top:0}
.og-mute{opacity:.55}
.og-mname{display:flex;flex-direction:column;gap:2px;min-width:0}
.og-w{font-style:normal;color:var(--og-muted);font-size:.78rem;margin-left:4px}
.og-info{font-size:.7rem;color:var(--og-muted);font-weight:500}
.og-spark{width:36px;height:16px;fill:var(--og-brand);opacity:.85}
.og-mbar{height:7px;background:var(--og-line);border-radius:99px;overflow:hidden}
.og-mbar i{display:block;height:100%;background:linear-gradient(90deg,#4f46e5,#14b8a6);border-radius:99px}
.og-mrow b{text-align:right}
.og-st{font-size:.68rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:3px 8px;border-radius:999px;
 background:color-mix(in srgb,var(--s,#64748b) 14%,transparent);color:var(--s,#64748b)}
.og-st-loaded{--s:#16a34a}.og-st-failed{--s:#dc2626}.og-st-skipped{--s:#64748b}
.og-cross{border-left:4px solid var(--accent,var(--og-line))}
.og-cross-head{display:flex;justify-content:space-between;align-items:center;gap:8px}
.og-story{margin:10px 0 6px;font-weight:600;color:var(--og-ink);line-height:1.45}
.og-advice{margin:8px 0 0;color:var(--og-muted);line-height:1.5}
.og-signals{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
.og-sig{font-size:.78rem;padding:4px 10px;border-radius:999px;background:var(--og-line);color:var(--og-ink)}
.og-sig-none{opacity:.7}
.og-scroll{overflow-x:auto}
.og-sys{width:100%;border-collapse:collapse;font-size:.85rem}
.og-sys th{text-align:left;color:var(--og-muted);font-weight:600;padding:6px 8px 6px 0}
.og-sys td{padding:7px 8px 7px 0;border-top:1px solid var(--og-line);vertical-align:top}
.og-warn{color:#d97706;font-weight:600}

/* --- ADDED: voice assistant ----------------------------------------------------------------------------- */
.og-assist-intro{background:color-mix(in srgb,var(--og-brand) 8%,var(--og-card));border:1px solid var(--og-line);border-radius:16px;
 padding:12px 16px;color:var(--og-ink);line-height:1.5}
.og-answer{background:var(--og-card);border:1px solid var(--og-line);border-left:5px solid var(--og-brand);border-radius:20px;
 padding:20px 22px;box-shadow:var(--og-shadow);animation:og-in-soft .35s both}
.og-answer p{margin:12px 0 0;font-size:1.35rem;line-height:1.55;color:var(--og-ink);font-weight:550}
.og-answer .og-chip{--accent:var(--og-brand)}
.og-you{margin:8px 2px;color:var(--og-muted);font-size:.95rem}
.og-hist{margin:0;padding-left:20px;display:grid;gap:10px;color:var(--og-ink)}
.og-hist .og-q{display:block;font-weight:650}.og-hist .og-a{display:block;color:var(--og-muted)}
.og-chips{gap:8px!important;flex-wrap:wrap!important}
.og-comfort{padding:10px 14px!important;border:1px solid var(--og-line)!important;border-radius:16px!important}
body.og-large .gradio-container{font-size:1.2rem}
body.og-large .og-answer p{font-size:1.85rem}
body.og-large .og-verdict h2{font-size:2rem}
body.og-large .og-cta{min-height:76px;font-size:1.3rem!important}
body.og-contrast{--og-bg:#000;--og-card:#000;--og-ink:#fff;--og-muted:#f1f1f1;--og-line:#fff;--og-brand:#ffd400}
body.og-contrast .og-result,body.og-contrast .og-answer,body.og-contrast .og-mtable,body.og-contrast .og-cross{border-width:2px}

/* --- ADDED: animation layers (existing keyframes and transitions are kept) ------------------------------- */
@keyframes og-in-soft{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@keyframes og-pulse-ring{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--og-brand) 45%,transparent)}100%{box-shadow:0 0 0 14px transparent}}
.og-ring--animated .og-ring-fg{animation:og-draw 1s cubic-bezier(.2,.8,.2,1) 1}
.og-pulse{animation:og-pulse-ring 1.2s ease-out infinite;border-radius:50%}
.og-cta,.og-chip,.og-tab,[role=tab]{transition:transform .28s cubic-bezier(0.34,1.56,0.64,1),box-shadow .28s cubic-bezier(0.34,1.56,0.64,1),background .2s}
.og-cta:active{transform:scale(.98)}
@media (prefers-reduced-motion: reduce){
 .og-mrow,.og-answer,.og-pulse,.og-ring--animated .og-ring-fg{animation:none!important}
 .og-cta,.og-chip,.og-tab,[role=tab]{transition:none!important}
}
"""

CSS = CSS + """
/* --- ADDED: three tabs on a phone. Tabs may wrap to two lines and share the width equally, so none is clipped
   (the hidden measuring copy Gradio uses for its overflow menu gets the same rule, so it no longer hides tabs). --- */
@media (max-width:560px){
 .tab-container button{white-space:normal!important;max-width:calc((100vw - 40px) / 3)!important;line-height:1.15!important;
  text-align:center;padding:7px 4px!important;font-size:.78rem!important}
}
.og-sys-wrap .og-ghost{align-self:center;width:auto!important;padding:0 20px!important}
.og-flush,.og-flush .prose{color:var(--og-ink)}
.og-brand,.og-assist-intro b{color:var(--og-ink)}
.og-comfort{background:var(--og-card)!important}
/* High contrast: switches on the dark palette (see the checkbox's JS) and pushes colours to black/white/yellow,
   including Gradio's own variables so inputs, labels and tabs follow. */
body.og-contrast{--body-background-fill:#000;--background-fill-primary:#000;--background-fill-secondary:#000;
 --block-background-fill:#000;--block-border-color:#fff;--border-color-primary:#fff;--input-border-color:#fff;
 --body-text-color:#fff;--body-text-color-subdued:#f2f2f2;--block-label-text-color:#fff;--block-title-text-color:#fff;
 --button-secondary-text-color:#fff;--button-secondary-background-fill:#000;--button-secondary-border-color:#fff;
 --input-background-fill:#000;--checkbox-label-background-fill:#000;--checkbox-label-text-color:#fff}
body.og-contrast .og-ghost,body.og-contrast .og-pill{border:2px solid #fff!important;color:#fff!important}
body.og-contrast [role=tab]{color:#fff!important}
body.og-contrast [role=tab].selected,body.og-contrast [role=tab][aria-selected=true]{color:#ffd400!important}
body.og-contrast .og-cta{background:#ffd400!important;color:#000!important;box-shadow:none!important}
"""

THEME = gr.themes.Base(primary_hue="indigo", neutral_hue="slate", radius_size="lg",
                       font=("ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"))

# WHY format="wav" on the inputs: Gradio converts .mp3/.m4a/.ogg uploads and mic
# recordings to .wav server-side (needs ffmpeg), because librosa 1.x can no longer
# decode .m4a (AAC) by itself.
with gr.Blocks(title="OMNIGUARD — Protect your voice, protect your people.") as demo:
    # --- ADDED: top bar (the original header card below it is unchanged) ---
    with gr.Row(elem_classes=["og-topbar"]):
        topbar = gr.HTML(topbar_html(), elem_classes="og-flush")
        theme_btn = gr.Button("🌓", size="sm", elem_classes="og-ghost og-theme", elem_id="og-theme")
    theme_btn.click(None, None, None, js="() => { document.body.classList.toggle('dark'); }")
    gr.HTML(HEADER, elem_classes="og-flush")
    def samples_row(target):
        """Same two example clips in both tabs; each row fills only its own tab
        (updating a hidden tab's input made Gradio jump back to the first tab)."""
        with gr.Row(elem_classes="og-samples"):
            gr.Markdown("Try a sample")
            real = gr.Button("Real voice", size="sm", elem_classes="og-ghost")
            fake = gr.Button("AI voice", size="sm", elem_classes="og-ghost")
        real.click(lambda: load_example("real"), None, target)
        fake.click(lambda: load_example("fake"), None, target)

    with gr.Tabs():
        with gr.Tab("🎙️ Deepfake Detector"):
            _w.open(gr.Row, elem_classes=["og-shell", "og-shell--wide"])
            _w.open(gr.Column, elem_classes=["og-col-input"])
            audio_a = gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav",
                               label="Voice clip")
            samples_row(audio_a)
            profile_a = gr.Radio([(v["label"], k) for k, v in rt.PROFILES.items()], value=rt.current_profile(),
                                 label="Speed vs accuracy", info="Fast: fewer detectors. Thorough: also checks the words.")
            run_a = gr.Button("Check this voice", variant="primary", size="lg", elem_classes="og-cta")
            _w.close()
            _w.open(gr.Column, elem_classes=["og-col-result"])
            card_a = gr.HTML(EMPTY_CARD, elem_classes="og-flush")
            table_a = gr.HTML("", elem_classes=["og-flush", "og-model-table"])
            cross_a = gr.HTML("", elem_classes="og-flush")
            _w.close(2)

        with gr.Tab("♿ Accessibility Assistant"):
            _w.open(gr.Row, elem_classes=["og-shell"])
            _w.open(gr.Column, elem_classes=["og-col-input"])
            audio_b = gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav",
                               label="Voice clip")
            samples_row(audio_b)
            run_b = gr.Button("🔊 Play Voice Alert", variant="primary", size="lg",
                              elem_id="play-alert", elem_classes="og-cta")
            _w.close()
            _w.open(gr.Column, elem_classes=["og-col-result"])
            card_b = gr.HTML(EMPTY_CARD, elem_classes="og-flush")
            cross_b = gr.HTML("", elem_classes="og-flush")
            message_b = gr.Markdown(elem_id="alert-text")
            speech_b = gr.Audio(label="Spoken safety alert", type="filepath",
                                autoplay=True, interactive=False)
            transcript_b = gr.Textbox(label="What was said (transcript)", lines=3,
                                      interactive=False)
            _w.close(2)

        # --- ADDED: the AI voice assistant for people with disabilities ---
        with gr.Tab("🗣️ Voice Assistant"):
            _w.open(gr.Row, elem_classes=["og-shell"])
            _w.open(gr.Column, elem_classes=["og-col-input"])
            gr.HTML(ASSIST_INTRO, elem_classes="og-flush")
            mic_v = gr.Audio(sources=["microphone"], type="filepath", format="wav",
                             label="Press the microphone, speak, then press stop")
            with gr.Row():
                q_text = gr.Textbox(label="Or type your question", lines=2, scale=4,
                                    placeholder="For example: I think I am being scammed")
                ask_btn = gr.Button("Ask", variant="primary", scale=1, elem_classes="og-cta")
            with gr.Row(elem_classes="og-chips"):
                chip_a = gr.Button("I think I'm being scammed", size="sm", elem_classes="og-ghost")
                chip_b = gr.Button("I already sent money", size="sm", elem_classes="og-ghost")
                chip_c = gr.Button("What can you do?", size="sm", elem_classes="og-ghost")
            attach_v = gr.Audio(sources=["upload"], type="filepath", format="wav",
                                label="Attach a recording for me to check (optional)")
            with gr.Group(elem_classes="og-comfort"):
                speed_v = gr.Slider(0.6, 1.5, value=1.0, step=0.1, label="Speaking speed")
                simple_v = gr.Checkbox(True, label="Use simple words")
                aloud_v = gr.Checkbox(True, label="Read answers aloud")
                large_v = gr.Checkbox(False, label="Large text")
                contrast_v = gr.Checkbox(False, label="High contrast")
            assist_state = gr.State({"history": [], "analysis": None, "reply": None})
            _w.close()
            _w.open(gr.Column, elem_classes=["og-col-result"])
            answer_v = gr.HTML(ANSWER_EMPTY, elem_classes="og-flush")
            speech_v = gr.Audio(label="Spoken answer", type="filepath", autoplay=True, interactive=False)
            cross_v = gr.HTML("", elem_classes="og-flush")
            heard_v = gr.HTML("", elem_classes="og-flush")
            hist_v = gr.HTML("", elem_classes="og-flush")
            _w.close(2)

    gr.HTML(FOOTER, elem_classes="og-flush")

    # --- ADDED: system & models panel (collapsible), refreshed on page load and on demand ---
    with gr.Column(elem_classes=["og-footer", "og-sys-wrap"]):
        sys_html = gr.HTML(system_html(), elem_classes="og-flush")
        refresh_btn = gr.Button("Refresh model status", size="sm", elem_classes="og-ghost")

    run_a.click(detect, audio_a, card_a)
    run_b.click(assist, audio_b, [card_b, transcript_b, message_b, speech_b])

    # --- ADDED: second listeners (they read the cached analysis, so nothing runs twice) and new wiring ---
    run_a.click(detail_a, audio_a, [table_a, cross_a])
    run_b.click(crosscheck_b, audio_b, cross_b)
    profile_a.change(set_profile, profile_a, None)
    _assist_in = [mic_v, q_text, attach_v, speed_v, simple_v, aloud_v, assist_state]
    _assist_out = [answer_v, speech_v, cross_v, heard_v, hist_v, assist_state, mic_v, q_text, speed_v, simple_v]
    # On phones the answer sits below the inputs, so bring it into view after each turn.
    _scroll = ("() => { if (window.innerWidth < 1025) { const a = document.querySelector('.og-answer'); "
               "if (a) a.scrollIntoView({behavior: 'smooth', block: 'center'}); } }")
    mic_v.stop_recording(assistant_turn, _assist_in, _assist_out).then(None, None, None, js=_scroll)
    ask_btn.click(assistant_turn, _assist_in, _assist_out).then(None, None, None, js=_scroll)
    q_text.submit(assistant_turn, _assist_in, _assist_out).then(None, None, None, js=_scroll)
    for _chip, _msg in ((chip_a, "I think I am being scammed"), (chip_b, "I already sent money"),
                        (chip_c, "What can you do?")):
        _chip.click(lambda m=_msg: m, None, q_text).then(assistant_turn, _assist_in, _assist_out).then(
            None, None, None, js=_scroll)
    large_v.change(None, large_v, None, js="(v) => document.body.classList.toggle('og-large', !!v)")
    contrast_v.change(None, contrast_v, None, js="""(v) => { const b = document.body;
        if (v) { b.dataset.ogWasDark = b.classList.contains('dark') ? '1' : '0'; b.classList.add('dark'); }
        else if (b.dataset.ogWasDark === '0') { b.classList.remove('dark'); }
        b.classList.toggle('og-contrast', !!v); }""")
    refresh_btn.click(system_html, None, sys_html)
    demo.load(lambda: (topbar_html(), system_html()), None, [topbar, sys_html])

if __name__ == "__main__":
    # Gradio 6: theme and css belong to launch(), not gr.Blocks().
    demo.queue().launch(theme=THEME, css=CSS)
