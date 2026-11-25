"""OMNIGUARD engine: detectors, scoring, calibration, speech models and the analysis pipeline.

No UI code lives here (see app.py). Everything is local: models run on this computer, the GPU is
shared one detector at a time so a 4 GB laptop GPU is enough, and every model can fail without taking
the others down.
"""
import json
import math
import os
import tempfile
import wave
from pathlib import Path

import librosa
import numpy as np
import torch

# --- Local modules: runtime helpers, scam-wording + assistant brain, channel log ------------------
import hashlib
import threading
import time

import guardian as gd
import omni_runtime as rt
from channel_log import LOG

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
MAX_SECONDS = 30                # never read more than this from a file (6 windows x 4 s fits)
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
# --- Extra detectors ------------------------------------------------------------------------------
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
# Choose detectors without editing code: OMNIGUARD_DETECTORS=arena,gary,melody
# Default: the three solid HF models. Optionals (pellav2, anti_small, etc.) only
# when the user names them or when no OMNIGUARD_DETECTORS is set and they are
# available at runtime (pellav2 needs the optional requirements).
_default = "arena,gary,melody"
_base = [k.strip() for k in os.environ.get("OMNIGUARD_DETECTORS", _default).split(",")
         if k.strip() in CATALOG]
# Only auto-append pellav2 when the user did not pin a list; anti_small is never
# auto-loaded (fairseq, unavailable under Transformers).
_new = [] if "OMNIGUARD_DETECTORS" in os.environ else [
    k for k in ("pellav2",) if k in CATALOG and k not in _base]
ACTIVE = _base + _new or ["arena", "gary", "melody"]
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


def _load_detector_hf(key):
    """Plain Transformers loader: arena = custom model code, the others = audio-classification."""
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


def _load_piper_voice():
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


# --- Hardware plan ------------------------------------------------------------------------------------
# GTX 1650-class laptops have ~4 GB of VRAM: detectors wait in RAM and borrow the GPU one at a time,
# and a model that would not fit the budget (OMNIGUARD_VRAM_GB, default 3.6) stays on the CPU.
SWAP_MODE = rt.want_swap(torch, DEVICE, _ZEROGPU, os.environ.get("OMNIGUARD_SWAP", "auto").lower())
HW = rt.hardware_summary(torch, DEVICE, SWAP_MODE)
RESIDENCY = rt.Residency(torch, SWAP_MODE, budget_gb=float(os.environ.get("OMNIGUARD_VRAM_GB", "3.6")))
_GPU_LOCK = threading.RLock()


def load_detector(key):
    """Load one detector. Skips unavailable/optional ones with a reason, times every load, and (on a
    small GPU) parks the model in RAM so detectors can take turns on the GPU."""
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
        det, err = _load_detector_hf(key)
    rt.record_load(key, time.perf_counter() - t0)
    if det is None:
        rt.set_status(key, "failed", err or "")
    else:
        if SWAP_MODE:
            RESIDENCY.release(det["model"])
        rt.set_status(key, "loaded", "")
    return det, err


def load_piper():
    t0 = time.perf_counter()
    voice, err = _load_piper_voice()
    rt.record_load("piper", time.perf_counter() - t0)
    rt.set_status("piper", "loaded" if voice else "failed", err or "")
    return voice, err


# --- Speech-to-text: one Whisper model stays in RAM; the mode decides which size ----------------------
WHISPER, WHISPER_SIZE, WHISPER_ERR = None, None, None
_WHISPER_LOCK = threading.RLock()
_WHISPER_TRIED = {}


def whisper_size_for(profile=None):
    """Whisper size for a mode (OMNIGUARD_WHISPER pins one size for every mode)."""
    pinned = os.environ.get("OMNIGUARD_WHISPER", "").strip()
    return pinned or rt.PROFILES[profile or rt.current_profile()]["whisper"]


def load_whisper(size=None):
    """faster-whisper on the CPU (int8). OMNIGUARD_WHISPER_DEVICE=cuda is opt-in (needs CUDA 12 cuBLAS
    and cuDNN 9) and falls back to the CPU if a test run fails. Returns (model, error)."""
    size = size or whisper_size_for()
    t0 = time.perf_counter()
    model, err = None, None
    try:
        from faster_whisper import WhisperModel
        if os.environ.get("OMNIGUARD_WHISPER_DEVICE", "cpu").lower() == "cuda":
            try:
                model = WhisperModel(size, device="cuda", compute_type="int8_float16")
                list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))[0])  # surfaces missing CUDA libs
            except Exception as exc:  # noqa: BLE001
                print(f"[OMNIGUARD] Whisper on CUDA failed ({type(exc).__name__}); using the CPU")
                model = None
        if model is None:
            model = WhisperModel(size, device="cpu", compute_type="int8")
    except Exception as exc:  # noqa: BLE001
        model, err = None, f"{type(exc).__name__}: {exc}"
    rt.record_load("whisper", time.perf_counter() - t0)
    return model, err


def ensure_whisper():
    """Switch to the Whisper size the current mode wants. If that fails (offline, never downloaded) the
    model already in memory keeps working. A failed size is not retried for a minute."""
    global WHISPER, WHISPER_SIZE, WHISPER_ERR
    want = whisper_size_for()
    if WHISPER is not None and want == WHISPER_SIZE:
        return WHISPER
    with _WHISPER_LOCK:
        if time.time() - _WHISPER_TRIED.get(want, 0) < 60:
            return WHISPER
        _WHISPER_TRIED[want] = time.time()
        model, err = load_whisper(want)
        if model is not None:
            WHISPER, WHISPER_SIZE, WHISPER_ERR = model, want, None
            rt.set_status("whisper", "loaded", f"size {want}")
            LOG.add("system", f"Speech model: {want}", f"loaded in {rt.LOAD_SECONDS.get('whisper', 0):.1f} s", "ok")
        else:
            WHISPER_ERR = err
            rt.set_status("whisper", "loaded" if WHISPER else "failed", err or "")
            LOG.add("system", f"Could not load speech model {want}",
                    (f"keeping {WHISPER_SIZE}. " if WHISPER else "") + (err or ""), "warn")
        return WHISPER


DETECTORS = []
for _key in ACTIVE:
    _det, _err = load_detector(_key)
    if _det:
        DETECTORS.append(_det)
    else:  # technical detail goes to the console, not to the user's screen
        print(f"[OMNIGUARD] detector '{_key}' unavailable -> {_err}")
ensure_whisper()
VOICE, VOICE_ERR = load_piper()
for _name, _err in (("whisper", WHISPER_ERR), ("piper", VOICE_ERR)):
    if _err:
        print(f"[OMNIGUARD] {_name} unavailable -> {_err}")
if not CALIBRATION:
    print("[OMNIGUARD] no calibration.json: raw model scores in use "
          "(run `python calibrate.py` with your own clips to tune them)")


# --- Status for everything not attempted, the "words" models, and a startup summary ------------------------
for _k, _spec in CATALOG.items():
    if _k not in rt.MODEL_STATUS:
        rt.set_status(_k, "skipped", _spec.get("unavailable") or (
            "optional: add it to OMNIGUARD_DETECTORS to load it" if _spec.get("optional") else "not in the active list"))
SEMANTIC = gd.Semantic()  # zero-shot scam-wording model: loaded on first use (Balanced/Thorough), see semantic_for_mode()
rt.set_status("semantic", "skipped", "loads on first use in Balanced and Thorough mode (OMNIGUARD_PRELOAD=1 loads it now)")
LLM = gd.Ollama()
print(f"[OMNIGUARD] device={DEVICE} swap={SWAP_MODE} detectors={[d['key'] for d in DETECTORS]} "
      f"whisper={WHISPER_SIZE} llm={'ready' if LLM.available() else 'off'} mode={rt.current_profile()}")
LOG.add("system", "OMNIGUARD started",
        f"{len(DETECTORS)} detector(s) ready; device {HW.get('gpu') or 'CPU'}; GPU sharing {'on' if SWAP_MODE else 'off'}; "
        f"mode {rt.PROFILES[rt.current_profile()]['label']}", "ok")
for _k in ACTIVE:
    _st = rt.MODEL_STATUS.get(_k, "skipped")
    LOG.add("system", f"Detector {CATALOG[_k]['name']}: {_st}",
            f"{rt.LOAD_SECONDS[_k]:.1f} s to load" if _k in rt.LOAD_SECONDS and _st == "loaded" else rt.STATUS_NOTE.get(_k, ""),
            "ok" if _st == "loaded" else ("error" if _st == "failed" else "info"))


def semantic_for_mode():
    """The scam-wording model, only in modes that use it. Loads on first use so startup stays quick;
    a failed load is remembered (SEMANTIC.error) and not retried."""
    if not rt.PROFILES[rt.current_profile()]["semantic"]:
        return None
    if SEMANTIC.pipe is None and SEMANTIC.error is None:
        t0 = time.perf_counter()
        ok = SEMANTIC.load()
        rt.record_load("semantic", time.perf_counter() - t0)
        rt.set_status("semantic", "loaded" if ok else "skipped", "" if ok else (SEMANTIC.error or ""))
        LOG.add("system", "Scam-wording model " + ("loaded" if ok else "unavailable"),
                f"{rt.LOAD_SECONDS['semantic']:.1f} s" if ok else (SEMANTIC.error or ""), "ok" if ok else "warn")
    return SEMANTIC if SEMANTIC.pipe is not None else None


def llm_for_mode():
    """(client, token budget) for the assistant: off in Fast mode or when Ollama is not running."""
    prof = rt.PROFILES[rt.current_profile()]
    return (LLM, prof["llm_tokens"]) if (prof["llm"] and LLM.available()) else (None, 0)


if os.environ.get("OMNIGUARD_PRELOAD", "0") == "1":
    semantic_for_mode()

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
    """Up to as many windows as the mode allows (2 / 4 / 6), spread evenly across the clip."""
    if audio.size <= WINDOW:
        return [audio]
    count = min(rt.PROFILES[rt.current_profile()]["max_windows"], math.ceil(audio.size / WINDOW))
    starts = np.linspace(0, audio.size - WINDOW, num=count).astype(int)
    return [audio[s:s + WINDOW] for s in starts]


def _fit(chunk):
    """Crop or tile to exactly WINDOW samples (what DF Arena's own extractor does)."""
    if chunk.size >= WINDOW:
        return chunk[:WINDOW]
    return np.tile(chunk, math.ceil(WINDOW / chunk.size))[:WINDOW]


def _score_window(det, chunk):
    """Probability (0-1) that one window is fake (model already on DEVICE). Used by the ZeroGPU path."""
    return _score_window_on(det, chunk, DEVICE)


@spaces.GPU(duration=15)
def _score_all_zerogpu(windows):
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


def _transcribe_whisper(audio):
    if ensure_whisper() is None:
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


# --- Orchestration: scoring, caching, assistant helpers ---------------------------------------------------
_WORK = threading.RLock()  # one analysis at a time: protects VRAM, lets a second UI listener reuse the result
_TL = threading.local()    # per-window scores and timings of the latest scoring call
_TL.timings = {}
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
                LOG.add("system", f"{CATALOG[det['key']]['name']}: GPU memory full, retried on the CPU", "", "warn")
                vals = [_score_window_on(det, w, "cpu") for w in windows]
        per_window = (time.perf_counter() - t0) / max(len(windows), 1)
        rt.record_latency(det["key"], per_window, device)
        timings = getattr(_TL, "timings", None)
        if timings is not None:
            timings[det["key"]] = (per_window, device)
        return vals
    except Exception as exc:  # noqa: BLE001 - one bad model must not sink the rest
        print(f"[OMNIGUARD] {det['key']} failed on this clip -> {type(exc).__name__}: {exc}")
        LOG.add("system", f"{CATALOG[det['key']]['name']} failed on this clip", f"{type(exc).__name__}: {exc}", "error")
        return None


def _score_all(windows):
    """Local scorer: mode-aware, one detector on the GPU at a time, timed, logged. ZeroGPU keeps its own."""
    if _ZEROGPU:
        return _score_all_zerogpu(windows)
    profile = rt.current_profile()
    chosen = rt.select_detectors(DETECTORS, profile, SPEED_RANK)
    windows = windows[: rt.window_limit(profile, len(windows))]
    out, _TL.timings = {}, {}
    with _GPU_LOCK, torch.no_grad():
        for i, det in enumerate(chosen):
            rt.report(0.15 + 0.7 * i / max(len(chosen), 1),
                      f"Checking the voice with {CATALOG[det['key']]['name']} ({i + 1} of {len(chosen)})")
            out[det["key"]] = _run_detector(det, windows)
    _TL.last_windows = out
    if _TL.timings:
        LOG.add("system", f"Scored {len(windows)} window(s) with {len(_TL.timings)} detector(s) ({rt.PROFILES[profile]['label']} mode)",
                "; ".join(f"{CATALOG[k]['name']} {t:.1f} s/window on {d}" for k, (t, d) in _TL.timings.items())
                + ("; GPU shared one at a time" if SWAP_MODE else ""))
    return out


def analyse(audio):
    """Weighted, cached, mode-aware ensemble. Safety first: disagreement is never green."""
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
        result = {"p": p, "key": verdict, "models": models, "spread": spread, "profile": rt.current_profile()}
        votes = ", ".join(f"{m['name']} {m['p'] * 100:.0f}%" for m in models)
        label = {"green": "likely real", "yellow": "uncertain", "red": "likely AI-generated"}[verdict]
        LOG.add("detection", f"{p * 100:.0f}% AI-generated: {label}",
                f"votes: {votes}" + ("; the detectors disagree" if spread > 0.5 else ""),
                {"green": "ok", "yellow": "warn", "red": "alert"}[verdict])
        if verdict == "red":
            LOG.add("alert", "Likely AI-generated voice", "Do not send money. Call back on a number you know.", "alert")
        return _remember(_CACHE["analyse"], key, result)


def transcribe(audio):
    """Whisper transcript, cached per clip so a second UI listener never repeats the work."""
    key = _digest(audio)
    with _WORK:
        hit = _CACHE["words"].get(key)
        if hit is None:
            hit = _transcribe_whisper(audio)
            if not hit.startswith("("):
                _remember(_CACHE["words"], key, hit)
        return hit


def transcribe_ex(audio):
    """(text, language, language probability) for the voice assistant; ("", None, 0) if Whisper is unavailable."""
    model = ensure_whisper()
    if model is None:
        return "", None, 0.0
    try:
        segments, info = model.transcribe(audio, beam_size=1, vad_filter=True,
                                          vad_parameters={"min_silence_duration_ms": 2500})
        text = " ".join(s.text.strip() for s in list(segments)).strip()
        return text, getattr(info, "language", None), float(getattr(info, "language_probability", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        return "", None, 0.0


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
    words = gd.assess_text(transcript, semantic_for_mode()) if heard else {"score": 0.0, "signals": [], "semantic": None}
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


