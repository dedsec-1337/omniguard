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
# Choose detectors without editing code:  OMNIGUARD_DETECTORS=arena,gary,melody
ACTIVE = [k.strip() for k in os.environ.get("OMNIGUARD_DETECTORS", "arena,gary").split(",")
          if k.strip() in CATALOG] or ["arena", "gary"]


def read_calibration():
    """Per-model scaling {a, b} written by calibrate.py (optional)."""
    try:
        data = json.loads((HERE / "calibration.json").read_text())
        return {k: {"a": float(v["a"]), "b": float(v["b"])} for k, v in data.items() if k in CATALOG}
    except Exception:  # noqa: BLE001 - missing/bad file just means "use defaults"
        return {}


CALIBRATION = read_calibration()


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

THEME = gr.themes.Base(primary_hue="indigo", neutral_hue="slate", radius_size="lg",
                       font=("ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"))

# WHY format="wav" on the inputs: Gradio converts .mp3/.m4a/.ogg uploads and mic
# recordings to .wav server-side (needs ffmpeg), because librosa 1.x can no longer
# decode .m4a (AAC) by itself.
with gr.Blocks(title="OMNIGUARD — Protect your voice, protect your people.") as demo:
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
            audio_a = gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav",
                               label="Voice clip")
            samples_row(audio_a)
            run_a = gr.Button("Check this voice", variant="primary", size="lg", elem_classes="og-cta")
            card_a = gr.HTML(EMPTY_CARD, elem_classes="og-flush")

        with gr.Tab("♿ Accessibility Assistant"):
            audio_b = gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav",
                               label="Voice clip")
            samples_row(audio_b)
            run_b = gr.Button("🔊 Play Voice Alert", variant="primary", size="lg",
                              elem_id="play-alert", elem_classes="og-cta")
            card_b = gr.HTML(EMPTY_CARD, elem_classes="og-flush")
            message_b = gr.Markdown(elem_id="alert-text")
            speech_b = gr.Audio(label="Spoken safety alert", type="filepath",
                                autoplay=True, interactive=False)
            transcript_b = gr.Textbox(label="What was said (transcript)", lines=3,
                                      interactive=False)

    gr.HTML(FOOTER, elem_classes="og-flush")

    run_a.click(detect, audio_a, card_a)
    run_b.click(assist, audio_b, [card_b, transcript_b, message_b, speech_b])

if __name__ == "__main__":
    # Gradio 6: theme and css belong to launch(), not gr.Blocks().
    demo.queue().launch(theme=THEME, css=CSS)
