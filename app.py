"""OMNIGUARD - Protect your voice, protect your people.

Tab 1: deepfake voice detector (wav2vec2).  Tab 2: accessibility assistant
(faster-whisper transcript + rule-based safety message + Piper spoken alert).
Runs on a plain laptop CPU (`python app.py`) and on Hugging Face ZeroGPU.
"""
import html
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
SAMPLE_RATE = 16_000            # wav2vec2 and Whisper both expect 16 kHz mono
MAX_SECONDS = 15                # model works best on ~2.5-13 s; also keeps CPU fast
MIN_SECONDS = 0.5               # anything shorter is not a usable clip
SILENCE_RMS = 1e-3              # below this loudness we call the clip silent
DETECTOR_ID = "garystafford/wav2vec2-deepfake-voice-detector"
PIPER_VOICE = "en_US-lessac-medium"
EXAMPLES = {
    "real": Path(__file__).parent / "examples" / "real_sample.wav",
    "fake": Path(__file__).parent / "examples" / "fake_sample.wav",
}
# WHY: on ZeroGPU, torch reports CUDA as available outside @spaces.GPU too
# (emulation), so moving the model once at startup is the documented pattern.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Model loading (once, at startup) ---------------------------------------
# WHY: each loader returns (model, error_text) instead of raising, so one failed
# download never crashes the app - the other features keep working.
def _fake_index(config):
    """Find which output index means 'fake'; the model card says index 1."""
    for idx, name in config.id2label.items():
        if any(w in str(name).lower() for w in ("fake", "spoof", "synthetic")):
            return int(idx)
    return 1


def load_detector():
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        extractor = AutoFeatureExtractor.from_pretrained(DETECTOR_ID)
        model = AutoModelForAudioClassification.from_pretrained(DETECTOR_ID)
        model = model.to(DEVICE).eval()
        return {"extractor": extractor, "model": model, "fake": _fake_index(model.config)}, None
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


DETECTOR, DETECTOR_ERR = load_detector()
WHISPER, WHISPER_ERR = load_whisper()
VOICE, VOICE_ERR = load_piper()
for _name, _err in (("detector", DETECTOR_ERR), ("whisper", WHISPER_ERR), ("piper", VOICE_ERR)):
    if _err:  # technical detail goes to the console, not to the user's screen
        print(f"[OMNIGUARD] {_name} unavailable -> {_err}")


# --- Audio helpers ------------------------------------------------------------
class AudioProblem(Exception):
    """An error whose message is safe and friendly to show on screen."""


def load_audio(path):
    """Read any common format, resample to 16 kHz mono, reject bad clips."""
    if not path:
        raise AudioProblem("No audio yet. Upload a clip or record one first.")
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
    except Exception:  # noqa: BLE001
        raise AudioProblem("That file could not be read as audio. "
                           "Try a .wav, .mp3, .m4a or .ogg clip.") from None
    if audio.size < SAMPLE_RATE * MIN_SECONDS or float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS:
        raise AudioProblem("The clip is silent or too short to analyse. Please try another one.")
    return audio.astype(np.float32)


@spaces.GPU(duration=10)
def _score_on_device(audio):
    """Fake probability (0-1). Only this small function needs the GPU."""
    inputs = DETECTOR["extractor"](audio, sampling_rate=SAMPLE_RATE,
                                   return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(DETECTOR["model"](**inputs).logits, dim=-1)[0]
    return float(probs[DETECTOR["fake"]])


def score_fake(audio):
    if DETECTOR is None:
        raise AudioProblem("The deepfake model is not available right now. "
                           "Connect to the internet once so it can download, then restart.")
    return _score_on_device(audio)


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


# --- Rule-based verdicts (no LLM) --------------------------------------------
VERDICTS = {
    "green": {
        "color": "#1b8a3a", "icon": "🟢", "label": "Likely a REAL human voice",
        "why": "The detector found no strong sign of synthetic speech in this clip.",
        "speech": "This voice sounds like a real person. But always verify by calling back on a known number.",
    },
    "yellow": {
        "color": "#b7791f", "icon": "🟡", "label": "Uncertain - inconclusive",
        "why": "The score sits in the grey zone. Noise, phone compression or a short clip can cause this.",
        "speech": "Be careful. I cannot tell if this voice is real or AI-generated. "
                  "Do not send money or share codes. Call back on a number you know.",
    },
    "red": {
        "color": "#c0392b", "icon": "🔴", "label": "Likely AI-GENERATED (deepfake)",
        "why": "AI voices often show unnatural pauses, overly smooth rhythm, odd spectral patterns "
               "and missing breath sounds. The detector scores these patterns; it does not point "
               "to exact moments.",
        "speech": "Warning. This voice is likely AI-generated. Do not send money. Call a family member first.",
    },
}
NO_VERDICT_SPEECH = ("I could not check whether this voice is real. Treat it with caution. "
                     "Do not send money. Call back on a number you know.")
TIP = "Never transfer money based on a single phone call. Call back on a known number to verify."


def classify(p_fake):
    if p_fake < 0.30:
        return "green"
    return "yellow" if p_fake <= 0.70 else "red"


def banner(color, big, small):
    return (f'<div class="og-banner" style="background:{color}" role="status">'
            f'<div class="og-big">{html.escape(big)}</div>'
            f'<div class="og-small">{html.escape(small)}</div></div>')


def verdict_banner(key, p_fake):
    v = VERDICTS[key]
    return banner(v["color"], f'{p_fake * 100:.1f}% fake', f'{v["icon"]} {v["label"]}')


def notice_banner(message):
    return banner("#4a5568", "Cannot analyse", message)


# --- Tab 1: Deepfake Detector -------------------------------------------------
def detect(path):
    try:
        audio = load_audio(path)
        p_fake = score_fake(audio)
    except AudioProblem as problem:
        return notice_banner(str(problem)), ""
    except Exception:  # noqa: BLE001 - never show a stack trace
        return notice_banner("Something went wrong while analysing this clip."), ""
    key = classify(p_fake)
    details = (f"**Confidence the voice is AI-generated:** {p_fake * 100:.1f}%  \n"
               f"**What this means:** {VERDICTS[key]['why']}  \n\n"
               f"**Safety tip:** {TIP}  \n\n"
               f"_Only the first {MAX_SECONDS} seconds are analysed. No detector is 100% accurate._")
    return verdict_banner(key, p_fake), details


# --- Tab 2: Accessibility Assistant -------------------------------------------
def assist(path):
    """Transcript + verdict + spoken alert. Works even if some models failed."""
    try:
        audio = load_audio(path)
    except AudioProblem as problem:
        return "", notice_banner(str(problem)), str(problem), None
    transcript = transcribe(audio)
    try:
        p_fake = score_fake(audio)
        key = classify(p_fake)
        top, message = verdict_banner(key, p_fake), VERDICTS[key]["speech"]
    except Exception:  # noqa: BLE001 - graceful degradation without the detector
        top, message = notice_banner("Voice check unavailable. Be careful."), NO_VERDICT_SPEECH
    speech = speak(message)
    if speech is None:
        message += "\n\n⚠️ Voice alert unavailable - please read this message instead."
    return transcript, top, f"### {message}", speech


# --- Shared examples ------------------------------------------------------------
def load_example(kind):
    """Drop one example clip into BOTH tabs; fail politely if it is missing."""
    path = EXAMPLES[kind]
    try:
        sf.info(str(path))
    except Exception:  # noqa: BLE001
        gr.Warning("Example not available - please upload your own clip")
        return gr.update(), gr.update()
    return str(path), str(path)


# --- UI -------------------------------------------------------------------------
CSS = """
.og-banner{border-radius:16px;color:#fff;text-align:center;padding:22px 12px;margin:8px 0}
.og-big{font-size:clamp(2rem,9vw,3.4rem);font-weight:800;line-height:1.1}
.og-small{font-size:clamp(1rem,4vw,1.4rem);margin-top:6px}
#play-alert{font-size:1.4rem;min-height:72px}
.gradio-container{max-width:900px!important;margin:auto}
"""

# WHY format="wav" on the inputs: Gradio converts .mp3/.m4a/.ogg uploads and mic
# recordings to .wav server-side (needs ffmpeg), because librosa 1.x can no longer
# decode .m4a (AAC) by itself.
with gr.Blocks(title="OMNIGUARD") as demo:
    gr.Markdown("# 🛡️ OMNIGUARD — Protect your voice, protect your people.")
    gr.Markdown("🔒 **Privacy:** your audio is analysed only by the models inside this app and is "
                "not sent to any third-party service. Run it on your own laptop and it never "
                "leaves your device.")
    with gr.Row():
        gr.Markdown("**Try an example (loads into both tabs):**")
        real_btn = gr.Button("▶ Real voice sample")
        fake_btn = gr.Button("▶ AI voice sample")

    with gr.Tabs():
        with gr.Tab("🎙️ Deepfake Detector"):
            audio_a = gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav",
                               label="Upload a clip or record from your microphone")
            run_a = gr.Button("Check this voice", variant="primary", size="lg")
            banner_a = gr.HTML()
            details_a = gr.Markdown()

        with gr.Tab("♿ Accessibility Assistant"):
            audio_b = gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav",
                               label="Upload a clip or record from your microphone")
            run_b = gr.Button("🔊 Play Voice Alert", variant="primary", size="lg",
                              elem_id="play-alert")
            banner_b = gr.HTML()
            transcript_b = gr.Textbox(label="What was said (transcript)", lines=3,
                                      interactive=False)
            message_b = gr.Markdown()
            speech_b = gr.Audio(label="Spoken safety alert", type="filepath",
                                autoplay=True, interactive=False)

    gr.Markdown("_No detector is 100% accurate. Always verify by calling back on a known number._")

    run_a.click(detect, audio_a, [banner_a, details_a])
    run_b.click(assist, audio_b, [transcript_b, banner_b, message_b, speech_b])
    real_btn.click(lambda: load_example("real"), None, [audio_a, audio_b])
    fake_btn.click(lambda: load_example("fake"), None, [audio_a, audio_b])

if __name__ == "__main__":
    # Gradio 6: theme and css belong to launch(), not gr.Blocks().
    demo.queue().launch(theme=gr.themes.Soft(), css=CSS)
