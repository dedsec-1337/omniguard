"""OMNIGUARD runtime layer (additive): model status, timing, accuracy profiles,
GPU "residency" for small-VRAM laptops, and the optional Pellav2 detector.

Nothing here replaces app.py behaviour; app.py calls into it and keeps its original
code paths as the fallback.
"""
import contextlib
import contextvars
import threading
import time
from collections import deque

import numpy as np

# --- Model status / timing report (rendered in the "System & models" panel) --------------
MODEL_STATUS = {}      # key -> "loaded" | "failed" | "skipped"
STATUS_NOTE = {}       # key -> short human reason
LOAD_SECONDS = {}      # key -> seconds the loader took
_LATENCY = {}          # key -> deque of (seconds per window, device)
_LOCK = threading.Lock()
SLOW_SECONDS_PER_WINDOW = 2.0


def set_status(key, status, note=""):
    with _LOCK:
        MODEL_STATUS[key] = status
        STATUS_NOTE[key] = note


def record_load(key, seconds):
    with _LOCK:
        LOAD_SECONDS[key] = seconds


def record_latency(key, seconds, device):
    with _LOCK:
        _LATENCY.setdefault(key, deque(maxlen=20)).append((float(seconds), device))


def avg_latency(key):
    """Rolling average seconds per window, or None if the model has not run yet."""
    with _LOCK:
        data = list(_LATENCY.get(key, ()))
    return float(np.mean([s for s, _ in data])) if data else None


def last_device(key):
    with _LOCK:
        data = _LATENCY.get(key)
        return data[-1][1] if data else None


def latency_warnings(limit=SLOW_SECONDS_PER_WINDOW):
    """Models that average more than `limit` s/window while running on the CPU."""
    out = []
    for key in list(_LATENCY):
        avg = avg_latency(key)
        if avg is not None and avg > limit and last_device(key) == "cpu":
            out.append(f"{key} averages {avg:.1f} s per window on CPU (limit {limit:.0f} s)")
    return out


# --- Accuracy profiles ("model switching" without reloading anything) ---------------------
PROFILES = {
    "fast": {"label": "Fast", "max_models": 2, "max_windows": 2, "crosscheck": False,
             "help": "Two quickest detectors, 2 windows. Best on a slow CPU."},
    "balanced": {"label": "Balanced", "max_models": None, "max_windows": 4, "crosscheck": False,
                 "help": "Every loaded detector, up to 4 windows (the original behaviour)."},
    "thorough": {"label": "Thorough", "max_models": None, "max_windows": 4, "crosscheck": True,
                 "help": "Balanced, plus a check of what was said (scam wording)."},
}
_profile = contextvars.ContextVar("og_profile", default=None)
_global = {"name": "balanced"}


def set_global_profile(name):
    """Process-wide default (set by the UI radio). Fine on a personal laptop."""
    if name in PROFILES:
        _global["name"] = name
    return _global["name"]


def current_profile():
    return _profile.get() or _global["name"]


@contextlib.contextmanager
def profile_scope(name):
    token = _profile.set(name if name in PROFILES else "balanced")
    try:
        yield
    finally:
        _profile.reset(token)


def window_limit(profile, available):
    return max(1, min(available, PROFILES.get(profile, PROFILES["balanced"])["max_windows"]))


def select_detectors(detectors, profile, speed_rank):
    """Subset of loaded detectors for this request. Fast = the quickest ones."""
    cap = PROFILES.get(profile, PROFILES["balanced"])["max_models"]
    if not cap or len(detectors) <= cap:
        return list(detectors)

    def cost(det):
        lat = avg_latency(det["key"])
        if lat is not None:
            return (0, lat)
        rank = speed_rank.index(det["key"]) if det["key"] in speed_rank else len(speed_rank)
        return (1, rank)
    return sorted(detectors, key=cost)[:cap]


# --- GPU residency: keep models in RAM, borrow the GPU one at a time -------------------------
class Residency:
    """On a 4 GB laptop GPU, three or four detectors cannot all sit in VRAM. With 32 GB of
    RAM they can all wait on the CPU and take turns on the GPU (peak VRAM = largest model).
    Any CUDA problem falls back to the CPU for that model instead of failing."""

    def __init__(self, torch, enabled):
        self.torch = torch
        self.enabled = bool(enabled)

    def _empty_cache(self):
        try:
            self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cache clearing is best effort
            pass

    def release(self, model):
        with contextlib.suppress(Exception):
            model.to("cpu")
        self._empty_cache()

    @staticmethod
    def device_of(model):
        try:
            return next(model.parameters()).device.type
        except Exception:  # noqa: BLE001
            return "cpu"

    @contextlib.contextmanager
    def borrow(self, model):
        if not self.enabled:
            yield self.device_of(model)
            return
        try:
            model.to("cuda")
            device = "cuda"
        except Exception:  # noqa: BLE001 - out of memory, driver trouble, anything
            self.release(model)
            device = "cpu"
        try:
            yield device
        finally:
            self.release(model)


def hardware_summary(torch, device, swap):
    info = {"device": device, "swap": swap, "gpu": None, "vram_gb": None}
    if device == "cuda":
        try:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.mem_get_info()[1] / 1e9, 1)
        except Exception:  # noqa: BLE001
            pass
    return info


def want_swap(torch, device, zerogpu, setting="auto"):
    """Use GPU residency on small GPUs (<= 8 GB). Never on ZeroGPU. 'on'/'off' override."""
    if zerogpu or device != "cuda" or setting == "off":
        return False
    if setting == "on":
        return True
    try:
        return torch.cuda.mem_get_info()[1] / 1e9 <= 8.0
    except Exception:  # noqa: BLE001
        return False


# --- Pellav2 (optional detector: w2v-BERT 2.0 backbone, custom .pt checkpoint) ------------------
# Model code adapted from the author's pellav2_infer.py (MIT licence, Sadanie/pellav2-audio-deepfake-detector).
PELLAV_SR = 16000
PELLAV_CROP = 4 * PELLAV_SR
PELLAV_LUFS = -23.0


def pellav_loudness(x):
    try:
        import pyloudnorm as pyln
        if len(x) >= int(0.4 * PELLAV_SR):
            value = pyln.Meter(PELLAV_SR).integrated_loudness(x.astype(np.float64))
            if np.isfinite(value):
                return float(value)
    except Exception:  # noqa: BLE001
        pass
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) - 0.691)


def pellav_preprocess(chunk):
    """Same pre-processing as the author's script: DC removal, 20 Hz high-pass, centre 4 s
    crop, loudness normalisation to -23 LUFS, zero padding."""
    from scipy.signal import butter, sosfiltfilt
    wav = np.asarray(chunk, dtype=np.float32)
    wav = wav - wav.mean()
    if len(wav) > 3 * PELLAV_SR // 100:
        wav = sosfiltfilt(butter(2, 20.0, btype="highpass", fs=PELLAV_SR, output="sos"), wav)
    wav = wav.astype(np.float32)
    if len(wav) > PELLAV_CROP:
        off = (len(wav) - PELLAV_CROP) // 2
        wav = wav[off: off + PELLAV_CROP]
    if len(wav):
        gain = float(np.clip(PELLAV_LUFS - pellav_loudness(wav), -40.0, 40.0))
        wav = np.clip(wav * 10 ** (gain / 20), -1.0, 1.0).astype(np.float32)
    if len(wav) < PELLAV_CROP:
        wav = np.concatenate([wav, np.zeros(PELLAV_CROP - len(wav), np.float32)])
    return wav


def build_pellav_class(torch):
    """Build the Pellav2 module class (needs real torch, so it is created lazily)."""
    nn = torch.nn

    class PellavDetector(nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
            # Config only: the checkpoint already contains the backbone weights, so there is
            # no need to download facebook/w2v-bert-2.0's weights a second time.
            self.backbone = Wav2Vec2BertModel(Wav2Vec2BertConfig.from_pretrained("facebook/w2v-bert-2.0"))
            self.layer_weights = nn.Parameter(torch.zeros(self.backbone.config.num_hidden_layers + 1))
            self.head = nn.Linear(self.backbone.config.hidden_size, 1)

        @staticmethod
        def features(wav):
            import torchaudio.compliance.kaldi as kaldi
            feats, lens = [], []
            with torch.autocast(device_type=wav.device.type, enabled=False):
                for x in wav.float():
                    nz = torch.nonzero(x).flatten()
                    n_s = int(nz[-1]) + 1 if len(nz) else len(x)
                    f = kaldi.fbank(x[None, :max(n_s, 400)] * 32768.0, num_mel_bins=80,
                                    sample_frequency=PELLAV_SR, dither=0.0, window_type="povey")
                    f = (f - f.mean(0, keepdim=True)) / torch.sqrt(f.var(0, keepdim=True) + 1e-7)
                    n = f.shape[0] // 2 * 2
                    feats.append(f[:n].reshape(n // 2, 160))
                    lens.append(n // 2)
            width = max(lens)
            out = torch.zeros(len(feats), width, 160, device=wav.device)
            mask = torch.zeros(len(feats), width, dtype=torch.long, device=wav.device)
            for i, (f, n) in enumerate(zip(feats, lens)):
                out[i, :n] = f
                mask[i, :n] = 1
            return out, mask

        def forward(self, wav):
            feats, mask = self.features(wav)
            hidden = self.backbone(input_features=feats, attention_mask=mask,
                                   output_hidden_states=True).hidden_states
            w = torch.softmax(self.layer_weights, dim=0)
            h = torch.stack(hidden).mul(w[:, None, None, None]).sum(0)
            m = mask[:, :, None].float()
            h = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.head(h).squeeze(-1)

    return PellavDetector


def load_pellav(torch, repo_id):
    """Download (once) and load Pellav2 on the CPU. Safe load: weights_only=True."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, "pellav2_detector.pt")
    model = build_pellav_class(torch)()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def missing_modules(names):
    """Names from `names` that cannot be imported (used to skip optional detectors cleanly)."""
    import importlib.util
    return [n for n in names if importlib.util.find_spec(n) is None]


def timed(fn):
    """Decorator: returns fn's result and records how long it took under the key it was given."""
    def wrapper(key, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(key, *args, **kwargs)
        finally:
            record_load(key, time.perf_counter() - t0)
    return wrapper
