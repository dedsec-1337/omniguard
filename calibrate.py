"""Tune OMNIGUARD on YOUR clips (the real fix when AI voices are called human).

Why: every detector is trained on specific AI voice engines. A new engine can
score "real" on one model while another model still catches it, and score ranges
shift between recording conditions. This script measures each detector on audio
you know the answer to and fits a per-model scaling (Platt scaling) so that the
app's percentages become honest probabilities for YOUR kind of audio.

Usage:
    calib/real/   <- 20+ recordings of real people (phone calls, voice notes...)
    calib/fake/   <- 20+ clips from the AI voice tools you want to catch
    python calibrate.py            # prints the table and writes calibration.json
    python calibrate.py --dry-run  # prints only
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import app  # loads exactly the same detectors the web app uses

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".webm"}


def list_clips(folder):
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in AUDIO_EXT)


def load_clip(path):
    """Use the app's own loader; fall back to ffmpeg for formats librosa cannot read."""
    try:
        return app.load_audio(str(path))
    except app.AudioProblem:
        if not shutil.which("ffmpeg"):
            raise
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "clip.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path),
                            "-ar", str(app.SAMPLE_RATE), "-ac", "1", str(wav)], check=True)
            return app.load_audio(str(wav))


def auc(fake, real):
    """Chance a random fake clip scores higher than a random real one (1.0 = perfect)."""
    wins = sum((f > r) + 0.5 * (f == r) for f in fake for r in real)
    return wins / (len(fake) * len(real))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_scaling(fake, real):
    """Fit p = sigmoid(a * logit(raw) + b). Classes are weighted equally; a small ridge keeps
    `a` near 1 (the identity), so tiny or perfectly separable data cannot explode."""
    x = _logit(list(fake) + list(real))
    y = np.array([1.0] * len(fake) + [0.0] * len(real))
    w = np.where(y == 1, 0.5 / len(fake), 0.5 / len(real)) * len(y)
    a, b, ridge = 1.0, 0.0, 0.05
    for _ in range(200):
        p = 1 / (1 + np.exp(-(a * x + b)))
        grad = np.array([np.sum(w * (p - y) * x) + ridge * (a - 1), np.sum(w * (p - y))])
        s = w * p * (1 - p)
        hess = np.array([[np.sum(s * x * x) + ridge, np.sum(s * x)], [np.sum(s * x), np.sum(s)]])
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        a, b = float(np.clip(a - step[0], 0.05, 12)), float(np.clip(b - step[1], -15, 15))
        if np.abs(step).max() < 1e-7:
            break
    return a, b


def apply_scaling(raw, a, b):
    return 1 / (1 + np.exp(-(a * _logit(raw) + b)))


def score_folder(folder, label):
    rows = []
    for path in list_clips(folder):
        try:
            rows.append((path.name, label, app.raw_scores(load_clip(path))))
        except Exception as exc:  # noqa: BLE001 - skip unreadable clips, keep going
            print(f"  skipped {path.name}: {exc}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default="calib/real")
    ap.add_argument("--fake", default="calib/fake")
    ap.add_argument("--dry-run", action="store_true", help="print results, do not write calibration.json")
    args = ap.parse_args(argv)

    rows = score_folder(args.real, "real") + score_folder(args.fake, "fake")
    n_real = sum(r[1] == "real" for r in rows)
    n_fake = sum(r[1] == "fake" for r in rows)
    if not n_real or not n_fake:
        raise SystemExit(f"Need clips in both folders (found {n_real} real, {n_fake} fake).")
    if min(n_real, n_fake) < 10:
        print(f"\n! Only {n_real} real / {n_fake} fake clips: calibration will be rough. Use 20+ of each.")

    keys = [d["key"] for d in app.DETECTORS]
    print(f"\n{'clip':34s} {'truth':5s} " + " ".join(f"{k:>7s}" for k in keys))
    for name, label, raw in rows:
        print(f"{name[:34]:34s} {label:5s} " + " ".join(f"{raw.get(k, float('nan')):7.2f}" for k in keys))

    thresholds = {}
    print("\nPer-detector result (AUC 1.00 = perfect separation; 0.50 = coin flip):")
    for k in keys:
        fake = [raw[k] for _, lab, raw in rows if lab == "fake" and k in raw]
        real = [raw[k] for _, lab, raw in rows if lab == "real" and k in raw]
        if not fake or not real:
            continue
        a, b = fit_scaling(fake, real)
        thresholds[k] = {"a": round(a, 4), "b": round(b, 4)}
        cut = float(1 / (1 + np.exp(b / a)))  # raw score that maps to 50%
        before = (np.mean([f >= 0.5 for f in fake]), np.mean([r >= 0.5 for r in real]))
        after = (np.mean([apply_scaling(f, a, b) >= 0.5 for f in fake]),
                 np.mean([apply_scaling(r, a, b) >= 0.5 for r in real]))
        print(f"  {k:6s} AUC {auc(fake, real):.2f} | raw score {cut:.2f} now means 50% | "
              f"AI caught {before[0]:.0%} -> {after[0]:.0%}, real wrongly flagged {before[1]:.0%} -> {after[1]:.0%}")

    app.CALIBRATION.update(thresholds)
    verdicts = {"real": {"green": 0, "yellow": 0, "red": 0}, "fake": {"green": 0, "yellow": 0, "red": 0}}
    for _, label, raw in rows:
        ps = [app.calibrated(r, k) for k, r in raw.items()]  # uses the freshly fitted scaling
        key = app.classify(float(np.mean(ps)))
        if key == "green" and max(ps) - min(ps) > 0.5:
            key = "yellow"
        verdicts[label][key] += 1
    print("\nWith this calibration the combined verdict is:")
    for label in ("real", "fake"):
        v = verdicts[label]
        print(f"  {label} clips -> green {v['green']}, yellow {v['yellow']}, red {v['red']}")

    if args.dry_run:
        print("\nDry run: calibration.json not written.")
        return thresholds
    Path(app.HERE / "calibration.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    print(f"\nSaved {app.HERE / 'calibration.json'}. Restart the app to use it.")
    return thresholds


if __name__ == "__main__":
    main()
