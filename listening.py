"""Continuous listening: decide when someone has finished speaking.

Designed for slow, hesitant or stuttering speech: an utterance only ends after a *long* silence
(default 3 s, up to 5 s), short pauses and repeated fragments stay in the same utterance, and the
audio just before the first sound is kept so soft starts are not clipped. Plain NumPy, no models, so
it costs almost nothing next to Whisper.
"""
import time

import numpy as np

SAMPLE_RATE = 16_000
FRAME = 480  # 30 ms at 16 kHz

PAUSE_CHOICES = [("Normal (1.5 s)", 1.5), ("Long (3 s)", 3.0), ("Very long (5 s)", 5.0)]


def to_mono16k(samples, sr):
    """Browser chunks arrive as int16 or float, any sample rate, maybe stereo -> float32 mono 16 kHz."""
    x = np.asarray(samples)
    if x.dtype.kind in "iu":  # scale integers first: averaging stereo would turn them into floats
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr and sr != SAMPLE_RATE and x.size:
        n_out = max(1, int(round(x.size * SAMPLE_RATE / sr)))
        x = np.interp(np.linspace(0, x.size - 1, n_out), np.arange(x.size), x).astype(np.float32)
    return x


class Listener:
    """Feed it audio chunks; it returns (status, utterance). status is 'idle', 'hearing', 'waiting' or 'muted'."""

    def __init__(self, pause_s=3.0, min_speech_s=0.4, max_utterance_s=45.0):
        self.pause_s = float(pause_s)
        self.min_speech_s = float(min_speech_s)
        self.max_utterance_s = float(max_utterance_s)
        self.reset()

    def reset(self):
        self._carry = np.zeros(0, np.float32)    # samples that did not fill a whole frame yet
        self._preroll = []                       # the last ~0.35 s before speech starts
        self._utt = []                           # frames of the utterance in progress
        self._speaking = False
        self._voiced_run = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._floor = 0.004                      # running estimate of the background noise level
        self._muted_until = 0.0

    # -- internals ---------------------------------------------------------------------------------
    def _threshold(self):
        return max(0.010, 2.8 * self._floor)

    def _finish(self):
        frames, voiced = self._utt, self._speech_frames
        trail = int(0.5 * SAMPLE_RATE / FRAME)  # keep half a second of the pause, drop the rest
        keep = frames[: max(0, len(frames) - self._silence_frames + trail)]
        audio = np.concatenate(keep) if keep else np.zeros(0, np.float32)
        ok = voiced * FRAME / SAMPLE_RATE >= self.min_speech_s
        self.reset_utterance()
        return audio if ok else None

    def reset_utterance(self):
        self._utt, self._speaking = [], False
        self._voiced_run = self._speech_frames = self._silence_frames = 0

    # -- public ---------------------------------------------------------------------------------------
    def mute(self, seconds):
        """Ignore the microphone for a while (the assistant is speaking; do not answer ourselves)."""
        self._muted_until = time.time() + float(seconds)
        self.reset_utterance()
        self._carry = np.zeros(0, np.float32)
        self._preroll = []

    @property
    def muted(self):
        return time.time() < self._muted_until

    def feed(self, samples, sr=SAMPLE_RATE):
        if self.muted:
            return "muted", None
        x = np.concatenate([self._carry, to_mono16k(samples, sr)])
        n = x.size // FRAME
        self._carry = x[n * FRAME:]
        utterance = None
        for i in range(n):
            frame = x[i * FRAME:(i + 1) * FRAME]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            voiced = rms > self._threshold()
            if not self._speaking:
                self._floor = 0.95 * self._floor + 0.05 * min(rms, 0.05)  # learn the room noise
                self._preroll.append(frame)
                self._preroll = self._preroll[-12:]
                self._voiced_run = self._voiced_run + 1 if voiced else 0
                if self._voiced_run >= 3:  # three voiced frames in a row = someone started talking
                    self._speaking = True
                    self._utt = list(self._preroll)
                    self._speech_frames = self._voiced_run
                    self._silence_frames = 0
                    self._preroll = []
                continue
            self._utt.append(frame)
            if voiced:
                self._speech_frames += 1
                self._silence_frames = 0
            else:
                self._silence_frames += 1
            too_long = len(self._utt) * FRAME / SAMPLE_RATE >= self.max_utterance_s
            if self._silence_frames * FRAME / SAMPLE_RATE >= self.pause_s or too_long:
                utterance = self._finish()
        return self.status, utterance

    @property
    def status(self):
        if not self._speaking:
            return "idle"
        return "waiting" if self._silence_frames * FRAME / SAMPLE_RATE > 0.8 else "hearing"
