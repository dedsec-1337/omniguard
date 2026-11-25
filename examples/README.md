# Example clips

`real_sample.wav` and `fake_sample.wav` are short 16 kHz mono placeholders so the
UI buttons work out of the box.

**Before a school exhibition or public demo you must replace them.**

1. Put 15–30 real human recordings in `../calib/real/`
2. Put 15–30 AI / TTS voices in `../calib/fake/`
3. Run `python calibrate.py` (this writes `../calibration.json`)
4. Copy two clips that actually score 🟢 and 🔴 into this folder as
   `real_sample.wav` and `fake_sample.wav`

Do not commit private recordings. The `calib/` folder is git-ignored.
