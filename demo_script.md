# OMNIGUARD — 2-minute exhibition script

**Before the crowd arrives (not spoken):** run `python calibrate.py` once on your
real and AI clips, restart the app, then run both example clips. Confirm
`real_sample.wav` shows 🟢 and `fake_sample.wav` shows 🔴. If either is off, swap
the clip for one the detectors handle. Start the app with
`HF_HUB_OFFLINE=1 python app.py` in case venue Wi-Fi fails.

---

## 0:00 – 0:25 | The problem

"Your phone rings. It is your son, panicking, asking for money right now. It
sounds exactly like him, but it is an AI clone built from a few seconds of his
voice. In a 2023 McAfee survey, 83% of Indian victims of AI voice scams lost
money, and 69% of Indians said they cannot tell a cloned voice from a real one.
Seniors are targeted, and blind or low-vision users cannot cross-check a call
with anything they can see. OMNIGUARD is built for them."

## 0:25 – 0:55 | Live demo, Tab 1: Deepfake Detector

- Click **Real voice**, then **Check this voice**.
  "A real human voice: green, low fake probability. Two detectors vote, and open 'How we decided' to see each one."
- Click **AI voice**, then **Check this voice**.
  "An AI voice: red. It shows the exact percentage, why this matters (unnatural
  pauses, smooth rhythm, missing breath sounds) and a safety tip: never transfer
  money on one phone call, call back on a known number."

## 0:55 – 1:25 | Live demo, Tab 2: Accessibility Assistant

- Switch to **♿ Accessibility Assistant**. Click **AI voice** to load the clip.
- Press the big **🔊 Play Voice Alert** button.
  "It transcribes what was said, then speaks the verdict aloud: 'Warning. This
  voice is likely AI-generated. Do not send money. Call a family member first.'
  No screen reading needed. The message is rule-based, no chatbot involved, so
  it is predictable and safe."

## 1:25 – 1:40 | The dual-impact story

"One app, two groups: it protects scam victims, and it gives blind and
low-vision users the same protection through sound instead of sight."

## 1:40 – 1:52 | Limitations

"No detector is 100% accurate. New cloning tools can slip through, and noisy
calls can give an uncertain yellow result. So our rule never changes: always
verify by calling back on a known number."

## 1:52 – 2:00 | Future roadmap

"Next: a native Android app that checks live calls, and a browser-only
JavaScript version so your voice never leaves your device. Thank you."
