# OMNIGUARD — 2-minute exhibition script

**Before the crowd arrives (not spoken)**

1. Activate the venv and start offline + Fast profile:
   ```bash
   source venv/bin/activate
   bash run_exhibition.sh
   # or: HF_HUB_OFFLINE=1 OMNIGUARD_PROFILE=fast python app.py
   ```
2. Click **Real voice** → **Check this voice**. Expect 🟢.
3. Click **AI voice** → **Check this voice**. Expect 🔴.
4. If either is wrong, swap the clip in `examples/` for one your detectors handle, or re-run `python calibrate.py` the night before.
5. Close any extra browser tabs. Have the Accessibility Assistant tab ready as the second demo.

Keep a phone hotspot as emergency only. With `HF_HUB_OFFLINE=1` the app will not touch the network.

---

## 0:00 – 0:25 | The problem

"Your phone rings. It is your son, panicking, asking for money right now. It
sounds exactly like him, but it is an AI clone built from a few seconds of his
voice. In a 2023 McAfee survey, 83 % of Indian victims of AI voice scams lost
money, and 69 % of Indians said they cannot tell a cloned voice from a real one.
Seniors are targeted, and blind or low-vision users cannot cross-check a call
with anything they can see. OMNIGUARD is built for them."

## 0:25 – 0:55 | Live demo — Deepfake Detector

- Click **Real voice**, then **Check this voice**.
  "A real human voice: green, low fake probability. Several detectors vote.
  Open 'How we decided' if anyone asks how it works."
- Click **AI voice**, then **Check this voice**.
  "An AI voice: red. It shows the percentage and a clear safety tip:
  never transfer money on one phone call — call back on a number you already know."

## 0:55 – 1:25 | Live demo — Accessibility Assistant

- Switch to **♿ Accessibility Assistant**. Click **AI voice** to load the clip.
- Press the big **🔊 Play Voice Alert** button.
  "It transcribes what was said, then speaks the verdict aloud.
  No screen reading needed. The message is rule-based — no chatbot —
  so it is predictable and safe."

## 1:25 – 1:40 | The dual-impact story

"One app, two groups: it protects people from voice scams, and it gives
blind and low-vision users the same protection through sound instead of sight.
Everything runs on this laptop. Nothing is sent to the cloud."

## 1:40 – 1:52 | Limitations (be honest)

"No detector is perfect. New cloning tools and noisy calls can still confuse it.
So the rule never changes: always verify by calling back on a known number."

## 1:52 – 2:00 | Close

"Next steps could be a phone app that checks live calls, or a browser-only version
so your voice never leaves the device. Thank you."

---

## Optional 30-second segment — Voice Assistant

Open the **🗣️ Voice Assistant** tab. Press the microphone and say:
"I think I am being scammed." Let it answer out loud, then say "slower"
and watch the speed change. Click **I already sent money** to show the
step-by-step plan (bank + cybercrime helpline 1930). Switch on **Large text**.

Closing line: "It runs entirely on this laptop, free, offline, and by voice
for people who cannot rely on a screen."
