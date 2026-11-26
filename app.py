"""OMNIGUARD - Protect your voice, protect your people.

The web app (Gradio UI). All models and analysis live in engine.py; the assistant's brain is in
guardian.py; the activity log is channel_log.py. Run it with `python app.py` (CPU is enough; a small
NVIDIA GPU is shared automatically). Everything stays on this computer.
"""
import html
import os
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

import gradio as gr
import soundfile as sf

import engine as eg
import guardian as gd
import listening
import omni_runtime as rt
from channel_log import FILTERS, LOG
from engine import AudioProblem

ASSETS = Path(__file__).parent / "assets"
LANG_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
              "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "es": "Spanish",
              "fr": "French", "de": "German", "pt": "Portuguese", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese"}

# --- Words for the three traffic-light verdicts (rule-based, no LLM) ----------------------------------------
VERDICTS = {
    "green": {
        "color": "var(--og-ok)", "label": "Likely real", "headline": "Sounds like a real person",
        "advice": "No strong sign of AI in this clip. Still, call back on a number you know before acting.",
        "why": "The detectors found no strong sign of synthetic speech in this clip.",
        "speech": "This voice sounds like a real person. But always verify by calling back on a known number.",
    },
    "yellow": {
        "color": "var(--og-warn)", "label": "Uncertain", "headline": "We can't be sure",
        "advice": "Do not send money or share codes. Hang up and call back on a number you know.",
        "why": "The score sits in the grey zone, or the detectors disagree. Noise, phone "
               "compression or a short clip can cause this.",
        "speech": "Be careful. I cannot tell if this voice is real or AI-generated. "
                  "Do not send money or share codes. Call back on a number you know.",
    },
    "red": {
        "color": "var(--og-bad)", "label": "Likely AI-generated", "headline": "This voice is likely AI-generated",
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
LEVEL_COLOR = {"low": "var(--og-ok)", "elevated": "var(--og-warn)", "high": "var(--og-bad)", "critical": "var(--og-crit)"}


# --- Small HTML pieces -------------------------------------------------------------------------------------------
def _icon(paths):
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{paths}</svg>'


ICONS = {
    "detect": _icon('<path d="M12 3l8 3v6c0 4.5-3.2 8-8 9-4.8-1-8-4.5-8-9V6l8-3z"/><path d="M9 12l2 2 4-4"/>'),
    "access": _icon('<circle cx="12" cy="4.5" r="1.8"/><path d="M5 8.5l7 1.5 7-1.5M12 10v5m-3.5 5L12 15l3.5 5"/>'),
    "voice": _icon('<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0014 0M12 18v3"/>'),
    "log": _icon('<path d="M3 12h4l3-8 4 16 3-8h4"/>'),
}
BRAND = """
<div class="og-brandlock">
  <svg class="og-logo" viewBox="0 0 48 48" aria-hidden="true">
    <defs><linearGradient id="ogg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1"/>
      <stop offset=".6" stop-color="#8b5cf6"/><stop offset="1" stop-color="#14b8a6"/></linearGradient></defs>
    <path d="M24 4 40 10v12c0 11-7 19-16 22C15 41 8 33 8 22V10z" fill="url(#ogg)"/>
    <path d="M15 24v-2m4.5 6V18m4.5 10V14m4.5 14V18m4.5 6v-2" stroke="#fff" stroke-width="2.6" stroke-linecap="round" fill="none"/>
  </svg>
  <div><div class="og-wordmark">OMNIGUARD</div><div class="og-tagline">Protect your voice, protect your people.</div></div>
</div>"""
_PAGES = (("detect", "Deepfake Detector"), ("access", "Accessibility Assistant"), ("voice", "Voice Assistant"))
NAV = ('<div class="og-rail-label">Pages</div><nav class="og-nav" aria-label="Pages">' + "".join(
    f'<button type="button" class="og-nav-item{" is-active" if i == 0 else ""}" onclick="ogNav({i})" '
    + ('aria-current="page" ' if i == 0 else "")
    + f'aria-label="{label} (Alt+{i + 1})">{ICONS[key]}<span>{label}</span><kbd>Alt+{i + 1}</kbd></button>'
    for i, (key, label) in enumerate(_PAGES)) + "</nav>")
SKIP = ('<a class="og-skip" href="#og-main" onclick="var m=document.getElementById(\'og-main\');'
        'm.setAttribute(\'tabindex\',\'-1\');m.focus();return false;">Skip to main content</a>')
FOOTER = ('<div class="og-fine" style="text-align:center;margin-top:8px">&#128274; Your audio is analysed only by the models on '
          'this computer. Nothing is sent to a cloud service.<br>No detector is 100% accurate. Always verify by calling back '
          'on a known number.</div>')
EMPTY_CARD = ('<div class="og-empty">Your result will appear here.<br>'
              '<small>Add a clip, then press the big button.</small></div>')
ANSWER_EMPTY = ('<div class="og-empty">My answer will appear here, in large text, and I will read it aloud.<br>'
                '<small>Try: &ldquo;I think I am being scammed&rdquo;.</small></div>')
INTENT_LABEL = {"lost_money": "Emergency", "live_call": "Emergency", "flow": "Step by step", "analyze_text": "Message check",
                "explain_last": "Result", "repeat": "Again", "help": "Help", "howto": "How to", "more": "More detail",
                "stop": "Stopped", "clear_memory": "Memory cleared", "problem": "Heads up"}


def ring_svg(p, color):
    circ = 2 * 3.14159265 * 52
    return (f'<svg viewBox="0 0 120 120" class="og-ring-svg" aria-hidden="true">'
            f'<circle cx="60" cy="60" r="52" class="og-ring-bg"/>'
            f'<circle cx="60" cy="60" r="52" class="og-ring-fg" transform="rotate(-90 60 60)" '
            f'style="stroke:{color};stroke-dasharray:{circ:.1f};--from:{circ:.1f};stroke-dashoffset:{circ * (1 - p):.1f}"/>'
            f'<text x="60" y="64" class="og-ring-num">{p * 100:.0f}%</text>'
            f'<text x="60" y="83" class="og-ring-sub">AI-GENERATED</text></svg>')


def result_card(res):
    v = VERDICTS[res["key"]]
    n = len(res["models"])
    agree = ("" if n < 2 else ("Detectors disagree" if res["spread"] > 0.5 else "Detectors agree") + f" &middot; {n} voting")
    return (f'<div class="og-result" role="status" style="--accent:{v["color"]}">'
            f'<div class="og-ring">{ring_svg(res["p"], v["color"])}</div>'
            f'<div class="og-verdict"><span class="og-chip"><i class="og-dot"></i>{v["label"]}</span>'
            f'<h2>{v["headline"]}</h2><p>{v["advice"]}</p>'
            f'{f"<span class=og-agree>{agree}</span>" if agree else ""}</div></div>')


def notice_card(message):
    return ('<div class="og-result og-notice" role="status"><div class="og-verdict">'
            f'<span class="og-chip">Cannot analyse</span><p>{html.escape(message)}</p></div></div>')


def details_html(res):
    split = ("<p><b>The detectors disagree,</b> so this is treated as uncertain.</p>"
             if res["spread"] > 0.5 and len(res["models"]) > 1 else "")
    return (f'<details class="og-more"><summary>How we decided</summary><div class="og-details">'
            f'{split}<p>{VERDICTS[res["key"]]["why"]}</p><p><b>Safety tip:</b> {TIP}</p>'
            f'<p class="og-fine">Scores are the share of evidence for "AI-generated". No detector is 100% accurate.</p>'
            f'</div></details>')


def _spark(values):
    if not values:
        return ""
    bars = "".join(f'<rect x="{i * 9}" y="{16 - max(2, v * 16):.1f}" width="6" height="{max(2, v * 16):.1f}" rx="2"/>'
                   for i, v in enumerate(values[:6]))
    return f'<svg class="og-spark" viewBox="0 0 {len(values[:6]) * 9} 16" aria-hidden="true">{bars}</svg>'


def _mrow(i, key, name, p, weight, windows):
    status = rt.MODEL_STATUS.get(key, "skipped")
    score = (f'<div class="og-mbar"><i style="width:{p * 100:.0f}%"></i></div><b>{p * 100:.0f}%</b>'
             if p is not None else '<div class="og-mbar"></div><b>&mdash;</b>')
    info = '<span class="og-info">Independent benchmark: variable</span>' if key == "melody" else ""
    return (f'<div class="og-mrow{"" if p is not None else " og-mute"}" style="--i:{i}">'
            f'<span class="og-mname">{html.escape(name)} <i class="og-w">&times;{weight:g}</i>{info}</span>'
            f'<span>{_spark(windows)}</span>{score}'
            f'<span class="og-st og-st-{status}" title="{html.escape(rt.STATUS_NOTE.get(key, ""))}">{status}</span></div>')


def model_table_html(res):
    rows = [_mrow(i, m["key"], m["name"], m["p"], m["weight"], m["windows"]) for i, m in enumerate(res["models"])]
    shown = {m["key"] for m in res["models"]}
    rows += [_mrow(len(rows) + j, k, eg.CATALOG[k]["name"], None, eg.effective_weight(k), [])
             for j, k in enumerate(k for k in eg.ACTIVE if k not in shown)]
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


def mode_card_html(name):
    p = rt.PROFILES[name]
    dets = "2 quickest" if p["max_models"] else f"all loaded ({len(eg.DETECTORS)})"
    llm = "off (rules only)" if not p["llm"] else f"on, {p['llm_tokens']} tokens"
    whisper = os.environ.get("OMNIGUARD_WHISPER", "").strip() or p["whisper"]
    rows = [("Detectors", dets), ("Analysis windows", p["max_windows"]), ("Speech model", whisper),
            ("Scam-wording model", "on" if p["semantic"] else "off"), ("Local LLM", llm), ("Animation", p["motion"])]
    dl = "".join(f"<dt>{k}</dt><dd>{html.escape(str(v))}</dd>" for k, v in rows)
    return f'<div class="og-modecard"><b>{p["label"]}</b> &middot; {p["blurb"]}<dl>{dl}</dl></div>'


def topbar_html():
    ready = len(eg.DETECTORS)
    failed = sum(1 for k in eg.ACTIVE if rt.MODEL_STATUS.get(k) == "failed")
    cls = "og-pill-warn" if (failed or not ready) else "og-pill-ok"
    pills = [f'<span class="og-pill {cls}"><i class="og-dot"></i>{ready} detector{"s" if ready != 1 else ""} ready'
             + (f" &middot; {failed} failed" if failed else "") + "</span>",
             f'<span class="og-pill og-pill-ok"><i class="og-dot"></i>{rt.PROFILES[rt.current_profile()]["label"]} mode</span>']
    llm, _ = eg.llm_for_mode()
    pills.append(f'<span class="og-pill og-pill-ok"><i class="og-dot"></i>Local LLM {html.escape(eg.LLM.model)}</span>' if llm
                 else '<span class="og-pill og-pill-off"><i class="og-dot"></i>Local LLM off</span>')
    if eg.SWAP_MODE:
        pills.append('<span class="og-pill og-pill-ok"><i class="og-dot"></i>GPU shared</span>')
    return ('<div class="og-topbar2"><div class="og-crumb"><b>Local</b> &middot; private &middot; free</div>'
            f'<div class="og-pills">{"".join(pills)}</div></div>')


def system_html():
    def chip(status):
        return f'<span class="og-st og-st-{status}">{status}</span>'

    def lat(key):
        v = rt.avg_latency(key)
        return f"{v:.1f} s/window on {rt.last_device(key)}" if v is not None else "not run yet"

    def row(label, sub, key):
        t = f"{rt.LOAD_SECONDS[key]:.1f} s" if key in rt.LOAD_SECONDS else "&mdash;"
        return (f'<tr><td>{html.escape(label)}<br><small>{html.escape(sub)}</small></td><td>{chip(rt.MODEL_STATUS.get(key, "skipped"))}</td>'
                f'<td>{t}</td><td>{lat(key) if key in eg.CATALOG else "&mdash;"}</td>'
                f'<td><small>{html.escape(rt.STATUS_NOTE.get(key, ""))}</small></td></tr>')

    rows = "".join(row(s["name"], s["id"], k) for k, s in eg.CATALOG.items())
    rows += row("Speech-to-text (faster-whisper)", f"size {eg.WHISPER_SIZE or 'not loaded'}", "whisper")
    rows += row("Voice (Piper)", "en_US-lessac-medium", "piper")
    rows += row("Scam-wording model (zero-shot)", gd.SEMANTIC_ID, "semantic")
    llm_txt = ("ready: " + eg.LLM.model) if eg.LLM.available() else "off (optional: install Ollama, see README)"
    hw = f'{eg.HW["gpu"]} ({eg.HW["vram_gb"]} GB)' if eg.HW.get("gpu") else "no GPU in use (CPU)"
    warn = "".join(f"<p class='og-warn'>&#9888; {html.escape(w)}</p>" for w in rt.latency_warnings())
    modes = "".join(f"<tr><td><b>{p['label']}</b></td><td>{'2 quickest' if p['max_models'] else 'all'}</td><td>{p['max_windows']}</td>"
                    f"<td>{p['whisper']}</td><td>{'on' if p['semantic'] else 'off'}</td><td>{'on' if p['llm'] else 'off'}</td>"
                    f"<td>{p['motion']}</td></tr>" for p in rt.PROFILES.values())
    keys = [("Alt+1 / 2 / 3", "Switch page"), ("Alt+M", "Focus the microphone"), ("/", "Focus the question box"),
            ("Alt+L", "Show or hide the Channel Log"), ("Alt+T", "Change appearance"), ("Esc", "Close the log")]
    kbds = "".join(f"<kbd>{k}</kbd><span>{v}</span>" for k, v in keys)
    return (f'<details class="og-more"><summary>System &amp; models</summary><div class="og-details">'
            f'<p><b>Hardware:</b> {hw} &middot; GPU sharing: {"on (one detector on the GPU at a time)" if eg.SWAP_MODE else "off"}'
            f' &middot; Mode: {rt.PROFILES[rt.current_profile()]["label"]} &middot; Local LLM: {html.escape(llm_txt)}</p>{warn}'
            f'<div class="og-scroll"><table class="og-sys"><thead><tr><th>Model</th><th>Status</th><th>Load time</th><th>Speed</th><th>Note</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            f'<p><b>Modes</b></p><div class="og-scroll"><table class="og-sys"><thead><tr><th>Mode</th><th>Detectors</th><th>Windows</th>'
            f'<th>Speech</th><th>Scam model</th><th>LLM</th><th>Animation</th></tr></thead><tbody>{modes}</tbody></table></div>'
            f'<p><b>Keyboard</b></p><div class="og-kbds">{kbds}</div>'
            f'<p class="og-fine">Models run on this computer. Nothing is sent to a cloud service.</p></div></details>')


# --- Tab 1: Deepfake Detector ---------------------------------------------------------------------------------------
def detect(path, progress=gr.Progress()):
    try:
        with rt.progress_scope(progress):
            rt.report(0.04, "Reading the audio")
            audio = eg.load_audio(path)
            res = eg.analyse(audio)
            rt.report(0.97, "Done")
    except AudioProblem as problem:
        return notice_card(str(problem))
    except Exception:  # noqa: BLE001 - never show a stack trace
        return notice_card("Something went wrong while analysing this clip.")
    return result_card(res) + details_html(res)


def detail_a(path):
    """Second listener on the Detector button: per-model table (+ the words check in Thorough mode).
    Reads the cached analysis, so the models do not run twice."""
    try:
        res = eg.analyse(eg.load_audio(path))
    except Exception:  # noqa: BLE001
        return "", ""
    cross = ""
    if rt.PROFILES[rt.current_profile()]["crosscheck"]:
        try:
            cross = crosscheck_card_html(eg.analyse_clip(path))
        except Exception:  # noqa: BLE001
            cross = ""
    return model_table_html(res), cross


# --- Tab 2: Accessibility Assistant (transcript + verdict + spoken alert) -------------------------------------------
def assist(path):
    """Transcript + verdict + spoken alert. Works even if some models failed."""
    try:
        audio = eg.load_audio(path)
    except AudioProblem as problem:
        return notice_card(str(problem)), "", str(problem), None
    transcript = eg.transcribe(audio)
    try:
        res = eg.analyse(audio)
        card, message = result_card(res), VERDICTS[res["key"]]["speech"]
    except Exception:  # noqa: BLE001 - graceful degradation without the detectors
        card, message = notice_card("Voice check unavailable. Be careful."), NO_VERDICT_SPEECH
    speech = eg.speak(message)
    if speech is None:
        message += "\n\n⚠️ Voice alert unavailable - please read this message instead."
    return card, transcript, f"## {message}", speech


def crosscheck_b(path):
    try:
        return crosscheck_card_html(eg.analyse_clip(path))
    except Exception:  # noqa: BLE001
        return ""


def load_example(kind):
    """Return an example clip path for one tab; fail politely if it is missing."""
    path = Path(__file__).parent / "examples" / f"{kind}_sample.wav"
    try:
        sf.info(str(path))
    except Exception:  # noqa: BLE001
        gr.Warning("Example not available - please upload your own clip")
        return gr.update()
    return str(path)


# --- Tab 3: Voice assistant -------------------------------------------------------------------------------------------
def _new_state():
    return {"history": [], "analysis": None, "info": None, "reply": None, "intent": None, "question": None, "flow": None}


def answer_card(text, intent, emergency=False):
    cls = "og-answer og-emerg" if emergency else "og-answer"
    return (f'<div class="{cls}" role="status"><span class="og-chip"><i class="og-dot"></i>{INTENT_LABEL.get(intent, "Guardian")}</span>'
            f'<p>{html.escape(text)}</p></div>')


def _history_html(items):
    if not items:
        return ""
    rows = "".join(f'<li><span class="og-q">{html.escape(q)}</span><span class="og-a">{html.escape(a)}</span></li>'
                   for q, a in reversed(items[-6:]))
    return f'<details class="og-more"><summary>Earlier in this conversation</summary><ol class="og-hist">{rows}</ol></details>'


def _memory_html(state):
    n = len(state["history"])
    flow = f' &middot; guided steps: {state["flow"]["name"].replace("_", " ")}' if state.get("flow") else ""
    return (f'<div class="og-memory"><span>Memory: {n} recent turn{"s" if n != 1 else ""}{flow}. Kept only in this tab, on this computer.</span></div>')


def listen_status_html(status):
    label = {"idle": "Listening", "hearing": "Hearing you", "waiting": "Waiting, take your time",
             "muted": "Paused while I speak", "off": "Not listening"}[status]
    return f'<span class="og-listen is-{status}"><i></i>{label}</span>'


COMMAND_INTENTS = {"repeat", "slower", "faster", "simpler", "normal", "stop", "clear_memory", "more", "flow", "empty", "problem"}


def _assistant_core(heard, lang, prob, problem, attach_path, speed, simple, aloud, region, state, clear_inputs=True):
    """One turn: understand -> check any attached clip -> answer -> speak -> remember -> log."""
    state = state or _new_state()
    speed = float(speed or 1.0)
    info = None
    if attach_path:
        try:
            info = eg.analyse_clip(attach_path)
            state["info"] = info
            state["analysis"] = {"summary": info["summary"], "summary_simple": info["summary_simple"]}
            if info["fused"]["level"] in ("high", "critical"):
                LOG.add("alert", f"Recording checked: overall risk {info['fused']['level']}", info["fused"]["story"], "alert")
        except AudioProblem as exc:
            problem = problem or str(exc)
        except Exception:  # noqa: BLE001
            problem = problem or "I could not check that recording."
    last = {**(state.get("analysis") or {}), "reply": state.get("reply"), "intent": state.get("intent"),
            "question": state.get("question")}
    llm, tokens = eg.llm_for_mode()
    if not heard and info is None:
        reply = {"text": problem or "Please speak or type a question.", "intent": "problem", "settings": {}, "flow": state.get("flow")}
    elif not heard:
        reply = {"text": info["summary_simple" if simple else "summary"], "intent": "explain_last", "settings": {}, "flow": state.get("flow")}
    else:
        facts = (state.get("analysis") or {}).get("summary", "")
        reply = gd.reply_to(heard, simple=bool(simple), last=last, llm=llm, semantic=eg.semantic_for_mode(), facts=facts,
                            region=region, flow=state.get("flow"), history=state["history"], max_tokens=tokens)
    change = reply.get("settings") or {}
    new_speed, new_simple = speed, bool(simple)
    if "speed_delta" in change:
        new_speed = round(min(1.5, max(0.6, speed + change["speed_delta"])), 1)
    if "simple" in change:
        new_simple = bool(change["simple"])
    stop = bool(change.get("stop_audio"))
    if change.get("clear_memory"):
        state = _new_state()
        LOG.add("voice", "Memory cleared", "The conversation memory was erased.", "ok")
    else:
        state["flow"] = reply.get("flow")
        if reply["intent"] not in COMMAND_INTENTS:
            state["reply"], state["intent"], state["question"] = reply["text"], reply["intent"], heard
        state["history"] = (state["history"] + [(heard or "(recording)", reply["text"])])[-8:]
    body = reply["text"]
    speech = None
    if aloud and not stop:
        speech = eg.speak_rate(body, new_speed)
        if speech is None:
            body += " (The voice is unavailable, please read this.)"
    if reply.get("alert"):
        LOG.add("alert", reply["alert"], f"Guardian: {body[:200]}", "alert")
    LOG.add("voice", f"You: {heard or '(recording)'}"[:150], f"Guardian ({reply['intent']}): {body[:240]}")
    emergency = bool(state.get("flow")) or reply["intent"] in ("lost_money", "live_call")
    lang_txt = ""
    if heard and lang:
        lang_txt = f' <small>Heard {html.escape(LANG_NAMES.get(lang, lang))}' + (f" ({prob:.0%})" if prob else "") + (
            "; I answer in English for now" if lang != "en" else "") + "</small>"
    you = f'<div class="og-you">You said: &ldquo;{html.escape(heard)}&rdquo;{lang_txt}</div>' if heard else ""
    history_html = _history_html(state["history"][:-1] if not change.get("clear_memory") else [])
    signal = f'<span data-og-stop="{time.time_ns()}" class="og-sr-only"></span>' if stop else ""  # unique value so the page notices every Stop
    keep = gr.skip() if not clear_inputs else None
    return (answer_card(body, reply["intent"], emergency), speech, crosscheck_card_html(info) if info else gr.skip(), you,
            history_html, _memory_html(state), state, keep, "" if clear_inputs else gr.skip(),
            gr.update(value=new_speed), gr.update(value=new_simple), signal)


def assistant_turn(mic_path, text, attach_path, speed, simple, aloud, region, state):
    """Push-to-talk or typed: both go through the same path, so typing is equal to speaking."""
    heard, lang, prob, problem = (text or "").strip(), None, 0.0, ""
    if not heard and mic_path:
        try:
            heard, lang, prob = eg.transcribe_ex(eg.load_audio(mic_path))
        except AudioProblem as exc:
            problem = str(exc)
        if not heard and not problem:
            problem = ("Speech recognition is unavailable. Please type your question." if eg.ensure_whisper() is None
                       else "I could not make out any words. Please try again, or type your question.")
    return _assistant_core(heard, lang, prob, problem, attach_path, speed, simple, aloud, region, state)


def quick_turn(word):
    """Buttons that send a fixed phrase (Done, Skip, Back, Stop, the emergency buttons)."""
    def run(attach_path, speed, simple, aloud, region, state):
        return _assistant_core(word, None, 0.0, "", attach_path, speed, simple, aloud, region, state)
    return run


def listen_chunk(chunk, lstate, pause_s, attach_path, speed, simple, aloud, region, state):
    """Continuous listening: one call per audio chunk. Returns (listener, status, *assistant outputs)."""
    skip = (gr.skip(),) * 12
    if chunk is None:
        return (lstate, gr.skip()) + skip
    sr, data = chunk
    if lstate is None or abs(lstate.pause_s - float(pause_s)) > 1e-6:
        lstate = listening.Listener(pause_s=float(pause_s))
    status, utterance = lstate.feed(data, sr)
    if utterance is None:
        return (lstate, listen_status_html(status)) + skip
    heard, lang, prob = eg.transcribe_ex(utterance)
    if not heard:
        return (lstate, listen_status_html("idle")) + skip
    out = _assistant_core(heard, lang, prob, "", attach_path, speed, simple, aloud, region, state, clear_inputs=False)
    speech = out[1]
    if speech:  # echo guard: do not "hear" the answer we are about to play
        try:
            with wave.open(speech) as w:
                lstate.mute(w.getnframes() / float(w.getframerate()) + 1.2)
        except Exception:  # noqa: BLE001
            lstate.mute(6.0)
    return (lstate, listen_status_html("muted" if speech else "idle")) + out


def toggle_listen_mode(mode):
    continuous = mode == "continuous"
    return gr.update(visible=not continuous), gr.update(visible=continuous), gr.update(visible=continuous), listen_status_html(
        "idle" if continuous else "off")


def make_report(state, region):
    state = state or _new_state()
    text = gd.build_report(state.get("info"), region=region, history=state["history"])
    path = Path(tempfile.gettempdir()) / f"omniguard_safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(text, encoding="utf-8")
    LOG.add("voice", "Safety report created", path.name, "ok")
    return gr.update(value=str(path), visible=True)


# --- Modes, log, export ------------------------------------------------------------------------------------------------
def set_mode(name):
    old = rt.current_profile()
    rt.set_global_profile(name)
    if name != old:
        p = rt.PROFILES[name]
        LOG.add("system", f"Mode changed to {p['label']}", p["help"], "ok")
    return mode_card_html(rt.current_profile()), topbar_html()


def log_tick(filter_name, seen):
    """Timer: refresh the dock only when something changed (cheap: a version number compare)."""
    key = (LOG.version, filter_name)
    if seen == key:
        return gr.skip(), gr.skip(), seen
    return LOG.summary_html(), LOG.render_html(filter_name), key


def export_log():
    path = Path(tempfile.gettempdir()) / f"omniguard_channel_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(LOG.export_text(), encoding="utf-8")
    return gr.update(value=str(path), visible=True)


def clear_log():
    LOG.clear()
    LOG.add("system", "Channel log cleared", "", "ok")
    return LOG.summary_html(), LOG.render_html("All"), "All"


# --- Build the UI ---------------------------------------------------------------------------------------------------------
gr.set_static_paths(paths=[ASSETS])
FONT_URL = "/gradio_api/file=" + str((ASSETS / "fonts" / "Inter-latin.woff2").resolve())
CSS = (ASSETS / "omniguard.css").read_text().replace("__FONT_URL__", FONT_URL)
HEAD = "<script>" + (ASSETS / "omniguard.js").read_text() + "</script>"
THEME = gr.themes.Base(primary_hue="indigo", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.Font("OGInter"), gr.themes.Font("ui-sans-serif"), gr.themes.Font("system-ui"),
                             gr.themes.Font("sans-serif")])
DOCK_TITLE = f'<div class="og-dock-title">{ICONS["log"]}Channel Log</div>'
DOCK_TOGGLE = ('<button type="button" class="og-dock-toggle" aria-expanded="false" aria-controls="og-dock-body" '
               'onclick="ogToggleDock()" aria-label="Show or hide the Channel Log (Alt+L)"><span>Log</span><span class="chev">&#9650;</span></button>')
MODE_JS = "(m) => { ogSetMotion(m); return m; }"

with gr.Blocks(title="OMNIGUARD — Protect your voice, protect your people.") as demo:
    gr.HTML(SKIP, elem_classes="og-flush")
    with gr.Row(elem_classes="og-app"):
        # ---------------- left rail: brand, pages, mode, appearance ----------------
        with gr.Column(elem_classes="og-rail"):
            gr.HTML(BRAND, elem_classes=["og-flush", "og-r-brand"])
            gr.HTML(NAV, elem_classes=["og-flush", "og-r-nav"])
            gr.HTML('<div class="og-rail-label">Mode</div>', elem_classes="og-flush")
            mode_v = gr.Radio([(v["label"], k) for k, v in rt.PROFILES.items()], value=rt.current_profile(), show_label=False,
                              elem_classes="og-modes", container=False)
            mode_card_v = gr.HTML(mode_card_html(rt.current_profile()), elem_classes="og-flush")
            gr.HTML('<div class="og-rail-label">Appearance</div>', elem_classes="og-flush")
            appearance_v = gr.Radio([("Dark", "dark"), ("Light", "light"), ("Contrast", "contrast")], value="dark", show_label=False,
                                    elem_classes="og-appearance", container=False)
        # ---------------- main area ----------------
        with gr.Column(elem_classes="og-main", elem_id="og-main"):
            topbar = gr.HTML(topbar_html(), elem_classes="og-flush")
            with gr.Tabs(elem_id="og-tabs", elem_classes="og-hidden-strip"):
                # ===== Deepfake Detector =====
                with gr.Tab("🎙️ Deepfake Detector", id="detect"):
                    gr.HTML('<div class="og-h"><span class="og-eyebrow">Deepfake detector</span><h1>Is this voice real?</h1>'
                            '<p>Several detectors listen to a clip and vote. Add a recording or use the microphone.</p></div>', elem_classes="og-flush")
                    with gr.Row(elem_classes="og-shell"):
                        with gr.Column(elem_classes="og-col-input"):
                            with gr.Column(elem_classes="og-panel"):
                                audio_a = gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav", label="Voice clip")
                                with gr.Row(elem_classes="og-chips"):
                                    gr.Markdown("Try a sample")
                                    real_a = gr.Button("Real voice", size="sm", elem_classes="og-ghost")
                                    fake_a = gr.Button("AI voice", size="sm", elem_classes="og-ghost")
                                run_a = gr.Button("Check this voice", variant="primary", size="lg", elem_classes="og-cta")
                        with gr.Column(elem_classes="og-col-result"):
                            card_a = gr.HTML(EMPTY_CARD, elem_classes="og-flush")
                            table_a = gr.HTML("", elem_classes=["og-flush", "og-model-table"])
                            cross_a = gr.HTML("", elem_classes="og-flush")
                # ===== Accessibility Assistant =====
                with gr.Tab("♿ Accessibility Assistant", id="access"):
                    gr.HTML('<div class="og-h"><span class="og-eyebrow">Accessibility assistant</span><h1>Hear the verdict</h1>'
                            '<p>Upload a suspicious clip. I write down what was said, check the voice, and read the result aloud.</p></div>',
                            elem_classes="og-flush")
                    with gr.Row(elem_classes="og-shell"):
                        with gr.Column(elem_classes="og-col-input"):
                            with gr.Column(elem_classes="og-panel"):
                                audio_b = gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav", label="Voice clip")
                                with gr.Row(elem_classes="og-chips"):
                                    gr.Markdown("Try a sample")
                                    real_b = gr.Button("Real voice", size="sm", elem_classes="og-ghost")
                                    fake_b = gr.Button("AI voice", size="sm", elem_classes="og-ghost")
                                run_b = gr.Button("🔊 Play Voice Alert", variant="primary", size="lg", elem_id="play-alert", elem_classes="og-cta")
                        with gr.Column(elem_classes="og-col-result"):
                            card_b = gr.HTML(EMPTY_CARD, elem_classes="og-flush")
                            cross_b = gr.HTML("", elem_classes="og-flush")
                            message_b = gr.Markdown(elem_id="alert-text")
                            speech_b = gr.Audio(label="Spoken safety alert", type="filepath", format="wav",
                                                autoplay=True, interactive=False,
                                                elem_classes="og-audio", elem_id="og-speech-b")
                            transcript_b = gr.Textbox(label="What was said (transcript)", lines=3, interactive=False, elem_classes="og-audio")
                # ===== Voice Assistant =====
                with gr.Tab("🗣️ Voice Assistant", id="voice"):
                    gr.HTML('<div class="og-h"><span class="og-eyebrow">Voice assistant</span><h1>Talk to Guardian</h1>'
                            '<p>Short, literal answers. Speak or type, they work the same. Say &ldquo;tell me more&rdquo; for details.</p></div>',
                            elem_classes="og-flush")
                    with gr.Row(elem_classes="og-shell"):
                        with gr.Column(elem_classes="og-col-input"):
                            with gr.Column(elem_classes="og-panel"):
                                listen_mode = gr.Radio([("Push to talk", "push"), ("Continuous listening", "continuous")], value="push",
                                                       label="How to talk", elem_classes="og-chips")
                                listen_status = gr.HTML(listen_status_html("off"), elem_classes="og-flush")
                                mic_v = gr.Audio(sources=["microphone"], type="filepath", format="wav",
                                                 label="Press the microphone, speak, then press stop")
                                mic_stream = gr.Audio(sources=["microphone"], type="numpy", streaming=True, visible=False,
                                                      label="Listening. Speak whenever you are ready")
                                pause_v = gr.Dropdown(listening.PAUSE_CHOICES, value=3.0, visible=False,
                                                      label="How long a pause before I answer")
                                with gr.Row():
                                    q_text = gr.Textbox(label="Or type your question", lines=2, scale=4,
                                                        placeholder="For example: I think I am being scammed")
                                    ask_btn = gr.Button("Ask", variant="primary", scale=1, elem_classes="og-cta")
                            with gr.Column(elem_classes="og-panel"):
                                gr.HTML('<div class="og-rail-label">One-tap help</div>', elem_classes="og-flush")
                                with gr.Row():
                                    sos_a = gr.Button("I think I'm being scammed", elem_classes="og-danger")
                                    sos_b = gr.Button("I already sent money", elem_classes="og-danger")
                                gr.HTML('<div class="og-rail-label">Guided steps</div>', elem_classes="og-flush")
                                with gr.Row(elem_classes="og-flowbar"):
                                    flow_done = gr.Button("Done", elem_classes="og-ghost")
                                    flow_skip = gr.Button("Skip", elem_classes="og-ghost")
                                    flow_back = gr.Button("Back", elem_classes="og-ghost")
                                    flow_stop = gr.Button("Stop", elem_classes="og-ghost")
                                with gr.Row(elem_classes="og-chips"):
                                    chip_help = gr.Button("What can you do?", size="sm", elem_classes="og-ghost")
                                    chip_more = gr.Button("Tell me more", size="sm", elem_classes="og-ghost")
                                    chip_repeat = gr.Button("Repeat", size="sm", elem_classes="og-ghost")
                            attach_v = gr.Audio(sources=["upload"], type="filepath", format="wav", elem_classes="og-audio",
                                                label="Attach a recording for me to check (optional)")
                            with gr.Column(elem_classes=["og-panel", "og-comfort"]):
                                speed_v = gr.Slider(0.6, 1.5, value=1.0, step=0.1, label="Speaking speed")
                                simple_v = gr.Checkbox(True, label="Short, simple answers")
                                aloud_v = gr.Checkbox(True, label="Read answers aloud")
                                region_v = gr.Dropdown(gd.REGION_CHOICES, value=gd.REGION if gd.REGION in gd.REGIONS else "XX",
                                                       label="Region (for helplines)")
                                large_v = gr.Checkbox(False, label="Large text")
                            assist_state = gr.State(_new_state())
                            listen_state = gr.State(None)
                        with gr.Column(elem_classes="og-col-result"):
                            answer_v = gr.HTML(ANSWER_EMPTY, elem_classes="og-flush")
                            speech_v = gr.Audio(label="Spoken answer", type="filepath", format="wav",
                                                autoplay=True, interactive=False,
                                                elem_classes="og-audio", elem_id="og-speech-v")
                            heard_v = gr.HTML("", elem_classes="og-flush")
                            cross_v = gr.HTML("", elem_classes="og-flush")
                            memory_v = gr.HTML(_memory_html(_new_state()), elem_classes="og-flush")
                            with gr.Row(elem_classes="og-chips"):
                                clear_mem = gr.Button("Clear memory", size="sm", elem_classes="og-ghost")
                                report_btn = gr.Button("Create safety report", size="sm", elem_classes="og-ghost")
                            report_file = gr.File(label="Your safety report (plain text)", visible=False, interactive=False, elem_classes="og-audio")
                            hist_v = gr.HTML("", elem_classes="og-flush")
                            signal_v = gr.HTML("", elem_classes="og-flush")
            gr.HTML(FOOTER, elem_classes="og-flush")
            sys_html = gr.HTML(system_html(), elem_classes="og-flush")
            refresh_btn = gr.Button("Refresh model status", size="sm", elem_classes="og-ghost")

    # ---------------- Channel Log dock (always visible at the bottom) ----------------
    with gr.Column(elem_classes="og-dock"):
        with gr.Row(elem_classes="og-dock-head"):
            gr.HTML(DOCK_TITLE, elem_classes="og-flush")
            dock_sum = gr.HTML(LOG.summary_html(), elem_classes=["og-flush", "og-dock-grow"])
            gr.HTML(DOCK_TOGGLE, elem_classes="og-flush")
        with gr.Column(elem_classes="og-dock-body", elem_id="og-dock-body"):
            log_filter = gr.Radio(list(FILTERS), value="All", show_label=False, elem_classes="og-log-filter", container=False)
            log_html = gr.HTML(LOG.render_html("All"), elem_classes="og-flush")
            with gr.Row(elem_classes="og-chips"):
                export_btn = gr.Button("Export as text", size="sm", elem_classes="og-ghost")
                clear_btn = gr.Button("Clear log", size="sm", elem_classes="og-ghost")
            log_file = gr.File(label="Channel log (plain text)", visible=False, interactive=False, elem_classes="og-audio")
    log_seen = gr.State(None)
    log_timer = gr.Timer(2.0)

    # ---------------- wiring ----------------
    # Force-play the latest <audio> after a user click (browsers block silent autoplay;
    # Gradio's autoplay flag alone is unreliable across versions).
    _play = "() => { if (window.ogForcePlay) window.ogForcePlay(); }"
    _scroll = ("() => { if (window.innerWidth < 1024) { const a = document.querySelector('.og-answer'); "
               "if (a) a.scrollIntoView({behavior: 'smooth', block: 'center'}); }"
               " if (window.ogForcePlay) window.ogForcePlay(); }")

    run_a.click(detect, audio_a, card_a, show_progress="minimal")
    run_a.click(detail_a, audio_a, [table_a, cross_a])
    run_b.click(assist, audio_b, [card_b, transcript_b, message_b, speech_b]).then(None, None, None, js=_play)
    run_b.click(crosscheck_b, audio_b, cross_b)
    for _btn, _kind, _target in ((real_a, "real", audio_a), (fake_a, "fake", audio_a), (real_b, "real", audio_b), (fake_b, "fake", audio_b)):
        _btn.click(lambda k=_kind: load_example(k), None, _target)

    mode_v.change(set_mode, mode_v, [mode_card_v, topbar])
    mode_v.change(None, mode_v, None, js=MODE_JS)
    appearance_v.change(None, appearance_v, None, js="(v) => { ogSetTheme(v); }")
    large_v.change(None, large_v, None, js="(v) => document.body.classList.toggle('og-large', !!v)")
    refresh_btn.click(system_html, None, sys_html)

    _assist_in = [mic_v, q_text, attach_v, speed_v, simple_v, aloud_v, region_v, assist_state]
    _assist_out = [answer_v, speech_v, cross_v, heard_v, hist_v, memory_v, assist_state, mic_v, q_text, speed_v, simple_v, signal_v]
    mic_v.stop_recording(assistant_turn, _assist_in, _assist_out).then(None, None, None, js=_scroll)
    ask_btn.click(assistant_turn, _assist_in, _assist_out).then(None, None, None, js=_scroll)
    q_text.submit(assistant_turn, _assist_in, _assist_out).then(None, None, None, js=_scroll)
    _quick_in = [attach_v, speed_v, simple_v, aloud_v, region_v, assist_state]
    for _btn, _phrase in ((sos_a, "I think I am being scammed"), (sos_b, "I already sent money"), (flow_done, "done"),
                          (flow_skip, "skip"), (flow_back, "back"), (flow_stop, "stop"), (chip_help, "What can you do?"),
                          (chip_more, "tell me more"), (chip_repeat, "repeat that"), (clear_mem, "clear memory")):
        _btn.click(quick_turn(_phrase), _quick_in, _assist_out).then(None, None, None, js=_scroll)
    report_btn.click(make_report, [assist_state, region_v], report_file)

    listen_mode.change(toggle_listen_mode, listen_mode, [mic_v, mic_stream, pause_v, listen_status])
    mic_stream.stream(listen_chunk, [mic_stream, listen_state, pause_v, attach_v, speed_v, simple_v, aloud_v, region_v, assist_state],
                      [listen_state, listen_status] + _assist_out, stream_every=0.5, time_limit=None, show_progress="hidden",
                      concurrency_limit=1)

    log_timer.tick(log_tick, [log_filter, log_seen], [dock_sum, log_html, log_seen])
    log_filter.change(lambda f: LOG.render_html(f), log_filter, log_html)
    export_btn.click(export_log, None, log_file)
    clear_btn.click(clear_log, None, [dock_sum, log_html, log_filter])
    demo.load(None, None, appearance_v, js="() => ogSavedTheme()")
    demo.load(None, mode_v, None, js=MODE_JS)
    demo.load(lambda: (topbar_html(), system_html(), mode_card_html(rt.current_profile())), None, [topbar, sys_html, mode_card_v])


if __name__ == "__main__":
    # Gradio 6: theme, css and head belong to launch(), not gr.Blocks().
    demo.queue().launch(theme=THEME, css=CSS, head=HEAD)
