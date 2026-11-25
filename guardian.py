"""OMNIGUARD Guardian: the "words" half of scam detection plus the voice assistant brain.

Design rule: every safety-critical decision is made by transparent code (rules, calibrated
detectors, a small zero-shot model). The optional local LLM (Ollama) only *explains and answers*;
it never sets a verdict, and its output is screened before it is spoken.

Everything runs locally. No paid API, no cloud service.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

REGION = os.environ.get("OMNIGUARD_REGION", "IN").upper()

# --- 1. Scam-language rules (explainable, instant, no download) -----------------------------
# (id, plain-English label, weight, patterns). Basic Hindi/Bengali keywords are included because
# Whisper transcribes those languages in their own script; coverage there is deliberately small.
SIGNALS = [
    ("credentials", "asks for an OTP, PIN, password or card details", 0.60, [
        r"\botp\b", r"one[- ]time (password|pin|code)", r"\b(upi|atm|card|net ?banking|secret) ?pin\b",
        r"\bpassword\b", r"\bcvv\b", r"card number", r"aadhaar|aadhar", r"verification code",
        r"ओटीपी", r"पासवर्ड", r"ওটিপি"]),
    ("money", "asks you to send or transfer money", 0.45, [
        r"\b(send|transfer|wire|pay|deposit)\b[^.?!]{0,40}\b(money|rupees|rs\.?|₹|amount|cash|fees?|fine|deposit)\b",
        r"\b(money|rupees|amount)\b[^.?!]{0,30}\b(send|transfer|wire|pay)\b",
        r"पैसे\s*(भेज|ट्रांसफर|दे)", r"টাকা\s*(পাঠান|পাঠাও|দিন)"]),
    ("urgency", "pushes you to act immediately", 0.25, [
        r"\b(urgent(ly)?|immediately|right now|at once|asap|hurry up|within (the )?(next )?\d+ (minutes?|hours?))\b",
        r"तुरंत", r"এখনই"]),
    ("secrecy", "tells you to keep it secret", 0.40, [
        r"don'?t tell (anyone|anybody|your|mom|dad|mum)", r"keep (this|it) (a )?secret", r"between us",
        r"do not (tell|inform) (anyone|anybody)", r"किसी को मत बताना", r"কাউকে বলবেন না"]),
    ("authority", "claims to be police, a bank, customs or another authority", 0.30, [
        r"\b(police|cbi|cyber ?crime|customs|narcotics|income tax|rbi|telecom authority|trai)\b",
        r"\bbank (manager|officer|executive|representative)\b", r"\bcustomer (care|support)\b",
        r"पुलिस", r"পুলিশ"]),
    ("parcel", "mentions a parcel or courier (weak signal on its own)", 0.15, [
        r"\b(fedex|dhl|courier|parcel)\b"]),
    ("threat", "threatens arrest, legal action or account closure", 0.40, [
        r"\b(arrest(ed)?|warrant|legal action|jail|digital arrest|case (has been|against))\b",
        r"\b(account|sim|card|wallet|fastag)\b[^.?!]{0,20}\b(blocked|suspended|frozen|deactivated|closed)\b",
        r"गिरफ्तार", r"खाता बंद", r"গ্রেপ্তার"]),
    ("family", "claims a relative is in trouble", 0.40, [
        r"\b(i'?m|i am|he|she|your \w+) (is |was |has been |got )?in (serious )?(trouble|an accident|the hospital|jail|danger)\b",
        r"\bhad an accident\b", r"\bin (the )?hospital\b[^.?!]{0,30}\b(money|pay|urgent)\b"]),
    ("remote", "asks you to install a remote-access or screen-sharing app", 0.55, [
        r"\b(anydesk|teamviewer|quicksupport|screen ?shar(e|ing)|remote (access|control))\b",
        r"\binstall (this |the )?app\b"]),
    ("gift", "asks for gift cards, crypto or wire transfer", 0.55, [
        r"\b(gift cards?|google play cards?|itunes cards?|bitcoin|crypto(currency)?|usdt|western union)\b"]),
    ("prize", "promises a prize, refund or cashback", 0.35, [
        r"\b(you('ve| have) (won|been selected)|lottery|lucky draw|cashback|refund)\b"]),
    ("kyc", "mentions a KYC update", 0.45, [r"\bkyc\b", r"केवाईसी", r"কেওয়াইসি"]),
]
_COMPILED = [(sid, label, w, [re.compile(p, re.I | re.UNICODE) for p in pats])
             for sid, label, w, pats in SIGNALS]


def scam_signals(text):
    """Rule-based scam score (0-1) and the signals that fired, with a short evidence snippet."""
    found, miss = [], 1.0
    for sid, label, weight, pats in _COMPILED:
        for pat in pats:
            m = pat.search(text or "")
            if m:
                found.append({"id": sid, "label": label, "weight": weight,
                              "evidence": m.group(0).strip()[:60]})
                miss *= 1.0 - weight
                break
    return {"score": round(min(1.0 - miss, 0.97), 3), "signals": found}


# --- 2. Optional semantic check (small zero-shot model, runs on the CPU) ------------------------
SEMANTIC_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
HYPOTHESES = {
    "money": "The speaker is asking for money or a payment urgently.",
    "credentials": "The speaker is asking for a one-time password, PIN or bank details.",
    "authority": "The speaker claims to be police, a bank, a courier or a government officer.",
    "threat": "The speaker threatens arrest, legal action or closing an account.",
    "secrecy": "The speaker tells the listener to keep this secret.",
    "family": "The speaker pretends to be a relative who is in trouble.",
}


class Semantic:
    """Lazy wrapper around a zero-shot classifier. `load()` never raises."""

    def __init__(self, model_id=SEMANTIC_ID):
        self.model_id, self.pipe, self.error = model_id, None, None

    def load(self):
        if os.environ.get("OMNIGUARD_SEMANTIC", "1") == "0":
            self.error = "disabled (OMNIGUARD_SEMANTIC=0)"
            return False
        try:
            from transformers import pipeline
            self.pipe = pipeline("zero-shot-classification", model=self.model_id, device=-1)
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def score(self, text):
        """{hypothesis id: probability} or None when unavailable."""
        if self.pipe is None or not (text or "").strip():
            return None
        try:
            ids = list(HYPOTHESES)
            out = self.pipe(text[:1200], candidate_labels=[HYPOTHESES[i] for i in ids],
                            hypothesis_template="{}", multi_label=True)
            by_label = dict(zip(out["labels"], out["scores"]))
            return {i: float(by_label[HYPOTHESES[i]]) for i in ids}
        except Exception:  # noqa: BLE001
            return None


def assess_text(text, semantic=None):
    """Combine the rules with the optional semantic model (rules stay the backbone)."""
    rules = scam_signals(text)
    sem = semantic.score(text) if semantic is not None else None
    score = rules["score"]
    if sem:
        top = max(sem.values())
        if top >= 0.7:  # the semantic model may only add evidence, and only when it is confident
            score = 1.0 - (1.0 - score) * (1.0 - 0.5 * top)
    return {"score": round(min(score, 0.97), 3), "signals": rules["signals"], "semantic": sem}


# --- 3. Referee: fuse "how the voice sounds" with "what the voice says" --------------------------
def fuse(acoustic_p, content_score):
    """Noisy-OR of the two independent signals. An AI-sounding voice alone tops out at 'high'
    (legitimate synthetic voices exist); it needs scam wording to reach 'critical'."""
    pa = None if acoustic_p is None else max(0.0, min(1.0, float(acoustic_p)))
    cs = max(0.0, min(1.0, float(content_score or 0.0)))
    risk = 1.0 - (1.0 - 0.8 * (pa or 0.0)) * (1.0 - cs)
    level = ("critical" if risk >= 0.85 else "high" if risk >= 0.60
             else "elevated" if risk >= 0.30 else "low")
    ai, words = (pa or 0) >= 0.70, cs >= 0.50
    if pa is None:
        story = "Only the words could be checked: the voice detectors are unavailable."
    elif ai and words:
        story = "The voice sounds AI-generated and the words match a scam script."
    elif ai:
        story = "The voice sounds AI-generated, but no scam wording was heard."
    elif words and (pa or 0) < 0.30:
        story = "The voice sounds real, but the words match a scam script. Human scammers exist too."
    elif words:
        story = "The voice is uncertain and the words match a scam script."
    else:
        story = "No strong sign in the voice or the words."
    advice = {
        "critical": "Hang up now. Do not send money or share codes. Call a family member or your bank on a number you know.",
        "high": "Do not send money or share codes. Hang up and call back on a number you already know.",
        "elevated": "Be careful. Verify by calling back on a number you already know before doing anything.",
        "low": "Nothing alarming, but always verify unexpected requests by calling back on a known number.",
    }[level]
    return {"risk": round(risk, 3), "level": level, "story": story, "advice": advice}


# --- 4. Local LLM through Ollama (optional; everything works without it) -------------------------
LLM_SYSTEM = (
    "You are Guardian, a calm voice assistant that helps people, including seniors and people with "
    "disabilities, stay safe from phone and message scams. Use short sentences and plain words. "
    "Never ask for or accept an OTP, PIN, password or Aadhaar number. Never tell the user to send "
    "money. If something may be a scam, tell them to hang up and call back on a number they already "
    "know. Use only the facts you are given about any analysis; if you do not know, say so. "
    "Do not use markdown, lists or emoji.")


def speakable(text):
    """Clean text for the TTS voice: no markdown/emoji, digits of helpline numbers spoken one by one."""
    t = re.sub(r"[*_`#>]+", "", text or "")
    t = re.sub(r"[^\w\s.,;:!?'\"()%/@₹+\-\u0900-\u097F\u0980-\u09FF]", " ", t)
    # Phone numbers and short helpline numbers are read digit by digit ("1930" -> "1 9 3 0").
    t = re.sub(r"\b0\d{3} \d{3} \d{4}\b", lambda m: ", ".join(" ".join(p) for p in m.group(0).split()), t)
    t = re.sub(r"\b\d[\d-]{2,}\d\b", lambda m: ", ".join(" ".join(p) for p in m.group(0).split("-")), t)
    t = re.sub(r"\b\d{3,4}\b", lambda m: " ".join(m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()


_UNSAFE = re.compile(
    r"\b(share|give|tell|send|provide|read out)\b[^.?!]{0,25}\b(otp|pin|password|cvv)\b"
    r"|\b(send|transfer|wire|pay)\b[^.?!]{0,25}\b(the )?(money|amount|fee|fine)\b", re.I)
_NEGATION = re.compile(r"\b(never|not|don'?t|do not|won'?t|shouldn'?t|must not|no one|nobody|avoid|stop|without)\b|n't", re.I)


def unsafe_advice(text):
    """True if a generated answer tells the user to hand over a code or send money.
    A negation earlier in the same sentence ("never share your OTP") makes it safe."""
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        for m in _UNSAFE.finditer(sentence):
            if not _NEGATION.search(sentence[:m.start()]):
                return True
    return False


class Ollama:
    """Tiny client for a local Ollama server. Uses only the standard library."""

    def __init__(self, host=None, model=None, timeout=60):
        host = host or os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434"
        if "://" not in host:
            host = "http://" + host
        self.base = host.replace("0.0.0.0", "127.0.0.1").rstrip("/")
        self.model = model or os.environ.get("OMNIGUARD_LLM", "qwen2.5:1.5b")
        self.timeout = timeout
        self.disabled = self.model.lower() in ("off", "none", "0", "")
        self._cache = (0.0, False)

    def available(self):
        """True if the server answers and has the model. Cached for 20 s (never blocks long)."""
        if self.disabled:
            return False
        stamp, value = self._cache
        if time.time() - stamp < 20:
            return value
        ok = False
        try:
            with urllib.request.urlopen(self.base + "/api/tags", timeout=0.8) as resp:
                names = [m.get("name", "") for m in json.loads(resp.read().decode()).get("models", [])]
            ok = any(n == self.model or n.split(":")[0] == self.model or n.startswith(self.model) for n in names)
        except Exception:  # noqa: BLE001
            ok = False
        self._cache = (time.time(), ok)
        return ok

    def chat(self, messages, max_tokens=160, temperature=0.2):
        """Reply text, or None on any failure."""
        if not self.available():
            return None
        body = json.dumps({"model": self.model, "messages": messages, "stream": False,
                           "keep_alive": "2m",
                           "options": {"temperature": temperature, "num_predict": max_tokens,
                                       "num_ctx": 2048}}).encode()
        req = urllib.request.Request(self.base + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())["message"]["content"].strip() or None
        except Exception:  # noqa: BLE001
            self._cache = (0.0, False)
            return None


def llm_answer(llm, question, facts="", simple=True, max_tokens=120, history=()):
    """Ask the local LLM; returns screened plain text or None (caller then uses the rule-based answer).
    `history` is the last few (question, answer) pairs so follow-ups make sense."""
    if llm is None or max_tokens <= 0:
        return None
    limit = 3 if simple else 5
    context = f"Facts from OMNIGUARD's analysis: {facts}\n" if facts else ""
    style = "Use very simple words and at most 3 short sentences." if simple else "Answer in at most 5 short sentences."
    messages = [{"role": "system", "content": LLM_SYSTEM}]
    for q, a in list(history)[-3:]:
        messages += [{"role": "user", "content": str(q)[:300]}, {"role": "assistant", "content": str(a)[:400]}]
    messages.append({"role": "user", "content": f"{context}{style}\nQuestion: {question}"})
    reply = llm.chat(messages, max_tokens=max_tokens)
    if not reply:
        return None
    reply = re.sub(r"[*_`#>]+", "", reply).strip()
    sentences = re.split(r"(?<=[.!?])\s+", reply)
    reply = " ".join(sentences[:limit]).strip()
    return None if (not reply or unsafe_advice(reply)) else reply


# --- 5. Regions: who to call -----------------------------------------------------------------------------
# Verified against official or agency pages in November 2026. The UK's Action Fraud was replaced by
# "Report Fraud" (same phone number) in December 2025.
REGIONS = {
    "IN": {"name": "India", "block": "block your card or UPI",
           "short": "Call 1930. This is the cyber crime helpline. Or go to cybercrime.gov.in.",
           "long": "Call the national cyber crime helpline 1930, or report at cybercrime.gov.in."},
    "US": {"name": "United States", "block": "block your card and stop the payment",
           "short": "Go to ReportFraud.ftc.gov, or call 1-877-382-4357. For online fraud also use ic3.gov.",
           "long": "Report it to the FTC at ReportFraud.ftc.gov or on 1-877-382-4357, and to the FBI's IC3 at ic3.gov if it happened online."},
    "GB": {"name": "United Kingdom", "block": "block your card and stop the payment",
           "short": "Go to reportfraud.police.uk, or call 0300 123 2040. In Scotland call 101. To reach your bank's fraud team, dial 159.",
           "long": "Report it to Report Fraud at reportfraud.police.uk or on 0300 123 2040. In Scotland, call Police Scotland on 101. Dial 159 to reach your bank's fraud team."},
    "AU": {"name": "Australia", "block": "block your card and stop the payment",
           "short": "Report it at scamwatch.gov.au and cyber.gov.au. If your identity may be exposed, call IDCARE on 1800 595 160.",
           "long": "Report it to Scamwatch at scamwatch.gov.au and to ReportCyber at cyber.gov.au. If your identity may be exposed, call IDCARE on 1800 595 160."},
    "CA": {"name": "Canada", "block": "block your card and stop the payment",
           "short": "Tell your local police. Then call the Canadian Anti-Fraud Centre on 1-888-495-8501.",
           "long": "Contact your local police first, then report it to the Canadian Anti-Fraud Centre on 1-888-495-8501 or at antifraudcentre-centreantifraude.ca."},
    "XX": {"name": "Other / not sure", "block": "block your card and stop the payment",
           "short": "Report it to your local police and your country's fraud service.",
           "long": "Report it to your local police and to your country's fraud-reporting service."},
}
REGION_CHOICES = [(info["name"], code) for code, info in REGIONS.items()]


def region_info(code=None):
    return REGIONS.get((code or REGION or "XX").upper(), REGIONS["XX"])


def _helpline(simple, region=None):
    info = region_info(region)
    return info["short" if simple else "long"]


# --- 6. Guided emergency flows (one step at a time, each step needs the user's confirmation) -------------
def _flow_steps(name, simple, region):
    r = region_info(region)
    if name == "sent_money":
        return [
            {"text": f"Call your bank now. Ask them to {r['block']} and flag the payment.", "prompt": "Say done when you have called, or skip."},
            {"text": "Report it. " + _helpline(simple, region), "prompt": "Say done when you have reported it, or skip."},
            {"text": "Keep every message, call log and screenshot. Do not delete anything.", "prompt": "Say done to continue."},
            {"text": "Change any password you shared. Start with your email and banking apps.", "prompt": "Say done to continue."},
            {"text": "Do not pay anyone who promises to get your money back. That is often a second scam.", "prompt": "Say done to finish."},
        ]
    return [  # "scammed_now"
        {"text": "If the caller is still on the line, hang up now. Do not call that number back.", "prompt": "Say done when you have hung up."},
        {"text": "Do not click links or install any app they mentioned. Do not share any code, PIN or password.", "prompt": "Say done to continue."},
        {"text": "Have you already shared a code or sent money? Say yes or no.", "prompt": "", "branch": "sent_money"},
        {"text": "Call someone you trust, like a family member, and tell them what happened.", "prompt": "Say done when you have."},
        {"text": "You can check the call here. Attach the recording, or tell me what they said.", "prompt": "Say done to finish."},
    ]


FLOW_INTRO = {
    "sent_money": "I am sorry this happened. We will go one step at a time. Say done after each step, skip to skip one, or stop to end.",
    "scammed_now": "Stay calm. We will go one step at a time. Say done after each step, skip to skip one, or stop to end.",
}
FLOW_DONE = {
    "sent_money": "That was the last step. You took the right actions. You can create a safety report to show your bank or the police.",
    "scammed_now": "That was the last step. You are safe for now. If anything changes, tell me, for example: I already sent money.",
}


def flow_step_text(state, simple=True, region=None):
    steps = _flow_steps(state["name"], simple, region)
    step = steps[state["i"]]
    prompt = (" " + step["prompt"]) if step.get("prompt") else ""
    return f"Step {state['i'] + 1} of {len(steps)}. {step['text']}{prompt}"


def flow_start(name, simple=True, region=None):
    state = {"name": name, "i": 0}
    return FLOW_INTRO[name] + " " + flow_step_text(state, simple, region), state


_FLOW_WORDS = {
    "done": r"done|next|finished|ok|okay|continue|go on|did it|i did|i have|all done|completed",
    "yes": r"yes|yeah|yep|i have|i did|sure",
    "no": r"no|nope|not yet|i have not|i haven't|i did not|i didn't",
    "skip": r"skip|pass|later",
    "back": r"back|previous|go back",
    "stop": r"stop|cancel|quit|end|exit|never ?mind|enough",
    "repeat": r"repeat|again|say again|what",
}


def flow_command(text):
    """Map a short utterance to a flow command (done, yes, no, skip, back, stop, repeat) or None."""
    t = re.sub(r"[^\w' ]", "", (text or "").lower()).strip()
    if not t or len(t.split()) > 4:
        return None
    for cmd in ("stop", "skip", "back", "repeat", "no", "yes", "done"):
        if re.fullmatch(rf"(?:please |ok |okay )?(?:{_FLOW_WORDS[cmd]})(?: please| now)?", t):
            return cmd
    return None


def flow_advance(state, cmd, simple=True, region=None):
    """Apply a flow command. Returns (text, new_state_or_None, started_flow_name_or_None)."""
    steps = _flow_steps(state["name"], simple, region)
    step = steps[state["i"]]
    if cmd == "stop":
        return "Okay, I stopped. I am here whenever you need me.", None, None
    if cmd == "repeat":
        return flow_step_text(state, simple, region), state, None
    if cmd == "back":
        state = {**state, "i": max(0, state["i"] - 1)}
        return flow_step_text(state, simple, region), state, None
    if step.get("branch") and cmd == "yes":
        text, new = flow_start(step["branch"], simple, region)
        return text, new, step["branch"]
    if cmd in ("done", "yes", "no", "skip"):
        if state["i"] + 1 >= len(steps):
            return FLOW_DONE[state["name"]], None, None
        state = {**state, "i": state["i"] + 1}
        return flow_step_text(state, simple, region), state, None
    return flow_step_text(state, simple, region), state, None


# --- 7. The assistant brain -----------------------------------------------------------------------------
def text_lost_money(simple, region=None):
    r = region_info(region)
    if simple:
        return (f"I am sorry this happened. Do this now. First, call your bank. Ask them to {r['block']}. "
                + _helpline(True, region) + " Keep all messages and screenshots. Do not delete them. "
                "Do not pay anyone who says they can get your money back.")
    return (f"I'm sorry this happened. Act fast. First, call your bank and ask them to {r['block']} and flag the "
            "transaction. " + _helpline(False, region) + " Keep every message, call log and screenshot "
            "as evidence. Do not pay anyone who promises to recover your money; that is a common second scam.")


def text_live_call(simple):
    if simple:
        return ("Do not share any code, PIN or password. Do not send money. Hang up. Then call back on a "
                "number you already know. Not the number that called you.")
    return ("Do not share any OTP, PIN or password, and do not send money. Hang up. Then call the person or "
            "the bank back on a number you already know, not the number that just called you.")


def text_help(simple):
    if simple:
        return ("I can help with scam calls and messages. Say: I think I am being scammed. Or: I already sent "
                "money. Or read me a message and ask if it is a scam. You can also attach a recording.")
    return ("I can help you stay safe from scam calls and messages. Try: I think I'm being scammed; I already "
            "sent money; or read me a message and ask whether it is a scam. You can also attach a recording "
            "and I will check the voice and the words. Say slower, faster, simple words, repeat, stop or "
            "clear memory at any time.")


def text_howto(simple):
    if simple:
        return ("To check a recording, attach it here. Or open the Deepfake Detector tab. "
                "Then press the big button. I will tell you if it is safe.")
    return ("To check a recording, attach it below or use the Deepfake Detector tab and press the big button. "
            "I will check how the voice sounds and what it says, and then tell you the risk.")


def text_unknown(simple):
    return "I am not sure how to help with that. " + text_help(simple)


def text_text_analysis(res, simple):
    score, sigs = res["score"], res["signals"]
    reasons = "; ".join(s["label"] for s in sigs[:3])
    if score >= 0.65:
        if simple:
            return ("This looks like a scam. " + reasons.capitalize() + ". Do not reply. Do not click links. "
                    "Do not share codes. Call your bank on a number you know.")
        return (f"This looks like a scam. It {reasons}. Do not reply, click links or share codes. "
                "Contact your bank or the real company using a number you already know.")
    if score >= 0.35:
        if simple:
            return "This has some scam signs. " + reasons.capitalize() + ". Be careful. Check on a number you know."
        return (f"This has some signs of a scam: it {reasons}. Be careful, and verify by contacting the sender "
                "on a number you already know.")
    return ("I do not see clear scam signs, but I cannot be sure. If it asks for money or codes, or says it is "
            "urgent, stop and check with the sender on a number you know.")


def text_explain_last(last, simple):
    if not last or not last.get("summary"):
        return ("I do not have a result yet. Attach a recording below, or use the Deepfake Detector tab, "
                "and then ask me again.")
    return last["summary_simple" if simple else "summary"]


_INTENTS = [  # (intent, pattern) in priority order; the first match wins
    ("clear_memory", r"\b(clear|erase|forget|delete|reset)\b.{0,25}\b(memory|history|conversation|everything|chat)\b"),
    ("lost_money", r"\b(sent|send|paid|transferred|gave|shared|lost|gone|stolen)\b.{0,40}\b(money|rupees|rs|otp|pin|password|bank|upi|card|details)\b"
                   r"|\b(i was|i got|i have been|i've been) (scammed|cheated|defrauded|tricked)\b|\balready (paid|sent|shared)\b"),
    ("live_call", r"\b(on a call|on the phone|calling me|called me|someone is calling|phone call|caller)\b.{0,60}\b(scam|money|otp|bank|police|asking|urgent|threat)"
                  r"|\b(being|getting) scammed\b|\bthink (i am|i'm|it is|it's) (a )?scam"),
    ("more", r"\b(tell me more|more detail|more details|explain more|expand|full (answer|plan|steps)|all the steps)\b|^go on$"),
    ("repeat", r"\b(repeat|say (that )?again|again please|come again)\b"),
    ("slower", r"\b(slower|slow down|too fast)\b"),
    ("faster", r"\b(faster|speed up|too slow)\b"),
    ("simpler", r"\b(simple|simpler|easy|easier|plain|short|shorter) (words|language|english|answers?)\b|\bexplain (it )?simply\b|\bi do not understand\b|\bi don't understand\b"),
    ("normal", r"\b(normal|full|detailed|longer) (words|language|answers?)\b"),
    ("help", r"\b(what can you do|help me|what do you do|who are you|how do you work)\b|^help$|^hello$|^hi$"),
    ("howto", r"\bhow (do|can) i (check|use|test|analy[sz]e)\b|\bhow does this work\b"),
    ("explain_last", r"\b(is|was) (this|that|my|the) (last )?(call|recording|voice|clip|result)\b|\bwhat does (the|this|that) result\b|\bwhy (red|yellow|green|high risk)\b|\bexplain (the|this|that|my) result\b"),
]


def interpret(text):
    """Map what the user said to an intent (keyword rules: fast, predictable, offline)."""
    t = (text or "").lower().strip()
    if not t:
        return "empty"
    if re.fullmatch(r"(please )?(stop|be quiet|quiet|silence|enough|cancel)( please)?", re.sub(r"[^\w' ]", "", t)):
        return "stop"
    for intent, pattern in _INTENTS:
        if re.search(pattern, t):
            return intent
    if re.search(r"\b(is this|is it|check (this|the|my)|read this|look at this)\b.{0,30}\b(scam|fake|real|safe|fraud|message|sms|email|text)\b", t) \
            or len(t.split()) >= 8:
        return "analyze_text"
    return "unknown"


def reply_to(text, *, simple=True, last=None, llm=None, semantic=None, facts="", region=None,
             flow=None, history=(), max_tokens=120):
    """Return {"text", "intent", "settings", "flow", "analysis"?, "alert"?}.
    Deterministic rules answer every safety-critical intent; the LLM only handles open questions.
    An active guided flow only advances when the user confirms (done / skip / back / stop)."""
    out = {"settings": {}, "flow": flow, "alert": None}

    # 1. A guided flow is running: commands move it, anything else is answered and the step is repeated.
    if flow:
        cmd = flow_command(text)
        if cmd:
            out["intent"] = "flow"
            out["text"], out["flow"], started = flow_advance(flow, cmd, simple, region)
            if cmd == "stop":
                out["settings"] = {"stop_audio": True}  # Stop also silences anything still playing
            if started:
                out["alert"] = f"Emergency flow started: {started}"
            return out

    intent = interpret(text)
    out["intent"] = intent
    if intent == "empty":
        out["text"] = "I did not hear anything. Please try again, or type your question."
    elif intent == "stop":
        out["settings"] = {"stop_audio": True}
        out["flow"] = None
        out["text"] = "Okay, I stopped." + (" The guided steps are ended." if flow else "")
    elif intent == "clear_memory":
        out["settings"] = {"clear_memory": True}
        out["flow"] = None
        out["text"] = "Done. I cleared my memory of this conversation."
    elif intent == "lost_money":
        out["text"], out["flow"] = flow_start("sent_money", simple, region)
        out["alert"] = "Emergency flow started: I already sent money"
    elif intent == "live_call":
        out["text"], out["flow"] = flow_start("scammed_now", simple, region)
        out["alert"] = "Emergency flow started: I think I'm being scammed"
    elif intent == "more":
        before = (last or {}).get("intent")
        if before == "lost_money":
            out["text"] = text_lost_money(False, region)
        elif before == "live_call":
            out["text"] = text_live_call(False)
        elif (last or {}).get("question"):
            return {**reply_to(last["question"], simple=False, last=last, llm=llm, semantic=semantic, facts=facts,
                               region=region, flow=None, history=history, max_tokens=max_tokens), "intent": "more"}
        else:
            out["text"] = "There is nothing to expand yet. " + text_help(False)
    elif intent == "help":
        out["text"] = text_help(simple)
    elif intent == "howto":
        out["text"] = text_howto(simple)
    elif intent == "explain_last":
        out["text"] = text_explain_last(last, simple)
    elif intent == "repeat":
        out["text"] = (last or {}).get("reply") or "There is nothing to repeat yet."
    elif intent == "slower":
        out["settings"] = {"speed_delta": -0.2}
        out["text"] = "Okay, I will speak more slowly."
    elif intent == "faster":
        out["settings"] = {"speed_delta": +0.2}
        out["text"] = "Okay, I will speak a little faster."
    elif intent == "simpler":
        out["settings"] = {"simple": True}
        out["text"] = "Okay. I will keep answers short and simple from now on."
    elif intent == "normal":
        out["settings"] = {"simple": False}
        out["text"] = "Okay. I will give fuller answers from now on."
    elif intent == "analyze_text":
        res = assess_text(text, semantic)
        out["analysis"] = res
        if res["score"] >= 0.65:
            out["alert"] = "Message looks like a scam"
        if res["score"] >= 0.35 or re.search(r"\b(scam|fake|fraud|safe|real)\b", text.lower()):
            out["text"] = text_text_analysis(res, simple)
        else:
            info = (facts + " " if facts else "") + f"scam-wording score {res['score']:.2f}"
            out["text"] = llm_answer(llm, text, info, simple, max_tokens, history) or text_unknown(simple)
    else:
        out["text"] = llm_answer(llm, text, facts, simple, max_tokens, history) or text_unknown(simple)

    if flow and out["flow"] is flow and intent not in ("stop", "clear_memory"):
        # an unrelated question during a flow: answer it, then remind where we are
        out["text"] += " " + flow_step_text(flow, simple, region)
    return out


# --- 8. Safety report (plain text, created on this computer) ---------------------------------------------
def build_report(info=None, *, region=None, history=(), when=None):
    """A short report the user can show a bank or the police. `info` is engine.analyse_clip()'s result."""
    from datetime import datetime
    r = region_info(region)
    stamp = (when or datetime.now()).strftime("%Y-%m-%d %H:%M")
    lines = ["OMNIGUARD safety report", f"Created: {stamp}", f"Region: {r['name']}", "", "WHAT WAS CHECKED"]
    if info:
        res, words, fused = info.get("res"), info.get("words", {}), info["fused"]
        if res:
            votes = ", ".join(f"{m['name']} {m['p'] * 100:.0f}%" for m in res["models"])
            lines.append(f"- Voice: {res['p'] * 100:.0f}% likely AI-generated ({res['key']}). Detectors: {votes}.")
        else:
            lines.append("- Voice: could not be checked (detectors unavailable).")
        sigs = "; ".join(s["label"] for s in words.get("signals", [])) or "none found"
        lines.append(f"- Scam wording: {sigs}.")
        if info.get("transcript"):
            lines.append(f'- Words heard: "{info["transcript"][:600]}"')
        lines += [f"- Overall risk: {fused['level'].upper()}. {fused['story']}", f"- Advice: {fused['advice']}"]
    else:
        lines.append("- No recording was checked in this session.")
    if history:
        lines += ["", "RECENT QUESTIONS"]
        lines += [f"- {q[:140]}" for q, _ in list(history)[-5:]]
    lines += ["", "WHAT TO DO NOW"]
    lines += [f"{i + 1}. {s['text']}" for i, s in enumerate(_flow_steps("sent_money", False, region))]
    lines += ["", "WHERE TO REPORT", r["long"], "",
              "NOTES", "- This report was created on your own computer. Nothing was sent anywhere.",
              "- No detector is 100% accurate. Always verify by calling back on a number you already know."]
    return "\n".join(lines) + "\n"
