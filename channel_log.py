"""Channel Log: a small, thread-safe activity log shared by the whole app.

Design goals: tiny (a bounded deque, no disk, no network), cheap to render (at most `limit` rows of
escaped HTML), and useful to read, filter and export. Nothing in here ever blocks a model call.
"""
import html
import itertools
import threading
import time
from collections import deque
from datetime import datetime

# Filter name shown in the UI -> event category (None = everything).
FILTERS = {"All": None, "Detections": "detection", "Voice": "voice", "System": "system", "Alerts": "alert"}
CATEGORY_LABEL = {"detection": "Detection", "voice": "Voice", "system": "System", "alert": "Alert"}
LEVELS = ("info", "ok", "warn", "error", "alert")


class ChannelLog:
    def __init__(self, maxlen=500):
        self._events = deque(maxlen=maxlen)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.version = 0  # bumps on every change, so the UI can skip redundant refreshes

    # -- writing ---------------------------------------------------------------------------
    def add(self, category, title, detail="", level="info"):
        if category not in CATEGORY_LABEL:
            category = "system"
        if level not in LEVELS:
            level = "info"
        event = {"id": next(self._ids), "ts": time.time(), "category": category, "level": level,
                 "title": str(title)[:160], "detail": str(detail)[:400]}
        with self._lock:
            self._events.append(event)
            self.version += 1
        return event

    def clear(self):
        with self._lock:
            self._events.clear()
            self.version += 1

    # -- reading ----------------------------------------------------------------------------
    def events(self, filter_name="All", limit=150):
        """Newest first."""
        wanted = FILTERS.get(filter_name)
        with self._lock:
            items = list(self._events)
        if wanted:
            items = [e for e in items if e["category"] == wanted]
        return items[::-1][:limit]

    def counts(self):
        with self._lock:
            items = list(self._events)
        out = {name: 0 for name in FILTERS}
        out["All"] = len(items)
        for e in items:
            for name, cat in FILTERS.items():
                if cat == e["category"]:
                    out[name] += 1
        return out

    # -- rendering ----------------------------------------------------------------------------
    @staticmethod
    def _clock(ts):
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

    def render_html(self, filter_name="All", limit=150):
        items = self.events(filter_name, limit)
        if not items:
            return ('<div class="og-log" role="log" aria-label="Channel log" tabindex="0">'
                    '<p class="og-log-empty">Nothing here yet. Detections, assistant conversations, system events '
                    'and safety alerts will appear as they happen.</p></div>')
        rows = []
        for e in items:
            detail = f'<span class="og-log-detail">{html.escape(e["detail"])}</span>' if e["detail"] else ""
            rows.append(
                f'<li class="og-log-row og-lv-{e["level"]}" data-cat="{e["category"]}">'
                f'<time>{self._clock(e["ts"])}</time>'
                f'<span class="og-log-cat og-cat-{e["category"]}">{CATEGORY_LABEL[e["category"]]}</span>'
                f'<span class="og-log-main"><b>{html.escape(e["title"])}</b>{detail}</span></li>')
        return ('<ul class="og-log" role="log" aria-label="Channel log, newest first" aria-live="off" tabindex="0">'
                + "".join(rows) + "</ul>")

    def summary_html(self):
        """One-line preview for the collapsed dock: newest event plus counts."""
        items = self.events("All", 1)
        c = self.counts()
        if items:
            e = items[0]
            newest = (f'<span class="og-dock-new og-lv-{e["level"]}"><time>{self._clock(e["ts"])}</time> '
                      f'{html.escape(e["title"])}</span>')
        else:
            newest = '<span class="og-dock-new">No activity yet</span>'
        alerts = (f'<span class="og-dock-alerts">{c["Alerts"]} alert{"s" if c["Alerts"] != 1 else ""}</span>'
                  if c["Alerts"] else "")
        return f'<div class="og-dock-sum">{newest}<span class="og-dock-count">{c["All"]} events</span>{alerts}</div>'

    def export_text(self, header=True):
        with self._lock:
            items = list(self._events)
        lines = []
        if header:
            lines += ["OMNIGUARD channel log", f"Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      f"{len(items)} events (oldest first)", "-" * 60]
        for e in items:
            stamp = datetime.fromtimestamp(e["ts"]).strftime("%Y-%m-%d %H:%M:%S")
            tail = f" - {e['detail']}" if e["detail"] else ""
            lines.append(f"[{stamp}] [{CATEGORY_LABEL[e['category']].upper()}/{e['level']}] {e['title']}{tail}")
        return "\n".join(lines) + "\n"


LOG = ChannelLog()
