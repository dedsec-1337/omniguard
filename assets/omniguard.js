/* OMNIGUARD front-end helpers: theme, motion level, navigation, Channel Log dock, keyboard shortcuts and a
   persistent screen-reader live region. Plain JavaScript, no dependencies, no network. */
(function () {
  "use strict";
  var MOTION = { fast: "none", balanced: "soft", thorough: "full" };
  var THEMES = ["dark", "light", "contrast"];
  function body() { return document.body; }
  function store(key, value) { try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch (e) { return null; } }

  /* ---- appearance: dark (default) / light / high contrast -------------------------------------------- */
  window.ogSavedTheme = function () { var t = store("og-theme"); return THEMES.indexOf(t) >= 0 ? t : "dark"; };
  window.ogSetTheme = function (name) {
    if (THEMES.indexOf(name) < 0) name = "dark";
    var b = body(); if (!b) return name;
    b.dataset.ogTheme = name;
    b.classList.toggle("dark", name !== "light");   // Gradio's own widgets follow
    b.classList.toggle("og-contrast", name === "contrast");
    store("og-theme", name);
    return name;
  };
  window.ogCycleTheme = function () {
    var next = THEMES[(THEMES.indexOf(body().dataset.ogTheme || "dark") + 1) % THEMES.length];
    window.ogSetTheme(next);
    var radio = document.querySelector('.og-appearance input[value="' + next + '"]');
    if (radio) radio.click();
    return next;
  };

  /* ---- motion level follows the mode: none / soft / full ------------------------------------------------ */
  window.ogSetMotion = function (mode) { var b = body(); if (b) b.dataset.ogMotion = MOTION[mode] || "soft"; return mode; };

  /* ---- navigation: our side rail drives Gradio's (hidden) tab strip ------------------------------------- */
  function tabButtons() { return Array.prototype.slice.call(document.querySelectorAll('#og-tabs [role="tab"], .og-hidden-strip [role="tab"]')); }
  window.ogNav = function (i) {
    var tabs = tabButtons(); if (!tabs[i]) return;
    tabs[i].click(); syncNav();
    var main = document.getElementById("og-main"); if (main && window.innerWidth < 1024) main.scrollIntoView({ block: "start" });
  };
  function syncNav() {
    var tabs = tabButtons();
    var active = tabs.findIndex(function (t) { return t.getAttribute("aria-selected") === "true"; });
    document.querySelectorAll(".og-nav-item").forEach(function (el, i) {
      var on = i === active; el.classList.toggle("is-active", on);
      if (on) el.setAttribute("aria-current", "page"); else el.removeAttribute("aria-current");
    });
  }

  /* ---- Channel Log dock (the open state lives on the dock element itself) -------------------------------------- */
  window.ogToggleDock = function (force) {
    var dock = document.querySelector(".og-dock"); if (!dock) return false;
    var open = typeof force === "boolean" ? force : !dock.classList.contains("is-open");
    dock.classList.toggle("is-open", open); body().classList.toggle("og-dock-open", open);
    document.querySelectorAll(".og-dock-toggle").forEach(function (el) { el.setAttribute("aria-expanded", String(open)); });
    return open;
  };
  window.ogStopAudio = function () { document.querySelectorAll("audio").forEach(function (a) { try { a.pause(); } catch (e) {} }); };

  /* Force-play the newest spoken-alert / spoken-answer audio after a user click.
     Gradio autoplay alone is blocked by browsers and often fails on the 2nd update. */
  window.ogForcePlay = function () {
    var tries = 0;
    function attempt() {
      tries += 1;
      var nodes = document.querySelectorAll(
        "#og-speech-b audio, #og-speech-v audio, .og-audio audio, audio"
      );
      var a = null;
      for (var i = nodes.length - 1; i >= 0; i--) {
        if (nodes[i] && nodes[i].src) { a = nodes[i]; break; }
      }
      if (!a) {
        if (tries < 12) setTimeout(attempt, 120);
        return;
      }
      try {
        a.muted = false;
        a.volume = 1;
        a.currentTime = 0;
        var p = a.play();
        if (p && typeof p.catch === "function") {
          p.catch(function () {
            /* still blocked — user can press the player controls */
          });
        }
      } catch (e) {}
    }
    setTimeout(attempt, 80);
  };

  /* ---- keyboard: Alt+1/2/3 pages, Alt+M microphone, Alt+L log, Alt+T theme, "/" question box, Esc ------------ */
  function typing(el) { return el && (/^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) || el.isContentEditable); }
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { if (document.querySelector(".og-dock.is-open")) window.ogToggleDock(false); return; }
    if (e.altKey && !e.ctrlKey && !e.metaKey) {
      if (e.key >= "1" && e.key <= "3") { e.preventDefault(); window.ogNav(parseInt(e.key, 10) - 1); }
      else if (e.key.toLowerCase() === "l") { e.preventDefault(); window.ogToggleDock(); }
      else if (e.key.toLowerCase() === "t") { e.preventDefault(); window.ogCycleTheme(); }
      else if (e.key.toLowerCase() === "m") {
        e.preventDefault();
        var rec = Array.prototype.slice.call(document.querySelectorAll("button")).find(function (b) { return b.offsetParent && /record|start/i.test((b.getAttribute("aria-label") || "") + " " + b.innerText); });
        if (rec) rec.focus();
      }
    } else if (e.key === "/" && !typing(document.activeElement)) {
      var box = Array.prototype.slice.call(document.querySelectorAll("textarea")).find(function (t) { return t.offsetParent; });
      if (box) { e.preventDefault(); box.focus(); }
    }
  });

  /* ---- a persistent live region: Gradio updates cards in place (it morphs the DOM), so a card's own aria-live is
          unreliable. We watch for changes inside result/answer cards and speak a summary through one permanent region. ---- */
  var live = null, timer = null, lastText = "", lastAt = 0;
  function announce(text) {
    if (!live || !text) return;
    var now = Date.now(); if (text === lastText && now - lastAt < 1500) return;
    lastText = text; lastAt = now; live.textContent = "";
    clearTimeout(timer); timer = setTimeout(function () { live.textContent = text; }, 120);
  }
  var WATCH = ".og-result, .og-answer, .og-cross, .og-notice, .og-emerg";
  function onMutations(muts) {
    var picked = null, stop = false;
    muts.forEach(function (m) {
      if (m.type === "attributes" && m.attributeName === "data-og-stop") { stop = true; return; }
      var target = m.target && m.target.nodeType === 1 ? m.target : (m.target && m.target.parentElement);
      if (target && target.closest && m.type !== "attributes") { var c = target.closest(WATCH); if (c) picked = c; }
      if (m.type === "attributes" && target && target.matches && target.matches(WATCH)) picked = target;
      (m.addedNodes || []).forEach(function (n) {
        if (n.nodeType !== 1) return;
        if ((n.matches && n.matches("[data-og-stop]")) || (n.querySelector && n.querySelector("[data-og-stop]"))) stop = true;
        var hit = (n.matches && n.matches(WATCH)) ? n : (n.querySelector ? n.querySelector(WATCH) : null);
        if (hit) picked = hit;
      });
    });
    if (stop) window.ogStopAudio();
    if (picked) announce(picked.innerText.replace(/\s+/g, " ").trim());
    syncNav();
  }

  function boot() {
    if (document.getElementById("og-live")) return;
    live = document.createElement("div");
    live.id = "og-live"; live.setAttribute("role", "status"); live.setAttribute("aria-live", "polite"); live.setAttribute("aria-atomic", "true");
    live.className = "og-sr-only"; document.body.appendChild(live);
    new MutationObserver(onMutations).observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ["class", "data-og-stop"] });
    new MutationObserver(function () {   // Gradio toggles its own dark class at startup: keep ours in charge
      var t = body().dataset.ogTheme; if (!t) return;
      var wantDark = t !== "light"; if (body().classList.contains("dark") !== wantDark) body().classList.toggle("dark", wantDark);
    }).observe(document.body, { attributes: true, attributeFilter: ["class"] });
    window.ogSetTheme(window.ogSavedTheme());
    window.ogSetMotion("balanced"); syncNav();
  }
  var tries = 0, wait = setInterval(function () {
    if (document.body && tabButtons().length) { clearInterval(wait); boot(); }
    else if (++tries > 120) { clearInterval(wait); if (document.body) boot(); }
  }, 100);
})();
