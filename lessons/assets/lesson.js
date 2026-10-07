/* lesson.js — the shared lesson shell. A lesson file calls S2S.lesson({...}) once; see lessons/AUTHORING.md.
   Builds: top bar (home, title, progress, tabs) · Learn (rail | story | pinned stage) · Simulate · Practice.
   Drawing: each step's scene(G) draws on a 640×450 dark stage; sim.draw(G, values, t) draws on a 640×260 chart. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const $ = (s, r) => (r || document).querySelector(s);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage blocked: progress just isn't remembered */ } },
  };

  /* ---------------- drawing API (G) ---------------- */
  function makeG(svg, prefix) {
    const timers = [];
    const c = (name) => (name && name[0] === "#" ? name : cssVar(`--${prefix}-${name}`) || name);
    const el = (tag, attrs, parent) => {
      const n = document.createElementNS(NS, tag);
      for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
      (parent || svg).appendChild(n);
      return n;
    };
    const G = {
      svg, c, el,
      W: +svg.viewBox.baseVal.width, H: +svg.viewBox.baseVal.height,
      clear() { timers.splice(0).forEach((t) => { clearTimeout(t); clearInterval(t); }); if (window.gsap) gsap.killTweensOf(svg.querySelectorAll("*")); while (svg.firstChild) svg.removeChild(svg.firstChild); },
      group(o = {}) { return el("g", { opacity: o.opacity, transform: o.transform }, o.parent); },
      rect(x, y, w, h, o = {}) {
        return el("rect", { x, y, width: Math.max(0, w), height: Math.max(0, h), rx: o.rx ?? 6, fill: o.fill ? c(o.fill) : "none",
          stroke: o.stroke ? c(o.stroke) : undefined, "stroke-width": o.sw ?? (o.stroke ? 1.5 : undefined), "stroke-dasharray": o.dash,
          opacity: o.opacity }, o.parent);
      },
      text(x, y, s, o = {}) {
        const t = el("text", { x, y, "font-size": o.size ?? 13, fill: c(o.color ?? "ink"), "text-anchor": o.anchor ?? "start",
          "font-weight": o.weight, opacity: o.opacity, "dominant-baseline": o.baseline }, o.parent);
        t.textContent = s; return t;
      },
      line(x1, y1, x2, y2, o = {}) {
        return el("line", { x1, y1, x2, y2, stroke: c(o.color ?? "muted"), "stroke-width": o.w ?? 1.5, "stroke-dasharray": o.dash,
          "stroke-linecap": o.cap ?? "round", opacity: o.opacity, "marker-end": o.arrow ? G._marker(o.color ?? "muted") : undefined }, o.parent);
      },
      arrow(x1, y1, x2, y2, o = {}) { return G.line(x1, y1, x2, y2, Object.assign({ arrow: true }, o)); },
      path(d, o = {}) {
        return el("path", { d, fill: o.fill ? c(o.fill) : "none", stroke: o.color ? c(o.color) : undefined, "stroke-width": o.w ?? 1.5,
          "stroke-dasharray": o.dash, opacity: o.opacity, "marker-end": o.arrow ? G._marker(o.color ?? "muted") : undefined }, o.parent);
      },
      circle(cx, cy, r, o = {}) {
        return el("circle", { cx, cy, r, fill: o.fill ? c(o.fill) : "none", stroke: o.stroke ? c(o.stroke) : undefined,
          "stroke-width": o.sw ?? 1.5, opacity: o.opacity }, o.parent);
      },
      /* a labelled box; fill + text color chosen for contrast */
      box(x, y, w, h, label, o = {}) {
        const g = G.group({ parent: o.parent });
        G.rect(x, y, w, h, { fill: o.fill, stroke: o.stroke ?? (o.fill ? undefined : "ink"), rx: o.rx ?? 8, dash: o.dash, opacity: o.boxOpacity, parent: g, sw: o.sw });
        if (label !== undefined && label !== "") G.text(x + w / 2, y + h / 2 + (o.size ?? 13) * 0.36, label, { anchor: "middle", size: o.size ?? 13,
          color: o.color ?? (o.fill ? "bg" : "ink"), weight: o.weight, parent: g });
        return g;
      },
      /* a row of equal boxes; returns the groups */
      row(x, y, labels, o = {}) {
        const w = o.w ?? 80, h = o.h ?? 36, gap = o.gap ?? 10;
        return labels.map((l, i) => G.box(x + i * (w + gap), y, w, h, l, Object.assign({}, o, { fill: typeof o.fill === "function" ? o.fill(i) : o.fill })));
      },
      /* grid of cells: fill(r,c) → color name or null */
      grid(x, y, rows, cols, cw, ch, fill, o = {}) {
        const out = [];
        for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
          const f = fill(r, k); if (!f) continue;
          out.push(G.rect(x + k * cw, y + r * ch, cw - (o.gap ?? 2), ch - (o.gap ?? 2), { fill: f, rx: o.rx ?? 3, opacity: o.opacity }));
        }
        return out;
      },
      /* vertical bars from baseline y0; values in [0, max] */
      bars(x, y0, values, o = {}) {
        const w = o.w ?? 14, gap = o.gap ?? 4, hmax = o.h ?? 150, max = o.max ?? Math.max(...values, 1);
        return values.map((v, i) => G.rect(x + i * (w + gap), y0 - (hmax * v) / max, w, (hmax * v) / max,
          { fill: typeof o.fill === "function" ? o.fill(i, v) : (o.fill ?? "k"), rx: o.rx ?? 2, opacity: o.opacity }));
      },
      /* axis helper for charts */
      axes(x, y, w, h, o = {}) {
        G.line(x, y + h, x + w, y + h, { color: "line" }); G.line(x, y, x, y + h, { color: "line" });
        if (o.xlabel) G.text(x + w, y + h + 18, o.xlabel, { anchor: "end", size: 11, color: "muted" });
        if (o.ylabel) G.text(x, y - 8, o.ylabel, { size: 11, color: "muted" });
      },
      caption(s) { return G.text(20, G.H - 14, s, { size: 13, color: "muted" }); },
      label(x, y, s, o = {}) { return G.text(x, y, s, Object.assign({ size: 12, color: "muted" }, o)); },
      /* animation helpers: no-ops when reduced motion or GSAP is missing */
      from(targets, vars) { if (window.gsap && !reduce && targets && (targets.length ?? 1)) gsap.from(targets, vars); },
      to(targets, vars) { if (window.gsap && !reduce && targets && (targets.length ?? 1)) gsap.to(targets, vars); },
      fromTo(targets, a, b) { if (window.gsap && !reduce && targets && (targets.length ?? 1)) gsap.fromTo(targets, a, b); },
      pulse(target, o = {}) { if (window.gsap && !reduce) gsap.fromTo(target, { opacity: o.lo ?? 0.25 }, { opacity: o.hi ?? 1, duration: o.d ?? 0.4, repeat: o.repeat ?? 5, yoyo: true }); },
      after(ms, fn) { timers.push(setTimeout(fn, reduce ? 0 : ms)); },
      every(ms, fn) { if (reduce) { fn(); return; } timers.push(setInterval(fn, ms)); },
      _marker(color) {
        const id = "m" + c(color).replace(/[^a-z0-9]/gi, "");
        if (!svg.querySelector("#" + id)) {
          let defs = svg.querySelector("defs") || el("defs", {}); if (defs.parentNode !== svg) svg.insertBefore(defs, svg.firstChild);
          const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
          el("path", { d: "M0,0 L10,5 L0,10 z", fill: c(color) }, m);
        }
        return `url(#${id})`;
      },
    };
    return G;
  }

  /* ---------------- formatting helpers for lessons ---------------- */
  const fmt = {
    bytes(b) { const a = Math.abs(b); return a >= 2 ** 40 ? (b / 2 ** 40).toFixed(2) + " TiB" : a >= 2 ** 30 ? (b / 2 ** 30).toFixed(2) + " GiB" : a >= 2 ** 20 ? (b / 2 ** 20).toFixed(1) + " MiB" : a >= 1024 ? (b / 1024).toFixed(1) + " KiB" : b.toFixed(0) + " B"; },
    num(n, d = 0) { return Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }); },
    si(n, d = 1) { const a = Math.abs(n); return a >= 1e12 ? (n / 1e12).toFixed(d) + " T" : a >= 1e9 ? (n / 1e9).toFixed(d) + " G" : a >= 1e6 ? (n / 1e6).toFixed(d) + " M" : a >= 1e3 ? (n / 1e3).toFixed(d) + " k" : n.toFixed(d) + " "; },
    ms(s) { return s >= 1 ? s.toFixed(2) + " s" : (s * 1e3).toFixed(s < 0.01 ? 2 : 1) + " ms"; },
  };

  /* ---------------- the lesson shell ---------------- */
  function lesson(L) {
    const KEY = "s2s:progress:" + L.id;
    const prog = store.get(KEY, { seen: [], done: [] });
    const seen = new Set(prog.seen), done = new Set(prog.done);
    const save = () => store.set(KEY, { seen: [...seen], done: [...done], steps: L.steps.length, checks: checkCount });
    const checkCount = L.steps.filter((s) => s.check).length;
    document.title = `${L.n} ${L.title}`;
    const app = $("#app") || document.body.appendChild(Object.assign(document.createElement("div"), { id: "app" }));
    const tabs = [["learn", "Learn"]].concat(L.sim ? [["sim", "Simulate"]] : []).concat(L.practice ? [["prac", "Practice"]] : []);

    app.innerHTML = `
<div class="bar">
  <a class="home" href="index.html" aria-label="Course home">☰ Course</a>
  <div class="t">${esc(L.n)} · ${esc(L.title)}<small>${esc(L.subtitle || "")}</small></div>
  <div class="meter" title="checkpoints answered"><div class="track"><div class="fill" id="meter"></div></div><span id="meterTxt"></span></div>
  <div class="tabs" role="tablist">${tabs.map(([id, t], i) => `<button role="tab" aria-selected="${i === 0}" aria-controls="${id}">${t}</button>`).join("")}</div>
</div>
<section role="tabpanel" id="learn" class="learn">
  <nav class="rail" id="rail" aria-label="Lesson path"></nav>
  <div class="story" id="story">
    <div class="intro"><div class="kicker">${esc(L.kicker || "Lesson")}</div><h1>${L.headline || esc(L.title)}</h1>${L.intro || ""}
      <div class="facts">${(L.facts || []).map((f) => `<span>${esc(f)}</span>`).join("")}</div></div>
    ${L.steps.map((s, i) => `
    <section class="step" data-s="${i}" id="step-${i + 1}">
      <div class="card"><div class="n">Step ${i + 1} · ${esc(s.rail)}</div><h2>${s.title}</h2>${s.body}</div>
      ${s.check ? `<div class="check" data-s="${i}"><div class="lab">CHECKPOINT</div><div class="q">${s.check.q}</div>
        <div class="opts">${s.check.options.map((o, j) => `<button data-i="${j}">${o}</button>`).join("")}</div><div class="fb"></div></div>` : ""}
    </section>`).join("")}
    <div class="outro">${L.outro || ""}
      <div class="btns">${L.sim ? `<button class="btn pri" data-go="sim">Open the simulator →</button>` : ""}${L.practice ? `<button class="btn" data-go="prac">Practice →</button>` : ""}</div></div>
  </div>
  <div class="stagewrap"><div class="stage">
    <div class="hd"><span id="stageTitle"></span><button class="replay" id="replay">Replay</button></div>
    <svg id="sv" viewBox="0 0 640 450" role="img" aria-label="Animated explainer"></svg>
    <div class="legend" id="legend"></div>
  </div></div>
</section>
${L.sim ? `<section role="tabpanel" id="sim" class="pane" hidden><div class="simgrid">
  <div class="panel"><div class="kicker">Simulator</div><h2>${L.sim.title}</h2><p class="lead">${L.sim.intro || ""}</p>
    <div class="ctl" id="ctl"></div>
    <div class="btns">${L.sim.run ? `<button class="btn pri" id="run">${esc(L.sim.run.label || "Run")}</button><button class="btn" id="stop">Stop</button>` : ""}</div>
    <div class="chart"><svg id="simsvg" viewBox="0 0 640 ${L.sim.height || 260}" role="img" aria-label="${esc(L.sim.title)}"></svg></div>
    ${L.sim.note ? `<p class="note">${L.sim.note}</p>` : ""}
  </div>
  <div class="side" id="panels"></div>
</div></section>` : ""}
${L.practice ? `<section role="tabpanel" id="prac" class="pane" hidden><div class="panel">
  <div class="kicker">Practice</div><h2>Exercises</h2><p class="lead">${L.practice.intro || "Run these from the repository root. Each has a starter file and a test; <code>S2S_SOLUTIONS=1</code> runs the same test against the reference solution."}</p>
  <div class="ex">${(L.practice.items || []).map((x, i) => `<div class="item"><b>${i + 1}. ${esc(x.title)}${x.tier ? `<span class="tier">${esc(x.tier)}</span>` : ""}</b>
    ${x.cmd ? `<button class="btn" data-copy="${i}">Copy command</button>` : "<span></span>"}<small>${x.goal}</small>${x.cmd ? `<code id="cmd${i}">${esc(x.cmd)}</code>` : ""}</div>`).join("")}</div>
  ${(L.practice.labs || []).length ? `<div class="labs"><h3 style="font:700 12px var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)">In the repository</h3><ul>${L.practice.labs.map((l) => `<li>${l.label}: <code>${esc(l.path)}</code></li>`).join("")}</ul></div>` : ""}
</div></section>` : ""}
<div class="foot">${L.prev ? `<a class="btn" href="${L.prev}.html">← Previous lesson</a>` : "<span></span>"}${L.next ? `<a class="btn pri" href="${L.next}.html">Next lesson →</a>` : ""}</div>`;

    /* ---- stage ---- */
    const sv = $("#sv"), G = makeG(sv, "s");
    let cur = -1;
    function show(i, force) {
      if (i === cur && !force) return;
      cur = i; G.clear();
      const s = L.steps[i];
      $("#stageTitle").textContent = `step ${i + 1} · ${s.rail}`;
      $("#legend").innerHTML = (s.legend || L.legend || []).map(([col, t]) => `<span><i style="background:${G.c(col)}"></i>${esc(t)}</span>`).join("");
      try { s.scene && s.scene(G); } catch (e) { G.text(20, 40, "scene error: " + e.message, { color: "hot" }); console.error(e); }
      seen.add(i); save(); rail();
    }
    $("#replay").onclick = () => show(Math.max(0, cur), true);

    /* ---- rail + progress ---- */
    function rail() {
      const r = $("#rail");
      r.innerHTML = '<div class="lbl">Lesson path</div>' + L.steps.map((s, i) =>
        `<a href="#step-${i + 1}" data-i="${i}" class="${seen.has(i) ? "seen" : ""} ${i === cur ? "now" : ""} ${done.has(i) ? "done" : ""}"><span class="dot">${done.has(i) ? "✓" : i + 1}</span>${esc(s.rail)}</a>`).join("");
      r.querySelectorAll("a").forEach((a) => a.onclick = (e) => { e.preventDefault(); $(`#step-${+a.dataset.i + 1}`).scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" }); });
      const n = [...done].length;
      $("#meter").style.width = checkCount ? (100 * n) / checkCount + "%" : "0";
      $("#meterTxt").textContent = `${n} / ${checkCount} checkpoints`;
    }

    /* ---- checkpoints ---- */
    app.querySelectorAll(".check").forEach((box) => {
      const i = +box.dataset.s, ck = L.steps[i].check;
      if (done.has(i)) { const b = box.querySelector(`button[data-i="${ck.answer}"]`); b && b.classList.add("right"); box.querySelector(".fb").textContent = ck.why; }
      box.querySelectorAll(".opts button").forEach((b) => b.onclick = () => {
        const ok = +b.dataset.i === ck.answer;
        b.classList.add(ok ? "right" : "wrong");
        box.querySelector(".fb").innerHTML = (ok ? "<b>Correct.</b> " : "<b>Not quite.</b> ") + ck.why;
        if (ok) { done.add(i); save(); rail(); }
      });
    });

    /* ---- scroll-driven stage ---- */
    const steps = [...app.querySelectorAll(".step")];
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (!e.isIntersecting) return;
      steps.forEach((x) => x.classList.remove("active")); e.target.classList.add("active"); show(+e.target.dataset.s);
    }), { rootMargin: "-45% 0px -45% 0px" });
    steps.forEach((s) => io.observe(s));
    steps[0].classList.add("active");

    /* ---- simulator ---- */
    let simTimer = null, simG = null;
    const vals = {};
    function simDraw(t) {
      if (!L.sim) return;
      simG.clear();
      const panels = L.sim.draw(simG, vals, t ?? 1) || [];
      $("#panels").innerHTML = panels.map((p) => `<div class="panel"><h3>${esc(p.title)}</h3>
        ${p.gauge ? `<div class="gauge">${p.gauge.map(([v, col]) => `<span style="width:${Math.max(0, v) * 100}%;background:${simG.c(col)}"></span>`).join("")}</div>` : ""}
        ${p.gaugeText ? `<div style="font:12px var(--f-mono);color:var(--muted)">${p.gaugeText}</div>` : ""}
        ${(p.rows || []).map(([k, v]) => `<div class="row"><span>${k}</span><b>${v}</b></div>`).join("")}
        ${p.chip ? `<div style="margin-top:8px"><span class="chip ${p.chip[0] ? "ok" : "no"}">${esc(p.chip[1])}</span></div>` : ""}
        ${p.html || ""}</div>`).join("");
    }
    if (L.sim) {
      simG = makeG($("#simsvg"), "c");
      const ctl = $("#ctl");
      ctl.innerHTML = L.sim.controls.map((c) => c.type === "select"
        ? `<label>${esc(c.label)}<select id="c_${c.id}">${c.options.map(([v, t]) => `<option value="${esc(v)}" ${v == c.value ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></label>`
        : `<label>${esc(c.label)}: <b id="v_${c.id}"></b><input id="c_${c.id}" type="range" min="${c.min}" max="${c.max}" step="${c.step || 1}" value="${c.value}"></label>`).join("");
      const read = () => L.sim.controls.forEach((c) => {
        const e = $("#c_" + c.id); vals[c.id] = c.type === "select" ? (isNaN(+e.value) ? e.value : +e.value) : +e.value;
        const v = $("#v_" + c.id); if (v) v.textContent = c.format ? c.format(vals[c.id]) : fmt.num(vals[c.id]);
      });
      ctl.addEventListener("input", () => { clearInterval(simTimer); read(); simDraw(1); });
      read();
      if (L.sim.run) {
        const frames = L.sim.run.frames || 40, ms = L.sim.run.ms || 60;
        $("#run").onclick = () => { clearInterval(simTimer); let f = 0; if (reduce) { simDraw(1); return; }
          simTimer = setInterval(() => { f++; simDraw(f / frames); if (f >= frames) clearInterval(simTimer); }, ms); };
        $("#stop").onclick = () => clearInterval(simTimer);
      }
    }

    /* ---- practice copy buttons ---- */
    app.querySelectorAll("[data-copy]").forEach((b) => b.onclick = () => {
      const c = $("#cmd" + b.dataset.copy);
      const sel = () => { const r = document.createRange(); r.selectNodeContents(c); getSelection().removeAllRanges(); getSelection().addRange(r); b.textContent = "Selected"; };
      if (navigator.clipboard) navigator.clipboard.writeText(c.textContent).then(() => b.textContent = "Copied", sel); else sel();
    });

    /* ---- tabs ---- */
    function tab(id) {
      tabs.forEach(([t]) => { $("#" + t).hidden = t !== id; });
      app.querySelectorAll("[role=tab]").forEach((b) => b.setAttribute("aria-selected", b.getAttribute("aria-controls") === id));
      if (id === "learn") show(Math.max(0, cur), true);
      if (id === "sim") simDraw(1);
      window.scrollTo(0, 0);
    }
    app.querySelectorAll("[role=tab]").forEach((b) => b.onclick = () => tab(b.getAttribute("aria-controls")));
    app.querySelectorAll("[data-go]").forEach((b) => b.onclick = () => tab(b.dataset.go));
    if (location.hash === "#simulate" && L.sim) tab("sim");
    if (location.hash === "#practice" && L.practice) tab("prac");

    const redraw = () => { show(Math.max(0, cur), true); if (L.sim && !$("#sim").hidden) simDraw(1); };
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
    new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    show(0); rail();
  }

  window.S2S = { lesson, fmt, makeG, store };
})();
