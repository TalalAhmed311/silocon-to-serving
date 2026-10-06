// Animation core: every animation defines STEPS (array of {caption, ...state}) and draw(svg, step, index).
// Controls: play/pause, step back/forward, reset, speed slider, keyboard (space, ←, →, r), theme toggle.
(function () {
  const $ = (s) => document.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  window.el = function (tag, attrs, parent, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  };
  window.css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  window.startAnimation = function (cfg) {
    const svg = $("#svg");
    let i = 0, playing = false, timer = null;
    const steps = cfg.steps;
    const render = () => {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      cfg.draw(svg, steps[i], i);
      $("#caption").innerHTML = steps[i].caption || "";
      $("#bar").style.width = (100 * i / Math.max(1, steps.length - 1)) + "%";
      $("#count").textContent = (i + 1) + " / " + steps.length;
    };
    const speed = () => +$("#speed").value;           // steps per second
    const tick = () => { if (i < steps.length - 1) { i++; render(); } else pause(); };
    const play = () => { if (i >= steps.length - 1) i = 0; playing = true; $("#play").textContent = "Pause"; schedule(); render(); };
    const schedule = () => { clearTimeout(timer); if (playing) timer = setTimeout(() => { tick(); schedule(); }, 1000 / speed()); };
    const pause = () => { playing = false; clearTimeout(timer); $("#play").textContent = "Play"; };
    $("#play").onclick = () => (playing ? pause() : play());
    $("#next").onclick = () => { pause(); if (i < steps.length - 1) i++; render(); };
    $("#prev").onclick = () => { pause(); if (i > 0) i--; render(); };
    $("#reset").onclick = () => { pause(); i = 0; render(); };
    $("#speed").oninput = () => { $("#speedv").textContent = speed() + "/s"; schedule(); };
    $("#theme").onclick = () => {
      const r = document.documentElement, dark = matchMedia("(prefers-color-scheme: dark)").matches;
      const cur = r.dataset.theme || (dark ? "dark" : "light");
      r.dataset.theme = cur === "dark" ? "light" : "dark"; render();
    };
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.key === " ") { e.preventDefault(); $("#play").click(); }
      if (e.key === "ArrowRight") $("#next").click();
      if (e.key === "ArrowLeft") $("#prev").click();
      if (e.key === "r") $("#reset").click();
    });
    window.rerender = render;   // lets an animation re-render after its own inputs change
    window.setSteps = (s) => { pause(); cfg.steps = s; steps.length = 0; s.forEach((x) => steps.push(x)); i = 0; render(); };
    $("#speedv").textContent = speed() + "/s";
    render();
  };
})();
