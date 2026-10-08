/* home.js — renders the course map from catalog.js; available.js (generated) lists built lessons. */
(function () {
  try { const t = JSON.parse(localStorage.getItem("s2s:theme") || "null"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
  document.getElementById("themebtn").onclick = () => {
    const r = document.documentElement, sys = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    const next = (r.dataset.theme || sys) === "dark" ? "light" : "dark"; r.dataset.theme = next;
    try { localStorage.setItem("s2s:theme", JSON.stringify(next)); } catch (e) {}
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const avail = new Set(window.S2S_AVAILABLE || []);
  const get = (k) => { try { return JSON.parse(localStorage.getItem(k) || "null"); } catch (e) { return null; } };
  let total = 0, built = 0;
  const html = window.S2S_CATALOG.map((ph) => `
  <section class="phase"><div class="phase-hd"><span class="tag">${esc(ph.phase)}</span><h2>${esc(ph.title)}</h2><span class="wk">weeks ${esc(ph.weeks)}</span><p>${esc(ph.blurb)}</p></div>
    <div class="lgrid">${ph.lessons.map((l) => {
      total++; const ok = avail.has(l.id); if (ok) built++;
      const p = get("s2s:progress:" + l.id);
      const frac = p && p.checks ? p.done.length / p.checks : p && p.steps ? p.seen.length / p.steps : 0;
      const st = ok ? `<span class="track"><span class="fill" style="width:${Math.round(frac * 100)}%;display:block"></span></span>${p ? Math.round(frac * 100) + "%" : "start"}` : "coming soon";
      return ok ? `<a class="lcard" href="${l.id}.html"><span class="num">${esc(l.n)}</span><b>${esc(l.title)}</b><span class="st">${st}</span></a>`
                : `<div class="lcard soon"><span class="num">${esc(l.n)}</span><b>${esc(l.title)}</b><span class="st">${st}</span></div>`;
    }).join("")}</div></section>`).join("");
  document.getElementById("map").innerHTML = html;
  document.getElementById("count").textContent = `${built} of ${total} lessons available`;
})();
