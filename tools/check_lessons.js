// check_lessons.js — open each lesson headlessly, visit every step, the simulator and practice; report errors.
// Run: PW=$(npm root -g)/playwright node tools/check_lessons.js [id ...]      (no ids = every lesson)
// Screenshots of each step go to /tmp/lesson-shots/<id>/ for review. GSAP may fail to load offline; that's ignored.
const fs = require("fs"), path = require("path");
const { chromium } = require(process.env.PW || "playwright");
const ROOT = path.resolve(__dirname, "../lessons");
const ids = process.argv.slice(2).length ? process.argv.slice(2) : fs.readdirSync(path.join(ROOT, "content")).filter((f) => f.endsWith(".js")).map((f) => f.slice(0, -3));
(async () => {
  const b = await chromium.launch({ executablePath: process.env.CHROME || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome" });
  let bad = 0;
  for (const id of ids) {
    const out = `/tmp/lesson-shots/${id}`; fs.mkdirSync(out, { recursive: true });
    const c = await b.newContext({ viewport: { width: 1400, height: 900 } }); const p = await c.newPage();
    const errs = [];
    p.on("pageerror", (e) => errs.push("pageerror: " + e.message));
    p.on("console", (m) => { if (m.type() === "error" && !/ERR_TUNNEL|ERR_NAME|net::|Failed to load resource/.test(m.text())) errs.push("console: " + m.text()); });
    await p.goto("file://" + path.join(ROOT, id + ".html")); await p.waitForTimeout(400);
    const n = await p.evaluate(() => document.querySelectorAll(".step").length);
    for (let i = 1; i <= n; i++) {
      await p.evaluate((i) => document.querySelector("#step-" + i).scrollIntoView({ block: "center" }), i);
      await p.waitForTimeout(350);
      const sceneErr = await p.evaluate(() => [...document.querySelectorAll("#sv text")].map((t) => t.textContent).find((t) => t.startsWith("scene error")));
      if (sceneErr) errs.push(`step ${i}: ${sceneErr}`);
      await p.screenshot({ path: `${out}/step-${i}.png` });
    }
    if (await p.$('[aria-controls="sim"]')) { await p.click('[aria-controls="sim"]'); if (await p.$("#run")) { await p.click("#run"); await p.waitForTimeout(2600); } await p.screenshot({ path: `${out}/sim.png` }); }
    if (await p.$('[aria-controls="prac"]')) { await p.click('[aria-controls="prac"]'); await p.screenshot({ path: `${out}/practice.png` }); }
    console.log(`${id}: ${n} steps ${errs.length ? "FAIL\n  " + errs.join("\n  ") : "ok"}`); if (errs.length) bad++;
    await c.close();
  }
  await b.close(); process.exit(bad ? 1 : 0);
})();
