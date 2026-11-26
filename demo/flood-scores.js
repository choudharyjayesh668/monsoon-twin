"use strict";
// Flood page: scores the twin's flood alerts on every weather window, live from /simulate (simulate() is in app.js).
(() => {
  const dialog = document.getElementById("scores");
  const sum = (rs, key) => rs.reduce((a, r) => a + r[key], 0);
  const share = (a, b) => (b ? Math.round(100 * a / b) + "%" : "–");
  async function row(w) {
    const out = await simulate(w, null), inRain = w.init_states.map(d => d[1]);
    const real = FloodRisk.levels(inRain, w.actual.map(d => d[1]), w.doy_sincos, false);
    const twin = FloodRisk.levels(inRain, out.rollout.map(d => d[1]), w.doy_sincos, true);
    const pers = FloodRisk.levels(inRain, w.actual.map(() => w.init_states[2][1]), w.doy_sincos, false);
    const days = (a, b, fc) => Array.from({ length: b - a + 1 }, (_, n) => FloodRisk.compare(real[a + n], fc[a + n]));
    const d13 = days(0, 2, twin), p13 = days(0, 2, pers), d46 = days(3, 5, twin);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${w.label}</td><td class="num">${sum(d13, "real")}</td><td class="num">${sum(d13, "twin")}</td>`
      + `<td class="num"><strong>${share(sum(d13, "hits"), sum(d13, "real"))}</strong></td>`
      + `<td class="num">${share(sum(d13, "twin") - sum(d13, "hits"), sum(d13, "twin"))}</td>`
      + `<td class="num">${share(sum(p13, "hits"), sum(p13, "real"))}</td><td class="num">${share(sum(d46, "hits"), sum(d46, "real"))}</td>`;
    return tr;
  }
  let filled = false;
  document.getElementById("open-scores").onclick = async () => {
    dialog.showModal();
    if (filled) return;
    filled = true;
    try {
      const rows = [];
      for (const w of WINDOWS) rows.push(await row(w));
      document.getElementById("floodrows").replaceChildren(...rows);
    } catch (e) {
      filled = false;
      document.getElementById("floodstatus").textContent = `Cannot reach the twin (${e.message}). Start it: python serve.py --port 8000`;
    }
  };
})();
