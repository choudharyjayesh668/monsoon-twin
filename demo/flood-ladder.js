"use strict";
// Flood page, "Early warning" tab: how early could the twin have warned? For each past flood (demo/data/ladders.js), start the twin 7, 6 ... 1
// days before the peak, live from /simulate (simulate() is in app.js), and read the state's alert level on the peak day.
(() => {
  const panel = document.getElementById("ladder"), LEADS = 7, LEVEL = ["No alert", "Watch", "Warning", "Severe"];
  const rain = days => days.map(d => d[1]);
  const levelOf = (lv, name) => (FloodRisk.stateAlerts(lv, cellStates()).find(s => s.name === name) || { level: 0 }).level;
  const peakText = iso => new Date(iso + "T00:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });

  async function ladder(f) {
    const d = f.days, peak = d.length - 1, steps = [];
    for (let lead = LEADS; lead >= 1; lead--) {                     // last input day: `lead` days before the peak
      const s = peak - lead, init = d.slice(s - 2, s + 1);
      const out = await simulate({ init_states: init, doy_sincos: f.sincos[s] }, null);
      const lv = FloodRisk.levels(rain(init), rain(out.rollout), f.sincos[s], true)[lead - 1];
      steps.push({ lead, level: levelOf(lv, f.state) });
    }
    const real = levelOf(FloodRisk.levels(rain(d.slice(peak - 5, peak - 2)), rain(d.slice(peak - 2, peak + 1)),
                                          f.sincos[peak - 3], false)[2], f.state);
    let warned = 0;                                                 // every run from this many days before gave a warning
    for (const st of [...steps].reverse()) { if (st.level < 2) break; warned = st.lead; }
    return { steps, real, warned, watched: Math.max(0, ...steps.filter(st => st.level >= 1).map(st => st.lead)) };
  }

  function render(f, r) {
    const div = document.createElement("div");
    div.className = "lad";
    const says = r.warned ? `The twin warned ${f.state} <strong>${r.warned} ${r.warned === 1 ? "day" : "days"}</strong> before the peak.`
      : r.watched ? `The twin gave a flood watch for ${f.state} ${r.watched} ${r.watched === 1 ? "day" : "days"} before the peak, but no warning.`
      : `The twin gave no alert for ${f.state} before the peak.`;
    div.innerHTML = `<h3>${f.label}</h3><p>Peak: ${peakText(f.peak)}. Real level on that day: <strong>${LEVEL[r.real]}</strong>. ${says}</p>`
      + `<ol class="steps">${r.steps.map(st => `<li class="lv${st.level}"><span>${st.lead} ${st.lead === 1 ? "day" : "days"} before</span>`
                                                + `<b>${LEVEL[st.level]}</b></li>`).join("")}</ol>`;
    return div;
  }

  let filled = false;
  async function fill() {
    if (filled) return;
    filled = true;
    const sum = document.getElementById("ladder-sum");
    try {
      const results = [];
      for (const f of window.LADDERS) results.push(await ladder(f));
      document.getElementById("ladders").replaceChildren(...window.LADDERS.map((f, n) => render(f, results[n])));
      const w = results.map(r => r.warned).filter(Boolean);
      sum.innerHTML = w.length ? `On ${w.length} of ${results.length} past floods, the twin gave a flood warning `
        + `<strong>${Math.min(...w)} to ${Math.max(...w)} days</strong> before the peak.` : "The twin gave no flood warning before these peaks.";
    } catch (e) {
      filled = false;
      sum.textContent = `Cannot reach the twin (${e.message}). Start it: python serve.py --port 8000`;
    }
  }
  const tabs = { map: document.getElementById("tab-map"), ladder: document.getElementById("tab-ladder") };
  function show(name) {                                             // the map, or the early-warning ladder over it
    Object.entries(tabs).forEach(([k, b]) => b.setAttribute("aria-selected", k === name));
    panel.hidden = name !== "ladder";
    document.body.classList.toggle("ladder-on", name === "ladder");
    if (name === "ladder") fill();
  }
  tabs.map.onclick = () => show("map");
  tabs.ladder.onclick = () => show("ladder");
})();
