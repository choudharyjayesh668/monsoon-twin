"use strict";
// Flood page: the digital twin. Two copies of the weather evolve one day at a time, live. Each day is one /simulate call
// per copy (steps = 1); the twin reads its own last 3 days and calculates the next. "Before" gets no change; "After" gets
// the changes you apply, at any day. Each day: the people and the states at flood risk, before and after. Uses app.js (simulate, paint, P, state ...) and twin-builder.js (Builder).
const Evolve = (() => {
  const MAX = 30, ev = { on: false, busy: false, n: 0, k: 0, A: null, B: null, events: [] };
  const STOP = { 7: STOPS.flood, 1: STOPS.rain, 0: STOPS.temp, 6: STOPS.speed, 5: STOPS.tcwv };
  const doy0 = () => { const [s, c] = W().doy_sincos; return Math.atan2(s, c) * 365 / (2 * Math.PI); };
  const sc = m => { const a = 2 * Math.PI * (doy0() + m) / 365; return [Math.sin(a), Math.cos(a)]; };   // m days after the last real day
  const dateOf = n => { const d = new Date(W().dates[0] + "T00:00:00"); d.setDate(d.getDate() + n - 1); return d; };
  const evCh = () => "7";                                       // flood risk only
  const clone = x => JSON.parse(JSON.stringify(x));
  const short = p => (p >= 1e6 ? `${(p / 1e6).toFixed(p >= 1e7 ? 0 : 1)}M` : p >= 1e3 ? `${Math.round(p / 1e3)}k` : String(p));   // the day strip

  function universe() { return { days: clone(W().init_states), lv: [], alerts: [], people: [] }; }
  function score(u, n) {                                           // flood alerts of day n (1..), from the 2 days before and day n
    const rain = u.days.slice(n - 1, n + 3).map(d => d[1]);
    const lv = FloodRisk.levels(rain.slice(0, 3), [rain[3]], sc(n - 1), true)[0];   // shortcut: day-1 lead factor for every day
    u.lv[n] = lv; u.alerts[n] = FloodRisk.stateAlerts(lv, cellStates()); u.people[n] = FloodRisk.people(lv);
  }
  function apply() {                                               // the selected change, to the After copy's latest day, now
    go(ev.n);                                                      // a change always goes on the newest day
    const spec = Builder.spec(), last = ev.B.days[ev.B.days.length - 1], d = shockDelta(spec, last);
    ev.B.days[ev.B.days.length - 1] = last.map((ch, c) => ch.map((row, i) => row.map((v, j) => v + d[c][i][j])));
    ev.events.push({ n: ev.n, text: $("change").selectedOptions[0].text });
    const [a0, a1, o0, o1] = spec.box; P[1].box.setBounds([[a0, o0], [a1, o1]]).addTo(P[1].map);
    render();
  }
  async function tick() {                                          // calculate one more day, live, for both copies
    const n = ev.n + 1;
    const [a, b] = await Promise.all([ev.A, ev.B].map(u => simulate({ init_states: u.days.slice(-3), doy_sincos: sc(n - 1) }, null, 1)));
    ev.A.days.push(a.rollout[0]); ev.B.days.push(b.rollout[0]);
    score(ev.A, n); score(ev.B, n); ev.n = n;
    $("status").textContent = "";                                 // the twin answers again
  }
  async function next() {                                          // →: a day already calculated, or calculate the next one
    if (ev.busy) return;
    if (ev.k < ev.n) return go(ev.k + 1);
    if (ev.n >= MAX) return;
    if (!ev.events.length) apply();                                // a different start for After, unless you gave one already
    ev.busy = true; buttons();
    try { await tick(); go(ev.n); }
    catch (e) { $("status").textContent = `Cannot reach the twin (${e.message}).`; }
    ev.busy = false; buttons();
  }
  function go(k) { ev.k = Math.max(0, Math.min(ev.n, k)); buttons(); render(); }
  function buttons() {
    $("ev-prev").disabled = ev.busy || ev.k === 0; $("ev-next").disabled = ev.busy || (ev.k === ev.n && ev.n >= MAX);
    $("ev-day").textContent = ev.busy ? "Calculating…" : ev.k ? `Day ${ev.k} of ${MAX}` : "Start";
  }

  function reset() {
    ev.n = ev.k = 0; ev.busy = false; ev.events = []; ev.A = universe(); ev.B = universe();
    P[1].box.remove(); buttons(); render();
  }
  function grid(u, ch) { return +ch === 7 ? (ev.k ? u.lv[ev.k] : zeros()) : layer(u.days[ev.k + 2], ch); }
  const zeros = () => Array.from({ length: NLAT }, () => new Array(NLON).fill(0));
  function render() {
    const ch = evCh(), n = ev.k, g = [grid(ev.A, ch), grid(ev.B, ch)];
    $("field").value = ch;                                         // the hover tooltip reads the field from here
    g.forEach((x, k) => paint(P[k], x, STOP[ch])); legend(STOP[ch], ch, false); state.grids = g;
    setWind(P[0], ev.A.days[n + 2]); setWind(P[1], ev.B.days[n + 2]);
    const ev0 = ev.events.filter(e => e.n === n).map(e => e.text);
    $("eyebrow").textContent = n ? `Digital twin · day ${n} of ${MAX} · ${dateText(dateOf(n).toISOString().slice(0, 10))}`
                                 : "Digital twin · start: the last real day";
    const risk = u => (u.alerts[n].length ? `${persons(u.people[n])} people in ${u.alerts[n].length} ${u.alerts[n].length === 1 ? "state" : "states"}` : "no state");
    $("headline").textContent = n ? (ev.A.alerts[n].length || ev.B.alerts[n].length
                                     ? `After the change: ${risk(ev.B)} at flood risk. Before: ${risk(ev.A)}.` : "No state at flood risk, before or after the change.")
                                  : "Press → to calculate the next day. The twin calculates one day at a time, from its own last 3 days.";
    $("captext").textContent = ev0.length ? `▲ Change applied on this day: ${ev0.join("; ")}.` : "";
    $("stats").innerHTML = n ? stat("People at risk", `${persons(ev.A.people[n])} → ${persons(ev.B.people[n])}`, "before → after · 2011 population", "hot")
      + stat("States at risk", `${ev.A.alerts[n].length} → ${ev.B.alerts[n].length}`, names(ev.B.alerts[n].map(s => s.name)) || "none after the change")
      + stat("Changes", String(ev.events.length), ev.events.map(e => `▲ day ${e.n}`).join(", ") || "none") : stat("Changes", String(ev.events.length), ev.events.map(e => e.text).join("; ") || "none yet");
    $("tlhead").textContent = "After the change: people at flood risk, by day";
    const top = Math.max(1, ...ev.B.people.slice(1));
    $("timeline").replaceChildren(...Array.from({ length: ev.n }, (_, k) => {
      const d = k + 1, b = document.createElement("button"), mark = ev.events.some(e => e.n === k) ? "▲" : "";
      b.className = "chip"; b.title = `Day ${d}: people at risk, before ${persons(ev.A.people[d])}, after ${persons(ev.B.people[d])}`;
      b.setAttribute("aria-current", d === n ? "true" : "false"); b.onclick = () => go(d);
      b.innerHTML = `<span class="track"><span class="bar" style="height:${Math.max(3, 28 * ev.B.people[d] / top)}px"></span></span>`
        + `<span class="val">${mark}${short(ev.B.people[d])}</span><span class="d">Day ${d}</span>`;
      return b;
    }).slice(Math.max(0, Math.min(ev.n, n + 5) - 10), Math.max(10, Math.min(ev.n, n + 5))));
  }

  function enter() {
    ev.on = true; document.body.classList.add("evolve-on");
    state.view = "universes"; $("wrap").className = "mapwrap two"; P.forEach(p => { p.map.invalidateSize(); p.marks.clearLayers(); p.box.remove(); });
    $("title0").textContent = "Before: no change"; $("title1").textContent = "After: with your change";
    reset();
    requestAnimationFrame(() => { P.forEach(p => p.map.invalidateSize()); fit(); });   // after the 2-map layout settles
  }
  $("ev-prev").onclick = () => go(ev.k - 1);
  $("ev-next").onclick = next;
  document.addEventListener("keydown", e => {                        // the arrow keys step too, outside form fields
    if (/^(SELECT|INPUT|TEXTAREA)$/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "ArrowRight") { e.preventDefault(); next(); }
    if (e.key === "ArrowLeft") { e.preventDefault(); go(ev.k - 1); }
  });
  $("ev-apply").onclick = apply;
  $("ev-reset").onclick = reset;
  $("win").addEventListener("change", () => { if (ev.on) reset(); });
  enter();                                                          // the flood page is the twin only
  return { on: () => ev.on };
})();
