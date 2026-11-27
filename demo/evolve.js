"use strict";
// Flood page: the digital twin. Two copies of the weather evolve one day at a time, live. Each day is one /simulate call
// per copy (steps = 1); the twin reads its own last 3 days and calculates the next. "Before" gets no change; "After" gets
// the changes you apply, at any day. At day 10 the run stops and shows how many days before the real flood the twin's
// 10-day forecast from the same start warned. Uses app.js (simulate, paint, P, state ...) and twin-builder.js (Builder).
const Evolve = (() => {
  const MAX = 30, GAP_MS = 350, ev = { on: false, playing: false, n: 0, A: null, B: null, events: [], req: 0 };
  const STOP = { 7: STOPS.flood, 1: STOPS.rain, 0: STOPS.temp, 6: STOPS.speed, 5: STOPS.tcwv };
  const doy0 = () => { const [s, c] = W().doy_sincos; return Math.atan2(s, c) * 365 / (2 * Math.PI); };
  const sc = m => { const a = 2 * Math.PI * (doy0() + m) / 365; return [Math.sin(a), Math.cos(a)]; };   // m days after the last real day
  const dateOf = n => { const d = new Date(W().dates[0] + "T00:00:00"); d.setDate(d.getDate() + n - 1); return d; };
  const evCh = () => "7";                                       // flood risk only
  const clone = x => JSON.parse(JSON.stringify(x));

  function universe() { return { days: clone(W().init_states), lv: [], alerts: [], people: [] }; }
  function score(u, n) {                                           // flood alerts of day n (1..), from the 2 days before and day n
    const rain = u.days.slice(n - 1, n + 3).map(d => d[1]);
    const lv = FloodRisk.levels(rain.slice(0, 3), [rain[3]], sc(n - 1), true)[0];   // shortcut: day-1 lead factor for every day
    u.lv[n] = lv; u.alerts[n] = FloodRisk.stateAlerts(lv, cellStates()); u.people[n] = FloodRisk.people(lv);
  }
  function apply() {                                               // the selected change, to universe B's latest day, now
    const spec = Builder.spec(), last = ev.B.days[ev.B.days.length - 1], d = shockDelta(spec, last);
    ev.B.days[ev.B.days.length - 1] = last.map((ch, c) => ch.map((row, i) => row.map((v, j) => v + d[c][i][j])));
    ev.events.push({ n: ev.n, text: $("change").selectedOptions[0].text });
    const [a0, a1, o0, o1] = spec.box; P[1].box.setBounds([[a0, o0], [a1, o1]]).addTo(P[1].map);
    render();
  }
  async function tick() {
    const n = ev.n + 1;
    const [a, b] = await Promise.all([ev.A, ev.B].map((u, k) => simulate({ init_states: u.days.slice(-3), doy_sincos: sc(n - 1) },
                                                                        null, 1, `evolve ${"AB"[k]}, day ${n}`)));
    ev.A.days.push(a.rollout[0]); ev.B.days.push(b.rollout[0]);
    score(ev.A, n); score(ev.B, n); ev.n = n;
    $("status").textContent = "";                                 // the twin answers again
    render();
  }
  async function play() {
    if (ev.playing) { ev.playing = false; return; }
    if (!state.lead) check().catch(() => {});                     // the twin was down when the page opened
    if (!ev.events.length) apply();                                // a different start for B, unless you gave one already
    ev.playing = true; button();
    try {
      $("leadcard").hidden = true;
      while (ev.playing && ev.n < MAX) {
        await tick();
        if (ev.n === DAYS && state.lead) { ev.playing = false; showLead(DAYS - 1, true); break; }   // the flood check, at day 10
        await new Promise(r => setTimeout(r, GAP_MS));
      }
    } catch (e) { $("status").textContent = `Cannot reach the twin (${e.message}).`; }
    ev.playing = false; button();
  }
  const button = () => { $("ev-play").textContent = ev.playing ? "Pause" : ev.n ? (ev.n >= MAX ? "Done" : "▶ Continue") : "▶ Evolve";
                         $("ev-play").disabled = ev.n >= MAX; };

  function reset() {
    ev.playing = false; ev.n = 0; ev.events = []; ev.A = universe(); ev.B = universe();
    P[1].box.remove(); $("leadcard").hidden = true; button(); render(); check().catch(e => { $("status").textContent = `Cannot reach the twin (${e.message}).`; });
  }
  async function check() {     // the 10-day forecast from the same start, for the day-10 alert (floodLead in app.js)
    const req = ++ev.req, base = await simulate(W(), null, DAYS, "10-day forecast");
    if (req !== ev.req) return;
    Object.assign(state, { base, shocked: null, flood: {} }); state.lead = floodLead();
  }
  function grid(u, ch) { return +ch === 7 ? (ev.n ? u.lv[ev.n] : u.lv[0] || zeros()) : layer(u.days[ev.n + 2], ch); }
  const zeros = () => Array.from({ length: NLAT }, () => new Array(NLON).fill(0));
  function render() {
    const ch = evCh(), n = ev.n, g = [grid(ev.A, ch), grid(ev.B, ch)];
    $("field").value = ch;                                         // the hover tooltip reads the field from here
    g.forEach((x, k) => paint(P[k], x, STOP[ch])); legend(STOP[ch], ch, false); state.grids = g;
    setWind(P[0], ev.A.days[n + 2]); setWind(P[1], ev.B.days[n + 2]);
    const ev0 = ev.events.filter(e => e.n === n).map(e => e.text);
    $("eyebrow").textContent = n ? `Digital twin · day ${n} of ${MAX} · ${dateText(dateOf(n).toISOString().slice(0, 10))}`
                                 : "Digital twin · start: the last real day";
    $("headline").textContent = n ? `After: ${describe(ev.B.alerts[n])}. Before: ${describe(ev.A.alerts[n])}.`
                                  : "Press Evolve. The twin calculates one day at a time, from its own last 3 days.";
    $("captext").textContent = ev0.length ? `▲ Change applied today: ${ev0.join("; ")}.` : "";
    $("stats").innerHTML = n ? stat("People in the alert area", `${persons(ev.A.people[n])} → ${persons(ev.B.people[n])}`, "before → after · 2011 population")
      + stat("States under an alert", `${ev.A.alerts[n].length} → ${ev.B.alerts[n].length}`, "before → after")
      + stat("Changes", String(ev.events.length), ev.events.map(e => `▲ day ${e.n}`).join(", ") || "none") : stat("Changes", String(ev.events.length), ev.events.map(e => e.text).join("; ") || "none yet");
    $("tlhead").textContent = "After: states under a flood alert, by day";
    const top = Math.max(1, ...ev.B.alerts.slice(1).map(a => a.length));
    $("timeline").replaceChildren(...Array.from({ length: n }, (_, k) => {
      const d = k + 1, b = document.createElement("span"), mark = ev.events.some(e => e.n === k) ? "▲" : "";
      b.className = "chip"; b.title = `Day ${d}: people in the alert area, before ${persons(ev.A.people[d])}, after ${persons(ev.B.people[d])}`;
      b.innerHTML = `<span class="track"><span class="bar" style="height:${Math.max(3, 28 * ev.B.alerts[d].length / top)}px"></span></span>`
        + `<span class="val">${mark}${ev.B.alerts[d].length}</span><span class="d">Day ${d}</span>`;
      return b;
    }).slice(-10));
  }

  function enter() {
    ev.on = true; document.body.classList.add("evolve-on");
    state.view = "universes"; $("wrap").className = "mapwrap two"; P.forEach(p => { p.map.invalidateSize(); p.marks.clearLayers(); p.box.remove(); });
    $("title0").textContent = "Before: no change"; $("title1").textContent = "After: with your change";
    $("leadcard").hidden = true; Live.collapse(); reset();
    requestAnimationFrame(() => { P.forEach(p => p.map.invalidateSize()); fit(); });   // after the 2-map layout settles
  }
  $("ev-play").onclick = play;
  $("ev-apply").onclick = apply;
  $("ev-reset").onclick = reset;
  $("win").addEventListener("change", () => { if (ev.on) reset(); });
  enter();                                                          // the flood page is the twin only
  return { on: () => ev.on };
})();
