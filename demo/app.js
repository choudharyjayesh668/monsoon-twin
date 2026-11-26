"use strict";
// Monsoon Twin demo map. Every forecast comes live from the twin server's /simulate; nothing is pre-rendered.

const SERVER = new URLSearchParams(location.search).get("server") || "http://localhost:8000";
const $ = id => document.getElementById(id);
const NLAT = 32, NLON = 36, LAT0 = 6.5, LON0 = 65.5, DAYS = 8, SECONDS_PER_DAY = 3.5;
const BOUNDS = [[6, 65], [38, 101]];                               // outer edges of the 1-degree cells
const BOXES = { warm_ocean: [8, 20, 82, 94], monsoon_stall: [8, 20, 65, 75], arabian_heat: [8, 20, 65, 75] };
const FIELDS = {
  1: { name: "rain", unit: "mm/day", conv: v => v, stops: "rain", digits: 0 },
  0: { name: "temperature", unit: "°C", conv: v => v - 273.15, stops: "temp", digits: 1 },
  6: { name: "wind speed", unit: "m/s", conv: null, stops: "speed", digits: 1 },          // from u10 and v10
  5: { name: "water vapour", unit: "kg/m²", conv: v => v, stops: "tcwv", digits: 0 },
  4: { name: "sea-level pressure", unit: "hPa", conv: v => v / 100, stops: "msl", digits: 0 },
};
const STOPS = {   // [value in display units, [r, g, b, alpha]]
  rain: [[0, [255, 255, 255, 0]], [0.5, [214, 236, 255, 0]], [1, [214, 236, 255, 150]], [2.5, [160, 205, 255, 190]],
         [5, [95, 165, 245, 210]], [10, [40, 110, 220, 220]], [20, [60, 60, 190, 228]], [40, [125, 45, 190, 235]],
         [80, [195, 40, 160, 240]], [150, [255, 80, 120, 245]]],
  temp: [[-10, [94, 60, 153, 205]], [0, [59, 111, 216, 205]], [10, [74, 184, 224, 205]], [18, [127, 214, 160, 205]],
         [24, [232, 228, 106, 205]], [30, [245, 167, 66, 210]], [36, [232, 97, 44, 215]], [42, [178, 24, 43, 220]],
         [48, [109, 10, 30, 225]]],
  speed: [[0, [255, 255, 255, 0]], [2, [220, 240, 238, 120]], [5, [150, 215, 205, 180]], [8, [80, 170, 190, 205]],
          [12, [60, 110, 190, 220]], [16, [110, 60, 170, 230]], [22, [190, 40, 120, 240]]],
  tcwv: [[0, [255, 255, 255, 0]], [10, [222, 240, 234, 140]], [25, [150, 212, 196, 190]], [40, [70, 172, 162, 210]],
         [55, [30, 122, 140, 222]], [70, [22, 72, 122, 232]]],
  msl: [[985, [110, 40, 150, 225]], [995, [80, 110, 210, 210]], [1005, [190, 215, 240, 170]], [1010, [245, 245, 245, 120]],
        [1015, [250, 205, 150, 170]], [1025, [210, 100, 40, 215]]],
};
const diverging = m => [[-m, [33, 102, 172, 235]], [-m / 2, [120, 175, 215, 205]], [-m * 0.06, [255, 255, 255, 0]],
                        [m * 0.06, [255, 255, 255, 0]], [m / 2, [240, 150, 120, 205]], [m, [178, 24, 43, 235]]];

const state = { base: null, shocked: null, view: "forecast", t: 0, playing: false, grid: null, req: 0 };

// ---- map -------------------------------------------------------------------------------------------------
const map = L.map("map", { zoomSnap: 0.25, minZoom: 3.5, maxZoom: 8, maxBounds: [[-10, 45], [55, 125]] });
map.zoomControl.setPosition("topright");
map.attributionControl.setPrefix(false).addAttribution(
  'Borders: <a href="https://www.naturalearthdata.com">Natural Earth</a> (India view) · <a href="https://leafletjs.com">Leaflet</a> · wind: leaflet-velocity');
const narrow = () => innerWidth < 720;
map.fitBounds(BOUNDS, { paddingTopLeft: [narrow() ? 10 : 340, 10], paddingBottomRight: [10, 10] });
for (const [name, z] of [["field", 350], ["wind", 420], ["lines", 450]]) {
  map.createPane(name).style.zIndex = z; map.getPane(name).style.pointerEvents = "none";
}
const overlay = L.imageOverlay("data:,", BOUNDS, { pane: "field", interactive: false }).addTo(map);
L.geoJSON(window.BORDERS, { pane: "lines", interactive: false, style: f => f.properties.name === "India"
  ? { color: "#1d1d1f", weight: 1.5, fill: false } : { color: "#8e8e93", weight: 0.8, fill: false } }).addTo(map);
L.rectangle(BOUNDS, { pane: "lines", color: "#c7c7cc", weight: 1, fill: false, interactive: false }).addTo(map);
const shockBox = L.rectangle([[0, 0], [0, 0]], { pane: "lines", color: "#1d1d1f", weight: 1.5, dashArray: "6 6",
                                                 fill: false, interactive: false });
let wind = null;

// ---- field image in Web Mercator rows, so every cell sits at its true latitude -------------------------
const merc = lat => Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360));
const IW = 720, IH = Math.round(IW * (merc(38) - merc(6)) / (36 * Math.PI / 180));
const ROW_LAT = Array.from({ length: IH }, (_, r) => {
  const y = merc(38) - (r + 0.5) / IH * (merc(38) - merc(6));
  return (2 * Math.atan(Math.exp(y)) - Math.PI / 2) * 180 / Math.PI;
});
const canvas = Object.assign(document.createElement("canvas"), { width: IW, height: IH });
const ctx = canvas.getContext("2d");

function sample(g, lat, lon) {                                      // bilinear, clamped to the grid
  const fi = Math.min(NLAT - 1, Math.max(0, lat - LAT0)), fj = Math.min(NLON - 1, Math.max(0, lon - LON0));
  const i = Math.min(NLAT - 2, Math.floor(fi)), j = Math.min(NLON - 2, Math.floor(fj)), a = fi - i, b = fj - j;
  return (g[i][j] * (1 - a) + g[i + 1][j] * a) * (1 - b) + (g[i][j + 1] * (1 - a) + g[i + 1][j + 1] * a) * b;
}
function colorAt(stops, v) {
  if (v <= stops[0][0]) return stops[0][1];
  for (let k = 1; k < stops.length; k++) if (v <= stops[k][0]) {
    const [v0, c0] = stops[k - 1], [v1, c1] = stops[k], t = (v - v0) / (v1 - v0);
    return c0.map((x, n) => x + (c1[n] - x) * t);
  }
  return stops[stops.length - 1][1];
}
function paint(g, stops, tf = v => v) {
  const img = ctx.createImageData(IW, IH), d = img.data;
  for (let r = 0; r < IH; r++) for (let c = 0; c < IW; c++) {
    const col = colorAt(stops, tf(sample(g, ROW_LAT[r], 65 + (c + 0.5) * 36 / IW))), p = (r * IW + c) * 4;
    d[p] = col[0]; d[p + 1] = col[1]; d[p + 2] = col[2]; d[p + 3] = col[3];
  }
  ctx.putImageData(img, 0, 0);
  overlay.setUrl(canvas.toDataURL());
}

// ---- data ------------------------------------------------------------------------------------------------
const W = () => WINDOWS[+$("win").value];
function layer(src, k, ch) {                                        // one day of one field, in display units
  if (+ch === 6) return src[k][2].map((row, i) => row.map((u, j) => Math.hypot(u, src[k][3][i][j])));
  return src[k][ch].map(row => row.map(FIELDS[ch].conv));
}
const sub = (a, b) => a.map((row, i) => row.map((v, j) => v - b[i][j]));
const lerp = (a, b, t) => a.map((row, i) => row.map((v, j) => v + (b[i][j] - v) * t));
function sources() {
  const fc = (state.shocked || state.base).rollout, base = state.base.rollout, real = W().actual;
  return { forecast: [fc], real: [real], error: [base, real], shock: [state.shocked && state.shocked.rollout, base] }[state.view];
}
function gridAt(k, ch) {
  const [a, b] = sources();
  return b ? sub(layer(a, k, ch), layer(b, k, ch)) : layer(a, k, ch);
}

// ---- drawing -----------------------------------------------------------------------------------------------
function draw(t) {
  if (!state.base) return;
  const ch = $("field").value, k0 = Math.min(DAYS - 1, Math.floor(t)), k1 = Math.min(DAYS - 1, k0 + 1);
  const g = k1 === k0 ? gridAt(k0, ch) : lerp(gridAt(k0, ch), gridAt(k1, ch), t - k0);
  const diff = state.view === "error" || state.view === "shock";
  let stops = STOPS[FIELDS[ch].stops], tf;
  if (diff) {           // scale to this day's largest change; a square-root curve keeps small changes visible
    let m = 1e-6;
    for (const row of g) for (const v of row) m = Math.max(m, Math.abs(v));
    stops = diverging(m); tf = v => Math.sign(v) * m * Math.sqrt(Math.min(1, Math.abs(v) / m));
  }
  state.grid = g; paint(g, stops, tf); legend(stops, ch, diff);
}

function legend(stops, ch, diff) {
  const c = $("bar"), x = c.getContext("2d"), lo = stops[0][0], hi = stops[stops.length - 1][0];
  x.clearRect(0, 0, c.width, c.height);
  for (let p = 0; p < c.width; p++) {
    const col = colorAt(stops, lo + (hi - lo) * p / (c.width - 1));
    x.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${Math.max(col[3], 40) / 255})`; x.fillRect(p, 0, 1, c.height);
  }
  const f = FIELDS[ch], fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(v) < 10 ? 1 : 0));
  $("legtitle").textContent = (diff ? (state.view === "error" ? "Forecast minus real weather: " : "Change from the shock: ") : "")
    + f.name[0].toUpperCase() + f.name.slice(1) + " (" + f.unit + ")";
  $("lo").textContent = fmt(lo); $("mid").textContent = fmt((lo + hi) / 2); $("hi").textContent = fmt(hi);
}

function updateWind(k) {
  const src = (state.view === "real" ? W().actual : (state.shocked || state.base).rollout)[k];
  const flat = ch => { const out = []; for (let i = NLAT - 1; i >= 0; i--) out.push(...src[ch][i]); return out; };  // north row first
  const head = n => ({ parameterCategory: 2, parameterNumber: n, nx: NLON, ny: NLAT, lo1: LON0, la1: LAT0 + NLAT - 1,
                       lo2: LON0 + NLON - 1, la2: LAT0, dx: 1, dy: 1 });
  const data = [{ header: head(2), data: flat(2) }, { header: head(3), data: flat(3) }];
  if (!$("wind").checked) { if (wind) { map.removeLayer(wind); wind = null; } return; }
  if (wind) { wind.setData(data); return; }
  wind = L.velocityLayer({ displayValues: false, data, paneName: "wind", maxVelocity: 18, velocityScale: 0.004,
    particleAge: 200, frameRate: 20, lineWidth: 1.8, particleMultiplier: 1 / 400, opacity: 0.9,
    colorScale: ["rgba(29,29,31,0.35)", "rgba(29,29,31,0.5)", "rgba(29,29,31,0.65)", "rgba(29,29,31,0.8)"] }).addTo(map);
}

// ---- captions: what is happening, read from the fields (ASD-STE100 style) ---------------------------------
function inRing(x, y, ring) {
  let inside = false;
  for (let a = 0, b = ring.length - 1; a < ring.length; b = a++) {
    const [xa, ya] = ring[a], [xb, yb] = ring[b];
    if ((ya > y) !== (yb > y) && x < (xb - xa) * (y - ya) / (yb - ya) + xa) inside = !inside;
  }
  return inside;
}
function country(lat, lon) {
  for (const f of window.BORDERS.features) for (const poly of f.geometry.coordinates)
    if (inRing(lon, lat, poly[0])) return f.properties.name;
  return null;
}
function place(lat, lon) {
  const c = country(lat, lon);
  if (c === "India") {
    if (lat >= 30 && lon < 80) return "north-west India";
    if (lon >= 88 && lat >= 21) return "north-east India";
    if (lat >= 24) return lon < 76 ? "north-west India" : "north India";
    if (lat >= 20) return "central India";
    return lon < 76.5 ? "the west coast of India" : lon >= 79 ? "the east coast of India" : "south India";
  }
  if (c) return c === "People's Republic of China" ? "China" : c;
  if (lon < 77 && lat < 25) return "the Arabian Sea";
  if (lon >= 92 && lat < 17) return "the Andaman Sea";
  if (lon >= 77.5 && lat < 23) return "the Bay of Bengal";
  return "the Indian Ocean";
}
const ll = (i, j) => [LAT0 + i, LON0 + j];
function extreme(g, sign) {                                         // sign 1: maximum, -1: minimum
  let best = -Infinity, at = [0, 0];
  g.forEach((row, i) => row.forEach((v, j) => { if (sign * v > best) { best = sign * v; at = [i, j]; } }));
  return { v: sign * best, i: at[0], j: at[1] };
}
function compass(dlat, dlon) {
  const a = (Math.atan2(dlat, dlon * Math.cos(22 * Math.PI / 180)) * 180 / Math.PI + 360) % 360;
  return ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"][Math.round(a / 45) % 8];
}
function centroid(g, thr) {
  let s = 0, si = 0, sj = 0;
  g.forEach((row, i) => row.forEach((v, j) => { if (v > thr) { s++; si += i; sj += j; } }));
  return s ? [si / s, sj / s, s] : null;
}
const km = deg => Math.round(deg * 111 / 10) * 10;

function caption(k) {
  const ch = $("field").value, f = FIELDS[ch], g = gridAt(k, ch), u = " " + f.unit, n = v => v.toFixed(f.digits);
  const at = e => place(...ll(e.i, e.j));
  if (state.view === "shock" || state.view === "error") {
    const hi = extreme(g, 1), lo = extreme(g, -1), e = Math.abs(hi.v) >= Math.abs(lo.v) ? hi : lo;
    if (Math.abs(e.v) < Math.pow(10, -f.digits)) return `The ${f.name} does not change on this day.`;
    if (state.view === "error") {
      let s = 0; g.forEach(r => r.forEach(v => { s += Math.abs(v); }));
      return `The largest error is ${n(Math.abs(e.v))}${u} in ${at(e)}. Here, the forecast is ${e.v > 0 ? "too high" : "too low"}. `
           + `The mean error is ${n(s / (NLAT * NLON))}${u}.`;
    }
    const [a0, a1, o0, o1] = BOXES[$("shock").value]; let far = 0;
    g.forEach((row, i) => row.forEach((v, j) => {
      if (Math.abs(v) < 0.2 * Math.abs(e.v)) return;
      const [la, lo2] = ll(i, j), dla = Math.max(a0 - la, 0, la - a1), dlo = Math.max(o0 - lo2, 0, lo2 - o1);
      far = Math.max(far, Math.hypot(dla, dlo));
    }));
    return `The shock ${e.v > 0 ? "increases" : "decreases"} the ${f.name} by up to ${n(Math.abs(e.v))}${u} in ${at(e)}. `
         + (far < 0.5 ? "The effect stays in the shock box." : `The effect is now ${km(far)} km from the shock box.`);
  }
  if (+ch === 1) {
    const e = extreme(g, 1), c = centroid(g, 10);
    let s = `The heaviest rain is ${n(e.v)}${u} in ${at(e)}.`;
    if (c) s += ` Heavy rain (more than 10 mm/day) covers ${Math.round(100 * c[2] / (NLAT * NLON))}% of the map.`;
    if (k > 0 && e.v >= 20) {                                     // follow the core: cells above half the peak
      const prev = gridAt(k - 1, ch), pe = extreme(prev, 1);
      const a = centroid(g, e.v / 2), b = pe.v >= 20 && centroid(prev, pe.v / 2);
      const d = a && b ? Math.hypot(a[0] - b[0], a[1] - b[1]) : 0;
      if (d >= 1 && d <= 6) s += ` The core of the heaviest rain moves ${km(d)} km ${compass(a[0] - b[0], a[1] - b[1])}.`;
    }
    return s;
  }
  if (+ch === 0) { const h = extreme(g, 1), c = extreme(g, -1);
    return `The hottest area is ${n(h.v)}${u} in ${at(h)}. The coldest area is ${n(c.v)}${u} in ${at(c)}.`; }
  if (+ch === 4) { const l = extreme(g, -1);
    return `The lowest pressure is ${n(l.v)}${u} over ${at(l)}.` + (l.v < 1000 ? " Pressure this low can show a cyclone or a strong low." : ""); }
  const e = extreme(g, 1);
  return +ch === 6 ? `The strongest wind is ${n(e.v)}${u} over ${at(e)}.` : `The most moist air is ${n(e.v)}${u} over ${at(e)}.`;
}

let shownDay = -1;
function setDay(k, force) {
  if (k === shownDay && !force) return;
  shownDay = k; $("day").value = k + 1; $("daylabel").textContent = `Day ${k + 1}`;
  const titles = { forecast: "Twin forecast", real: "Real weather", error: "Forecast error", shock: "Shock effect" };
  $("captitle").textContent = `Day ${k + 1} of ${DAYS} · ${titles[state.view]}`;
  const cap = $("cap"); cap.classList.remove("in"); void cap.offsetWidth;
  $("captext").textContent = caption(k); cap.classList.add("in");
  updateWind(k);
}

// ---- running the twin ------------------------------------------------------------------------------------
function shockDelta(name, last) {                                   // the organisers' shocks, plus the +2.5 °C test
  if (name === "none") return null;
  const d = Array.from({ length: 6 }, () => Array.from({ length: NLAT }, () => new Array(NLON).fill(0)));
  const [a0, a1, o0, o1] = BOXES[name];
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
    const [la, lo] = ll(i, j);
    if (la < a0 || la > a1 || lo < o0 || lo > o1) continue;
    if (name === "warm_ocean") { d[0][i][j] = 3; d[5][i][j] = 0.15 * last[5][i][j]; }
    if (name === "monsoon_stall") { d[2][i][j] = -0.4 * last[2][i][j]; d[3][i][j] = -0.4 * last[3][i][j]; }
    if (name === "arabian_heat") d[0][i][j] = 2.5;
  }
  return d;
}
async function simulate(w, perturbation) {
  const r = await fetch(SERVER + "/simulate", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ init_states: w.init_states, doy_sincos: w.doy_sincos, steps: 10, perturbation }) });
  if (!r.ok) throw new Error("the server answered " + r.status);
  return r.json();
}
async function run() {
  const req = ++state.req, w = W(), name = $("shock").value;
  $("status").textContent = "The twin calculates 10 days…";
  try {
    const base = await simulate(w, null);
    const shocked = name === "none" ? null : await simulate(w, shockDelta(name, w.init_states[2]));
    if (req !== state.req) return;                                  // a newer request started meanwhile
    Object.assign(state, { base, shocked });
    $("status").textContent = `Live from the twin server: 10 days in ${base.inference_ms.toFixed(0)} ms.`;
    if (name === "none") { shockBox.remove(); } else {
      const [a0, a1, o0, o1] = BOXES[name]; shockBox.setBounds([[a0, o0], [a1, o1]]).addTo(map); }
    $("v-shock").disabled = name === "none";
    if (name === "none" && state.view === "shock") setView("forecast"); else if (name !== "none") setView("shock");
    else { draw(state.t); setDay(Math.floor(state.t), true); }
  } catch (e) {
    $("status").textContent = `The page cannot connect to the twin (${e.message}). Start it: python serve.py --port 8000`;
  }
}
function setView(v) {
  state.view = v;
  document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-pressed", b.dataset.view === v));
  draw(state.t); setDay(Math.floor(state.t), true);
}

// ---- play: slow motion between days ------------------------------------------------------------------------
let last = 0, lastPaint = 0;
function frame(now) {
  if (!state.playing) return;
  state.t += (now - last) / 1000 / SECONDS_PER_DAY; last = now;
  if (state.t >= DAYS - 1) { state.t = DAYS - 1; draw(state.t); setDay(DAYS - 1); togglePlay(false); return; }
  if (now - lastPaint > 120) { draw(state.t); lastPaint = now; }   // about 8 frames a second is smooth enough
  setDay(Math.floor(state.t));
  requestAnimationFrame(frame);
}
function togglePlay(on) {
  state.playing = on ?? !state.playing;
  $("play").textContent = state.playing ? "❚❚" : "▶"; $("play").setAttribute("aria-label", state.playing ? "Pause" : "Play");
  if (state.playing) { if (state.t >= DAYS - 1) state.t = 0; last = performance.now(); requestAnimationFrame(frame); }
}

// ---- controls ----------------------------------------------------------------------------------------------
WINDOWS.forEach((w, i) => $("win").add(new Option(w.label, i)));
$("win").onchange = $("shock").onchange = () => { togglePlay(false); run(); };
$("field").onchange = () => { draw(state.t); setDay(Math.floor(state.t), true); };
$("wind").onchange = () => updateWind(Math.floor(state.t));
$("day").oninput = () => { togglePlay(false); state.t = +$("day").value - 1; draw(state.t); setDay(Math.floor(state.t)); };
$("play").onclick = () => togglePlay();
document.querySelectorAll(".seg button").forEach(b => { b.onclick = () => setView(b.dataset.view); });
map.on("mousemove", e => {
  const { lat, lng } = e.latlng, f = FIELDS[$("field").value];
  if (!state.grid || lat < 6 || lat > 38 || lng < 65 || lng > 101) { $("readout").textContent = "Point at the map to read a value."; return; }
  $("readout").textContent = `${lat.toFixed(1)}° N  ${lng.toFixed(1)}° E   ·   ${sample(state.grid, lat, lng).toFixed(f.digits)} ${f.unit}`;
});
if (narrow()) $("panel").removeAttribute("open");               // phones: the map first, controls on demand
run();
