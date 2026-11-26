"use strict";
// Climate twin page: the change to the starting conditions. One of the organisers' shocks, or your own change: a
// region, a field and an amount. The limits keep the start close to weather the twin has seen. Loaded before app.js;
// app.js calls Builder.spec() for each run and Builder.click() when the map is clicked.
const Builder = (() => {
  const el = id => document.getElementById(id);
  const REGIONS = {                                   // [name, box: lat0, lat1, lon0, lon1 on cell edges]
    bay: ["Bay of Bengal", [8, 20, 82, 94]],
    arabian: ["Arabian Sea", [8, 20, 65, 75]],
    ghats: ["Western Ghats and Konkan coast", [8, 21, 72, 77]],
    gangetic: ["Indo-Gangetic Plain", [24, 29, 75, 88]],
    central: ["Central India", [18, 26, 74, 84]],
    northeast: ["North-east India", [22, 29, 89, 97]],
    thar: ["Thar Desert", [24, 30, 69, 75]],
    click: ["Where I click on the map (8° × 8°)", null],
  };
  const WHAT = {                                      // [name, unit, min, max, step, start value, change]
    t2m: ["Temperature", "°C", -6, 6, 0.5, 3, v => ({ t2m: v })],
    tcwv: ["Water vapour", "%", -40, 40, 5, 15, v => ({ tcwv: v / 100 })],
    wind: ["Wind speed", "%", -60, 60, 10, -40, v => ({ wind: v / 100 })],
    msl: ["Sea-level pressure", "hPa", -10, 10, 1, -5, v => ({ msl: v * 100 })],
  };
  let clickBox = REGIONS.central[1];

  for (const [k, [name]] of Object.entries(REGIONS)) el("where").add(new Option(name, k));
  for (const [k, [name]] of Object.entries(WHAT)) el("what").add(new Option(name, k));
  const amountText = () => {
    const v = +el("amount").value, u = WHAT[el("what").value][1];
    el("amount-out").textContent = `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v)}${u === "%" ? "%" : " " + u}`;
  };
  function setWhat() {                                // a new field: its own limits and a typical amount
    const [, , min, max, step, start] = WHAT[el("what").value];
    Object.assign(el("amount"), { min, max, step, value: start });
    amountText();
  }
  setWhat();

  const custom = () => el("change").value === "custom";
  el("change").onchange = () => { el("custom").hidden = !custom(); run(); };
  el("where").onchange = () => run();
  el("what").onchange = () => { setWhat(); run(); };
  el("amount").oninput = amountText;
  el("amount").onchange = () => run();                // when the slider is let go, not while it moves

  return {
    spec() {
      if (!custom()) return PRESETS[el("change").value];
      const w = el("where").value;
      return { box: w === "click" ? clickBox : REGIONS[w][1], ...WHAT[el("what").value][6](+el("amount").value) };
    },
    click({ lat, lng }) {                             // an 8 x 8 degree box around the click, inside the grid
      if (!custom() || el("where").value !== "click") return;
      const la = Math.min(30, Math.max(6, Math.round(lat) - 4)), lo = Math.min(93, Math.max(65, Math.round(lng) - 4));
      clickBox = [la, la + 8, lo, lo + 8];
      run();
    },
  };
})();
