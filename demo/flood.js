"use strict";
// Flood-risk module: turns a rain forecast from the twin's /simulate into flood alert levels, per grid cell and per state.
// It is a rainfall-driven hazard index on the 1-degree grid. It knows no rivers, drainage, dams or people.
// Thresholds: demo/data/flood.js, written by demo/make_flood.py. People: demo/data/population.js (WorldPop 2011).
const FloodRisk = (() => {
  const F = window.FLOOD, NAMES = ["None", "Watch", "Warning", "Severe"];
  const NO_SCALE = [1, 1, 1];
  const doyOf = ([s, c]) => ((Math.round(Math.atan2(s, c) * 365 / (2 * Math.PI)) % 365) + 365) % 365 || 365;   // 1..365
  const monthOf = doy => new Date(Date.UTC(2001, 0, doy)).getUTCMonth();                                     // 0..11, a non-leap year

  // inputRain: the 3 input days [3][32][36]; rain: the days after them [days][32][36], mm/day; sincos: of the last
  // input day. forecast: true for the twin's own output, which uses the lead-scaled thresholds. -> [days][32][36] of 0..3
  function levels(inputRain, rain, sincos, forecast) {
    const all = [...inputRain, ...rain], d0 = doyOf(sincos);
    return rain.map((_, k) => {
      const thr = F.thr[monthOf((d0 + k) % 365 + 1)], f = forecast ? F.lead[k] : NO_SCALE;
      return F.land.map((row, i) => row.map((isLand, j) => {
        if (!isLand) return 0;
        const r3 = all[k + 1][i][j] + all[k + 2][i][j] + all[k + 3][i][j];         // this day and the 2 before it
        let lv = 0;
        for (let n = 0; n < 3; n++) if (r3 >= thr[n][i][j] * f[n]) lv = n + 1;
        return lv;
      }));
    });
  }

  // Real against forecast alerts (level >= min) on land: counts, the share of real alerts found, the share of false alarms.
  function compare(real, fc, min = 1) {
    let r = 0, t = 0, h = 0;
    F.land.forEach((row, i) => row.forEach((isLand, j) => {
      if (!isLand) return;
      const a = real[i][j] >= min, b = fc[i][j] >= min;
      r += a; t += b; h += a && b;
    }));
    return { real: r, twin: t, hits: h, found: r ? h / r : null, falseAlarm: t ? (t - h) / t : null };
  }

  // Each state's area at level >= min and its share of the state, largest area first (a share alone would put tiny
  // union territories first). cellStates[i][j]: [[state name, area], ...]
  function byState(lv, cellStates, min = 1) {
    const total = {}, hit = {};
    cellStates.forEach((row, i) => row.forEach((parts, j) => parts.forEach(([name, a]) => {
      total[name] = (total[name] || 0) + a;
      if (lv[i][j] >= min) hit[name] = (hit[name] || 0) + a;
    })));
    return Object.keys(total).map(name => ({ name, total: total[name], area: hit[name] || 0, share: (hit[name] || 0) / total[name] }))
                             .sort((x, y) => y.area - x.area);
  }

  // The states under an alert: a state's level is the highest level that covers at least 10% of its area. Union
  // territories smaller than half a grid cell are left out, because one sample point would decide them.
  // -> [{name, level 1..3, share of the state at that level}], the worst first
  const MIN_SHARE = 0.1, MIN_AREA = 0.5;
  function stateAlerts(lv, cellStates) {
    const by = [1, 2, 3].map(n => Object.fromEntries(byState(lv, cellStates, n).map(s => [s.name, s])));
    return Object.values(by[0]).filter(s => s.total >= MIN_AREA).map(s => {
      let level = 0;
      for (let n = 0; n < 3; n++) if (by[n][s.name].share >= MIN_SHARE) level = n + 1;
      return { name: s.name, level, share: level ? by[level - 1][s.name].share : 0 };
    }).filter(s => s.level).sort((a, b) => b.level - a.level || b.share - a.share);
  }

  // The people who live in cells at level >= min. Population: WorldPop 2011, a fixed map; the alerts are live.
  function people(lv, min = 1) {
    let n = 0;
    lv.forEach((row, i) => row.forEach((v, j) => { if (v >= min) n += window.POPULATION.people[i][j]; }));
    return n;
  }

  return { NAMES, levels, compare, byState, stateAlerts, people };
})();
