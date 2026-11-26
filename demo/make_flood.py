"""Writes demo/data/flood.js: the thresholds of the flood-risk module (demo/flood.js).

  python demo/make_flood.py

A cell's alert level compares its 3-day rain total with the totals that are rare for that cell in that month
(2005-2021, the month and its 2 neighbours pooled), with a floor in mm so that dry places and seasons do not raise an
alert for a little rain. The twin's forecast extremes weaken with lead time, so the twin's thresholds are scaled by
lead: the ratio of the twin's forecast percentiles to the real ones (June-September, land), so that an alert in the
twin is as rare as an alert in the real weather.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train  # noqa: E402
import twin  # noqa: E402

LEVELS = ["Watch", "Warning", "Severe"]
PCT = [97, 99, 99.7]                     # local percentile of the 3-day rain total, per level
FLOOR = [40, 70, 100]                    # mm in 3 days, per level: the minimum for an alert anywhere
DAYS, EVERY = 10, 3                      # forecast days; one forecast start every 3 days for the lead factors

X, days, static = train.load()
tp, land = X[:, 1], static[0] > 0.5
r3 = np.full_like(tp, np.nan)
r3[2:] = tp[2:] + tp[1:-1] + tp[:-2]
month = np.array([d.month for d in days])
ok = ~np.isnan(r3[:, 0, 0])

thr = []
for m in range(1, 13):
    sel = ok & np.isin(month, [(m - 2) % 12 + 1, m, m % 12 + 1])
    p = np.percentile(r3[sel], PCT, axis=0)                                       # [3,32,36]
    thr.append(np.maximum(p, np.array(FLOOR)[:, None, None]))
thr = np.array(thr)                                                                # [12,3,32,36]
assert (np.diff(thr, axis=1) >= 0).all(), "levels must not overlap"

# lead factors: forecast 3-day totals (2 real input days, then the twin) against the real ones
torch.set_num_threads(4)
model = twin.load()
starts = np.arange(2, len(days) - DAYS - 1, EVERY)
fc = np.zeros((len(starts), DAYS, 32, 36), "f4")
for s in range(0, len(starts), 256):
    idx = starts[s:s + 256]
    a = [2 * math.pi * days[i].timetuple().tm_yday / 365 for i in idx]
    fc[s:s + 256] = model.forecast(np.stack([X[i - 2:i + 1] for i in idx]), np.c_[np.sin(a), np.cos(a)], DAYS)[:, :, 1]
wet = np.isin(month[starts], [6, 7, 8, 9])
full = np.concatenate([np.stack([tp[i - 1:i + 1] for i in starts]), fc], 1)          # [n, 2 + DAYS, 32, 36]
lead = []
for k in range(DAYS):
    f3 = (full[:, k] + full[:, k + 1] + full[:, k + 2])[wet][:, land]
    t3 = np.stack([r3[i + 1 + k] for i in starts])[wet][:, land]
    lead.append(np.percentile(f3, PCT) / np.percentile(t3, PCT))
lead = np.minimum(np.array(lead), 1.0)                                              # never stricter than the real

out = {"levels": LEVELS, "pct": PCT, "floor_mm": FLOOR,
       "thr": np.round(thr, 1).tolist(), "lead": np.round(lead, 3).tolist(), "land": land.astype(int).tolist()}
Path(__file__).with_name("data").joinpath("flood.js").write_text(
    "// Flood-risk thresholds, written by demo/make_flood.py. thr[month][level][lat][lon]: mm in 3 days;\n"
    "// lead[day][level]: the factor for the twin's forecast on that day; land: 1 = land.\n"
    "window.FLOOD = " + json.dumps(out, separators=(",", ":")) + ";\n")
print("lead factors (watch, warning, severe):\n", np.round(lead, 2))
