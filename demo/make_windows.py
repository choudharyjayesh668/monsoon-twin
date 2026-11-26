"""Writes demo/windows.js: a few real 3-day input windows plus the 8 days that followed, for the demo page.

  python demo/make_windows.py
"""
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train  # noqa: E402

WINDOWS = [  # (last input day, label)
    (dt.date(2020, 5, 17), "Cyclone Amphan, May 2020"),
    (dt.date(2021, 7, 15), "Monsoon, July 2021"),
    (dt.date(2021, 1, 20), "Winter, January 2021"),
]

X, days, static = train.load()
out = []
for last, label in WINDOWS:
    i = int(np.where(days == last)[0][0])
    a = 2 * math.pi * last.timetuple().tm_yday / 365
    out.append({"label": label, "doy_sincos": [round(math.sin(a), 6), round(math.cos(a), 6)],
                "init_states": np.round(X[i - 2:i + 1].astype("f8"), 2).tolist(),
                "actual": np.round(X[i + 1:i + 9].astype("f8"), 2).tolist()})
Path(__file__).with_name("windows.js").write_text(
    "// Real input windows (3 days, oldest first) and the 8 days that followed, in physical units.\n"
    "window.WINDOWS = " + json.dumps(out, separators=(",", ":")) + ";\n"
    "// Land-sea mask (1 = land), to draw the coastline.\n"
    "window.LAND = " + json.dumps((static[0] > 0.5).astype(int).tolist(), separators=(",", ":")) + ";\n")
print("wrote", len(out), "windows")
