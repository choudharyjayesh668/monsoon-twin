"""Writes demo/windows.js: real 3-day input windows plus the 10 days that followed, for the demo pages.

  python demo/make_windows.py

Each window also carries rmse_clim [day][channel]: the error of the climatology forecast, from the organisers' kit
functions, so that the page can score the twin's live forecast exactly as the kit does (it needs the kit wheel).
"""
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
from monsoon_r1.scoring import rmse_lat
from monsoon_r1.stats import clim_for, climatology

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train  # noqa: E402

WINDOWS = [  # (last input day, label): one window for each kind of hidden test the organisers name
    (dt.date(2021, 6, 1), "Monsoon onset over Kerala, June 2021"),
    (dt.date(2021, 7, 19), "Heavy rain on the Konkan coast, July 2021"),
    (dt.date(2020, 5, 23), "Heatwave in north-west India, May 2020"),
    (dt.date(2020, 5, 16), "Cyclone Amphan, Bay of Bengal, May 2020"),
    (dt.date(2020, 1, 3), "Winter rain in north-west India, January 2020"),
    # past floods for the flood-risk view, each about 3 days before its peak (all in the shipped model's training years)
    (dt.date(2013, 6, 14), "Flood: Uttarakhand, June 2013"),
    (dt.date(2014, 9, 2), "Flood: Jammu and Kashmir, September 2014"),
    (dt.date(2018, 8, 13), "Flood: Kerala, August 2018"),
    (dt.date(2019, 8, 3), "Flood: Kolhapur and Sangli, Maharashtra, August 2019"),
]
DAYS = 10                                                   # the twin's 10-day forecast, compared day by day

X, days, static = train.load()
clim = climatology(X, np.array([int(d.strftime("%Y%m%d")) for d in days]))      # the grader's: all of 2005-2021
out = []
for last, label in WINDOWS:
    i = int(np.where(days == last)[0][0])
    a = 2 * math.pi * last.timetuple().tm_yday / 365
    out.append({"label": label, "doy_sincos": [round(math.sin(a), 6), round(math.cos(a), 6)],
                "init_states": np.round(X[i - 2:i + 1].astype("f8"), 2).tolist(),
                "actual": np.round(X[i + 1:i + 1 + DAYS].astype("f8"), 2).tolist(),
                "dates": [(last + dt.timedelta(days=k)).isoformat() for k in range(1, DAYS + 1)],
                "rmse_clim": [np.round(rmse_lat(clim_for(clim, last + dt.timedelta(days=k)), X[i + k]).astype("f8"), 4).tolist()
                              for k in range(1, DAYS + 1)]})
Path(__file__).with_name("windows.js").write_text(
    "// Real input windows (3 days, oldest first) and the 10 days that followed, in physical units.\n"
    "window.WINDOWS = " + json.dumps(out, separators=(",", ":")) + ";\n"
    "// Land-sea mask (1 = land), to draw the coastline.\n"
    "window.LAND = " + json.dumps((static[0] > 0.5).astype(int).tolist(), separators=(",", ":")) + ";\n")
print("wrote", len(out), "windows")

