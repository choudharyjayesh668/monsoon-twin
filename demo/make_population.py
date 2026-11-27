"""Writes demo/data/population.js: the people in each cell of the twin's grid, for the flood page (demo/flood.js).

  python demo/make_population.py path/to/ind_ppp_2011_1km_Aggregated.tif

Source: the WorldPop 2011 estimate for India on a 1 km grid (CC BY 4.0), built from census counts. Download it by hand
(nothing here goes to the network):
  https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2011/IND/ind_ppp_2011_1km_Aggregated.tif
Each 1 km pixel is added to the 1-degree cell that holds its centre. Needs rasterio (BSD-3), for local use only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

NLAT, NLON, LAT_EDGE, LON_EDGE = 32, 36, 6.0, 65.0       # the twin's grid: south and west outer edges, 1-degree cells

with rasterio.open(sys.argv[1]) as src:
    people = src.read(1, masked=True).filled(0).astype("f8")
    t = src.transform
assert t.b == 0 and t.d == 0 and t.e < 0, "expected a north-up grid in degrees"
rows, cols = np.indices(people.shape)
lat, lon = t.f + (rows + 0.5) * t.e, t.c + (cols + 0.5) * t.a          # pixel centres
i, j = np.floor(lat - LAT_EDGE).astype(int), np.floor(lon - LON_EDGE).astype(int)
ok = (people > 0) & (i >= 0) & (i < NLAT) & (j >= 0) & (j < NLON)
grid = np.zeros((NLAT, NLON))
np.add.at(grid, (i[ok], j[ok]), people[ok])

total = people[people > 0].sum()
assert abs(grid.sum() - total) < 1e-6 * total, "every Indian pixel must fall inside the grid"
assert 1.0e9 < total < 1.4e9, f"India in 2011 had about 1.2 billion people, the file gives {total:.3g}"

out = {"year": 2011, "source": "WorldPop 2011, India, 1 km (CC BY 4.0)", "people": np.round(grid).astype(int).tolist()}
Path(__file__).with_name("data").joinpath("population.js").write_text(
    "// People per cell of the twin's grid, written by demo/make_population.py from WorldPop 2011 (CC BY 4.0).\n"
    "window.POPULATION = " + json.dumps(out, separators=(",", ":")) + ";\n")
print(f"total {total / 1e9:.3f} billion people in {int((grid > 0).sum())} cells; largest cell {grid.max() / 1e6:.1f} million")
