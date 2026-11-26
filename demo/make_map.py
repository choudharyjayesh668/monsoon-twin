"""Writes demo/data/borders.js: country borders around the twin's grid, for the demo map.

Source: Natural Earth 1:10m admin-0 countries, India point of view (public domain):
https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries_ind.geojson

  python demo/make_map.py path/to/ne_10m_admin_0_countries_ind.geojson
"""
import json
import sys
from pathlib import Path

BOX = (58.0, 0.0, 108.0, 44.0)      # lon_min, lat_min, lon_max, lat_max: the grid (65-101 E, 6-38 N) plus a margin


def rings(geom):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [r for p in polys for r in p]


def thin(ring):
    """Round to 0.02 degrees (about 2 km) and drop repeated points."""
    out = []
    for x, y in ring:
        p = [round(x, 2), round(y, 2)]
        if not out or p != out[-1]:
            out.append(p)
    return out if len(out) >= 4 else None


src = json.loads(Path(sys.argv[1]).read_text())
features = []
for f in src["features"]:
    keep = []
    for r in rings(f["geometry"]):
        xs, ys = [p[0] for p in r], [p[1] for p in r]
        if max(xs) >= BOX[0] and min(xs) <= BOX[2] and max(ys) >= BOX[1] and min(ys) <= BOX[3]:
            t = thin(r)
            if t:
                keep.append([t])
    if keep:
        features.append({"type": "Feature", "properties": {"name": f["properties"].get("NAME_EN") or f["properties"]["NAME"]},
                         "geometry": {"type": "MultiPolygon", "coordinates": keep}})
out = Path(__file__).parent / "data" / "borders.js"
out.write_text("// Country borders, Natural Earth 1:10m, India point of view (public domain).\nwindow.BORDERS = "
               + json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")) + ";\n")
print(len(features), "countries:", ", ".join(sorted(f["properties"]["name"] for f in features)), f"| {out.stat().st_size // 1024} KB")
