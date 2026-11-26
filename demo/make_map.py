"""Writes the demo map's borders, thinned to a light size:

  demo/data/borders.js  neighbouring countries: Natural Earth 1:10m admin-0, India point of view (public domain)
  demo/data/india.js    India's outline and its 28 states and 8 union territories, as India draws them:
                        DataMeet India community (CC BY 4.0), https://github.com/datameet/maps

Sources (download them first; nothing is fetched here):
  https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries_ind.geojson
  https://github.com/datameet/maps/tree/master/States       Admin2.shp, Admin2.dbf
  https://github.com/datameet/maps/blob/master/Country/india-composite.geojson

  python demo/make_map.py NE.geojson path/to/Admin2 path/to/india-composite.geojson
"""
import json
import struct
import sys
from pathlib import Path

BOX = (58.0, 0.0, 108.0, 44.0)      # lon_min, lat_min, lon_max, lat_max: the grid (65-101 E, 6-38 N) plus a margin
UNION_TERRITORIES = {"Andaman & Nicobar", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
                     "Jammu & Kashmir", "Ladakh", "Lakshadweep", "Puducherry"}


def thin(ring, step):
    """Round to `step` degrees and drop repeated points; None if the ring collapses."""
    out = []
    for x, y in ring:
        p = [round(round(x / step) * step, 3), round(round(y / step) * step, 3)]
        if not out or p != out[-1]:
            out.append(p)
    return out if len(out) >= 4 else None


def multipolygon(rings_by_polygon, step):
    polys = []
    for rings in rings_by_polygon:
        t = [r for r in (thin(r, step) for r in rings) if r]
        if t:
            polys.append(t)
    return polys


def read_dbf(path):
    b = Path(path).read_bytes()
    n, header, size = struct.unpack("<IHH", b[4:12])
    fields = [(b[o:o + 11].split(b"\0")[0].decode(), b[o + 16]) for o in range(32, header - 1, 32)]
    rows = []
    for r in range(n):
        rec, pos, row = b[header + r * size + 1: header + (r + 1) * size], 0, {}
        for name, width in fields:
            row[name] = rec[pos:pos + width].decode("latin-1").strip(); pos += width
        rows.append(row)
    return rows


def read_shp(path):
    """Polygon shapefile -> list of shapes, each a list of rings [[x, y], ...] (each ring kept as its own polygon)."""
    b, o, shapes = Path(path).read_bytes(), 100, []
    while o < len(b):
        _, words = struct.unpack(">ii", b[o:o + 8]); o += 8
        rec = b[o:o + 2 * words]; o += 2 * words
        if struct.unpack("<i", rec[:4])[0] == 0:
            shapes.append([]); continue
        nparts, npoints = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts])) + [npoints]
        xy = struct.unpack(f"<{2 * npoints}d", rec[44 + 4 * nparts:44 + 4 * nparts + 16 * npoints])
        shapes.append([[[xy[2 * k], xy[2 * k + 1]] for k in range(parts[p], parts[p + 1])] for p in range(nparts)])
    return shapes


def neighbours(ne_path):
    src = json.loads(Path(ne_path).read_text())
    features = []
    for f in src["features"]:
        name = f["properties"].get("NAME_EN") or f["properties"]["NAME"]
        if name == "India":
            continue                                        # India comes from india.js
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        near = [p for p in polys if any(BOX[0] <= x <= BOX[2] and BOX[1] <= y <= BOX[3] for x, y in p[0])]
        if near:
            mp = multipolygon([[p[0]] for p in near], 0.02)
            if mp:
                features.append({"type": "Feature", "properties": {"name": name}, "geometry": {"type": "MultiPolygon", "coordinates": mp}})
    return features


def india(states_base, outline_path):
    rows, shapes = read_dbf(states_base + ".dbf"), read_shp(states_base + ".shp")
    states = []
    for row, rings in zip(rows, shapes):
        name = row["ST_NM"]
        mp = multipolygon([[r] for r in rings], 0.01)
        states.append({"type": "Feature", "properties": {"name": name,
                       "kind": "union territory" if name in UNION_TERRITORIES else "state"},
                       "geometry": {"type": "MultiPolygon", "coordinates": mp}})
    outline = json.loads(Path(outline_path).read_text())
    polys = []
    for f in outline["features"]:
        g = f["geometry"]
        polys += [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    mp = multipolygon([[p[0]] for p in polys], 0.01)
    return {"outline": {"type": "Feature", "properties": {"name": "India"}, "geometry": {"type": "MultiPolygon", "coordinates": mp}},
            "states": {"type": "FeatureCollection", "features": states}}


if __name__ == "__main__":
    ne, states_base, outline = sys.argv[1:4]
    data = Path(__file__).parent / "data"
    n = neighbours(ne)
    (data / "borders.js").write_text("// Neighbouring countries, Natural Earth 1:10m, India point of view (public domain).\nwindow.BORDERS = "
                                     + json.dumps({"type": "FeatureCollection", "features": n}, separators=(",", ":")) + ";\n")
    ind = india(states_base, outline)
    (data / "india.js").write_text("// India's outline, states and union territories: DataMeet India community, CC BY 4.0,\n"
                                   "// https://github.com/datameet/maps\nwindow.INDIA = " + json.dumps(ind, separators=(",", ":")) + ";\n")
    kinds = [f["properties"]["kind"] for f in ind["states"]["features"]]
    print(f"{len(n)} neighbours | {kinds.count('state')} states, {kinds.count('union territory')} union territories | "
          f"borders.js {(data / 'borders.js').stat().st_size // 1024} KB, india.js {(data / 'india.js').stat().st_size // 1024} KB")
