"""Export SwellCast's live instrument for the website: float16 weights, the six real windows as the network sees them
(the NDBC reading pipeline runs here, in Python), and reference outputs for the browser parity test.

Each window is exported 48 h wider on both sides (16 extra 3-hour steps) so the browser can slide the window in time.
For Hurricane Olivia at buoy 51004 the extra hours are real (cut from the raw NDBC files by pipeline.window); the five
pre-cut exam windows have nothing outside their 240 hours, so their margins are missing data (NaN), which is what the
pipeline itself writes for hours the buoy did not report.

    python models/swellcast/web/export_instrument.py      # writes site/public/instruments/swellcast/
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import pipeline as PL  # noqa: E402
from swellcast import SwellCast, features  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("swellcast")
NT, PAD = 80, 16  # window steps (3 h) and the extra steps exported on each side

# NDBC station names (for the picker)
BUOY = {"51004": "SE Hawaii", "41010": "Canaveral East", "41047": "NE Bahamas", "44008": "Nantucket SE",
        "46025": "Santa Monica Basin", "46047": "Tanner Bank"}


def wide_window(rec, centre_hour):
    """(3, NT + 2 PAD, 34) — the same 3-hour blocks pipeline.window cuts, over a longer span"""
    tc = int(np.floor(centre_hour / PL.DT_H) * PL.DT_H)
    blocks = [PL.window(rec, tc + s * PL.DT_H * NT)[0] for s in (-1, 0, 1)]
    w = np.concatenate(blocks, axis=1)[:, NT - PAD:2 * NT + PAD]
    for s in (-PAD, -7, 0, 5, PAD):  # check: every slice equals the pipeline's own window at that centre
        a = w[:, PAD + s:PAD + s + NT]; b = PL.window(rec, tc + s * PL.DT_H)[0]
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b)), s
    return w.astype(np.float32), tc


def variants(w):
    """deterministic parity variants: (name, (3, 80, 34) window)"""
    x = w[:, PAD:PAD + NT]
    end = x.copy(); end[:, 48:] = np.nan  # the buoy feed stops 24 h after the window centre
    shift = w[:, PAD + 8:PAD + 8 + NT]     # window slid 24 h later
    return [("raw", x), ("end24", end), ("shift24", shift)]


def main():
    net = SwellCast()
    W = dict(net.W)
    net.W = roundtrip(W, "f16")  # the browser sees float16 weights; reference outputs use the same values
    meta = dict(net.meta)
    write_pack(OUT / "weights", W, dtype="f16", meta=meta)

    items = []
    d = os.path.join(MODEL, "examples", "raw")
    rec = PL.read_ndbc(os.path.join(d, "51004w2018.txt.gz"), os.path.join(d, "51004d2018.txt.gz"), os.path.join(d, "51004j2018.txt.gz"))
    om = json.load(open(os.path.join(d, "olivia.json")))
    w, tc = wide_window(rec, om["window_centre_hour"])
    t = om["truth"]
    items.append((w, tc, dict(name="Olivia", buoy="51004", basin="EP", year=2018, raw=True,
                              truth=dict(D=t["D"], t0=t["t0"], dir=t["dir"]))))
    for f in sorted(glob.glob(os.path.join(MODEL, "examples", "*.npz"))):
        z = np.load(f); t = json.loads(str(z["truth"]))
        w = np.full((3, NT + 2 * PAD, 34), np.nan, np.float32); w[:, PAD:PAD + NT] = z["x"]
        items.append((w, int(z["centre_hour"]), dict(name=t["name"].title(), buoy=t["buoy"], basin=t["basin"], year=t["year"], raw=False,
                                                     truth=dict(D=t["D"], t0=t["t0"], dir=t["dir"]))))

    xs, ex, cases = {}, [], []
    for i, (w, tc, m) in enumerate(items):
        xs[f"x{i}"] = w
        for variant, x in variants(w):
            o = net.forward(features(x[None]))[0]
            cases.append({"example": i, "variant": variant, "outputs": o.tolist()})
        r = net.read(w[:, PAD:PAD + NT])
        ex.append({**m, "place": BUOY.get(m["buoy"], ""), "centre_hour": tc, "centre": PL.utc(tc),
                   "python": {k: r[k] for k in ("p_swell", "D_km", "logD_sig", "t0", "t0_sig", "dir", "kappa", "hs", "loghs_sig")}})
    write_pack(OUT / "examples", xs, dtype="f32", meta={"pad": PAD, "nt": NT, "fgrid": PL.FGRID.tolist(), "dt_h": PL.DT_H, "examples": ex})
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    size = sum(os.path.getsize(p) for p in OUT.iterdir()) / 1e6
    print(f"wrote {OUT}: {len(ex)} windows, {len(cases)} parity cases, {size:.2f} MB in all")
    for e in ex:
        p = e["python"]
        print(f"  {e['name']:9s} {e['buoy']}  score {p['p_swell']:.2f}  D {p['D_km']:5.0f} (truth {e['truth']['D']:5.0f})  "
              f"t0 {p['t0']:+5.0f} (truth {e['truth']['t0']:+4.0f})  dir {p['dir']:4.0f} (truth {e['truth']['dir']:4.0f})")


if __name__ == "__main__":
    main()
