"""Export PlantDetective's live instrument for the website: the 3-network ensemble in float16, the three real exam units
in examples/ (EPA CAMPD hourly gross output 2018-2025, with the ERCOT hub price and Henry Hub gas the exam fed), and
reference outputs for the browser parity test.

    python models/plantdetective/web/export_instrument.py      # writes site/public/instruments/plantdetective/

The browser recomputes everything from the hourly output: the 170 window features, the ensemble, the pooling over
windows and the readings. The controls change the INPUT: which stretch of the record the Detective sees (30 days,
91 days, one calendar year or all eight years), and whether it is also given prices and the EIA-860/923 facts.
cut() below is the one definition of what each control does (ui.ts mirrors it).
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

import pdcore  # noqa: E402
from plantdetective import Detective, window_vector, ens_forward, pool, tech_rule  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("plantdetective")
FLEX = ["baseload", "load-following", "cycling", "peaking", "unknown"]
LENGTH_H = {"month": 30 * 24, "quarter": 91 * 24}


def month_index(t):
    """UTC calendar month of each hour, counted from January 1970."""
    return np.asarray(t, "int64").astype("datetime64[s]").astype("datetime64[M]").astype(int)


def cut(n, t, p):
    """The instrument's window control -> (start, length) of the stretch the Detective is given.
    p: span 'all' | 'year' | 'quarter' | 'month'; start: first UTC month (months since Jan 1970) or year (span 'year').
    month = 30 days and quarter = 91 days from the first hour of that month; year = that calendar year (UTC)."""
    if p["span"] == "all":
        return 0, n
    if p["span"] == "year":
        yr = np.asarray(t, "int64").astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
        i = np.flatnonzero(yr == p["start"])
        return int(i[0]), int(len(i))
    s = int(np.flatnonzero(month_index(t) == p["start"])[0])
    return s, min(LENGTH_H[p["span"]], n - s)


def outputs(det, x, t, price, gas, ctx):
    """Parity vector: the features of every window the Detective cuts, the pooled ensemble outputs (technology
    probabilities before the EIA-860 rule, transformed means, raw sigmas), then the reading: technology after the rule,
    each estimate's value / lo68 / hi68 / lo95 / hi95, capacity (MW) and the flexibility class index."""
    wins = det.windows(t)
    X = []
    for s, L in wins:
        sl = slice(s, s + L)
        X.append(window_vector(x[sl], t[sl], None if price is None else price[sl], None if gas is None else gas[sl], ctx))
    X = np.array(X)
    p, m, sg = ens_forward(det.Ws, X)
    pc, mm, ss = pool(p, m, sg)
    r = det.infer(x, t, price, gas, ctx)
    est = [v for k in pdcore.REG for v in (r["estimates"][k][q] for q in ("value", "lo68", "hi68", "lo95", "hi95"))]
    cap = r["capacity_MW"] if np.isfinite(r["capacity_MW"]) else -1.0   # NaN (fewer than 11 running hours) -> -1
    fin = [r["tech"][k] for k in pdcore.TECHS] + est + [cap, FLEX.index(r["flex_class"])]
    vec = list(X.ravel()) + list(pc) + list(mm) + list(ss) + fin
    return [float(v) for v in vec], r, len(wins)


def main():
    full = Detective()  # untouched float32 weights, to report what float16 costs
    det = Detective()
    z = np.load(os.path.join(MODEL, "plantdetective_weights.npz"), allow_pickle=False)
    W, keep = {}, {}
    for i in range(int(z["n_members"])):
        for k in ("mu", "sd", "l0w", "l0b", "l1w", "l1b", "l2w", "l2b", "l3w", "l3b"):
            W[f"m{i}_{k}"] = z[f"m{i}_{k}"]
        keep[f"m{i}_mu"] = keep[f"m{i}_sd"] = "f32"  # normalisation stays float32 (exact); layers go float16
    meta = {"members": int(z["n_members"]), "factors": det.factors, "techs": pdcore.TECHS, "reg": pdcore.REG,
            "n_params": int(sum(v.size for v in W.values())), "flex": FLEX,
            # exact float64 constants the features compare against (np.linspace edges, not recomputed in JS)
            "loadbins": pdcore.LOADBINS.tolist(), "run_edges": pdcore.RUN_EDGES.tolist(), "zbins": pdcore.ZBINS.tolist(),
            "ctx_keys": pdcore.CTX_KEYS, "hot_h": pdcore.HOT_H, "warm_h": pdcore.WARM_H, "part_min": pdcore.PART_MIN}
    write_pack(OUT / "weights", W, dtype="f16", meta=meta, keep=keep)

    # the browser sees float16 layers: reference outputs use the same values
    R = roundtrip(W, "f16")
    det.Ws = [{k: (W if keep.get(f"m{i}_{k}") == "f32" else R)[f"m{i}_{k}"]
               for k in ("mu", "sd", "l0w", "l0b", "l1w", "l1b", "l2w", "l2b", "l3w", "l3b")} for i in range(len(det.Ws))]

    files = sorted(glob.glob(os.path.join(MODEL, "examples", "*.npz")))
    zs = [np.load(f, allow_pickle=False) for f in files]
    t = zs[0]["t"].astype("int64")
    for q in zs:   # one ERCOT price / Henry Hub series and one hourly UTC grid serve all three units
        assert (q["t"] == t).all() and (q["price"] == zs[0]["price"]).all() and (q["gas"] == zs[0]["gas"]).all()
    assert (np.diff(t) == 3600).all(), "regular hourly grid: the browser rebuilds t from t0"
    price = zs[0]["price"].astype(float); gas = zs[0]["gas"].astype(float)
    xs = {"price": zs[0]["price"], "gas": zs[0]["gas"]}
    meta_ex, cases, f16diff = [], [], []
    mi = month_index(t)
    for i, q in enumerate(zs):
        m = json.loads(str(q["meta"]))
        x = q["x"].astype(float)
        xs[f"x{i}"] = q["x"].astype(np.float32)   # integer MW, exact in float32
        variants = [
            ("all", {"span": "all", "ctx": True, "price": True}),
            ("all-output-only", {"span": "all", "ctx": False, "price": False}),
            ("year2024", {"span": "year", "start": 2024, "ctx": True, "price": True}),
            ("quarter2023-07", {"span": "quarter", "start": int(mi[np.flatnonzero(t >= 1688169600)[0]]), "ctx": True, "price": True}),
            ("month2025-01", {"span": "month", "start": int(mi[np.flatnonzero(t >= 1735689600)[0]]), "ctx": True, "price": True}),
            ("month2019-04-no-ctx", {"span": "month", "start": int(mi[np.flatnonzero(t >= 1554076800)[0]]), "ctx": False, "price": True}),
        ]
        for name, p in variants:
            s, L = cut(len(t), t, p)
            sl = slice(s, s + L)
            ctx = m["context"] if p["ctx"] else None
            o, r, nw = outputs(det, x[sl], t[sl], price[sl] if p["price"] else None, gas[sl] if p["price"] else None, ctx)
            cases.append({"example": i, "variant": name, "p": p, "start": s, "hours": L, "windows": nw, "outputs": o})
            if name == "all":
                py = r
                rf = full.infer(x, t, price, gas, m["context"])
                f16diff.append(max(abs(r["estimates"][k]["value"] / rf["estimates"][k]["value"] - 1) for k in pdcore.REG))
        meta_ex.append({
            "id": m["id"], "name": m["name"], "unit": m["unit"], "tech": m["tech"], "years": m["years"],
            "context": m["context"], "truth": m["truth"], "flex_true": m["flex_true"],
            "python": {"tech": py["tech"], "estimates": {k: py["estimates"][k]["value"] for k in pdcore.REG},
                       "flex": py["flex_class"]},
        })
    write_pack(OUT / "examples", xs, dtype="f32", meta={"t0": int(t[0]), "n": int(len(t)), "units": meta_ex})
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    tot = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {len(zs)} units, {len(cases)} parity cases, {tot / 1e6:.2f} MB total; "
          f"float16 vs float32 weights, estimates on all years: max {max(f16diff):.1e} relative")


if __name__ == "__main__":
    main()
