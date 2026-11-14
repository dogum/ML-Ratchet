"""Export GridPulse's live instrument for the website: the main network in float16, real GB exam events as 81-sample
1 s windows (-20..+60 s around the trip), and reference outputs for the browser parity test.

    python models/gridpulse/web/export_instrument.py      # writes site/public/instruments/gridpulse/

Events: the six in examples/gb_exam_examples.npz (the release's seeded draw, two per drop-size stratum) and four more
drawn with seed 20261114 from the exam events that carry a medium-confidence loss label and NESO inertia (needs
shared/grid/data; never dev). Windows are made the way exam.py made them (griddata onset, training/measure.window).
The extra-noise control adds a real quiet NESO slice from the frozen-set noise (lab/noise.json, 2022-01, mHz).
perturb() below is the one definition of what each control does (net.ts mirrors it).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, REPO)

from gridpulse import GridPulse, PRE, POST  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("gridpulse")
SEED, N_EXTRA = 20261114, 4
NAMES = ["loss", "E", "beta", "head", "ffr", "nadir", "settle"]


def _platform():
    sys.path.insert(0, os.path.join(MODEL, "training"))
    import _paths  # noqa: F401  (puts shared/grid on sys.path)
    import griddata as gd
    import measure as M
    return gd, M


def window_of(eid):
    """evaluate.real_gb without torch: griddata's onset (event_t0_utc) -> training/measure.window dict."""
    import pandas as pd
    gd, M = _platform()
    meta, t, f = gd.event_trace(eid)
    off = int(round((pd.Timestamp(meta["event_t0_utc"]) - pd.Timestamp(meta["start_utc"])).total_seconds()))
    x = (f - 50.0)[None, :].astype(np.float64)
    if np.isnan(x).any():
        x = pd.Series(x[0]).interpolate(limit_direction="both").values[None, :]
    return M.window(x, np.array([int(np.where(np.round(t) == off)[0][0])]))


def extra_events(skip):
    """N_EXTRA more exam events with a medium loss label and NESO inertia, drawn by seed (sorted by id first)."""
    import pandas as pd
    gd, _ = _platform()
    ev = gd.events("GB", split="exam")
    ev = ev[(ev.loss_confidence == "medium") & ev.inertia_outturn_GVAs.notna() & ~ev.event_id.isin(skip)]
    ev = ev.sort_values("event_id").reset_index(drop=True)
    pick = np.sort(np.random.default_rng(SEED).choice(len(ev), N_EXTRA, replace=False))
    gm = gd.gb_genmix().set_index("time_utc")["GENERATION"]
    out = []
    for _, r in ev.iloc[pick].iterrows():
        m = window_of(r.event_id)
        out.append(dict(id=r.event_id, w=m["w"][0].astype(np.float64), demand=float(gm.get(pd.Timestamp(r.t0_utc).floor("30min"), np.nan)),
                        E_true=float(r.inertia_outturn_GVAs), loss=float(r.loss_MW), tier="medium",
                        nadir=float(m["nadir"][0]), settle=float(m["settle"][0]), source=f"seeded draw {SEED} (medium labels)"))
    return out


def perturb(w, demand, noise, p):
    """The instrument's controls on the model's inputs. p: N (seconds after the trip the network sees, 2..60),
    demand (False = unknown: the network runs without it), noise (scale on a real quiet NESO slice, mHz -> Hz,
    added to the whole window)."""
    w = np.asarray(w, np.float64).copy()
    if p.get("noise"):
        w = w + p["noise"] * np.asarray(noise, np.float64) / 1000.0
    d = demand if p.get("demand", True) else np.nan
    return w, int(p.get("N", 60)), d


def raw(gp, w, N, d):
    """Raw network outputs (target space, sigma widened) for one window: [mu x7, sigma x7]."""
    mu, sg = gp.forward(w[None], np.array([N]), np.array([1.0]), np.array([50.0]), np.array([d]))
    return [float(v) for v in mu[0]] + [float(v) for v in sg[0]]


def main():
    gp = GridPulse()
    W = {k: v for k, v in gp.W.items() if k != "ctx"}
    keep = {k: "f32" for k in ("xm", "xs", "ym", "ys", "widen")}  # normalisation stays float32; layers go float16
    n_params = int(sum(v.size for k, v in W.items() if k not in keep))
    write_pack(OUT / "weights", W, dtype="f16", keep=keep,
               meta={"ctx": bool(gp.ctx), "n_params": n_params, "names": NAMES, "pre": PRE, "post": POST,
                     "layers": sum(1 for k in W if k.endswith("_w") and k.startswith("b"))})

    full = GridPulse()  # untouched weights, to report what float16 costs
    R = roundtrip(W, "f16")
    for k in W:
        gp.W[k] = R[k] if k not in keep else W[k]

    Z = np.load(os.path.join(MODEL, "examples", "gb_exam_examples.npz"))
    evs = [dict(id=str(Z["event_id"][i]), w=Z["w"][i].astype(np.float64), demand=float(Z["demand"][i]), E_true=float(Z["E_true"][i]),
                loss=float(Z["loss_label"][i]), tier=str(Z["tier"][i]), nadir=float(Z["nadir"][i]), settle=float(Z["settle"][i]),
                source="examples/ (release draw, seed 7)") for i in range(len(Z["event_id"]))]
    # the extra windows are made the way the release examples were: check on every release event
    for e in evs:
        m = window_of(e["id"])
        assert np.abs(m["w"][0] - e["w"]).max() < 1e-6 and abs(m["nadir"][0] - e["nadir"]) < 1e-6, e["id"]
    evs += extra_events([e["id"] for e in evs])
    noise = json.load(open(os.path.join(MODEL, "lab", "noise.json")))
    nz = [np.asarray(noise[i], np.float64)[60 - PRE:60 + POST + 1] for i in range(len(evs))]

    xs, meta_ex, cases, f16 = {}, [], [], []
    variants = [("raw", {}), ("N5", {"N": 5}), ("N10", {"N": 10}), ("N2", {"N": 2}), ("N10-nodemand", {"N": 10, "demand": False}),
                ("N10-noise2", {"N": 10, "noise": 2.0}), ("N60-noise3-nodemand", {"N": 60, "noise": 3.0, "demand": False})]
    for i, e in enumerate(evs):
        xs[f"w{i}"] = np.asarray(e["w"], np.float32)
        xs[f"n{i}"] = nz[i].astype(np.float32)
        we = np.asarray(xs[f"w{i}"], np.float64)  # the browser sees float32 windows
        for name, p in variants:
            w, N, d = perturb(we, e["demand"], xs[f"n{i}"], p)
            cases.append({"example": i, "variant": name, "p": p, "outputs": raw(gp, w, N, d)})
            f16.append(np.abs(np.array(raw(gp, w, N, d)) - np.array(raw(full, w, N, d))).max())
        r60 = full.read_windows(we[None], 60, "GB", e["demand"])
        meta_ex.append({
            "id": e["id"], "t": f"{e['id'][3:7]}-{e['id'][7:9]}-{e['id'][9:11]} {e['id'][12:14]}:{e['id'][14:16]}",
            "demand_MW": e["demand"], "E_true_GVAs": e["E_true"],
            "loss_label_MW": None if not np.isfinite(e["loss"]) else e["loss"], "tier": None if e["tier"] == "nan" else e["tier"],
            "nadir_Hz": e["nadir"], "settle_Hz": e["settle"], "source": e["source"],
            "python_N60": {k: float(v[0]) for k, v in r60.items()},
        })
    write_pack(OUT / "examples", xs, dtype="f32", meta=meta_ex)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    tot = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {len(evs)} events, {len(cases)} parity cases, {tot / 1e6:.2f} MB total; "
          f"float16 vs float32 weights: max raw-output difference {max(f16):.1e}")


if __name__ == "__main__":
    main()
