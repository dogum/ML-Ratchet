"""Export BuildingDetective's live instrument for the website: the three networks (hourly, monthly, annual) in float16,
the real exam units of examples/ (3 BDG2 buildings and 1 HUE home with hourly 2017 meters and weather; 4 RECS and 4
CBECS respondents with annual totals), and reference outputs for the browser parity test.

    python models/buildingdetective/web/export_instrument.py      # writes site/public/instruments/buildingdetective/

The browser computes the features itself (site/src/instruments/buildingdetective/net.ts ports features.py, starting
from the shipped JS port training/detective.js) and runs the network live. perturb() below is the one definition of
what each control does to the input (net.ts mirrors it):
  mode     'hourly' | 'monthly' | 'annual': which network reads the building. Monthly bills and the annual total are
           made from the same hourly meters the way the exam made them (training/realfeats.py).
  months   keep only the first k local calendar months of meter data (3, 6, 9 or 12; the exam also had 3- and 6-month
           copies); weather stays. One month is not offered: with only January the change-point fit is degenerate (every
           heating balance point above the warmest day fits equally well) and numpy and JS break the tie differently.
  elec     True: drop every meter but electricity (the exam's electricity-only copy).
  dT       add dT kelvin to the outdoor temperature: the same meters in a warmer or colder year.
  noise    multiplicative meter noise, x * exp(noise * z), z a fixed standard-normal series per meter (seed 20261117).
BDG2-derived data are CC BY-SA 4.0: the exporter writes LICENSE-DATA.md next to them.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, REPO)

import common as C  # noqa: E402
import features as F  # noqa: E402
from buildingdetective import Detective  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("buildingdetective")
SEED = 20261117
MODES = ("hourly", "monthly", "annual")
LAYER = ("W0", "b0", "W1", "b1", "Wr", "br", "Wc", "bc")
NORM = ("xm", "xs", "ym", "ys", "widen", "temp")

LICENCE = """# Data licence: CC BY-SA 4.0 (Building Data Genome 2) and CC BY 4.0 (HUE)

`examples.json` and `examples.bin` (BuildingDetective's live instrument) hold the hourly 2017 meters, site weather and
metadata of three exam buildings derived from the Building Data Genome 2 dataset (BDG2): Miller, C., et al. (2020),
"The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition",
Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Changes: cleaned by BuildingData (bad hours NaN, timestamps fixed
to UTC), local calendar year 2017, steam and hot water summed into one district-heat channel, float32, local time
fields added, a fixed noise series added for the noise control. You may share and adapt them under the same licence,
with this credit.

The same files hold one home from HUE (Makonin, S. (2019), Data in Brief 23, 103744; Harvard Dataverse
doi:10.7910/DVN/N3HGRN; CC BY 4.0) and eight respondents of EIA RECS 2020 and CBECS 2018 (public domain).
This licence covers the data files only; the code in this repository is MIT.
"""


def load_units():
    ex = json.load(open(os.path.join(MODEL, "examples", "examples.json")))
    bd = json.load(open(os.path.join(MODEL, "examples", "bdg2", "bdg2_hourly.json")))
    z = dict(np.load(os.path.join(MODEL, "examples", "examples.npz")))
    z.update(np.load(os.path.join(MODEL, "examples", "bdg2", "bdg2_hourly.npz")))
    units = []
    for u in bd["hourly"] + ex["hourly"]:
        loc = pd.DatetimeIndex(z[u["key"] + "_utc"].astype("datetime64[s]"), tz="UTC").tz_convert(u["tz"])
        units.append(dict(u, kind="bdg2" if u["key"].startswith("b") else "hue", loc=loc,
                          T=z[u["key"] + "_T"], G=z[u["key"] + "_GHI"],
                          meters={c: z[u["key"] + "_" + c] for c in u["channels"]}))
    for u in ex["annual"]:
        units.append(dict(u, kind=u["id"].split()[0].lower()))
    return units


def perturb(u, p, zs):
    """The controls on one unit's input. Returns (mode, feature vector) as the release code computes it."""
    mode = p.get("mode", "hourly" if "loc" in u else "annual")
    if "loc" not in u:                                    # survey respondent: annual totals only
        kwh = {c: float(v) for c, v in u["kwh"].items() if not p.get("elec") or c == "elec"}
        return "annual", F.feat_annual(kwh, u["hdd"], u["cdd"])
    loc = u["loc"]
    T = np.asarray(u["T"], np.float64) + float(p.get("dT", 0.0))
    G = np.asarray(u["G"], np.float64)
    mon = np.asarray(loc.month)
    m = {}
    for c, x in u["meters"].items():
        if p.get("elec") and c != "elec":
            continue
        x = np.asarray(x, np.float64)
        if p.get("noise"):
            x = x * np.exp(float(p["noise"]) * np.asarray(zs[c], np.float64))
        if p.get("months", 12) < 12:
            x = np.where(mon <= int(p["months"]), x, np.nan)
        m[c] = x
    if mode == "hourly":
        return mode, F.feat_hourly(m, T, G, loc, u["lat"])
    if mode == "monthly":
        mm, Tm, Hm, Cm, nd = F.monthly_from_hourly(m, T, loc)
        return mode, F.feat_monthly(mm, Tm, Hm, Cm, nd, u["lat"])
    kwh = {c: (float(np.nanmean(x) * 8760) if np.isfinite(x).mean() > 0.2 else 0.0) for c, x in m.items()}
    d = np.asarray(loc.normalize().asi8)
    _, inv = np.unique(d, return_inverse=True)
    Td = np.bincount(inv, weights=np.nan_to_num(T, nan=np.nanmean(T))) / np.bincount(inv)
    hdd, cdd = float(np.maximum(F.HB - Td, 0).sum()), float(np.maximum(Td - F.HB, 0).sum())
    return mode, F.feat_annual(kwh, hdd, cdd)


def outputs(net, X):
    """Raw outputs: 12 means, 12 widened sigmas (target space), then the class probabilities (use 12, heat 5,
    cool 4) after the dev-fitted temperatures."""
    mu, sg, lg = net.raw(X)
    nu, nh = len(C.USES), len(C.HEAT)
    sm = lambda v, t: (lambda e: e / e.sum())(np.exp((v - v.max()) / t))  # noqa: E731
    pr = np.concatenate([sm(lg[0, :nu], net.temp[0]), sm(lg[0, nu:nu + nh], net.temp[1]), sm(lg[0, nu + nh:], net.temp[2])])
    return [float(v) for v in np.concatenate([mu[0], sg[0], pr])]


def main():
    det = Detective()
    W, keep = {}, {}
    for mode in MODES:
        n = det.nets[mode]
        assert n.n == 2
        src = dict(W0=n.W[0], b0=n.b[0], W1=n.W[1], b1=n.b[1], Wr=n.Wr, br=n.br, Wc=n.Wc, bc=n.bc,
                   xm=n.xm, xs=n.xs, ym=n.ym, ys=n.ys, widen=n.widen, temp=n.temp)
        for k, v in src.items():
            W[f"{mode}_{k}"] = v
            if k in NORM:
                keep[f"{mode}_{k}"] = "f32"     # normalisation and temperatures stay float32; layers go float16
    n_params = {m: int(sum(W[f"{m}_{k}"].size for k in LAYER)) for m in MODES}
    write_pack(OUT / "weights", W, dtype="f16", keep=keep, meta={"n_params": n_params, "uses": C.USES, "heat": C.HEAT,
                                                                 "cool": C.COOL, "reg": C.REG})

    # the browser's weights: float16 layers, float32 normalisation
    R16 = roundtrip({k: v for k, v in W.items() if k not in keep}, "f16")
    R32 = roundtrip({k: v for k, v in W.items() if k in keep}, "f32")
    full = Detective()
    for mode in MODES:
        n = det.nets[mode]
        g = lambda k: np.asarray({**R16, **R32}[f"{mode}_{k}"], np.float64)  # noqa: E731
        n.W, n.b = [g("W0"), g("W1")], [g("b0"), g("b1")]
        n.Wr, n.br, n.Wc, n.bc = g("Wr"), g("br"), g("Wc"), g("bc")
        n.xm, n.xs, n.ym, n.ys, n.widen, n.temp = g("xm"), g("xs"), g("ym"), g("ys"), g("widen"), g("temp")

    units = load_units()
    rng = np.random.default_rng(SEED)
    xs, meta, cases, f16 = {}, [], [], []
    hv = [("hourly", {}), ("monthly", {"mode": "monthly"}), ("annual", {"mode": "annual"}),
          ("hourly-elec", {"elec": True}), ("hourly-3months", {"months": 3}), ("hourly-dT+3", {"dT": 3.0}),
          ("monthly-dT-3", {"mode": "monthly", "dT": -3.0}), ("hourly-noise0.2", {"noise": 0.2}),
          ("annual-6months-noise0.1-dT+2", {"mode": "annual", "months": 6, "noise": 0.1, "dT": 2.0}),
          ("hourly-elec-6months", {"elec": True, "months": 6}), ("monthly-9months-elec", {"mode": "monthly", "months": 9, "elec": True})]
    av = [("annual", {}), ("annual-elec", {"elec": True})]
    for i, u in enumerate(units):
        zs = {}
        if "loc" in u:
            loc = u["loc"]
            xs[f"T{i}"] = np.asarray(u["T"], np.float32)
            xs[f"G{i}"] = np.asarray(u["G"], np.float32)
            xs[f"hr{i}"], xs[f"dow{i}"], xs[f"mon{i}"] = np.asarray(loc.hour), np.asarray(loc.day_of_week), np.asarray(loc.month)
            xs[f"day{i}"] = loc.tz_localize(None).values.astype("datetime64[D]").astype(np.int64) - 17000
            for c, x in u["meters"].items():
                xs[f"m{i}_{c}"] = np.asarray(x, np.float32)
                xs[f"z{i}_{c}"] = rng.standard_normal(len(x)).astype(np.float32)
                zs[c] = xs[f"z{i}_{c}"]
            # the browser sees float32 series: use exactly those
            u = dict(u, T=xs[f"T{i}"], G=xs[f"G{i}"], meters={c: xs[f"m{i}_{c}"] for c in u["meters"]})
        for name, p in (hv if "loc" in u else av):
            mode, X = perturb(u, p, zs)
            o = outputs(det.nets[mode], X)
            cases.append({"example": i, "variant": name, "p": p, "outputs": o})
            f16.append(np.abs(np.array(o) - np.array(outputs(full.nets[mode], X))).max())
        _, X = perturb(u, {}, zs)
        py = full._run("hourly" if "loc" in u else "annual", X)[0]
        tr = dict(u["truth"])
        m = dict(id=u["id"], kind=u["kind"], truth={k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                                    for k, v in tr.items()},
                 python={k: py[k][0] for k in ("floor_area_m2", "year_built", "UA_W_K", "Tb_heat_C")})
        if "loc" in u:
            m.update(tz=u["tz"], lat=u["lat"], channels=list(u["meters"]), n=len(u["T"]),
                     start_local=str(u["loc"][0])[:16])
        else:
            m.update(kwh=u["kwh"], hdd=u["hdd"], cdd=u["cdd"])
        meta.append(m)
    keepx = {k: ("u8" if k[:2] in ("hr", "do", "mo") else "i32" if k.startswith("day") else "f32") for k in xs}
    write_pack(OUT / "examples", xs, dtype="f32", keep=keepx, meta=meta)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    (OUT / "LICENSE-DATA.md").write_text(LICENCE, encoding="utf-8", newline="\n")
    tot = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {len(units)} units, {len(cases)} parity cases, {tot / 1e6:.2f} MB total; "
          f"float16 vs float64 weights: max raw-output difference {max(f16):.1e}")


if __name__ == "__main__":
    main()
