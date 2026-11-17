"""Export LoadCast's live instrument for the website: the cold and warm networks in float16, 16 real exam
building-meters (BDG2, local 2017) with their site weather, and reference outputs for the browser parity test.

    python models/loadcast/web/export_instrument.py      # writes site/public/instruments/loadcast/

The 16 meters are the release page's fixed draw (training/webdata.py, seed 20261117, stratified by meter type: 6
electricity, 4 chilled water, 3 steam, 2 hot water, 1 gas; never chosen by result). Their inputs (site weather from
2 Dec 2016 to the end of 2017, the local calendar with holidays, metadata features, the 2016 history encodings for the
three history lengths, the real 2017 meter and BuildingData's change-point baseline fitted on 2016) were written by
that script into the original page, lab/index.html, which this exporter reads: the BDG2 meters and weather live in
the shared building platform, which is not in the repository yet (training/_paths.py).

The browser recomputes the weather and calendar features, the warm-mode history features and both networks on every
change. frame() and predict() below are the one definition of what each control does (net.ts mirrors them); they call
loadcast.py's own feature functions and np_forward. Before anything is written, predict() is checked against the
release's Python predictions stored in the page (first two weeks of every meter and mode).

All data written here are derived from BDG2 (CC BY-SA 4.0); LICENSE-DATA.md goes into the output folder.
"""
import base64
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, REPO)

import loadcast as lc  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("loadcast")
VARIANTS = ["full", "w8", "w2"]                 # history: local 2016, its last 8 weeks, its last 2 weeks
SCHED = ["unknown", "standard", "extended", "24x7"]
MODES = [("cold", "full"), ("warm", "full"), ("warm", "w8"), ("warm", "w2")]

LICENSE = """# Data licence: CC BY-SA 4.0 (Building Data Genome 2)

The files in this folder (examples.bin, examples.json, reference.json) are derived from the Building Data Genome 2
dataset (BDG2): Miller, C., et al. (2020), "The Building Data Genome Project 2, energy meter data from the ASHRAE Great
Energy Predictor III competition", Scientific Data 7, 368, https://doi.org/10.1038/s41597-020-00712-x; data at
https://github.com/buds-lab/building-data-genome-project-2. Licensed CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). Changes: cleaned by BuildingData (timestamps fixed to UTC, quality
flags), subset to 16 exam building-meters at 8 sites; local 2017 meter readings (quantised to int16 steps); site
weather from 2 Dec 2016 to the end of 2017 at 0.01 degC / 0.01 m/s / 1 W/m2 (gaps filled from Iowa Environmental
Mesonet ASOS, irradiance from NASA POWER, by BuildingData); building metadata; 2016 history encodings and
change-point baselines fitted on the 2016 meters; LoadCast outputs on these inputs. You may share and adapt them under
the same licence, with this credit. This licence covers the data files only; the code in this repository is MIT.
The network weights (weights.bin, weights.json) were trained on simulations only.
"""


def page_data():
    s = open(os.path.join(MODEL, "lab", "index.html"), encoding="utf-8").read()
    js = re.findall(r"<script[^>]*>(.*?)</script>", s, re.S)[1]
    dec = json.JSONDecoder()
    get = lambda name: dec.raw_decode(js[js.index(f"const {name} = ") + len(f"const {name} = "):])[0]  # noqa: E731
    return get("DATA"), get("WTS")


def b64(s, dt):
    return np.frombuffer(base64.b64decode(s), dt)


# ---- what the browser computes ---------------------------------------------------------------------------------------
def frame(X, s, dT=0.0, closed_wk=None):
    """Features over a site's whole stored frame (lead-in from 2 Dec 2016, then local 2017), as LoadCast._frame does,
    on the stored weather. dT (K) shifts the outdoor temperature (dewpoint capped at the new temperature, as
    LoadCast._frame caps it); closed_wk marks that week of 2017 as a holiday (the building closed)."""
    lead = X["lead"]
    T = X["wx"][s, 0] / 100.0 + dT
    dew = np.minimum(X["wx"][s, 1] / 100.0, T)
    ghi = X["wx"][s, 2].astype(np.float64)
    wind = X["wx"][s, 3] / 100.0
    day = X["day"][s].astype(int)
    hour, dow, hol, month, mday = (X["cal"][s, k].astype(int) for k in range(5))
    hol = hol.copy()
    if closed_wk is not None:
        hol[lead + 168 * closed_wk: lead + 168 * (closed_wk + 1)] = 1
    wx = lc.weather_features(T, dew, ghi, wind, day)
    cal = lc.calendar_features(hour, dow, hol, month, mday, X["doy"][s].astype(int))
    Tday = np.bincount(day, T) / np.bincount(day)
    return dict(T=T, Tday=Tday[day], T24=lc._rolling_mean_past(T, 24), wx=wx, cal=cal, tow=dow * 24 + hour,
                clim=lc.annual_climate(T[lead:]))


def predict(W, X, u, F, a, b, mode="cold", var="full", sched="unknown", widen=None):
    """kW and sigma (kW, dev-widened) for target hours [a, b) of local 2017 (0 = first hour of 1 Jan)."""
    i = np.arange(a, b) + X["lead"]
    st = X["static"][u].astype(np.float64).copy()
    st[34:37] = F["clim"]
    if sched != "unknown":
        st[30:34] = 0
        st[30 + SCHED.index(sched)] = 1
    parts = [np.broadcast_to(st, (len(i), 38)), F["wx"][i], F["cal"][i]]
    Q, base = 1.0, np.zeros(len(i))
    if mode == "warm":
        v = VARIANTS.index(var)
        H = dict(S=float(X["H_S"][u, v]), coef=X["H_coef"][u, v], cp=X["H_cp"][u, v], prof=X["H_prof"][u, v],
                 scal=X["H_scal"][u, v])
        hh = lc.history_hourly(H, F["T"][i], F["tow"][i], F["Tday"][i], F["T24"][i])
        parts.append(hh)
        Q = max(H["S"], lc.QMIN)
        base = hh[:, 0] * H["S"] / Q
    mu, ls = lc.np_forward(W[mode], np.concatenate(parts, 1).astype(np.float64), base)
    k = (widen or {}).get(f"{mode}_{var}_{X['meter'][u]}", 1.0)
    unit = float(X["unit"][u])
    return np.clip(mu, 0, None) * Q * unit, np.exp(ls) * Q * unit * k


def cp_baseline(X, cp, u, F):
    """BuildingData's change-point model fitted on 2016 daily means, times the 2016 time-of-week shape (kW)."""
    i = np.arange(8760) + X["lead"]
    p, k, Td = cp["params"], cp["kind"], F["Tday"][i]
    if k == "2P":
        d = p["b0"] + p["b1"] * Td
    elif k == "3PH":
        d = p["base"] + p["slope"] * np.maximum(p["Tb"] - Td, 0)
    elif k == "3PC":
        d = p["base"] + p["slope"] * np.maximum(Td - p["Tb"], 0)
    elif k == "4P":
        d = p["base"] + p["slope_lo"] * np.minimum(Td - p["Tb"], 0) + p["slope_hi"] * np.maximum(Td - p["Tb"], 0)
    else:
        d = p["base"] + p["heat_slope"] * np.maximum(p["Tb_heat"] - Td, 0) + p["cool_slope"] * np.maximum(Td - p["Tb_cool"], 0)
    return np.maximum(d * X["cp_shape"][u][F["tow"][i]], 0)


def g14(y, f):
    ok = np.isfinite(y)
    y, f = y[ok], f[ok]
    return 100 * np.sqrt(np.mean((y - f) ** 2)) / y.mean(), 100 * np.mean(y - f) / y.mean()


def main():
    D, WP = page_data()
    sites = list(D["sites"])
    units = D["units"]
    S, U, N = len(sites), len(units), 9480
    assert all(D["sites"][k]["n"] == N and D["sites"][k]["lead"] == 720 for k in sites)

    # weights: the release file, checked against the page's copy
    Z = np.load(os.path.join(MODEL, "loadcast_weights.npz"))
    widen = {str(k): float(v) for k, v in zip(Z["widen_keys"], Z["widen_vals"])}
    assert widen.keys() == WP["widen"].keys() and all(abs(widen[k] - WP["widen"][k]) < 1e-6 * WP["widen"][k] for k in widen)
    page_vs_release = max(float(np.max(np.abs(b64(WP[m][k]["b64"], "<f4").reshape(WP[m][k]["shape"]) - Z[f"{m}/{k}"])))
                          for m in ("cold", "warm") for k in WP[m])
    Wt = {f"{m}_{k.replace('.', '_')}": Z[f"{m}/{k}"] for m in ("cold", "warm") for k in WP[m]}
    keep = {k: "f32" for k in Wt if k.endswith("_mu_x") or k.endswith("_sd_x")}
    nparams = {m: int(sum(Z[f"{m}/{k}"].size for k in WP[m] if k.startswith(("body", "head")))) for m in ("cold", "warm")}
    exam = json.load(open(os.path.join(MODEL, "training", "exam_r3.json")))
    rows = {"cold": "LoadCast cold", "warm_full": "LoadCast warm (2016)", "warm_w8": "LoadCast warm (last 8 weeks)",
            "warm_w2": "LoadCast warm (last 2 weeks)", "dumb": "dumb cold (dev median)", "towt": "TOWT (B)",
            "cp": "change-point hourly"}
    exam_meta = {m: {"n": e["LoadCast cold"]["n"], **{k: e[r]["cv_med"] for k, r in rows.items()},
                     "cov_warm": e["LoadCast warm (2016)"]["coverage90_mean"], "cov_cold": e["LoadCast cold"]["coverage90_mean"]}
                 for m, e in exam["by_meter"].items()}
    write_pack(OUT / "weights", Wt, dtype="f16", keep=keep,
               meta=dict(widen=widen, n_params=nparams, z90=lc.Z90, qmin=lc.QMIN, sched=SCHED, variants=VARIANTS))
    R = roundtrip(Wt, "f16")
    for k in keep:
        R[k] = Wt[k].astype(np.float32)
    split = lambda src: {m: {k: src[f"{m}_{k.replace('.', '_')}"].astype(np.float64) for k in WP[m]} for m in ("cold", "warm")}  # noqa: E731
    W16, W32 = split(R), split(Wt)

    # examples: exactly the arrays the browser reads (everything below is computed from these)
    T = {"wx": np.zeros((S, 4, N), np.int32), "cal": np.zeros((S, 5, N), np.uint8), "doy": np.zeros((S, N)),
         "day": np.zeros((S, N)), "y": np.full((U, 8760), np.nan), "static": np.zeros((U, 38)),
         "H_S": np.zeros((U, 3)), "H_coef": np.zeros((U, 3, 174)), "H_cp": np.zeros((U, 3, 3)),
         "H_prof": np.zeros((U, 3, 168)), "H_scal": np.zeros((U, 3, 14)), "cp_shape": np.zeros((U, 168)),
         "unit": np.zeros(U)}
    for j, k in enumerate(sites):
        d = D["sites"][k]
        for q, f in enumerate(("T", "dew", "ghi", "wind")):
            T["wx"][j, q] = b64(d[f], "<i2")
        c = b64(d["cal"], "<i4")
        for q, (sh, mask) in enumerate(((0, 31), (5, 7), (8, 1), (9, 15), (13, 31))):
            T["cal"][j, q] = (c >> sh) & mask
        T["doy"][j] = (c >> 18) & 511
        T["day"][j] = b64(d["day"], "<i2")
    for i, u in enumerate(units):
        yq = b64(u["y"], "<i2").astype(np.float64)
        T["y"][i] = np.where(yq == -32768, np.nan, yq * u["yscale"])
        T["static"][i] = u["static"]
        for v, var in enumerate(VARIANTS):
            h = u["H"][var]
            T["H_S"][i, v] = h["S"]
            T["H_coef"][i, v], T["H_cp"][i, v], T["H_prof"][i, v], T["H_scal"][i, v] = h["coef"], h["cp"], h["prof"], h["scal"]
        T["cp_shape"][i] = u["cp"]["shape"]
        T["unit"][i] = u["unit"]
    pack_keep = {"wx": "i32", "cal": "u8", "doy": "f16", "day": "f16"}
    meta_units = [dict(id=u["id"], bid=u["bid"], site=sites.index(u["site"]), use=u["use"], area=u["area"],
                       yb=u["yb"], meter=u["meter"], meters=u["meters"], hvac=u["hvac"], cz=u["cz"], cp=dict(kind=u["cp"]["kind"], params=u["cp"]["params"]))
                  for u in units]
    meta_sites = [dict(name=k, tz=D["sites"][k]["tz"], lat=D["sites"][k]["lat"], country=D["sites"][k]["country"],
                       start_utc=D["sites"][k]["start_utc"]) for k in sites]
    write_pack(OUT / "examples", T, dtype="f32", keep=pack_keep,
               meta=dict(lead=720, sites=meta_sites, units=meta_units, exam=exam_meta, exam_n=exam["n"]))
    # what the browser sees, as float64
    X = {k: v.astype(np.float32).astype(np.float64) if pack_keep.get(k) not in ("i32", "u8") else v for k, v in T.items()}
    X.update(lead=720, meter=[u["meter"] for u in units])

    # check 1: the release's Python predictions in the page (first 336 h of 2017, full-precision weather, float32
    # weights) against predict() on the stored inputs. The difference is the weather stored at 0.01 degC.
    ref_dev, year = 0.0, []
    for i, u in enumerate(units):
        F = frame(X, sites.index(u["site"]))
        cp = cp_baseline(X, u["cp"], i, F)
        for mode, var in MODES:
            mu, sd = predict(W32, X, i, F, 0, 8760, mode, var, widen=widen)
            r = u["ref"][f"{mode}_{var}"]
            ref_dev = max(ref_dev, float(np.max(np.abs(mu[:336] - r["mu"]) / max(np.max(np.abs(r["mu"])), 1e-9))))
            year.append((u["id"], mode, var, g14(T["y"][i], mu)[0], r["cv"], g14(T["y"][i], cp)[0]))

    # reference cases for the parity test: one meter per type, five control settings, one week each
    picks = [next(i for i, u in enumerate(units) if u["meter"] == m) for m in lc.METERS]
    variants = [("cold, as recorded, week 27", dict(mode="cold", var="full", wk=27)),
                ("warm 2016, as recorded, week 27", dict(mode="warm", var="full", wk=27)),
                ("warm 2 weeks, +3 C, week 2", dict(mode="warm", var="w2", wk=2, dT=3.0)),
                ("warm 8 weeks, 24x7, closed, week 10", dict(mode="warm", var="w8", wk=10, sched="24x7", closed=True)),
                ("cold, -5 C, standard hours, week 40", dict(mode="cold", var="full", wk=40, dT=-5.0, sched="standard"))]
    cases, f16_dev = [], 0.0
    for i in picks:
        s = sites.index(units[i]["site"])
        for name, p in variants:
            wk = p["wk"]
            F = frame(X, s, p.get("dT", 0.0), wk if p.get("closed") else None)
            mu, sd = predict(W16, X, i, F, 168 * wk, 168 * (wk + 1), p["mode"], p["var"], p.get("sched", "unknown"), widen)
            mu32, _ = predict(W32, X, i, F, 168 * wk, 168 * (wk + 1), p["mode"], p["var"], p.get("sched", "unknown"), widen)
            f16_dev = max(f16_dev, float(np.max(np.abs(mu - mu32)) / max(np.max(mu32), 1e-9)))
            cases.append({"example": i, "variant": name, "p": p, "outputs": [float(v) for v in np.r_[mu, sd]]})
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    (OUT / "LICENSE-DATA.md").write_text(LICENSE, encoding="utf-8", newline="\n")

    tot = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {U} exam meters at {S} sites, {len(cases)} parity cases, {tot / 1e6:.2f} MB total")
    print(f"page weights vs loadcast_weights.npz: max {page_vs_release:.1e}; parameters cold {nparams['cold']:,}, warm {nparams['warm']:,}")
    print(f"predict() on the stored inputs vs the release's Python predictions (first 336 h, 16 meters x 4 modes): "
          f"max {ref_dev:.1e} of the meter's peak (weather stored to 0.01 degC)")
    print(f"float16 vs float32 weights, reference weeks: max {f16_dev:.1e} of the week's peak")
    worst = max(abs(a - b) for _, _, _, a, b, _ in year)
    print(f"2017 CV(RMSE) from the stored inputs vs the release's: max difference {worst:.2f} points over {len(year)} meter-modes")
    for uid, mode, var, cv, cvr, cvcp in year:
        if mode == "warm" and var == "full":
            print(f"  {uid:40s} warm 2016 CV {cv:6.1f} % (release {cvr:6.1f}); change point {cvcp:6.1f} %")


if __name__ == "__main__":
    main()
