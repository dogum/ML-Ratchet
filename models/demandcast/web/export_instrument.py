"""Export DemandCast's live instrument for the website: the three MLPs in float16, 112 real exam days (2023-2025) and
reference outputs for the browser parity test.

    python models/demandcast/web/export_instrument.py      # writes site/public/instruments/demandcast/

The days are the release page's fixed draw (training/webdata.py, seed 2024): one random day per grid per quarter of
each exam year, plus each grid's hottest and coldest exam day, 14 per grid. Their per-day inputs (weather for the 38 h
from issue to the end of day D, filter states at issue, occupancy schedules, the network's other inputs, wind and solar
physics tabulated over weather multipliers, the truth and the grid's own day-ahead forecast) were written by that script
into the original page, lab/index.html, which this exporter reads, so it runs without the pipeline's weather bundles
(those need NASA POWER series at every weather point from GridData's data rebuild, not the v1.0 zip; NOTEBOOK.md).

The browser re-runs the building-stock physics for those 38 h and the three networks on every change. perturb() below is
the one definition of what each control does (ui.ts and net.ts mirror it); physics() is checked here against the
model's own training/buildings.py to machine precision before anything is written.
"""
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.join(MODEL, "training"))  # buildings.py (numpy/scipy/pandas) for the physics check
sys.path.insert(0, REPO)

import buildings as B  # noqa: E402
from demandcast import DemandCast, align_solar_x, mask_dark_solar, solar_offset  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("demandcast")
WX = ["T", "q", "ghi", "clr", "btm", "res", "com", "ind", "occ", "gr"]   # 38 h each, issue (10:00 LST D-1) to end of D
TZ = {"ERCO": "America/Chicago", "CISO": "America/Los_Angeles", "PJM": "America/New_York", "MISO": "America/Chicago",
      "SWPP": "America/Chicago", "ISNE": "America/New_York", "NYIS": "America/New_York", "BPAT": "America/Los_Angeles"}
REF_DAYS_PER_BA = 2  # parity cases: the first random day of 2023 and 2025 per grid, every variant


def page_data():
    s = open(os.path.join(MODEL, "lab", "index.html"), encoding="utf-8").read()
    js = re.findall(r"<script[^>]*>(.*?)</script>", s, re.S)[0]
    dec = json.JSONDecoder()
    i = js.index("const DATA = ") + len("const DATA = ")
    return dec.raw_decode(js[i:])[0]


def solar_offsets(m, days):
    """v1.1: the solar reporting offset each page day's 168 h history shows at issue. The page stores only the first 156
    solar features, so the offset is read in Python from the full feature vector of the same day in the actual-weather
    exam bank (training/realbank.py exam -> work/data/banks/real_exam.npz) and stored with the day; the browser then
    aligns the history, runs the network and moves the forecast exactly as demandcast.py does."""
    B = np.load(os.path.join(MODEL, "work", "data", "banks", "real_exam.npz"))
    key = {(b, d): i for i, (b, d, f) in enumerate(zip(B["meta_ba"], B["meta_day"], B["meta_fc"])) if f == 0}
    out = np.zeros(len(days), int)
    for j, d in enumerate(days):
        i = key.get((d["ba"], d["day"]))
        if i is not None and "s" in d:
            out[j] = solar_offset(np.nan_to_num(B["xs"][i].astype(np.float64)), m.offset_ref(d["ba"], 1),
                                  rmax=m.solar_rmax)[0][0]
    return out


def f32(a):
    """What the browser sees: the page's values stored as float32 (None -> NaN)."""
    return np.array([np.nan if v is None else v for v in np.ravel(a)], np.float32).astype(np.float64)


# ---- the controls ---------------------------------------------------------------------------------------------------
def perturb(p):
    """Controls -> multipliers. p: dT (K added to the 38 h after issue; specific humidity follows at constant relative
    humidity, x exp(0.065 dT)), hum (x specific humidity), sun (x irradiance, capped at max(1.05 x clear sky, actual);
    rooftop PV and utility solar follow), wind (x wind speed at every weather point). Same as Region.sample's forecast-
    error path, with the forecast-weather flag left at 0."""
    return float(p.get("dT", 0.0)), float(p.get("hum", 1.0)), float(p.get("sun", 1.0)), float(p.get("wind", 1.0))


def physics(d, st, prm, dT=0.0, hum=1.0, sun=1.0):
    """buildings.demand() stepped hour by hour from the filter states at issue over the 38 h to the end of day D.
    Returns the 24 h of D: components (MW), total, and the weather the network reads (T, q, ghi)."""
    p = prm
    a = lambda tau: 1 - math.exp(-1.0 / max(tau, 0.05))  # noqa: E731
    a30, a1, a2, ag = a(240.0), a(p["tau1"]), a(p["tau2"]), a(2.0)
    T30, L1, L2, G = st["T30"], st["L1"], st["L2"], st["G"]
    sh_ind = max(1 - p["sh_res"] - p["sh_com"], 0.0)
    e_other = max(1 - p["e_res"] - p["e_hp"], 0.0)
    out = {k: [] for k in ("base", "light", "heat", "cool", "btm", "total", "T", "q", "ghi")}
    for t in range(38):
        T = d["T"][t] + dT
        q = d["q"][t] * hum * math.exp(0.065 * dT)
        g = min(d["ghi"][t] * sun, max(d["clr"][t] * 1.05, d["ghi"][t]))
        bcf = min(1.0, max(0.0, d["btm"][t] * sun))
        T30 += a30 * (T - T30)
        season = 1 + p["wh"] * (15.0 - T30) / 10.0
        P0 = p["P0"] * d["gr"][t]
        res, com, ind, occ = d["res"][t], d["com"][t], d["ind"][t], d["occ"][t]
        base = P0 * (p["sh_res"] * res + p["sh_com"] * com + sh_ind * ind) * season
        dark = 1 - min(1.0, max(0.0, g / 1000.0 / 0.25))
        light = P0 * p["light"] * (0.5 * res + 0.5 * com) * dark
        Tsa = T + p["a_sol"] * g / 1000.0
        L1 += a1 * (Tsa - L1)
        L2 += a2 * (Tsa - L2)
        G += ag * (g / 1000.0 - G)
        Teff = p["f1"] * L1 + (1 - p["f1"]) * L2 + p["g_int"] * (0.6 + 0.4 * occ) + p["g_win"] * G
        mh = p["mu_h"] - p["setback"] * (1 - occ)
        mc = p["mu_c"] + p["setup"] * (1 - occ)
        UA = p["ua"] * P0
        zh, zc = (mh - Teff) / p["sd_h"], (Teff - mc) / p["sd_c"]
        heat_th = UA * ((mh - Teff) * norm.cdf(zh) + p["sd_h"] * norm.pdf(zh))
        cool_th = UA * ((Teff - mc) * norm.cdf(zc) + p["sd_c"] * norm.pdf(zc) + p["lam"] * max(q - p["q_in"], 0.0) * norm.cdf(zc))
        heat = heat_th * (p["e_res"] + p["e_hp"] / min(4.5, max(1.0, 2.6 + 0.08 * T)) + 0.03 * e_other)
        cool = cool_th * p["ac"] / min(5.5, max(2.0, 4.2 - 0.07 * (T - 25.0)))
        btm = p["btm"] * P0 * bcf
        if t >= 14:
            for k, v in (("base", base), ("light", light), ("heat", heat), ("cool", cool), ("btm", btm),
                         ("total", base + light + heat + cool - btm), ("T", T), ("q", q), ("ghi", g)):
                out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def interp(tab, grid, m):
    """Physics tabulated in Python at grid multipliers (rows of tab); linear in between, clamped at the ends."""
    step = grid[1] - grid[0]
    k = max(0, min(len(grid) - 2, math.floor((m - grid[0]) / step)))
    f = max(0.0, min(1.0, (m - grid[k]) / (grid[k + 1] - grid[k])))
    return tab[k] * (1 - f) + tab[k + 1] * f


def inputs(d, st, prm, MW, MS, p):
    """The three networks' inputs for control settings p (features.py layout; everything not set here is the page's
    stored input: the 168 h of real output against the physics, calendar, descriptors)."""
    dT, hum, sun, wind = perturb(p)
    ph = physics(d, st, prm, dT, hum, sun)
    s = d["sd"]
    xd = d["xd"].copy()
    for j, c in enumerate(("base", "light", "heat", "cool", "btm")):
        xd[100 + 24 * j:124 + 24 * j] = ph[c] / s
    xd[220:244] = ph["total"] / s - 1
    xd[244:268] = ph["T"] / 10
    xd[268:292] = ph["q"] / 10
    xd[292:316] = ph["ghi"] / 1000
    o = dict(ph=ph, xd=xd)
    if d["hasw"]:
        pt, lo, hi = interp(d["wtab"], MW, wind), interp(d["wtab"], MW, 0.9 * wind), interp(d["wtab"], MW, 1.1 * wind)
        xw = d["xw"].copy()
        xw[0:24] = pt
        xw[24:48] = hi - lo
        xw[136:160] = d["ws"] * wind / 10
        o.update(xw=xw, pw=pt)
    if d["hass"]:
        pt, lo, hi = interp(d["stab"], MS, sun), interp(d["stab"], MS, 0.8 * sun), interp(d["stab"], MS, 1.2 * sun)
        xs = d["xs"].copy()
        xs[0:24] = pt
        xs[24:48] = hi - lo
        clr = d["clr"][14:] / 1000.0
        xs[128:152] = np.where(clr > 0.02, np.clip(d["ghi"][14:] * sun / 1000.0 / np.maximum(clr, 1e-3), 0, 1.2), 0)
        o.update(xs=xs, ps=pt)
    return o


def outputs(m, d, st, prm, MW, MS, p):
    """Parity vector: physics total (24); demand raw mu, raw sigma, MW, lo95, hi95; then for wind and for solar (when the
    grid reports them) the physics CF, raw mu, raw sigma, CF, lo95, hi95 (24 each)."""
    o = inputs(d, st, prm, MW, MS, p)
    from demandcast import _forward
    vec = list(o["ph"]["total"])
    mu, sg = _forward(m.W["d"], o["xd"][None], "d", z_clip=m.z_clip)
    r = m.predict_demand(o["xd"][None], o["ph"]["total"][None])
    vec += list(mu[0]) + list(sg[0]) + list(r["mw"][0]) + list(r["lo95"][0]) + list(r["hi95"][0])
    res = dict(demand=r, ph=o["ph"])
    for k, x, pk in (("w", "xw", "pw"), ("s", "xs", "ps")):
        if x not in o:
            continue
        night = d["night"][None] if k == "s" else None
        xin = o[x][None] if k == "w" else align_solar_x(o[x][None], np.array([d["soff"]]))
        if k == "s" and m.mask_dark:
            xin = mask_dark_solar(xin)
        mu, sg = _forward(m.W[k], xin, k, o[pk][None], z_clip=m.z_clip)
        r = m.predict_cf(k, o[x][None], o[pk][None], night=night) if k == "w" else             m.predict_cf(k, o[x][None], o[pk][None], night=night, offset=d["soff"])
        vec += list(o[pk]) + list(mu[0]) + list(sg[0]) + list(r["cf"][0]) + list(r["lo95"][0]) + list(r["hi95"][0])
        res[k] = r
    return [float(v) for v in vec], res


def check_stepper(D):
    """physics() vs buildings.demand() on a real page day's weather and a real calendar: 26 copies of the 38 h (988 h)
    for every grid, states taken from buildings.rc_filter at hour 900, then stepped 38 h."""
    worst = 0.0
    for d in D["days"][::14]:
        ba, prm = d["ba"], D["params"][d["ba"]]
        n = 38 * 26
        times = pd.date_range(f"{d['day']} 00:00", periods=n, freq="h", tz="UTC") - pd.Timedelta(hours=300)
        cal = B.calendar(times, TZ[ba])
        rep = lambda k: np.tile(f32(d[k]), 26)  # noqa: E731
        wx = dict(T=rep("T"), q=rep("q"), ghi=rep("ghi"), btm_cf=rep("btm"))
        full = B.demand(wx, cal, prm, t_ref=prm.get("t_ref", 2020.0))
        k0 = 900
        res, com, ind, occ = B.occupancy({k: v[k0:k0 + 38] for k, v in cal.items()}, prm)
        gr = 1 + prm["trend"] * (cal["year"][k0:k0 + 38] - prm.get("t_ref", 2020.0))
        g = wx["ghi"] / 1000.0
        Tsa = wx["T"] + prm["a_sol"] * g
        st = dict(T30=B.rc_filter(wx["T"][:k0], 240.0)[-1], L1=B.rc_filter(Tsa[:k0], prm["tau1"])[-1],
                  L2=B.rc_filter(Tsa[:k0], prm["tau2"])[-1], G=B.rc_filter(g[:k0], 2.0)[-1])
        dd = dict(T=wx["T"][k0:k0 + 38], q=wx["q"][k0:k0 + 38], ghi=wx["ghi"][k0:k0 + 38], clr=np.full(38, 1e4),
                  btm=wx["btm_cf"][k0:k0 + 38], res=res, com=com, ind=ind, occ=occ, gr=gr)
        ph = physics(dd, st, prm)
        for c in ("base", "light", "heat", "cool", "btm", "total"):
            worst = max(worst, float(np.max(np.abs(ph[c] - full[c][k0 + 14:k0 + 38]) / np.maximum(np.abs(full["total"][k0 + 14:k0 + 38]), 1))))
    return worst


def main():
    D = page_data()
    MW, MS = [float(v) for v in D["MW"]], [float(v) for v in D["MS"]]
    calib = json.load(open(os.path.join(MODEL, "calib", "demand_dev.json")))
    assert all(D["params"][ba] == calib[ba]["params"] for ba in D["params"]), "page params = released dev calibration"
    m = DemandCast()
    assert abs(D["sigma_scale"]["w"] - m.sigma_scale["w"]) < 1e-12   # solar re-fitted for v1.1, demand for v1.1.1
    soff = solar_offsets(m, D["days"])
    step_err = check_stepper(D)
    assert step_err < 1e-9, step_err

    # weights: layers float16, input standardisation float32
    Z = np.load(os.path.join(MODEL, "demandcast_weights.npz"))
    W = {n.replace(".", "_"): Z[n] for n in Z.files}
    keep = {k: "f32" for k in W if k.endswith("_xm") or k.endswith("_xs")}
    meta = dict(sigma_scale=m.sigma_scale, n_params=m.meta["n_params"], inputs=m.meta["inputs"], MW=MW, MS=MS,
                params=D["params"], floor={"d": 0.004, "w": 0.005, "s": 0.003}, z_clip=m.z_clip,
                mask_dark_solar=m.mask_dark)
    write_pack(OUT / "weights", W, dtype="f16", meta=meta, keep=keep)
    R = roundtrip(W, "f16")
    for k in keep:
        R[k] = W[k].astype(np.float32)
    full = DemandCast()
    m.W = {k: {n.split(".", 1)[1]: R[n.replace(".", "_")].astype(np.float64) for n in Z.files if n.startswith(k + ".")} for k in "dws"}

    days = D["days"]
    N = len(days)
    X = {"wx": np.zeros((N, len(WX), 38)), "xd": np.zeros((N, 329)), "truth": np.full((N, 4, 24), np.nan),
         "xw": np.zeros((N, 164)), "wtab": np.zeros((N, len(MW), 24)), "ws": np.zeros((N, 24)), "yw": np.full((N, 24), np.nan),
         "xs": np.zeros((N, 156)), "stab": np.zeros((N, len(MS), 24)), "ys": np.full((N, 24), np.nan), "night": np.ones((N, 24))}
    meta_ex, cases, dev = [], [], dict(prior=0.0, xd=0.0, f16=0.0)
    ape = {"model": [], "da": [], "pers": [], "prior": []}
    for i, d in enumerate(days):
        for j, k in enumerate(WX):
            X["wx"][i, j] = f32(d[k])
        X["xd"][i] = f32(d["xd"])
        for j, k in enumerate(("y", "df", "pers", "prior")):
            X["truth"][i, j] = f32(d[k])
        hasw, hass = "w" in d, "s" in d
        if hasw:
            X["xw"][i], X["ws"][i], X["yw"][i] = f32(d["w"]["x"]), f32(d["w"]["ws"]), f32(d["w"]["y"])
            X["wtab"][i] = np.array([f32(r) for r in d["w"]["tab"]])
        if hass:
            X["xs"][i], X["ys"][i], X["night"][i] = f32(d["s"]["x"]), f32(d["s"]["y"]), f32(d["s"]["night"])
            X["stab"][i] = np.array([f32(r) for r in d["s"]["tab"]])
        dd = {k: X["wx"][i, j] for j, k in enumerate(WX)}
        dd.update(sd=float(d["sd"]), xd=X["xd"][i], hasw=hasw, hass=hass, xw=X["xw"][i], wtab=X["wtab"][i], ws=X["ws"][i],
                  xs=X["xs"][i], stab=X["stab"][i], night=X["night"][i], soff=int(soff[i]))
        st, prm = d["st"], D["params"][d["ba"]]
        o = inputs(dd, st, prm, MW, MS, {})
        # neutral controls reproduce the page's stored inputs (rounded there to 1e-5 and to 1 MW)
        dev["prior"] = max(dev["prior"], float(np.max(np.abs(o["ph"]["total"] / X["truth"][i, 3] - 1))))
        dev["xd"] = max(dev["xd"], float(np.max(np.abs(o["xd"][100:316] - X["xd"][i, 100:316]))))
        vec, r = outputs(m, dd, st, prm, MW, MS, {})
        rf = full.predict_demand(o["xd"][None], o["ph"]["total"][None])
        dev["f16"] = max(dev["f16"], float(np.max(np.abs(r["demand"]["mw"][0] / rf["mw"][0] - 1))))
        y, df, pers = X["truth"][i, 0], X["truth"][i, 1], X["truth"][i, 2]
        ok = np.isfinite(y) & np.isfinite(df) & np.isfinite(pers) & (y > 0)
        for k, v in (("model", r["demand"]["mw"][0]), ("da", df), ("pers", pers), ("prior", o["ph"]["total"])):
            ape[k] += list(np.abs(v[ok] - y[ok]) / y[ok])
        meta_ex.append(dict(ba=d["ba"], day=d["day"], tag=d["tag"], st=st, sd=float(d["sd"]), hasw=hasw, hass=hass,
                            capw=float(d["w"]["cap"]) if hasw else 0.0, caps=float(d["s"]["cap"]) if hass else 0.0,
                            soff=int(soff[i]),
                            T_mean=float(np.mean(dd["T"][14:])), T_min=float(np.min(dd["T"][14:])), T_max=float(np.max(dd["T"][14:]))))
        if d["tag"] == "random" and int(d["day"][:4]) in (2023, 2025) and \
                sum(1 for c in meta_ex if c["ba"] == d["ba"] and c["tag"] == "random" and c["day"][:4] == d["day"][:4]) == 1:
            for name, p in (("raw", {}), ("+3C", {"dT": 3.0}), ("cold-8C,humid", {"dT": -8.0, "hum": 1.2}),
                            ("cloudy,windy", {"sun": 0.55, "wind": 1.27}), ("hot,sunny,calm", {"dT": 6.5, "sun": 1.2, "wind": 0.65})):
                v, _ = outputs(m, dd, st, prm, MW, MS, p)
                cases.append({"example": i, "variant": name, "p": p, "outputs": v})
    keep_ex = {}
    write_pack(OUT / "examples", {k: v.astype(np.float32) for k, v in X.items()}, dtype="f32", meta=meta_ex, keep=keep_ex)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    tot = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    mape = {k: 100 * float(np.mean(v)) for k, v in ape.items()}
    print(f"wrote {OUT}: {N} exam days, {len(cases)} parity cases, {tot / 1e6:.2f} MB total")
    print(f"physics() vs buildings.demand(): max {step_err:.1e} of total demand")
    print(f"neutral controls vs the page's stored inputs: prior max {dev['prior']:.1e} relative (stored to 1 MW), "
          f"physics features max {dev['xd']:.1e} (weather stored to 0.01 K and 0.001 g/kg)")
    print(f"float16 vs float32 weights, demand MW on all days: max {dev['f16']:.1e} relative")
    print(f"spot check on the {N} page days ({len(ape['model'])} hours), demand MAPE %: DemandCast {mape['model']:.2f}, "
          f"grid day-ahead {mape['da']:.2f}, persistence 24 h {mape['pers']:.2f}, physics alone {mape['prior']:.2f}")


if __name__ == "__main__":
    main()
