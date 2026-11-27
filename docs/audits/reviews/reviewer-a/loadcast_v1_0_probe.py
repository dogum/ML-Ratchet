"""Read-only forensic audit probe for LoadCast v1.0.

Reproduces all measured numbers in docs/audits/reviews/reviewer-a/loadcast-v1.0.md:
  1. Shipped weight, standardization & calibration health (loadcast_weights.npz)
  2. PyTorch (training/lcnet.py) vs NumPy (loadcast.py) exact forward parity
  3. Isolated has_electricity=0 (z[18] = -1000.0) & default meters_present=None blow-up
  4. Counterfactual ablations for night_ratio (sd=23.374) and unclipped T_below (|z| up to 29.8)
  5. Operational boundary checks: unsliced history leakage, same-day Tdmax/Tday & future annual_climate perturbations
  6. Rung decision arithmetic replay on training/ratchet.json and training/r2_ladder.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
MDIR = os.path.join(ROOT, "models", "loadcast")
sys.path.insert(0, MDIR)
sys.path.insert(0, os.path.join(MDIR, "training"))

import loadcast as lc  # noqa: E402
import lcnet  # noqa: E402

COLD_NAMES = lc.STATIC_NAMES + lc.WX_NAMES + lc.CAL_NAMES
WARM_NAMES = COLD_NAMES + lc.HIST_NAMES


def load_example(rec):
    base = os.path.join(MDIR, "examples", "bdg2")
    w = pd.read_csv(os.path.join(base, rec["weather_file"]), index_col=0, parse_dates=True)
    y = pd.read_csv(os.path.join(base, rec["meter_file"]), index_col=0, parse_dates=True).iloc[:, 0]
    if w.index.tz is None:
        w.index = w.index.tz_localize("UTC")
    if y.index.tz is None:
        y.index = y.index.tz_localize("UTC")
    return w, y


def check_weights_and_parity():
    print("=== 1. WEIGHTS, STANDARDIZATION, WIDEN FACTORS & TORCH<->NUMPY PARITY ===")
    z = np.load(os.path.join(MDIR, "loadcast_weights.npz"), allow_pickle=False)
    rng = np.random.default_rng(42)
    for mode, names in (("cold", COLD_NAMES), ("warm", WARM_NAMES)):
        mu = z[f"{mode}/mu_x"]
        sd = z[f"{mode}/sd_x"]
        w0 = z[f"{mode}/body.0.weight"]
        cn = np.linalg.norm(w0, axis=0)
        nan_ct = sum(int(np.isnan(z[k]).sum() + np.isinf(z[k]).sum()) for k in z.files if k.startswith(f"{mode}/"))
        dead_rows = sum(
            int((np.linalg.norm(z[k], axis=1) < 1e-4).sum())
            for k in z.files
            if k.startswith(f"{mode}/") and k.endswith(".weight") and z[k].ndim == 2
        )
        bkeys = sorted(int(k.split(".")[1]) for k in z.files if k.startswith(f"{mode}/body.") and k.endswith(".weight"))
        dropout = 0.1 if (len(bkeys) > 1 and bkeys[1] - bkeys[0] == 3) else 0.0
        net = lcnet.MLP(len(names), width=w0.shape[0], depth=len(bkeys), dropout=dropout)
        state = {k[len(mode) + 1 :]: torch.from_numpy(z[k].astype(np.float32)) for k in z.files if k.startswith(f"{mode}/")}
        net.load_state_dict(state, strict=True)
        net.eval()

        x = (rng.normal(0, 1, (256, len(names))) * sd + mu).astype(np.float32)
        base = rng.uniform(0, 2, 256).astype(np.float32)
        with torch.no_grad():
            mt, lst = net(torch.from_numpy(x), torch.from_numpy(base))
            mt, lst = mt.numpy(), lst.numpy()
        W = {k[len(mode) + 1 :]: z[k].astype(np.float64) for k in z.files if k.startswith(f"{mode}/")}
        mn, lsn = lc.np_forward(W, x.astype(np.float64), base.astype(np.float64))
        print(
            f"  {mode:4s}: nin={len(names):2d} nan/inf={nan_ct} dead_rows={dead_rows} "
            f"sd_min={sd.min():.4g} ({names[int(np.argmin(sd))]}) sd_max={sd.max():.4g} ({names[int(np.argmax(sd))]}) | "
            f"parity mu={np.abs(mt - mn).max():.2e} ls={np.abs(lst - lsn).max():.2e}"
        )
        for col_name in ("has_electricity", "night_ratio", "T_below"):
            if col_name in names:
                i = names.index(col_name)
                print(f"    {mode} idx={i:2d} {col_name:16s}: mu={mu[i]:+.6f} sd={sd[i]:.6g} ||W0[:,col]||={cn[i]:.4f}")
    widen = {str(k): float(v) for k, v in zip(z["widen_keys"], z["widen_vals"])}
    capped = {k: v for k, v in widen.items() if v >= 30.0}
    print(f"  widen factors: min={min(widen.values()):.2f} max={max(widen.values()):.2f} capped_at_30={capped}")


def check_has_electricity_and_boundary():
    print("\n=== 2. has_electricity=0 BLOW-UP, DEFAULT meters_present=None & UNSLICED HISTORY ===")
    m = lc.LoadCast()
    recs = json.load(open(os.path.join(MDIR, "examples", "bdg2", "examples.json")))
    for rec in recs:
        w, y = load_example(rec)
        tz = rec["tz"]
        t_split = pd.Timestamp("2017-01-01", tz=tz).tz_convert("UTC")
        hist = y[y.index < t_split]
        mp = rec["meters_present"]
        hv = lc.hvac_class_from_meters(mp)

        fc_c = m.forecast(w, tz, rec["lat"], rec["use"], rec["area_m2"], rec["meter"], mp, "2017-01-01", "2018-01-01", None, rec.get("year_built"), hvac=hv)
        fc_w = m.forecast(w, tz, rec["lat"], rec["use"], rec["area_m2"], rec["meter"], mp, "2017-01-01", "2018-01-01", hist, rec.get("year_built"), hvac=hv)
        fc_leak = m.forecast(w, tz, rec["lat"], rec["use"], rec["area_m2"], rec["meter"], mp, "2017-01-01", "2018-01-01", y, rec.get("year_built"), hvac=hv)
        yt = y.reindex(fc_c.index).values
        cv_c, nm_c, _ = lc.g14(yt, fc_c["kW"].values)
        cv_w, nm_w, _ = lc.g14(yt, fc_w["kW"].values)
        cv_l, nm_l, _ = lc.g14(yt, fc_leak["kW"].values)
        print(f"  {rec['building']} ({rec['meter']}, true_mean={np.nanmean(yt):.2f} kW):")
        print(f"    baseline    : cold mean={fc_c['kW'].mean():7.2f} CV={cv_c:5.1f}% | warm mean={fc_w['kW'].mean():7.2f} CV={cv_w:5.1f}%")
        print(f"    unsliced y  : warm mean={fc_leak['kW'].mean():7.2f} CV={cv_l:5.1f}% (target-year leakage into history_encode)")
        if rec["meter"] != "electricity":
            mp_noelec = [x for x in mp if x != "electricity"]
            fc_c0 = m.forecast(w, tz, rec["lat"], rec["use"], rec["area_m2"], rec["meter"], mp_noelec, "2017-01-01", "2018-01-01", None, rec.get("year_built"), hvac=hv)
            fc_w0 = m.forecast(w, tz, rec["lat"], rec["use"], rec["area_m2"], rec["meter"], mp_noelec, "2017-01-01", "2018-01-01", hist, rec.get("year_built"), hvac=hv)
            fc_cn = m.forecast(w, tz, rec["lat"], rec["use"], rec["area_m2"], rec["meter"], None, "2017-01-01", "2018-01-01", None, rec.get("year_built"))
            fc_wn = m.forecast(w, tz, rec["lat"], rec["use"], rec["area_m2"], rec["meter"], None, "2017-01-01", "2018-01-01", hist, rec.get("year_built"))
            cv_c0, _, _ = lc.g14(yt, fc_c0["kW"].values)
            cv_w0, _, _ = lc.g14(yt, fc_w0["kW"].values)
            cv_cn, _, _ = lc.g14(yt, fc_cn["kW"].values)
            cv_wn, _, _ = lc.g14(yt, fc_wn["kW"].values)
            print(
                f"    has_elec=0  : cold mean={fc_c0['kW'].mean():7.2f} ({fc_c0['kW'].mean()/fc_c['kW'].mean():.2f}x) CV={cv_c0:5.1f}% | "
                f"warm mean={fc_w0['kW'].mean():7.2f} ({fc_w0['kW'].mean()/fc_w['kW'].mean():.2f}x) CV={cv_w0:5.1f}% (same hvac={hv})"
            )
            print(
                f"    mp=None def : cold mean={fc_cn['kW'].mean():7.2f} CV={cv_cn:5.1f}% | "
                f"warm mean={fc_wn['kW'].mean():7.2f} CV={cv_wn:5.1f}%"
            )


def check_ablations_and_causality():
    print("\n=== 3. COUNTERFACTUAL ABLATIONS (night_ratio, T_below) & FUTURE-WEATHER PERTURBATIONS ===")
    m = lc.LoadCast()
    recs = json.load(open(os.path.join(MDIR, "examples", "bdg2", "examples.json")))
    nr_idx = WARM_NAMES.index("night_ratio")
    wr_idx = WARM_NAMES.index("we_ratio")
    tb_idx = WARM_NAMES.index("T_below")

    for rec in recs:
        w, y = load_example(rec)
        tz = rec["tz"]
        t_split = pd.Timestamp("2017-01-01", tz=tz).tz_convert("UTC")
        hist = y[y.index < t_split]
        f = m._frame(w, tz)
        idx = f["idx"]
        t0 = pd.Timestamp("2017-01-01").tz_localize(tz).tz_convert("UTC")
        t1 = pd.Timestamp("2018-01-01").tz_localize(tz).tz_convert("UTC")
        tsel = np.flatnonzero((idx >= t0) & (idx < t1))
        yrs = np.unique(f["year"][tsel])
        clim = lc.annual_climate(f["T"][np.isin(f["year"], yrs)])
        hv = lc.hvac_class_from_meters(rec["meters_present"])
        st = lc.static_features(rec["use"], rec["area_m2"], rec.get("year_built"), rec["meter"], rec["meters_present"], hv, "unknown", clim, rec["lat"])
        unit = rec["area_m2"] * lc.IREF[rec["meter"]] / 1000.0
        h = hist.reindex(idx).values.astype(float) / unit
        hi = np.flatnonzero((idx >= hist.index.min()) & (idx <= hist.index.max()))
        H = lc.history_encode(h[hi], f["T"][hi], f["tow"][hi], f["dcode"][hi], f["hol"][hi], f["hour"][hi], 1.0)
        hh = lc.history_hourly(H, f["T"][tsel], f["tow"][tsel], f["Tday"][tsel], f["T24"][tsel])
        X = np.concatenate([np.broadcast_to(st, (len(tsel), len(st))), f["wx"][tsel], f["cal"][tsel], hh], 1).astype(np.float64)
        Q = max(H["S"], lc.QMIN)
        base = hh[:, 0].astype(np.float64) * H["S"] / Q
        kw0 = np.clip(lc.np_forward(m.W["warm"], X, base)[0], 0, None) * Q * unit

        def sweep(col, lo, hi_v):
            xlo = X.copy(); xlo[:, col] = lo
            xhi = X.copy(); xhi[:, col] = hi_v
            klo = np.clip(lc.np_forward(m.W["warm"], xlo, base)[0], 0, None) * Q * unit
            khi = np.clip(lc.np_forward(m.W["warm"], xhi, base)[0], 0, None) * Q * unit
            return float(np.abs(khi - klo).max() / kw0.mean() * 100)

        nr_s = sweep(nr_idx, 0.3, 1.5)
        wr_s = sweep(wr_idx, 0.3, 1.5)
        Z = (X - m.W["warm"]["mu_x"]) / m.W["warm"]["sd_x"]
        Xc = np.clip(Z, -8.0, 8.0) * m.W["warm"]["sd_x"] + m.W["warm"]["mu_x"]
        kwc = np.clip(lc.np_forward(m.W["warm"], Xc, base)[0], 0, None) * Q * unit
        yt = y.reindex(idx[tsel]).values
        imax = int(np.argmax(np.abs(Z[:, tb_idx])))
        print(
            f"  {rec['building']:24s}: night_ratio(0.3->1.5)={nr_s:4.2f}% vs we_ratio={wr_s:6.2f}% | "
            f"T_below max|z|={abs(Z[imax, tb_idx]):5.2f} (hr {imax}: true={yt[imax]:.1f}, raw={kw0[imax]:.1f}, clip8={kwc[imax]:.1f})"
        )

        # Future-weather perturbation at Jul 1 06:00 and Jan 15 12:00 local
        t_jul = pd.Timestamp("2017-07-01 06:00", tz=tz).tz_convert("UTC")
        t_day_end = pd.Timestamp("2017-07-02 00:00", tz=tz).tz_convert("UTC")
        t_jan = pd.Timestamp("2017-01-15 12:00", tz=tz).tz_convert("UTC")
        w_same = w.copy()
        w_same.loc[(w_same.index > t_jul) & (w_same.index < t_day_end), "T_out"] += 10.0
        w_fut = w.copy()
        w_fut.loc[w_fut.index >= pd.Timestamp("2017-08-01", tz=tz).tz_convert("UTC"), "T_out"] += 10.0
        for mname, harg in (("cold", None), ("warm", hist)):
            f0 = m.forecast(w, tz, rec["lat"], rec["use"], rec["area_m2"], rec["meter"], rec["meters_present"], "2017-01-01", "2018-01-01", harg, rec.get("year_built"))
            fs = m.forecast(w_same, tz, rec["lat"], rec["use"], rec["area_m2"], rec["meter"], rec["meters_present"], "2017-01-01", "2018-01-01", harg, rec.get("year_built"))
            ff = m.forecast(w_fut, tz, rec["lat"], rec["use"], rec["area_m2"], rec["meter"], rec["meters_present"], "2017-01-01", "2018-01-01", harg, rec.get("year_built"))
            dj = (fs.loc[t_jul, "kW"] - f0.loc[t_jul, "kW"]) / max(f0.loc[t_jul, "kW"], 1e-9) * 100
            da = (ff.loc[t_jan, "kW"] - f0.loc[t_jan, "kW"]) / max(f0.loc[t_jan, "kW"], 1e-9) * 100
            print(f"    causal({mname}): Jul1 06h same-day(+10C 07-23h)={dj:+6.2f}% | Jan15 12h Aug-Dec(+10C)={da:+7.2f}%")


def check_rungs():
    print("\n=== 4. RUNG DECISION ARITHMETIC REPLAY (ratchet.json & r2_ladder.json) ===")
    rj = json.load(open(os.path.join(MDIR, "training", "ratchet.json")))
    inc_obj = None
    inc_dev = None
    ok_ct = 0
    for r in rj:
        obj = r["frozen"]["objective"]
        dev = r["dev"]["score"]
        acc = r["accepted"]
        exp = True if inc_obj is None else (obj > inc_obj + 0.01 and dev <= inc_dev * 1.02)
        ok_ct += int(acc == exp)
        print(f"  v1 {r['run']:3s}: obj={obj:+.5f} dev={dev:6.3f} logged={str(acc):5s} expected={str(exp):5s}")
        if exp:
            inc_obj, inc_dev = obj, dev

    r2 = json.load(open(os.path.join(MDIR, "training", "r2_ladder.json")))
    r3_dev = [r["dev"]["score"] for r in rj if r["run"] == "r3"][0]
    r2_ok = 0
    for name, fset, keys in (
        ("r6", "v2_val_p11", ["r6s0", "r6s1", "r6s2"]),
        ("r7", "v2_val", ["r7s0", "r7s1", "r7s2"]),
        ("r8", "v2_val", ["r8s0", "r8s1", "r8s2"]),
    ):
        inc = r2[fset]["R3"]["objective"]
        mo = float(np.mean([r2[fset][k]["objective"] for k in keys]))
        md = float(np.mean([r2[fset][k]["dev_score"] for k in keys]))
        exp = (mo > inc + 0.01) and (md <= r3_dev * 1.02)
        r2_ok += int(exp is False)  # all 3 logged rejected
        print(f"  r2 {name:3s} ({fset:10s}): mean_obj={mo:+.5f} (vs {inc:+.5f}) mean_dev={md:6.3f} (gate {r3_dev*1.02:.3f}) expected={exp}")
    print(f"  Summary: v1 {ok_ct}/{len(rj)} match, r2 {r2_ok}/3 match")


if __name__ == "__main__":
    check_weights_and_parity()
    check_has_electricity_and_boundary()
    check_ablations_and_causality()
    check_rungs()
