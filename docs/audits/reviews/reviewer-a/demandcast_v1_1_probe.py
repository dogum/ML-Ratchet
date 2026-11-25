"""Read-only forensic audit probe for DemandCast v1.1.

Reproduces all measured numbers in docs/audits/reviews/reviewer-a/demandcast-v1.1.md:
  1. Shipped weight, standardization & calibration health (demandcast_weights.npz, demandcast_meta.json)
  2. PyTorch (training/model.py) vs NumPy (demandcast.py) exact forward parity
  3. Solar twilight column (a_last24[10], LST 20:00, idx=66) amplification & BPAT exam blow-up + single-scalar ablation
  4. Demand miss_frac (idx=328, xs=0.001) unclipped z-score blow-up counterfactual on 16 real exam days
  5. CISO solar +2h offset vs solar_offset {-1, 0, +1} clamp on shipped exam days
  6. Rung decision arithmetic replay on training/ratchet.json
"""
import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
MDIR = os.path.join(ROOT, "models", "demandcast")
sys.path.insert(0, MDIR)
sys.path.insert(0, os.path.join(MDIR, "training"))

import demandcast as dc  # noqa: E402
import model as M  # noqa: E402


def check_weights_and_parity():
    print("=== 1. WEIGHTS, STANDARDIZATION & TORCH<->NUMPY PARITY ===")
    z = np.load(os.path.join(MDIR, "demandcast_weights.npz"), allow_pickle=False)
    ex = np.load(os.path.join(MDIR, "examples", "exam_days.npz"), allow_pickle=False)
    m = dc.DemandCast()

    for kind, xk, pk in (("d", "xd", None), ("w", "xw", "pw"), ("s", "xs", "ps")):
        xm = z[f"{kind}.xm"]
        xs = z[f"{kind}.xs"]
        nin = xm.shape[0]
        nan_ct = sum(int(np.isnan(z[k]).sum() + np.isinf(z[k]).sum()) for k in z.files if k.startswith(f"{kind}."))
        dead_mu = np.where(np.linalg.norm(z[f"{kind}.mu.weight"], axis=1) < 1e-4)[0].tolist()
        h0 = z[f"{kind}.body.0.weight"].shape[0]
        h1 = z[f"{kind}.body.2.weight"].shape[0]
        net = M.Net(nin, hidden=(h0, h1))
        sd = {k[2:]: torch.from_numpy(z[k].astype(np.float32)) for k in z.files if k.startswith(f"{kind}.")}
        net.load_state_dict(sd, strict=True)
        net.eval()
        X = ex[xk][:, :nin].astype(np.float32)
        P = ex[pk].astype(np.float32) if pk else None
        with torch.no_grad():
            mt, st = M.head(kind, net, torch.from_numpy(X), torch.from_numpy(P) if P is not None else None)
            mt, st = mt.numpy(), st.numpy()
        mn, sn = dc._forward(m.W[kind], X.astype(np.float64), kind, P.astype(np.float64) if P is not None else None)
        print(
            f"  {kind}: nin={nin:3d} nan/inf={nan_ct} xs_min={xs.min():.4g} xs_max={xs.max():.4g} "
            f"dead_mu_rows={dead_mu} sigma_scale={m.sigma_scale[kind]:.4f} | "
            f"parity mu={np.abs(mt - mn).max():.2e} sig={np.abs(st - sn).max():.2e}"
        )

    w0_s = z["s.body.0.weight"]
    v66, v90 = w0_s[:, 66], w0_s[:, 90]
    cos_20h = float(np.dot(v66, v90) / (np.linalg.norm(v66) * np.linalg.norm(v90)))
    print(
        f"  solar LST 20h columns: ||W0[:,66 (a_20h)]||={np.linalg.norm(v66):.3f} (xm={z['s.xm'][66]:.5f}, xs={z['s.xs'][66]:.5f}) | "
        f"||W0[:,90 (p_20h)]||={np.linalg.norm(v90):.3f} | cos={cos_20h:+.4f}"
    )
    print(
        f"  demand miss_frac col 328: xm={z['d.xm'][328]:.5f} xs={z['d.xs'][328]:.5f} "
        f"||W0[:,328]||={np.linalg.norm(z['d.body.0.weight'][:, 328]):.4f}"
    )


def check_bpat_solar_twilight():
    print("\n=== 2. SOLAR LST 20:00 TWILIGHT BLOW-UP & SINGLE-SCALAR ABLATION ON BPAT ===")
    m = dc.DemandCast()
    ex = np.load(os.path.join(MDIR, "examples", "exam_days.npz"), allow_pickle=False)
    for idx in (14, 15):
        X0 = ex["xs"][idx : idx + 1].astype(np.float64)
        P = ex["ps"][idx : idx + 1].astype(np.float64)
        y = ex["ys"][idx]
        ok = np.isfinite(y)
        night = ex["night"][idx : idx + 1]
        base = m.predict_cf("s", X0, P, night, align=True, ba=ex["ba"][idx : idx + 1])
        mu_raw, _ = dc._forward(m.W["s"], X0[:, :156], "s", P)

        # Ablation A: zero ONLY column 66 (a_last24[10], LST 20:00)
        X1 = X0.copy()
        X1[0, 66] = 0.0
        o1 = m.predict_cf("s", X1, P, night, align=True, ba=ex["ba"][idx : idx + 1])
        mr1, _ = dc._forward(m.W["s"], X1[:, :156], "s", P)

        # Ablation B: zero all nighttime a_last24 where p_last24 == 0
        Xn = X0.copy()
        Xn[0, 56:80] = np.where(X0[0, 80:104] <= 1e-4, 0.0, X0[0, 56:80])
        on = m.predict_cf("s", Xn, P, night, align=True, ba=ex["ba"][idx : idx + 1])
        mrn, _ = dc._forward(m.W["s"], Xn[:, :156], "s", P)

        def nmae(cf):
            return float(np.abs(cf[0][ok] - y[ok]).sum() / y[ok].sum() * 100)

        print(f"  [{idx}] {ex['ba'][idx]} {ex['day'][idx]} (a_20h={X0[0,66]:.4f}, p_20h={X0[0,90]:.4f}, phys_nMAE={nmae(P):.1f}%):")
        print(
            f"    original       : nMAE={nmae(base['cf']):5.1f}%  raw_max_mu={mu_raw.max():.3f}  "
            f"pinned_1.2_hrs={int((base['cf'][0] >= 1.1999).sum()):2d}  max_sig={base['sigma'].max():.3f}  max_hi95={base['hi95'].max():.3f}"
        )
        print(
            f"    zero col 66    : nMAE={nmae(o1['cf']):5.1f}%  raw_max_mu={mr1.max():.3f}  "
            f"pinned_1.2_hrs={int((o1['cf'][0] >= 1.1999).sum()):2d}  max_sig={o1['sigma'].max():.3f}  max_hi95={o1['hi95'].max():.3f}"
        )
        print(
            f"    zero night a24 : nMAE={nmae(on['cf']):5.1f}%  raw_max_mu={mrn.max():.3f}  "
            f"pinned_1.2_hrs={int((on['cf'][0] >= 1.1999).sum()):2d}  max_sig={on['sigma'].max():.3f}  max_hi95={on['hi95'].max():.3f}"
        )


def check_demand_miss_frac():
    print("\n=== 3. DEMAND miss_frac (COL 328, xs=0.001) UNCLIPPED COUNTERFACTUAL ===")
    m = dc.DemandCast()
    ex = np.load(os.path.join(MDIR, "examples", "exam_days.npz"), allow_pickle=False)
    z = np.load(os.path.join(MDIR, "demandcast_weights.npz"), allow_pickle=False)
    xd0 = ex["xd"][:, :329].copy()
    y = ex["d_true"]
    base = m.predict_demand(xd0, ex["pd_total"])

    def mape(p):
        return float(np.mean(np.abs(p - y) / y) * 100)

    print(f"  0/168 missing (baseline): MAPE={mape(base['mw']):6.2f}%  mean_log_sigma={base['log_sigma'].mean():.4f}")
    for n_miss in (1, 4, 8, 24, 48, 168):
        frac = n_miss / 168.0
        xd = xd0.copy()
        xd[:, 328] = frac
        out = m.predict_demand(xd, ex["pd_total"])
        rel = np.abs(out["mw"] - base["mw"]) / base["mw"] * 100
        ratio95 = (out["hi95"] / np.maximum(out["mw"], 1e-9)).max()
        print(
            f"  {n_miss:3d}/168 ({frac:5.3f}, z={frac/z['d.xs'][328]:6.1f}): "
            f"MAPE={mape(out['mw']):6.2f}%  max_hr_shift={rel.max():6.2f}%  "
            f"mean_log_sigma={out['log_sigma'].mean():.4f}  max_hi95/mw={ratio95:.3g}"
        )


def check_ciso_offset_clamp():
    print("\n=== 4. CISO SOLAR +2h LAG vs solar_offset {-1, 0, +1} CLAMP ===")
    m = dc.DemandCast()
    ex = np.load(os.path.join(MDIR, "examples", "exam_days.npz"), allow_pickle=False)
    refs = m.offset_ref(ex["ba"], len(ex["ba"]))
    r_rel, rh_rel = dc.solar_offset(ex["xs"], refs)
    _, rh_abs = dc.solar_offset(ex["xs"], 0.0)
    for i in (2, 3):
        ps, ys = ex["ps"][i], ex["ys"][i]
        c0 = float(np.corrcoef(ps, ys)[0, 1])
        c1 = float(np.corrcoef(np.roll(ps, 1), ys)[0, 1])
        c2 = float(np.corrcoef(np.roll(ps, 2), ys)[0, 1])
        o10 = m.predict_cf("s", ex["xs"][i : i + 1], ex["ps"][i : i + 1], ex["night"][i : i + 1], align=False)
        o11 = m.predict_cf("s", ex["xs"][i : i + 1], ex["ps"][i : i + 1], ex["night"][i : i + 1], align=True, ba=ex["ba"][i : i + 1])
        ok = np.isfinite(ys)
        e10 = float(np.abs(o10["cf"][0][ok] - ys[ok]).sum() / ys[ok].sum() * 100)
        e11 = float(np.abs(o11["cf"][0][ok] - ys[ok]).sum() / ys[ok].sum() * 100)
        print(
            f"  [{i}] {ex['ba'][i]} {ex['day'][i]}: r_hat_abs={rh_abs[i]:+.3f} ref={refs[i]:+.3f} r_hat_rel={rh_rel[i]:+.3f} -> r={r_rel[i]:+d} | "
            f"corr(roll(ps, 0/+1/+2), ys) = {c0:.4f} / {c1:.4f} / {c2:.4f} | nMAE v1.0={e10:.1f}% -> v1.1={e11:.1f}%"
        )


def check_rungs():
    print("\n=== 5. RUNG DECISION ARITHMETIC REPLAY (training/ratchet.json) ===")
    rj = json.load(open(os.path.join(MDIR, "training", "ratchet.json")))
    inc_v3 = None
    v3_ok = 0
    v3_n = 0
    for r in rj:
        vs = r["val_set"]
        ck = r["ckpt"]
        obj = r["objective"]
        acc = r["accepted"]
        if vs == "val_v3":
            v3_n += 1
            if inc_v3 is None:
                exp = True
                rel = None
            else:
                rel = float(np.mean([r["dev"][k]["err"] / max(inc_v3["dev"][k]["err"], 1e-9) for k in ("d", "w", "s")]))
                exp = (obj > inc_v3["objective"] + 0.01) and (rel <= 1.02)
            v3_ok += int(acc == exp)
            print(f"  [{vs}] {ck:10s}: obj={obj:.5f} dev_rel={f'{rel:.4f}' if rel else 'None':6s} logged={str(acc):5s} expected={str(exp):5s}")
            if exp:
                inc_v3 = r
        else:
            inc_obj = r["incumbent_rescored"]["objective"]
            rel = r["dev_rel"]
            exp = (obj > inc_obj + 0.01) and (rel <= 1.02)
            print(
                f"  [{vs}] {ck:10s}: obj={obj:.5f} (vs {inc_obj:.5f}) dev_rel={rel:.4f} "
                f"logged={str(acc):5s} rule_pass={str(exp):5s} superseded_by={r.get('superseded_by')}"
            )
    print(f"  Summary: val_v3 {v3_ok}/{v3_n} exact match; val_v4 r9a documented as superseded by r9b before release")


if __name__ == "__main__":
    check_weights_and_parity()
    check_bpat_solar_twilight()
    check_demand_miss_frac()
    check_ciso_offset_clamp()
    check_rungs()
