"""Read-only forensic audit probe for BuildingDetective v1.0.

Reproduces all measured numbers in docs/audits/reviews/reviewer-a/buildingdetective-v1.0.md:
  1. Shipped weight & standardization health across hourly.npz, monthly.npz, annual.npz
  2. PyTorch (training/detective.py) vs NumPy (buildingdetective.py) exact forward parity
  3. Partial-year truncation counterfactual on shipped HUE_2 (365d -> 330d -> 180d -> 90d)
  4. Rare-channel zero-fraction |z| blow-up & |z|<=8 clipping counterfactual on full-year BDG2 examples
  5. Release vs training features.py rolling-center parity check
  6. Rung decision arithmetic replay on training/ratchet.json
"""
import importlib.util
import json
import os
import sys
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
MDIR = os.path.join(ROOT, "models", "buildingdetective")
WDIR = os.path.join(MDIR, "weights")
sys.path.insert(0, MDIR)

import buildingdetective as bd  # noqa: E402
import common as C  # noqa: E402
import features as rel_feat  # noqa: E402


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.append(os.path.join(MDIR, "training"))
trn_mod = _load_mod("bd_train", os.path.join(MDIR, "training", "train.py"))
trn_feat = _load_mod("bd_trn_features", os.path.join(MDIR, "training", "features.py"))


def check_weights_and_parity():
    print("=== 1. WEIGHTS, STANDARDIZATION, CALIBRATION & TORCH<->NUMPY PARITY ===")
    rng = np.random.default_rng(42)
    for mode in ("hourly", "monthly", "annual"):
        fn = os.path.join(WDIR, f"{mode}.npz")
        z = np.load(fn, allow_pickle=False)
        xm, xs = z["xm"], z["xs"]
        w0 = z["W0"]
        cn = np.linalg.norm(w0, axis=0)
        n_layers = int(z["n_layers"])
        nan_ct = sum(int(np.isnan(z[k]).sum() + np.isinf(z[k]).sum()) for k in z.files if np.issubdtype(z[k].dtype, np.number))
        dead_rows = sum(
            int((np.linalg.norm(z[k], axis=1) < 1e-4).sum())
            for k in z.files
            if k in [f"W{i}" for i in range(n_layers)] + ["Wr", "Wc"]
        )
        # Reconstruct PyTorch Net (Linear, GELU, Dropout -> step 3)
        hidden = w0.shape[0]
        net = trn_mod.Net(w0.shape[1], hidden=hidden, layers=n_layers, drop=0.1)
        sd = {}
        for i in range(n_layers):
            sd[f"body.{3 * i}.weight"] = torch.from_numpy(z[f"W{i}"].astype(np.float32))
            sd[f"body.{3 * i}.bias"] = torch.from_numpy(z[f"b{i}"].astype(np.float32))
        sd["reg.weight"] = torch.from_numpy(z["Wr"].astype(np.float32))
        sd["reg.bias"] = torch.from_numpy(z["br"].astype(np.float32))
        sd["cls.weight"] = torch.from_numpy(z["Wc"].astype(np.float32))
        sd["cls.bias"] = torch.from_numpy(z["bc"].astype(np.float32))
        net.load_state_dict(sd, strict=True)
        net.eval()

        X = (rng.normal(0, 1, (64, w0.shape[1])) * xs + xm).astype(np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(((X - xm) / xs).astype(np.float32))
            mu_r, ls_r, lg_t = net(xt)
            mu_t = mu_r.numpy() * z["ys"] + z["ym"]
            sg_t = np.exp(ls_r.numpy()) * z["ys"] * z["widen"]
            lg_t = lg_t.numpy()

        np_net = bd.Net(fn)
        mu_n, sg_n, lg_n = np_net.raw(X)
        print(
            f"  {mode:7s}: nin={w0.shape[1]:4d} layers={n_layers}x{hidden} nan/inf={nan_ct} dead_rows={dead_rows} "
            f"xs_min={xs.min():.3e} (col {int(np.argmin(xs))}) xs_max={xs.max():.3e} | "
            f"parity mu={np.abs(mu_t - mu_n).max():.2e} sg={np.abs(sg_t - sg_n).max():.2e} lg={np.abs(lg_t - lg_n).max():.2e} | "
            f"temp={np.round(z['temp'], 2).tolist()} widen_min={z['widen'].min():.3f} widen_max={z['widen'].max():.3f}"
        )
        if mode == "hourly":
            for col, label in [(6, "span (nd/365)"), (7, "log10(nd)/3"), (904, "dheat zero_frac"), (1197, "dcool zero_frac")]:
                print(f"    hourly col {col:4d} ({label:16s}): xm={xm[col]:+.6f} xs={xs[col]:.6e} ||W0[:,col]||={cn[col]:.4f}")


def check_partial_year_hue2():
    print("\n=== 2. PARTIAL-YEAR TRUNCATION COUNTERFACTUAL ON SHIPPED HUE_2 ===")
    ex = np.load(os.path.join(MDIR, "examples", "examples.npz"), allow_pickle=False)
    ex_meta = json.load(open(os.path.join(MDIR, "examples", "examples.json")))["hourly"][0]
    net = bd.Net(os.path.join(WDIR, "hourly.npz"))
    k = ex_meta["key"]
    T = ex[f"{k}_T"]
    G = ex[f"{k}_GHI"]
    lat = float(ex_meta["lat"])
    loc = pd.to_datetime(ex[f"{k}_utc"], unit="s", utc=True).tz_convert(ex_meta["tz"])
    elec = ex[f"{k}_elec"]

    for days in (365, 330, 180, 90):
        n = min(days * 24, len(T))
        X = rel_feat.feat_hourly({"elec": elec[:n]}, T[:n], G[:n], loc[:n], lat)
        Z = (X - net.xm) / net.xs
        mu, sg, lg = net.raw(X)
        area = 10.0 ** float(mu[0, 0])
        lo = 10.0 ** float(mu[0, 0] - sg[0, 0])
        hi = 10.0 ** float(mu[0, 0] + sg[0, 0])
        yb = float(mu[0, 1])
        u_probs = bd._softmax(lg[0, : len(C.USES)] / net.temp[0])
        top_u = C.USES[int(np.argmax(u_probs))]
        print(
            f"  {days:3d}d: z[6]={Z[6]:+7.1f} z[7]={Z[7]:+8.1f} | "
            f"floor_area={area:6.1f} m2 (68% [{lo:5.1f}, {hi:5.1f}]) year={yb:6.1f} use={top_u} ({u_probs.max():.2f})"
        )


def check_bdg2_unclipped_z():
    print("\n=== 3. FULL-YEAR BDG2 EXAMPLES: UNCLIPPED |z| vs CLIPPED |z| <= 8 ===")
    bz = np.load(os.path.join(MDIR, "examples", "bdg2", "bdg2_hourly.npz"), allow_pickle=False)
    meta = json.load(open(os.path.join(MDIR, "examples", "bdg2", "bdg2_hourly.json")))["hourly"]
    net = bd.Net(os.path.join(WDIR, "hourly.npz"))
    for rec in meta:
        k = rec["key"]
        T = bz[f"{k}_T"]
        G = bz[f"{k}_GHI"]
        loc = pd.to_datetime(bz[f"{k}_utc"], unit="s", utc=True).tz_convert(rec["tz"])
        meters = {ch: bz[f"{k}_{ch}"] for ch in rec["channels"]}
        X = rel_feat.feat_hourly(meters, T, G, loc, float(rec["lat"]))
        Z = (X - net.xm) / net.xs
        jmax = int(np.argmax(np.abs(Z)))
        mu0, sg0, _ = net.raw(X)
        Xc = np.clip(Z, -8.0, 8.0) * net.xs + net.xm
        muc, sgc, _ = net.raw(Xc)
        a0 = 10.0 ** float(mu0[0, 0])
        ac = 10.0 ** float(muc[0, 0])
        true_a = float(rec["truth"]["floor_area_m2"])
        print(
            f"  {k} {rec['id']:26s} (true {true_a:7.1f} m2, nd={len(T)//24}): "
            f"max|z|={abs(Z[jmax]):6.2f} (col {jmax:4d}, xs={net.xs[jmax]:.3e}) | "
            f"unclipped={a0:7.1f} m2 ({(a0-true_a)/true_a*100:+5.1f}%) vs "
            f"clipped(|z|<=8)={ac:7.1f} m2 ({(ac-true_a)/true_a*100:+5.1f}%, shift {(ac-a0)/a0*100:+5.1f}%)"
        )

    rec0 = meta[0]
    k0 = rec0["key"]
    loc0 = pd.to_datetime(bz[f"{k0}_utc"], unit="s", utc=True).tz_convert(rec0["tz"])
    meters0 = {ch: bz[f"{k0}_{ch}"] for ch in rec0["channels"]}
    x_rel = rel_feat.feat_hourly(meters0, bz[f"{k0}_T"], bz[f"{k0}_GHI"], loc0, float(rec0["lat"]))
    x_trn = trn_feat.feat_hourly(meters0, bz[f"{k0}_T"], bz[f"{k0}_GHI"], loc0, float(rec0["lat"]))
    print(f"  release vs training feat_hourly max|diff| on {k0}: {np.abs(x_rel - x_trn).max():.3e}")


def check_rung_arithmetic():
    print("\n=== 4. RUNG DECISION ARITHMETIC REPLAY (training/ratchet.json) ===")
    rj = json.load(open(os.path.join(MDIR, "training", "ratchet.json")))
    rungs = rj["rungs"]
    inc_sim = {}
    inc_dev = {}
    matches = 0
    exceptions = []
    for r in rungs:
        mode = r["mode"]
        name = r["name"]
        sim = r["val_skill"]
        kept = bool(r["kept"])
        if name.startswith("rebaseline"):
            # Re-scoring incumbent on updated frozen val / label hygiene (no dev_gate)
            inc_sim[mode] = sim
            matches += int(kept)
            continue
        dev = r["dev_gate"]
        if mode not in inc_sim:
            exp = True
        else:
            exp = (sim > inc_sim[mode] + 0.01) and (dev >= inc_dev[mode] - 0.02)
        if kept == exp:
            matches += 1
        else:
            exceptions.append((name, mode, round(sim, 4), round(dev, 4), kept, exp, r.get("reason", "")))
        if kept:
            inc_sim[mode] = sim
            inc_dev[mode] = dev
    print(f"  Replayed {len(rungs)} entries (24 candidate rungs + 6 rebaselines): {matches}/{len(rungs)} match strict rule")
    for ex_item in exceptions:
        print(f"    exception: {ex_item}")


if __name__ == "__main__":
    check_weights_and_parity()
    check_partial_year_hue2()
    check_bdg2_unclipped_z()
    check_rung_arithmetic()
