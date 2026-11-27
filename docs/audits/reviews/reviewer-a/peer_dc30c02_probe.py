"""Independent second-pass verification probe (`reviewer-a`) for commits d92a17b and dc30c02.

Covers all five open first-pass audits from reviewer-b:
  - gridpulse:     weight/ERCOT-head health, live-cutoff API barrier & f0 check, and raw-trace availability
  - firecast:      +112-cell outward padding boundary-exploitation test across all 3 shipped Mode-F fire-days
  - wakenet:       weight health, node/edge feature ranges, and offline-turbine-0 permutation invariance
  - quakenet-v2:   zero/noise waveform ablations, +-15s / +-60s p_guess jitter, weight health & GNN Torch<->NumPy parity
  - batterypilot:  weight health, real node-year |z| maxima, and exact Torch<->NumPy parity

Usage:
  OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_dc30c02_probe.py [all|gridpulse|firecast|wakenet|quakenet-v2|batterypilot]
"""
import glob
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]


def _use_model_dir(mdir: Path, extra: Path | None = None):
    for mod in ("pipeline", "demo", "train", "tt", "gnn", "common", "physics", "quakenet", "quakenet2", "batterypilot", "gridpulse", "firecast", "wakenet"):
        sys.modules.pop(mod, None)
    sys.path = [p for p in sys.path if not str(p).startswith(str(ROOT / "models"))]
    if extra is not None:
        sys.path.insert(0, str(extra))
    sys.path.insert(0, str(mdir))


def check_gridpulse():
    print("=== GRIDPULSE v1.0 (SECOND PASS) ===")
    mdir = ROOT / "models/gridpulse"
    z = np.load(mdir / "gridpulse_weights.npz", allow_pickle=False)
    e = np.load(mdir / "ercot_head.npz", allow_pickle=False)
    ex = np.load(mdir / "examples/gb_exam_examples.npz", allow_pickle=True)
    tr = pd.read_csv(mdir / "examples/trace_example.csv")

    dead_rows = sum(
        int((np.linalg.norm(z[k], axis=1) < 1e-4).sum())
        for k in z.files
        if k.endswith("_w") and z[k].ndim == 2
    )
    print(f"  weights finite={all(np.isfinite(z[k]).all() for k in z.files)} dead_rows={dead_rows} xs=[{z['xs'].min():.4f}, {z['xs'].max():.4f}]")
    for h in ("loss", "dfnad", "E"):
        print(f"  ercot_head {h:5s}: xs=[{e[f'{h}__xs'].min():.4f}, {e[f'{h}__xs'].max():.4f}]")
    print(
        f"  shipped traces: gb_exam_examples.npz has {len(ex['w'])} pre-windowed 81s rows (nans={int(np.isnan(ex['w']).sum())}); "
        f"trace_example.csv has {len(tr)} rows (nans={int(tr.isna().sum().sum())})"
    )

    _use_model_dir(mdir)
    import gridpulse as gp  # noqa: E402

    m = gp.GridPulse()
    t = tr["t_s"].values
    f = tr["f_Hz"].values
    # Check 60s post-onset requirement on live +10s prefix with fixed onset
    try:
        m.read(t[:91], f[:91], system="GB", N=10, onset_index=80)
        live_err = "NONE"
    except ValueError as err:
        live_err = str(err)
    print(f"  live +10s prefix (len=91, onset=80) ValueError: {live_err}")
    # Check f0 normalization difference: read() passes nominal 50.0 vs measured f_pre_Hz
    w_win, f0_meas = gp.window(f, 80)
    r_nom = m.forward(w_win[None], [10], [1.0], [50.0], [np.nan])[0][0]
    r_meas = m.forward(w_win[None], [10], [1.0], [f0_meas], [np.nan])[0][0]
    print(f"  f0_meas={f0_meas:.4f} Hz vs nominal 50.0 Hz: max|d_mu|={np.abs(r_nom - r_meas).max():.4e}")


def check_firecast():
    print("\n=== FIRECAST v1.0 (SECOND PASS: BOUNDARY-EXPLOITATION TEST) ===")
    mdir = ROOT / "models/firecast"
    z = np.load(mdir / "firecast_weights.npz", allow_pickle=False)
    dead_conv = sum(
        int((np.linalg.norm(z[k].reshape(z[k].shape[0], -1), axis=1) < 1e-4).sum())
        for k in z.files
        if z[k].ndim == 4
    )
    print(f"  weights finite={all(np.isfinite(z[k]).all() for k in z.files)} dead_conv_filters={dead_conv}")

    _use_model_dir(mdir)
    import firecast as fc  # noqa: E402
    from demo import load_days  # noqa: E402

    net = fc.FireCastNP()
    pad = 112  # ceil(10000 / 90), covers 100% of nominal 10 km window
    for day in load_days(mdir / "examples/demo_days.npz"):
        now, nxt = day["now"], day["nxt"]
        H, W = now.shape
        p0 = fc.predict(net, day, land=False)
        win0 = fc.score_window(now)
        cand0 = win0 & ~now
        s0 = fc.nextday(now, nxt, p0, fc.CELL_KM2, net.thr, win0)

        day_p = dict(day)
        day_p["now"] = np.pad(now, pad, constant_values=False)
        day_p["nxt"] = np.pad(nxt, pad, constant_values=False)
        if day["prev"] is not None:
            day_p["prev"] = np.pad(day["prev"], pad, constant_values=False)
        if day["fl"] is not None:
            day_p["fl"] = np.pad(day["fl"], pad, constant_values=False)
        for pk in ("pix_pm", "pix_am"):
            dp = {k: np.asarray(v).copy() for k, v in day[pk].items()}
            for c in ("row", "col"):
                if c in dp:
                    dp[c] = dp[c] + pad
            day_p[pk] = dp

        pp = fc.predict(net, day_p, land=False)
        pp_crop = pp[pad : pad + H, pad : pad + W]
        diff = np.abs(pp_crop[cand0] - p0[cand0])
        sp_int = fc.nextday(now, nxt, pp_crop, fc.CELL_KM2, net.thr, win0)
        cross = int(((p0[cand0] >= net.thr) != (pp_crop[cand0] >= net.thr)).sum())

        win_p = fc.score_window(day_p["now"])
        cand_p = win_p & ~day_p["now"]
        in_orig = np.zeros_like(cand_p)
        in_orig[pad : pad + H, pad : pad + W] = True
        ext_cand = cand_p & ~in_orig
        ext_max = float(pp[ext_cand].max()) if ext_cand.any() else 0.0
        ext_thr = int((pp[ext_cand] >= net.thr).sum()) if ext_cand.any() else 0
        sp_full = fc.nextday(day_p["now"], day_p["nxt"], pp, fc.CELL_KM2, net.thr, win_p)

        print(
            f"  {day['meta']['name']:10s} ({H}x{W}, pad=+112): interior max|dp|={diff.max():.4f} cross@0.10={cross} | "
            f"AP orig={s0['ap']:.4f} int_pad={sp_int['ap']:.4f} full10km={sp_full['ap']:.4f} | "
            f"F1 orig={s0['f1']:.4f} full10km={sp_full['f1']:.4f} | "
            f"ext_cells={int(ext_cand.sum())} ext_max_p={ext_max:.4f} ext>=0.10={ext_thr}"
        )


def check_wakenet():
    print("\n=== WAKENET v1.0.1 (SECOND PASS) ===")
    mdir = ROOT / "models/wakenet"
    z = np.load(mdir / "wakenet_weights.npz", allow_pickle=False)
    dead_rows = sum(
        int((np.linalg.norm(z[k], axis=1) < 1e-4).sum())
        for k in z.files
        if k != "meta" and z[k].ndim == 2
    )
    print(f"  weights finite={all(np.isfinite(z[k]).all() for k in z.files if k != 'meta')} dead_rows={dead_rows}")

    _use_model_dir(mdir)
    import pipeline as wpl  # noqa: E402
    import wakenet as wn  # noqa: E402

    wmod = wn.WakeNet()
    wmeta = json.loads((mdir / "examples/examples.json").read_text())
    recs = wmeta["examples"] if isinstance(wmeta, dict) and "examples" in wmeta else wmeta
    nx_lo, nx_hi, ex_lo, ex_hi = np.inf, -np.inf, np.inf, -np.inf
    for rec in recs:
        ez = np.load(mdir / "examples" / rec["file"], allow_pickle=True)
        turb = wn.WP.TURBINES[rec["turbine_model"]]
        on = ez["online"].astype(bool)
        g = wpl.build_graph(
            ez["east_m"], ez["north_m"], ez["hub_h_m"], turb,
            u_inf=float(rec["u_inf_ms"]), wind_dir_deg=float(rec["wind_dir_deg"]),
            ti_inf=float(rec["ti_inf"]), sigma_wd_deg=float(rec["sigma_wd_deg"]),
            yaw_deg=ez["yaw_deg"], online=on, elev_m=ez["elev_m"],
            air_density=float(rec["air_density"]),
        )
        nx_lo = min(nx_lo, float(g["node_x"].min()))
        nx_hi = max(nx_hi, float(g["node_x"].max()))
        ex_lo = min(ex_lo, float(g["edge_x"].min()))
        ex_hi = max(ex_hi, float(g["edge_x"].max()))
        if on.sum() < len(on):
            off_i = int(np.where(~on)[0][0])
            perm = np.arange(len(on))
            perm[0], perm[off_i] = off_i, 0
            r0 = wmod.read(
                ez["east_m"], ez["north_m"], ez["hub_h_m"], turb,
                float(rec["u_inf_ms"]), float(rec["wind_dir_deg"]), float(rec["ti_inf"]),
                float(rec["sigma_wd_deg"]), ez["yaw_deg"], on, ez["elev_m"],
                float(rec["air_density"]), float(rec["p_front_kw"]),
            )
            rp = wmod.read(
                ez["east_m"][perm], ez["north_m"][perm],
                ez["hub_h_m"][perm] if np.ndim(ez["hub_h_m"]) else ez["hub_h_m"], turb,
                float(rec["u_inf_ms"]), float(rec["wind_dir_deg"]), float(rec["ti_inf"]),
                float(rec["sigma_wd_deg"]),
                ez["yaw_deg"][perm] if np.ndim(ez["yaw_deg"]) else ez["yaw_deg"],
                on[perm], ez["elev_m"][perm] if np.ndim(ez["elev_m"]) else ez["elev_m"],
                float(rec["air_density"]), float(rec["p_front_kw"]),
            )
            print(
                f"  offline-turbine-0 swap ({rec['file']}): "
                f"farm_wake_sig diff={abs(rp['farm_wake_sig'] - r0['farm_wake_sig']):.2e} "
                f"max|eta diff|={np.abs(rp['eta'][perm] - r0['eta']).max():.2e}"
            )
    print(f"  6-snapshot feature ranges: node_x=[{nx_lo:+.2f}, {nx_hi:+.2f}], edge_x=[{ex_lo:+.2f}, {ex_hi:+.2f}]")


def check_quakenet_v2():
    print("\n=== QUAKENET v2 (SECOND PASS: WAVEFORM ABLATIONS & GNN PARITY) ===")
    mdir = ROOT / "models/quakenet-v2"
    z1 = np.load(mdir / "quakenet_weights.npz", allow_pickle=False)
    z2 = np.load(mdir / "quakenet2_weights.npz", allow_pickle=False)
    meta2 = json.loads(str(z2["meta"]))
    for label, z in (("station_v12", z1), ("gnn_v2", z2)):
        nans = sum(int(np.isnan(z[k]).sum() + np.isinf(z[k]).sum()) for k in z.files if np.issubdtype(z[k].dtype, np.number))
        dead = sum(int((np.linalg.norm(z[k].reshape(z[k].shape[0], -1), axis=1) < 1e-4).sum()) for k in z.files if z[k].ndim >= 2)
        print(f"  {label:11s}: keys={len(z.files)} nan/inf={nans} dead_units={dead}")
    print(f"  sigma_factor={meta2['sigma_factor']}")

    _use_model_dir(mdir, mdir / "training")
    import quakenet2 as qn2  # noqa: E402

    sys.modules["tt"] = types.SimpleNamespace(KM_PER_DEG=qn2.KM)
    import gnn as qgnn  # noqa: E402

    fn = z2["enc_0_w"].shape[1] - (32 if meta2["use_emb"] else 0)
    h_dim = z2["enc_0_w"].shape[0]
    fe = z2["msg0_0_w"].shape[1] - 2 * h_dim
    tgnn = qgnn.GNN(fn=fn, fe=fe, h=h_dim, rounds=meta2["rounds"], use_emb=meta2["use_emb"])
    sd = {}
    if meta2["use_emb"]:
        sd["proj.weight"] = torch.from_numpy(z2["proj_w"])
        sd["proj.bias"] = torch.from_numpy(z2["proj_b"])

    def map_mlp(prefix_np, prefix_pt, n_lin):
        for j in range(n_lin):
            sd[f"{prefix_pt}.{2 * j}.weight"] = torch.from_numpy(z2[f"{prefix_np}_{j}_w"])
            sd[f"{prefix_pt}.{2 * j}.bias"] = torch.from_numpy(z2[f"{prefix_np}_{j}_b"])

    map_mlp("enc", "enc", 2)
    for r in range(meta2["rounds"]):
        map_mlp(f"msg{r}", f"msg.{r}", 2)
        map_mlp(f"upd{r}", f"upd.{r}", 2)
    map_mlp("out", "out", 3)
    tgnn.load_state_dict(sd, strict=True)
    tgnn.eval()

    qnet = qn2.QuakeNet2()
    rng = np.random.default_rng(7)
    max_pdiff = 0.0
    for fn_ex in sorted(glob.glob(str(mdir / "examples/*.npz"))):
        zf = np.load(fn_ex)
        m = json.loads(str(zf["meta"]))
        e = m["event"]
        st = [dict(zne=x, lat=s["lat"], lon=s["lon"], t_start=s["t_start"], p_guess=s["p_guess"]) for x, s in zip(zf["zne"], m["stations"])]
        rd = [qnet.read_station(s["zne"], s["p_guess"]) for s in st]
        slat = np.array([s["lat"] for s in st], float)
        slon = np.array([s["lon"] for s in st], float)
        tpick = np.array([s["t_start"] + r["pick"] for s, r in zip(st, rd)])
        R = {k: np.array([r["R"][k] for r in rd], float) for k in rd[0]["R"]}
        tp = tpick - float(np.min(tpick))
        A = qn2.combine_v12(slat, slon, R, tp)
        xn, xe, _ = qn2.features(slat, slon, R, tp, [r["ls"] for r in rd], [r["peak"] for r in rd], [r["pz"] for r in rd], A, meta2["use_time"], meta2["use_cc"])
        emb = np.stack([r["emb"] for r in rd]) if meta2["use_emb"] else None
        o_np = qnet.gnn(xn, xe, emb)
        with torch.no_grad():
            o_pt = tgnn(
                torch.from_numpy(xn[None]),
                torch.from_numpy(xe[None]),
                torch.ones((1, len(xn))),
                torch.from_numpy(emb[None]) if emb is not None else None,
            ).numpy()[0]
        max_pdiff = max(max_pdiff, float(np.abs(o_np - o_pt).max()))

        r_real = qnet.locate(st)
        r_zero = qnet.locate([dict(s, zne=np.zeros_like(s["zne"])) for s in st])
        r_noise = qnet.locate(
            [dict(s, zne=rng.normal(0, max(float(np.std(s["zne"])), 1.0), s["zne"].shape).astype(s["zne"].dtype)) for s in st]
        )
        rng15 = np.random.default_rng(123)
        r_j15 = qnet.locate([dict(s, p_guess=float(s["p_guess"] + rng15.uniform(-15, 15))) for s in st])
        rng60 = np.random.default_rng(123)
        r_j60 = qnet.locate([dict(s, p_guess=float(s["p_guess"] + rng60.uniform(-60, 60))) for s in st])

        def miss(r):
            return qn2.gc_dist(e["lat"], e["lon"], r["lat"], r["lon"]) * qn2.KM

        print(
            f"  {e['id']}: real={miss(r_real):5.1f}km (Mw {r_real['mw']:.2f}) | "
            f"zero_wf={miss(r_zero):7.1f}km (Mw {r_zero['mw']:.2f}) | "
            f"noise_wf={miss(r_noise):6.1f}km (Mw {r_noise['mw']:.2f}) | "
            f"jit15s={miss(r_j15):5.1f}km | jit60s={miss(r_j60):5.1f}km"
        )
    print(f"  GNN Torch<->NumPy max|diff| across 4 exam events: {max_pdiff:.3e}")


def check_batterypilot():
    print("\n=== BATTERYPILOT v1.3 (SECOND PASS: WEIGHTS, PARITY & REAL |z|) ===")
    mdir = ROOT / "models/batterypilot"
    w = np.load(mdir / "batterypilot_weights.npz", allow_pickle=False)
    nans = sum(int(np.isnan(w[k]).sum() + np.isinf(w[k]).sum()) for k in w.files if np.issubdtype(w[k].dtype, np.number))
    dead = sum(int((np.linalg.norm(w[k], axis=1) < 1e-4).sum()) for k in w.files if w[k].ndim == 2)
    print(f"  weights nan/inf={nans} dead_rows={dead} x_sd=[{w['x_sd'].min():.4f}, {w['x_sd'].max():.4f}]")

    _use_model_dir(mdir, mdir / "training")
    import batterypilot as bp  # noqa: E402
    import train as bptrain  # noqa: E402

    pnet = bp.PolicyNet(mdir / "batterypilot_weights.npz")
    K = len(bp.SOC_KNOTS)
    nin_total = w["x_mu"].shape[0]
    hybrid = bool(int(w["hybrid"]))
    tnet = bptrain.Net(
        n_in=nin_total - (K if hybrid else 0),
        hidden=w["W0"].shape[0],
        depth=2,
        K=K,
        hybrid=hybrid,
        min_step=float(w["min_step"]),
    )
    sd = {
        "mu": torch.from_numpy(w["x_mu"].astype(np.float32)),
        "sd": torch.from_numpy(w["x_sd"].astype(np.float32)),
        "f.0.weight": torch.from_numpy(w["W0"].astype(np.float32)),
        "f.0.bias": torch.from_numpy(w["b0"].astype(np.float32)),
        "f.2.weight": torch.from_numpy(w["W1"].astype(np.float32)),
        "f.2.bias": torch.from_numpy(w["b1"].astype(np.float32)),
        "f.4.weight": torch.from_numpy(w["W2"].astype(np.float32)),
        "f.4.bias": torch.from_numpy(w["b2"].astype(np.float32)),
    }
    tnet.load_state_dict(sd, strict=True)
    tnet.eval()

    rng = np.random.default_rng(7)
    xr = (rng.normal(0, 1, (256, nin_total)) * w["x_sd"] + w["x_mu"]).astype(np.float32)
    with torch.no_grad():
        yt = tnet(torch.from_numpy(xr)).numpy()
    print(f"  random-batch Torch<->NumPy parity (256 rows): max|diff|={np.abs(yt - pnet.z_knots(xr)).max():.3e}")

    for csv in sorted(glob.glob(str(mdir / "examples/*.csv"))):
        df = pd.read_csv(csv)
        bat = bp.Battery()
        n = len(df)
        t = np.arange(n)
        ps = bp.rolling_scale(df["da"].values.astype(float))
        deg = bp.linear_wear(bat, df["T"].values.astype(float)) if hasattr(bp, "linear_wear") else np.full(n, 2.0)
        X, ps = bp.build(df, t, bat.d, deg, ps=ps)
        lam = bp.value_curves(df["da"].values.astype(float), df["hour"].values, df["day"].values, deg, bat.d, P=bat.P, eta_rt=bat.eta_rt)
        X = np.concatenate([X, np.arcsinh(lam / ps[:, None])], 1).astype(np.float32)
        Z = (X - w["x_mu"]) / w["x_sd"]
        j = int(np.argmax(np.abs(Z).max(0)))
        with torch.no_grad():
            yt_r = tnet(torch.from_numpy(X)).numpy()
        print(
            f"  {os.path.basename(csv):26s}: max|z|={np.abs(Z).max():5.2f} (col {j:3d}, sd={w['x_sd'][j]:.3f}) "
            f"| real parity={np.abs(yt_r - pnet.z_knots(X)).max():.3e}"
        )


def main():
    torch.set_num_threads(1)
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    runners = {
        "gridpulse": check_gridpulse,
        "firecast": check_firecast,
        "wakenet": check_wakenet,
        "quakenet-v2": check_quakenet_v2,
        "batterypilot": check_batterypilot,
    }
    if arg == "all":
        for fn in runners.values():
            fn()
    elif arg in runners:
        runners[arg]()
    else:
        raise SystemExit(f"Unknown target {arg!r}; choose from all, {', '.join(runners)}")


if __name__ == "__main__":
    main()
