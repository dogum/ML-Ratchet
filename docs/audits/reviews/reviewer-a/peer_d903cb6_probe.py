"""Second-pass cross-check probe for reviewer-b's d903cb6 audits:
ThermoPilot v1.1, TsunamiNet v1.0, and ChirpNet v2.1.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_d903cb6_probe.py [thermopilot|tsunaminet|chirpnet|all]
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]


def check_thermopilot():
    print("=== SECOND-PASS CHECK: THERMOPILOT v1.1 ===")
    sys.path[:0] = [
        str(ROOT / "models/thermopilot"),
        str(ROOT / "models/thermopilot/training"),
        str(ROOT / "shared/building/buildingsim"),
    ]
    import thermopilot_core as tp
    from tp.features import full_card, card_from_model
    from tp.mpc import Model
    from tp import env as tp_env
    from buildingsim import Simulator
    from buildingsim.weather import SIGMA, clear_cache

    # 1. Weight health (dead rows/cols) for optional v1.0 MLP
    W_tp = np.load(ROOT / "models/thermopilot/thermopilot_v1.npz")
    for k in W_tp.files:
        arr = W_tp[k]
        if arr.ndim == 2:
            dr = int((np.max(np.abs(arr), axis=1) < 1e-8).sum())
            dc = int((np.max(np.abs(arr), axis=0) < 1e-8).sum())
            assert np.isfinite(arr).all() and dr == 0 and dc == 0, f"bad weight {k}"
    print("  weights: all finite, 0 dead rows/cols across W0/W1/W2")

    # 2. Scope of shared buildingsim weather cache across models/
    rg_out = subprocess.check_output(
        ["rg", "-l", "buildingsim", "models/"], cwd=ROOT, text=True
    ).strip().splitlines()
    bs_models = sorted(set(p.split("/")[1] for p in rg_out))
    print(f"  buildingsim consumers in models/: {bs_models} ({len(rg_out)} files)")

    # 3. Closed-loop 30-day BuildingSim episodes comparing full-episode card vs causal prior-month card
    def make_weather(start_str, n_hours, T_base, T_amp, ghi_peak, wind_arr, lat, lon, elev, cloud=0.4):
        idx = pd.date_range(start_str, periods=n_hours, freq="h", tz="UTC")
        h = np.arange(n_hours, dtype=float)
        T = T_base + T_amp * np.sin(2 * np.pi * (h - 9) / 24.0) + 3.0 * np.sin(2 * np.pi * h / (24.0 * 7.0))
        G = ghi_peak * np.maximum(0.0, np.sin(np.pi * ((h % 24) - 6) / 12.0))
        df = pd.DataFrame(
            dict(
                T_out=T,
                RH=60.0,
                GHI=G,
                DNI=G,
                DHI=G * 0.3,
                wind_speed=wind_arr,
                wind_dir=180.0,
                pressure=101325.0,
                IR_horiz=SIGMA * (T + 273.15) ** 4 * 0.8,
                cloud=cloud,
            ),
            index=idx,
        )
        return df, dict(lat=lat, lon=lon, elev=elev, tz="UTC")

    def run_episode_pair(name, spec, wx_eval, wx_prior, hi_day=25.0, days=30, seed=0):
        n_ev = days * 24
        idx_e, idx_p = wx_eval[0].index, wx_prior[0].index

        clear_cache()
        m_full = Model(Simulator(spec, wx_eval, idx_e[0], idx_e[n_ev + 25], dt="1h", warmup_days=0, record_trend=False))
        clear_cache()
        m_prior = Model(Simulator(spec, wx_prior, idx_p[0], idx_p[n_ev + 25], dt="1h", warmup_days=0, record_trend=False))

        h = np.arange(n_ev + 25)
        wd = np.array([t.weekday() < 5 for t in idx_e[: n_ev + 25]])
        price = np.where(wd & (h % 24 >= 15) & (h % 24 < 20), 0.48, 0.10)
        lo = np.where((h % 24 >= 22) | (h % 24 < 6), 18.0, 20.0)
        hi = np.where((h % 24 >= 22) | (h % 24 < 6), 26.0, hi_day)
        ctx_e = tp.make_ctx(idx_e[: n_ev + 25], price, lo, hi, 20.0, hi_day, np.zeros(n_ev + 25))
        ctx_p = tp.make_ctx(idx_p[: n_ev + 25], price, lo, hi, 20.0, hi_day, np.zeros(n_ev + 25))

        cf = full_card(m_full, ctx_e, hi_day)
        cp = full_card(m_prior, ctx_p, hi_day)
        cp_map = dict(cf, map=cp["map"])

        cm_f = card_from_model(m_full, hi_day)
        cm_p = card_from_model(m_prior, hi_day)

        lam = 3.0
        fc = tp_env.Forecaster(m_full.To, m_full.ghi, 24, seed, perfect=False, scale=1.0)
        W_net = tp.load(ROOT / "models/thermopilot/thermopilot_v1.npz")

        def run_strat(fn):
            clear_cache()
            sim = Simulator(spec, wx_eval, idx_e[0], idx_e[n_ev], dt="1h", warmup_days=0, record_trend=False)
            obs = sim.reset()
            info = None
            k = 0
            acts = []
            while obs is not None:
                a = fn(k, obs, info)
                acts.append((a["heat_sp"], a["cool_sp"]))
                obs, info = sim.step(dict(heat_sp=float(a["heat_sp"]), cool_sp=float(a["cool_sp"])))
                k += 1
            ts = sim.finish()["ts"]
            el = ts["meter_electricity_kW"].values
            gas = ts["meter_gas_kW"].values
            T = ts["T_zone"].values
            bill = float((el * price[: len(ts)]).sum() + gas.sum() * tp_env.GAS_RES)
            dh = float((np.maximum(lo[: len(ts)] - T, 0.0) + np.maximum(T - hi[: len(ts)], 0.0)).sum())
            return dict(bill=bill, DH=dh, J=bill + lam * dh, actions=np.array(acts))

        def mk(card, mode="mpc"):
            p = tp.pilot(card, ctx_e, lam, m_full.base_W, mode=mode, W=(W_net if mode != "mpc" else None))
            return lambda k, o, i: p.step(
                k,
                float(o["T_zone"]),
                0.0 if i is None else float(i.get("Q_hvac_sens_kW", 0.0)) * 1000.0,
                0.0 if i is None else float(i.get("meter_electricity_kW", 0.0)) * 1000.0,
                *fc.get(k),
            )

        res = {
            "fixed": run_strat(lambda k, o, i: dict(heat_sp=20.0, cool_sp=hi_day)),
            "setback": run_strat(lambda k, o, i: dict(heat_sp=float(lo[k]), cool_sp=float(hi[k]))),
            "v1.1_full": run_strat(mk(cf, "mpc")),
            "v1.1_prior_map": run_strat(mk(cp_map, "mpc")),
            "v1.1_prior_all": run_strat(mk(cp, "mpc")),
            "v1.0_full": run_strat(mk(cf, "network")),
            "v1.0_prior_all": run_strat(mk(cp, "network")),
        }
        d11 = int(np.any(np.abs(res["v1.1_full"]["actions"] - res["v1.1_prior_map"]["actions"]) > 1e-6, axis=1).sum())
        d10 = int(np.any(np.abs(res["v1.0_full"]["actions"] - res["v1.0_prior_all"]["actions"]) > 1e-6, axis=1).sum())
        dJ = res["v1.1_prior_all"]["J"] - res["v1.1_full"]["J"]
        print(
            f"  {name}: At {cf['map']['At']:.5f}->{cp['map']['At']:.5f}, UA {cm_f['UA']:.2f}->{cm_p['UA']:.2f} W/K | "
            f"diff_hrs v1.1={d11}/720, v1.0={d10}/720 | "
            f"J fixed=${res['fixed']['J']:.2f} setback=${res['setback']['J']:.2f} "
            f"v1.1_full=${res['v1.1_full']['J']:.2f} v1.1_prior=${res['v1.1_prior_all']['J']:.2f} (dJ={dJ:+.2f}, DH={res['v1.1_prior_all']['DH']:.2f}) "
            f"v1.0_full=${res['v1.0_full']['J']:.2f} v1.0_prior=${res['v1.0_prior_all']['J']:.2f}"
        )

    n_h = 30 * 24 + 30
    spec_win = dict(
        primary_use="residential_single", floor_area_m2=150, year_built=2000,
        climate_zone="5A", system="heat_pump_air", ventilation=dict(natural_ventilation=False),
    )
    spec_sum = dict(spec_win, climate_zone="2B")
    w_step = np.full(n_h, 2.0)
    w_step[48:] = 8.0

    run_episode_pair(
        "winter_8vs2",
        spec_win,
        make_weather("2019-01-14", n_h, -3.0, 7.0, 300.0, np.full(n_h, 8.0), 41.9, -87.6, 200.0),
        make_weather("2018-12-14", n_h, -3.0, 7.0, 300.0, np.full(n_h, 2.0), 41.9, -87.6, 200.0),
    )
    run_episode_pair(
        "winter_2vs8",
        spec_win,
        make_weather("2019-01-14", n_h, -3.0, 7.0, 300.0, np.full(n_h, 2.0), 41.9, -87.6, 200.0),
        make_weather("2018-12-14", n_h, -3.0, 7.0, 300.0, np.full(n_h, 8.0), 41.9, -87.6, 200.0),
    )
    run_episode_pair(
        "winter_step2to8",
        spec_win,
        make_weather("2019-01-14", n_h, -3.0, 7.0, 300.0, w_step, 41.9, -87.6, 200.0),
        make_weather("2018-12-14", n_h, -3.0, 7.0, 300.0, np.full(n_h, 2.0), 41.9, -87.6, 200.0),
    )
    run_episode_pair(
        "summer_8vs2",
        spec_sum,
        make_weather("2019-07-01", n_h, 31.0, 8.0, 650.0, np.full(n_h, 8.0), 33.4, -112.0, 330.0, cloud=0.2),
        make_weather("2019-06-01", n_h, 31.0, 8.0, 650.0, np.full(n_h, 2.0), 33.4, -112.0, 330.0, cloud=0.2),
        hi_day=24.5,
    )


def check_tsunaminet():
    print("\n=== SECOND-PASS CHECK: TSUNAMINET v1.0 ===")
    sys.path.insert(0, str(ROOT / "models/tsunaminet"))
    import tsunaminet as tn

    # 1. Verify zero rows in RealNVP coupling output heads match masked pass-through dims exactly
    W_tn = np.load(ROOT / "models/tsunaminet/tsunaminet_weights.npz")
    for k in W_tn.files:
        arr = W_tn[k]
        if not np.issubdtype(arr.dtype, np.number):
            continue
        assert np.isfinite(arr).all(), f"non-finite {k}"
        if arr.ndim >= 2:
            axes = tuple(range(1, arr.ndim))
            dead = np.where(np.max(np.abs(arr), axis=axes) < 1e-8)[0]
            if k.startswith("f") and k.endswith("w2"):
                idx = int(k[1:-2])
                m = W_tn[f"f{idx}m"]
                expected = np.r_[np.where(m > 0.5)[0], np.where(m > 0.5)[0] + len(m)]
                assert np.array_equal(dead, expected), f"unexpected dead rows in {k}"
            else:
                assert len(dead) == 0, f"dead rows in {k}: {dead}"
    print("  weights: all numeric arrays finite; 0 dead active units (RealNVP masked pass-through rows verified 8/8)")

    # 2. Held-out target amplitude shift when input waveforms are zeroed (complements reviewer-b's Mw control)
    net = tn.TsunamiNet()
    gf = tn.GreensFunctions()
    gf.cal = net.cal
    wf = net.cal.get("widen", {}).get("amp", 1.0)
    ex_dir = ROOT / "models/tsunaminet/examples"
    for fn in sorted(f for f in os.listdir(ex_dir) if f.endswith(".npz")):
        z = np.load(ex_dir / fn, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        recs = {s: (z["eta"][i], z["mask"][i]) for i, s in enumerate(meta["sites"])}
        recs_zero = {s: (np.zeros_like(z["eta"][i]), z["mask"][i]) for i, s in enumerate(meta["sites"])}
        ev_full = gf.event(meta["lon"], meta["lat"], recs)
        ev_zero = gf.event(meta["lon"], meta["lat"], recs_zero)
        p_full = net.posterior(ev_full, n=512, seed=0)
        p_zero = net.posterior(ev_zero, n=512, seed=0)
        if not meta["targets"]:
            print(f"  {fn}: 0 targets | Mw {p_full['mw']:.2f} -> {p_zero['mw']:.2f}")
            continue
        t = meta["targets"][0]
        i = meta["sites"].index(t["site"])
        m = z["mask"][i][t["a"] : t["b"] + 1]
        f_full = gf.forecast(ev_full, p_full, t["site"], t["a"], t["b"], m, widen=wf)
        f_zero = gf.forecast(ev_zero, p_zero, t["site"], t["a"], t["b"], m, widen=wf)
        print(
            f"  {fn}: target={t['site'].split('_')[0]} | Mw {p_full['mw']:.2f}->{p_zero['mw']:.2f} | "
            f"amp_med {100*f_full['median']:.2f}->{100*f_zero['median']:.2f} cm (obs {100*t['obs']:.2f} cm)"
        )


def check_chirpnet():
    print("\n=== SECOND-PASS CHECK: CHIRPNET v2.1 ===")
    # 1. Dead-unit audit across all 4 shipped weight archives
    for wname in [
        "chirpnet_weights.npz",
        "chirpnet_coinc_weights.npz",
        "chirpnet_pe_weights.npz",
        "chirpnet_bns_weights.npz",
    ]:
        W_c = np.load(ROOT / "models/chirpnet" / wname)
        dead_unexpected = []
        for k in W_c.files:
            arr = W_c[k]
            if not np.issubdtype(arr.dtype, np.number):
                continue
            assert np.isfinite(arr).all(), f"non-finite {wname}:{k}"
            if arr.ndim >= 2:
                axes = tuple(range(1, arr.ndim))
                dead = np.where(np.max(np.abs(arr), axis=axes) < 1e-8)[0]
                if wname == "chirpnet_pe_weights.npz" and k.startswith("f") and k.endswith("w2"):
                    assert len(dead) == 6, f"unexpected PE flow dead rows in {k}: {dead}"
                elif len(dead):
                    dead_unexpected.append((k, arr.shape, len(dead)))
        assert not dead_unexpected, f"dead units in {wname}: {dead_unexpected}"
        print(f"  {wname}: finite, 0 dead active units")

    # 2. Lower-bound clustered O3 false-alarm rate from distinct O3 dev background pieces
    o3 = json.loads((ROOT / "models/chirpnet/training/logs/o3_dev_background.json").read_text())
    pieces_ge7 = [(k, v["max"]) for k, v in o3["pieces"].items() if v["max"] >= 7.0]
    livetime_h = o3["livetime_s"] / 3600.0
    lb_far = len(pieces_ge7) / livetime_h
    print(
        f"  O3 dev background: {len(pieces_ge7)}/{len(o3['pieces'])} distinct 32 s pieces exceed 7.0 "
        f"({o3['n_above_7']} windows in {livetime_h:.2f} h) -> clustered FAR lower bound >= {lb_far:.3f}/h"
    )
    for k, mx in pieces_ge7:
        print(f"    {k}: max={mx:.3f}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg in ("thermopilot", "all"):
        check_thermopilot()
    if arg in ("tsunaminet", "all"):
        check_tsunaminet()
    if arg in ("chirpnet", "all"):
        check_chirpnet()
