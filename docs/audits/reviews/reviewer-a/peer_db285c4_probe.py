"""Second-pass independent cross-check probes for reviewer-b's db285c4 audits:
HybridPilot v2.2, FaultFinder v1.1, and StormNet v1.0.1.

Run all checks:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_db285c4_probe.py
Or a single target:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_db285c4_probe.py hybridpilot
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]


def check_hybridpilot():
    print("=== SECOND-PASS CHECK: HybridPilot v2.2 ===")
    # 1. Shared RoadSim scope across models/
    roadsim_models = set()
    roadsim_files = 0
    for p in (REPO / "models").rglob("*.py"):
        txt = p.read_text(errors="ignore")
        if "roadsim" in txt:
            roadsim_files += 1
            roadsim_models.add(p.relative_to(REPO / "models").parts[0])
    print(f"RoadSim consumers in models/: {sorted(roadsim_models)} ({roadsim_files} files)")
    assert roadsim_models == {"energynet", "hybridpilot", "lawfinder", "tripdetective"}

    # 2. Full-trip closed-loop fuel & SOC impact: RoadSim bidirectional interp vs causal ffill
    # on the exact same 10 s grade gap + +4% post-gap step (preview_s=0, nav=False, soc0=0.55)
    sys.path.insert(0, str(REPO / "shared/vehicle/roadsim"))
    sys.path.insert(0, str(REPO / "models/hybridpilot"))
    import hybridpilot as hp
    from pipeline import load_example
    from roadsim.sim import simulate

    policy = hp.make_roadsim_strategy(preview_s=0.0, nav=False)
    trips = [("VED_125_1275", 41), ("VED_492_2826", 33), ("VED_428_465", 28)]
    for tid, gap_start in trips:
        spec, trip = load_example(tid)
        gap_end = gap_start + 10
        g_true = trip["grade"].copy()
        g_true[gap_end:] += 0.04

        g_gap = g_true.copy()
        g_gap[gap_start:gap_end] = np.nan

        g_ff = g_true.copy()
        g_ff[gap_start:gap_end] = g_true[gap_start - 1]

        def run_grade(g_arr):
            tr = dict(trip)
            tr["grade"] = g_arr.copy()
            return simulate(spec, tr, policy, init_soc=0.55)

        o_true = run_grade(g_true)
        o_interp = run_grade(g_gap)
        o_ff = run_grade(g_ff)

        dfuel = float(o_interp["summary"]["fuel_g"] - o_ff["summary"]["fuel_g"])
        dsoc = float(o_interp["soc"][-1] - o_ff["soc"][-1])
        p_diff_earlier = float(np.max(np.abs(o_interp["eng_power"][:gap_end] - o_ff["eng_power"][:gap_end]))) / 1000.0
        flips = int(np.sum(o_interp["eng_on"][:gap_end] != o_ff["eng_on"][:gap_end]))
        print(
            f"  {tid}: fuel true={o_true['summary']['fuel_g']:.3f}g, "
            f"interp={o_interp['summary']['fuel_g']:.3f}g, ffill={o_ff['summary']['fuel_g']:.3f}g "
            f"(dFuel={dfuel:+.3f}g, dSOC={dsoc:+.5f}, max_earlier_P={p_diff_earlier:.2f}kW, flips={flips})"
        )
        if tid == "VED_125_1275":
            assert flips >= 1 and abs(dfuel) > 5.0
        if tid == "VED_428_465":
            assert p_diff_earlier > 10.0 and abs(dfuel) > 4.0


def check_faultfinder():
    print("\n=== SECOND-PASS CHECK: FaultFinder v1.1 ===")
    sys.path.insert(0, str(REPO / "models/faultfinder"))
    import faultfinder as ff

    model = ff.FaultFinder()
    ex_dir = REPO / "models/faultfinder/examples"
    refs_map = json.loads((ex_dir / "references.json").read_text())
    cases = [
        ("flexlab", "flexlab_lbnl_szvav_flexlab_2017-09-11.csv"),
        ("iowa_ers", "iowa_ers_lbnl_mzvav2_ers_exp_2009-01-19.csv"),
        ("nesbitt", "nesbitt_NES_20180711_AHU-1.csv"),
        ("ornl_rtu", "ornl_rtu_lbnl_rtu_ornl_2018-01-31.csv"),
    ]
    alarm_drops_6h = 0
    for name, day_f in cases:
        df_day = pd.read_csv(ex_dir / day_f)
        ref_dfs = [pd.read_csv(ex_dir / rp) for rp in refs_map.get(day_f, [])]
        base = model.read_day(df_day, ref_days=ref_dfs or None)
        p_base = float(base["p_fault"])

        # 1. Self-reference leakage: passing target day itself as reference
        self_ref = model.read_day(df_day, ref_days=[df_day])
        p_self = float(self_ref["p_fault"])

        # 2. Truncated partial-day prefixes (6h = 24 rows, 12h = 48 rows)
        r6 = model.read_day(df_day.iloc[:24], ref_days=ref_dfs or None)
        r12 = model.read_day(df_day.iloc[:48], ref_days=ref_dfs or None)
        p6 = float(r6["p_fault"])
        p12 = float(r12["p_fault"])
        if p_base >= 0.50 and p6 < 0.50:
            alarm_drops_6h += 1
        print(
            f"  {name:10s}: base={p_base:.5f} | self_ref={p_self:.5f} ({p_self - p_base:+.5f}) | "
            f"6h={p6:.5f} | 12h={p12:.5f}"
        )
    assert alarm_drops_6h == 3, f"Expected all 3 faulty-classified examples to drop < 0.50 at 6h, got {alarm_drops_6h}"


def check_stormnet():
    print("\n=== SECOND-PASS CHECK: StormNet v1.0.1 ===")
    sys.path.insert(0, str(REPO / "models/stormnet"))
    import stormnet as sn

    model = sn.StormNet()

    # 1. All-NaN / missing input guardrails
    for label, wind, dst in [
        ("all_nan", np.full((1, 36, 3), np.nan), np.array([np.nan])),
        ("nan_wind_valid_dst", np.full((1, 36, 3), np.nan), np.array([-100.0])),
        ("quiet_wind_nan_dst", np.tile([0.0, 400.0, 5.0], (1, 36, 1)), np.array([np.nan])),
    ]:
        raised = False
        try:
            model.predict(wind, dst)
        except ValueError as e:
            raised = True
            print(f"  guardrail {label:20s}: raised ValueError ({e})")
        assert raised, f"{label} should raise ValueError"

    # 2. Solar-wind vs Dst persistence counterfactual ablation on all 721 gallery issues
    meta = json.loads((REPO / "models/stormnet/examples/storms.json").read_text())
    z = np.load(REPO / "models/stormnet/examples/storms.npz")
    winds, dsts, trues = [], [], []
    for idx, st in enumerate(meta):
        w_arr = z[f"wind{idx}"]
        d_arr = z[f"dst{idx}"]
        start = int(st["issue_from"])
        T = len(d_arr)
        for i in range(start + 1, T - 3):
            w = w_arr[i - 35 : i + 1]
            d0 = float(d_arr[i])
            y = float(np.min(d_arr[i + 1 : i + 4]))
            if w.shape == (36, 3) and np.all(np.isfinite(w)) and np.isfinite(d0) and np.isfinite(y):
                winds.append(w)
                dsts.append(d0)
                trues.append(y)
    W = np.asarray(winds, dtype=float)
    D0 = np.asarray(dsts, dtype=float)
    Y = np.asarray(trues, dtype=float)
    assert len(Y) == 721, f"Expected 721 gallery issues, got {len(Y)}"

    pred_full = model.predict(W, D0)["min_dst"]
    W_quiet = np.tile([0.0, 400.0, 5.0], (len(Y), 36, 1))
    pred_quiet = model.predict(W_quiet, D0)["min_dst"]
    pred_zero_dst = model.predict(W, np.zeros_like(D0))["min_dst"]

    for label, mask in [
        ("all (n=721)", np.ones(len(Y), dtype=bool)),
        ("storm <= -50 nT", Y <= -50.0),
        ("extreme <= -150 nT", Y <= -150.0),
    ]:
        mae_full = float(np.mean(np.abs(pred_full[mask] - Y[mask])))
        mae_pers = float(np.mean(np.abs(D0[mask] - Y[mask])))
        mae_quiet = float(np.mean(np.abs(pred_quiet[mask] - Y[mask])))
        mae_zdst = float(np.mean(np.abs(pred_zero_dst[mask] - Y[mask])))
        print(
            f"  {label:20s} (n={int(mask.sum()):3d}): full={mae_full:.2f} nT | "
            f"pers={mae_pers:.2f} nT | quiet_SW={mae_quiet:.2f} nT | zero_Dst={mae_zdst:.2f} nT"
        )
        assert mae_full < mae_pers and mae_full < mae_quiet


def main():
    targets = set(sys.argv[1:]) or {"hybridpilot", "faultfinder", "stormnet"}
    if "hybridpilot" in targets:
        check_hybridpilot()
    if "faultfinder" in targets:
        check_faultfinder()
    if "stormnet" in targets:
        check_stormnet()
    print("\nALL SECOND-PASS CHECKS PASSED.")


if __name__ == "__main__":
    main()
