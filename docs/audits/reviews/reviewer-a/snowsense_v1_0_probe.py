"""Forensic audit probe for SnowSense v1.0 (Reviewer A, first pass).

Checks:
  1. Weight health (finiteness, dead rows/cols) and widen-only calibration factors (k_r6.json).
  2. Torch (training/model.py SnowNet) vs NumPy (snowsense.py SnowSense.forward) parity.
  3. Causal boundary on missing temperature:
     - complete-prefix control (perturb P +50 mm, T +15 C strictly after 2022-03-01)
     - synthetic 2-day T gap straddling 2022-03-01 with +15 C applied only from 2022-03-02 onward
     - actual existing SNOTEL T gap at 497_WY_SNTL (2022-01-19, next obs 2022-01-21)
     - causal forward-fill (ffill) replay vs bidirectional interpolation on all 3 packaged exam stations
  4. Physical counterfactual ablations: P=0 (no precipitation) and T+15 C warming.
  5. Replay of all 6 logged rungs in training/ratchet.json on val_p4.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/snowsense_v1_0_probe.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / "models/snowsense"),
    str(ROOT / "models/snowsense/training"),
]
import snowsense as ss
import model as ss_tr_mod


def causal_predict_swe(ss_np, df, lat, elev_m):
    """Replay SnowSense with causal forward-fill for T gaps, preserving mT flags identically."""
    d = df.asfreq("D")
    mP = d["P"].isna().astype(np.float32)
    P = d["P"].clip(lower=0).fillna(0.0)
    tm_raw = d["T_mean"].fillna((d["T_min"] + d["T_max"]) / 2)
    mT = tm_raw.isna().astype(np.float32)
    tn_raw = d["T_min"].fillna(tm_raw - 5.0)
    tx_raw = d["T_max"].fillna(tm_raw + 5.0)
    c_ff = pd.DataFrame(index=d.index)
    c_ff["mP"] = mP
    c_ff["P"] = P
    c_ff["mT"] = mT
    c_ff["T_mean"] = tm_raw.ffill().fillna(0.0)
    c_ff["T_min"] = np.minimum(tn_raw.ffill().fillna(-5.0), c_ff["T_mean"])
    c_ff["T_max"] = np.maximum(tx_raw.ffill().fillna(5.0), c_ff["T_mean"])

    _, D_w, wys = ss_np.windows(df, lat, elev_m)
    X_ff = []
    for y in wys:
        idx = pd.date_range(f"{y - 1}-08-01", periods=ss.L, freq="D")
        w = c_ff.reindex(idx)
        Tm = w["T_mean"].ffill().fillna(0.0).values
        Tn = w["T_min"].ffill().fillna(-5.0).values
        Tx = w["T_max"].ffill().fillna(5.0).values
        mp = np.where(w["mP"].isna(), 1.0, w["mP"].values)
        mt = np.where(w["mT"].isna(), 1.0, w["mT"].values)
        X_ff.append(ss.features(w["P"].fillna(0).values, Tm, Tn, Tx, mp, mt, idx, lat, elev_m))
    q_ff = ss_np.forward(np.stack(X_ff).astype(np.float64))
    parts = []
    for j, y in enumerate(wys):
        idx = D_w[j]
        inwy = (idx.year + (idx.month >= 10)) == y
        parts.append(pd.Series(100.0 * np.maximum(q_ff[j, inwy, 2], 0.0) ** 2, index=idx[inwy]))
    return pd.concat(parts), int(mT.sum())


def main():
    w_path = ROOT / "models/snowsense/snowsense_weights.npz"
    w = np.load(w_path)
    for k in w.files:
        arr = w[k]
        assert np.isfinite(arr).all(), f"non-finite {k}"
        if arr.ndim == 2:
            assert (np.max(np.abs(arr), axis=1) >= 1e-8).all(), f"dead row {k}"
            assert (np.max(np.abs(arr), axis=0) >= 1e-8).all(), f"dead col {k}"
    k_r6 = json.loads((ROOT / "models/snowsense/training/k_r6.json").read_text())
    print(f"weights: finite, 0 dead rows/cols | k={w['k'].tolist()} | k_r6={k_r6}")

    # Build Torch mirror
    snet_t = ss_tr_mod.SnowNet(hidden=64, layers=2, nin=10, head="quant")
    sd = snet_t.state_dict()
    with torch.no_grad():
        sd["inp.weight"].copy_(torch.tensor(w["inp.weight"]))
        sd["inp.bias"].copy_(torch.tensor(w["inp.bias"]))
        for l in range(2):
            for pn in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                sd[f"gru.{pn}_l{l}"].copy_(torch.tensor(w[f"gru.{pn}_l{l}"]))
        sd["head.weight"].copy_(torch.tensor(w["head.weight"]))
        sd["head.bias"].copy_(torch.tensor(w["head.bias"]))
    snet_t.eval()

    ss_np = ss.SnowSense(str(w_path))
    smeta = json.loads((ROOT / "models/snowsense/examples/stations.json").read_text())

    max_torch_diff = 0.0
    print("\nPackaged SNOTEL exam stations:")
    for sid, rec in smeta.items():
        df = pd.read_csv(ROOT / "models/snowsense/examples" / f"{sid}.csv", parse_dates=["date"]).set_index("date")
        X_w, D_w, wys = ss_np.windows(df, rec["lat"], rec["elev_m"])
        q_np = ss_np.forward(X_w.astype(np.float64))
        with torch.no_grad():
            q_t = snet_t.quantiles(torch.tensor(X_w, dtype=torch.float32)).numpy()
        td = float(np.max(np.abs(q_np - q_t)))
        max_torch_diff = max(max_torch_diff, td)

        out0 = ss_np.predict(df, lat=rec["lat"], elev_m=rec["elev_m"])
        daily_col = out0["daily"].columns[0]

        # 3a. Complete-prefix control (perturb strictly after 2022-03-01)
        cut = pd.Timestamp("2022-03-01")
        df_c = df.copy()
        after = df_c.index > cut
        df_c.loc[after, "P"] = df_c.loc[after, "P"].fillna(0.0) + 50.0
        for c in ("T_mean", "T_min", "T_max"):
            df_c.loc[after, c] = df_c.loc[after, c] + 15.0
        out_c = ss_np.predict(df_c, lat=rec["lat"], elev_m=rec["elev_m"])
        ctrl_diff = float(np.max(np.abs(out_c["daily"].loc[:cut, daily_col] - out0["daily"].loc[:cut, daily_col])))
        assert ctrl_diff == 0.0, f"complete-prefix leak on {sid}: {ctrl_diff}"

        # 3b. Synthetic 2-day T gap (Feb 28 - Mar 1) + future T +15 C from Mar 2 onward
        df_g0 = df.copy()
        gap_mask = (df_g0.index >= "2022-02-28") & (df_g0.index <= "2022-03-01")
        for c in ("T_mean", "T_min", "T_max"):
            df_g0.loc[gap_mask, c] = np.nan
        df_g1 = df_g0.copy()
        for c in ("T_mean", "T_min", "T_max"):
            df_g1.loc[df_g1.index >= "2022-03-02", c] = df_g1.loc[df_g1.index >= "2022-03-02", c] + 15.0
        s_g0 = float(ss_np.predict(df_g0, lat=rec["lat"], elev_m=rec["elev_m"])["daily"].loc[cut, daily_col])
        s_g1 = float(ss_np.predict(df_g1, lat=rec["lat"], elev_m=rec["elev_m"])["daily"].loc[cut, daily_col])

        # 3c. Causal ffill replay on actual shipped gaps
        swe_ff, n_t_nan = causal_predict_swe(ss_np, df, rec["lat"], rec["elev_m"])
        obs_eod = df["SWE_obs"].shift(-1).reindex(out0["daily"].index)
        valid = obs_eod.notna()
        y_o = obs_eod[valid].values
        y_p = out0["daily"].loc[valid, daily_col].values
        y_ff = swe_ff.reindex(out0["daily"].index).loc[valid].values
        nse_i = 1.0 - np.sum((y_p - y_o) ** 2) / np.sum((y_o - y_o.mean()) ** 2)
        nse_f = 1.0 - np.sum((y_ff - y_o) ** 2) / np.sum((y_o - y_o.mean()) ** 2)
        max_ff_shift = float(np.max(np.abs(y_p - y_ff)))

        # 4. Physical ablations (P=0 and T+15 C)
        df_noP = df.copy()
        df_noP["P"] = 0.0
        df_hot = df.copy()
        for c in ("T_mean", "T_min", "T_max"):
            df_hot[c] = df_hot[c] + 15.0
        pk0 = float(out0["annual"]["peak_swe_mm"].mean())
        pk_noP = float(ss_np.predict(df_noP, lat=rec["lat"], elev_m=rec["elev_m"])["annual"]["peak_swe_mm"].mean())
        pk_hot = float(ss_np.predict(df_hot, lat=rec["lat"], elev_m=rec["elev_m"])["annual"]["peak_swe_mm"].mean())

        print(
            f"  {sid}: torch_diff={td:.2e} | ctrl={ctrl_diff:.1e} | gap_leak={s_g0:.2f}->{s_g1:.2f} ({s_g1-s_g0:+.2f} mm) | "
            f"nT_NaN={n_t_nan}, NSE interp={nse_i:.5f} vs ffill={nse_f:.5f} (max_day={max_ff_shift:.2f} mm) | "
            f"peak {pk0:.1f} -> P=0:{pk_noP:.1f}, T+15:{pk_hot:.1f} mm"
        )

    # 3d. Actual real gap test at 497_WY_SNTL (2022-01-19 missing T, next obs 2022-01-21)
    df_wy = pd.read_csv(ROOT / "models/snowsense/examples/497_WY_SNTL.csv", parse_dates=["date"]).set_index("date")
    rec_wy = smeta["497_WY_SNTL"]
    gap_day = pd.Timestamp("2022-01-19")
    assert df_wy.loc[gap_day, ["T_mean", "T_min", "T_max"]].isna().all()
    df_wy_fut = df_wy.copy()
    for c in ("T_mean", "T_min", "T_max"):
        df_wy_fut.loc[df_wy_fut.index >= "2022-01-21", c] = df_wy_fut.loc[df_wy_fut.index >= "2022-01-21", c] + 15.0
    o_wy0 = ss_np.predict(df_wy, lat=rec_wy["lat"], elev_m=rec_wy["elev_m"])["daily"]
    o_wy1 = ss_np.predict(df_wy_fut, lat=rec_wy["lat"], elev_m=rec_wy["elev_m"])["daily"]
    dc = o_wy0.columns[0]
    w0, w1 = float(o_wy0.loc[gap_day, dc]), float(o_wy1.loc[gap_day, dc])
    print(f"\nreal gap 497_WY_SNTL (2022-01-19, perturb >=2022-01-21 by +15 C): {w0:.2f} -> {w1:.2f} mm ({w1-w0:+.2f} mm)")
    print(f"max Torch-vs-NumPy quantile diff: {max_torch_diff:.3e}")

    # 5. Ratchet replay
    rj = json.loads((ROOT / "models/snowsense/training/ratchet.json").read_text())
    inc_s, inc_d = None, None
    for row in rj["rungs"]:
        s, d_nse = row["sim"]["val_p4"], row["dev"]["nse"]
        kept = True if inc_s is None else (s > inc_s + 0.01) and ((1.0 - d_nse) <= 1.05 * (1.0 - inc_d))
        assert kept == row["kept"], f"ratchet mismatch on {row['rung']}"
        if kept:
            inc_s, inc_d = s, d_nse
    print(f"ratchet.json replay: {len(rj['rungs'])}/{len(rj['rungs'])} rungs verified -> incumbent={rj['incumbent']}")


if __name__ == "__main__":
    main()
