"""EnergyNet (Vehicle / RoadSim physics v2, VED real trips) adapter for TreeTwin + ConformalRatchet.

- Simulator : RoadSim physics v2 via models/energynet/training/{fleet,bank}.py (generated fleets
              driven over real VED train_noise routes + synthetic drives + dyno cycles; zero real
              training labels).
- Features  : Exact 151-feature EnergyNet tabular vector (90 route + 12 physics integrals + 49
              vehicle/tier features).
- Target    : log(E_true / E_phys) — correction to the generic physics energy estimate E_phys.
              Headline metric is mean per-trip MdAPE (%) across ICE, HEV, PHEV (EV reported
              alongside, matching EnergyNet's card and Prior Wars).
- Monotone  : +1 on EPA window-sticker fuel & electric log-ratios (higher lab consumption -> higher
              correction relative to generic E_phys) and -1 on mean ambient temperature (colder air
              -> higher density, rolling resistance and cabin/warm-up energy).
- Real splits: Frozen RoadData VED splits (dev = 16 cars; exam = ~350 unseen cars).
- Strata    : Powertrain group (ICE, HEV, PHEV, EV) for Mondrian conformal calibration.
- Incumbent : Released EnergyNet v2.1 5-MLP ensemble (models/energynet/weights/*.npz).
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import tt_paths as P

EN_TRAIN = P.REPO / "models" / "energynet" / "training"
EN_MODEL = P.REPO / "models" / "energynet"

NAME = "energynet"
TITLE = "EnergyNet (VED real trips)"
DOMAIN = "Transportation"
TARGET_NAME = "log_energy_ratio"
UNIT = "log(E / E_phys)"
SHAP_SCALE = 100.0   # delta log(E / E_phys) -> approx % energy correction
SHAP_UNIT = "% energy correction"

# Exact 151-feature index map (see models/energynet/training/features.py):
#   0..89   : route_vec R (90)
#   90..101 : phys_vec  F (12)
#   102..150: vehicle_vec V (49)
FEATURE_NAMES = (
    [
        "log_dur", "log_dist", "v_mean", "idle_frac", "stop_density",
        "v_start", "v_end", "acc_rms", "kin_energy_per_m", "v3_per_m",
    ]
    + [f"va_hist_{i:02d}" for i in range(56)]
    + [f"grade_hist_{i}" for i in range(7)]
    + ["climb_per_km", "descent_per_km", "net_elev_per_km"]
    + ["T_mean", "T_min", "T_max", "T_subzero_frac", "T_hot_frac"]
    + ["headwind_v", "crosswind_v", "wind_mean", "rho_air"]
    + ["surf_wet", "surf_snow", "surf_gravel", "surf_concrete", "hwy_frac"]
    + [
        "log_E_phys", "log_Ep_over_E", "log_En_over_E",
        "aero_share", "rolling_share", "grade_pos_share", "grade_neg_share", "inertia_share",
        "log_specific_wheel_E", "log_Eh_over_E", "log_Eb_over_E", "phev_cd_frac",
    ]
    + [f"arch_{i}" for i in range(7)]
    + [f"cls_{i}" for i in range(7)]
    + ["log_mass", "mass_exact", "mass_rep", "log_cda", "log_crr", "rl_known"]
    + ["log_disp", "has_disp", "log_power", "has_power", "diesel"]
    + [f"trans_{i}" for i in range(6)]
    + ["ngears", "has_ngears", "log_batt_cap", "has_batt_cap"]
    + ["lab_fuel_ratio", "lab_fuel_city_hwy", "has_lab_fuel"]
    + ["lab_elec_ratio", "lab_elec_city_hwy", "has_lab_elec"]
    + ["soc_start", "has_soc", "warm_start", "cold_start", "obd_maf", "dyno", "model_year", "has_model_year"]
)
assert len(FEATURE_NAMES) == 151

# 8 physical feature groups partitioning all 151 features (model-families.ts Section 07).
# Indices follow models/energynet/training/features.py exactly (see FEATURE_NAMES above):
# log_cda = 119 and log_crr = 120 are road-load terms; 122..126 are engine size/power/diesel.
# Fixed 27 Nov 2026 (Codex, PR #19): the first version filed 122-124 (engine) under aero and
# rolling, 119-120 under mass, and powertrain, EPA labels, battery and SOC under "Model year".
FEATURE_GROUPS = {
    # v3_per_m, wind (head, cross, mean), air density, highway share, aero work share, CdA
    "Aero work": [9, 81, 82, 83, 84, 89, 93, 119],
    # wet/snow/gravel/concrete surface, rolling work share, Crr, road-load-known flag
    "Rolling work": [85, 86, 87, 88, 94, 120, 121],
    # grade histogram, climb/descent/net elevation, uphill and downhill work shares
    "Grade work": list(range(66, 76)) + [95, 96],
    # mean speed, stop density, start/end speed, accel RMS, kinetic energy per m, speed-accel
    # histogram, inertia work share, regen-able and specific wheel energy
    "Speed & stops": [2, 4, 5, 6, 7, 8] + list(range(10, 66)) + [92, 97, 98],
    # T mean/min/max, sub-zero and hot fractions, warm and cold start flags
    "Ambient temperature": [76, 77, 78, 79, 80, 145, 146],
    # trip duration and distance, idle share, physics energy scale and propulsive share
    "Duration & idle": [0, 1, 3, 90, 91],
    # vehicle class one-hots, mass and how it is known
    "Mass & class": list(range(109, 116)) + [116, 117, 118],
    # architecture, engine size/power/diesel, transmission, battery, EPA labels, SOC,
    # hybrid/battery energy shares, PHEV charge-depleting share, OBD MAF and dyno flags, model year
    "Powertrain, labels & year": list(range(102, 109)) + list(range(122, 145)) + [99, 100, 101, 147, 148, 149, 150],
}
# Verify exact disjoint partition of all 151 features (preserves exact Shapley efficiency!)
_all_idx = [i for idxs in FEATURE_GROUPS.values() for i in idxs]
assert len(_all_idx) == 151 and len(set(_all_idx)) == 151, f"bad partition: {len(_all_idx)}, {len(set(_all_idx))}"

# Physically justified monotone constraints on log(E / E_phys):
#   - T_mean (76), T_min (77): -1 (colder ambient -> higher energy relative to room-T E_phys)
#   - lab_fuel_ratio (137), lab_elec_ratio (140): +1 (higher EPA sticker consumption -> higher correction)
MONOTONE_CST = [0] * 151
MONOTONE_CST[76] = -1
MONOTONE_CST[77] = -1
MONOTONE_CST[137] = 1
MONOTONE_CST[140] = 1

FLOOR = {"ICE": 1 * 0.042, "HEV": 1 * 0.042, "EV": 0.01 * 3.6, "PHEV": 0.05}
HEADLINE_GROUPS = ("ICE", "HEV", "PHEV")
EXAM_PER_CAR = 12      # fixed subsample cap per exam car for fast bootstrap + full-car balance
EXAM_SEED = 20261124


def _en_env() -> dict[str, str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    # Route EnergyNet training outputs cleanly into experiments/treetwin/work/energynet
    env["ML_RATCHET_WORK"] = str(P.work("energynet_root"))
    return env


def _en_work() -> Path:
    return P.work("energynet_root", "energynet")


def build_banks_and_real(n_train_veh: int = 800, n_val_veh: int = 200, workers: int = 2) -> None:
    """Build RoadSim v2 train/val banks and featurize VED dev & exam using EnergyNet's own scripts."""
    ew = _en_work()
    (ew / "data").mkdir(parents=True, exist_ok=True)
    env = _en_env()

    def _run(args: list[str]) -> None:
        print(f"[energynet] Running {' '.join(args)} ...", flush=True)
        subprocess.run([sys.executable] + args, cwd=str(EN_TRAIN), env=env, check=True)

    # 1. Real VED dev & exam featurization (uses official frozen VED splits)
    if not (ew / "data" / "real_ved_dev.npz").exists():
        _run(["realfeat.py", "ved_dev"])
    if not (ew / "data" / "real_ved_exam.npz").exists():
        _run(["realfeat.py", "ved_exam"])

    # 2. RoadSim physics v2 simulated train & val fleets + banks (zero real training labels!)
    if not (ew / "data" / "fleet_tt_val.json").exists():
        _run(["fleet.py", "tt_val", str(n_val_veh), "202", str(workers)])
    if not (ew / "data" / "fleet_tt_train.json").exists():
        _run(["fleet.py", "tt_train", str(n_train_veh), "201", str(workers)])
    if len(list((ew / "data" / "bank_tt_val").glob("chunk_*.npz"))) < 4:
        _run(["bank.py", "tt_val", "8", "777", str(workers)])
    if len(list((ew / "data" / "bank_tt_train").glob("chunk_*.npz"))) < 16:
        _run(["bank.py", "tt_train", "10", "101", str(workers)])


def _load_en_dataset():
    os.environ["ML_RATCHET_WORK"] = str(P.work("energynet_root"))
    P.add_sys_path(EN_TRAIN)
    import dataset as DS
    return DS


def _assemble_bank(name: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    DS = _load_en_dataset()
    B = DS.load_bank(name)
    T = DS.draw_tiers(B, np.random.default_rng(seed))
    X, E = DS.assemble(B, T)
    # Use EnergyNet v2.1's exact target ('e': RoadSim v2 OBD label model + PHEV patch)
    y = DS.target(B, T, "e")
    ok = (y > 0.02) & (E > 0) & np.all(np.isfinite(X), axis=1)
    arch = np.array(["conv", "hev_parallel_p2", "hev_series", "hev_powersplit", "phev_powersplit", "phev_series", "bev"])[B["arch"].astype(int)]
    grp = np.where(arch == "conv", "ICE", np.where(arch == "bev", "EV", np.where(np.char.startswith(arch, "phev"), "PHEV", "HEV")))
    lr = np.log(y[ok] / E[ok])
    return X[ok].astype(np.float32), lr.astype(np.float64), grp[ok].astype(str), E[ok].astype(np.float64)


def sim_train_val() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    build_banks_and_real()
    X_tr, y_tr, st_tr, _ = _assemble_bank("tt_train", seed=1)
    X_va, y_va, st_va, _ = _assemble_bank("tt_val", seed=777)
    return X_tr, y_tr, st_tr, X_va, y_va, st_va


def _load_real_ved(split: str) -> dict:
    build_banks_and_real()
    DS = _load_en_dataset()
    ew = _en_work()
    B = dict(np.load(ew / "data" / f"real_ved_{split}.npz"))
    T = {k: B["T_" + k] for k in DS.TIER_KEYS}
    M = pd.read_parquet(ew / "data" / f"real_ved_{split}_meta.parquet")
    X, E = DS.assemble(B, T)
    Es = E * B["vr"]
    y_mj = B["y"].astype(float)
    grp = M["group"].values.astype(str)
    fl = np.array([FLOOR[g] for g in grp], dtype=float)
    ok = (y_mj > fl) & (Es > 1e-4) & np.all(np.isfinite(X), axis=1)
    X = X[ok].astype(np.float32)
    y_mj = y_mj[ok]
    Es = Es[ok]
    M = M[ok].reset_index(drop=True)
    lr = np.log(y_mj / Es).astype(np.float64)
    return {
        "X": X,
        "y": lr,
        "E_phys": Es.astype(np.float64),
        "y_mj": y_mj.astype(np.float64),
        "strata": M["group"].values.astype(str),
        "unit_ids": M["vehicle_id"].values.astype(str),
        "trip_ids": M["trip_id"].values.astype(str),
        "T_mean_C": (X[:, 76] * 20.0).astype(np.float64),
        "stop_density": (X[:, 4] * 3.0).astype(np.float64),
        "hwy_frac": X[:, 89].astype(np.float64),
    }


def real_dev() -> dict:
    return _load_real_ved("dev")


def real_exam() -> dict:
    d = _load_real_ved("exam")
    # Use all exam trips (fast for GBDT and MLP; unit-grouped bootstrap over all ~350 cars)
    return d


@functools.lru_cache(maxsize=1)
def _incumbent_ensemble():
    P.add_sys_path(EN_MODEL)
    import glob
    from model_np import Ensemble
    ps = sorted(glob.glob(str(EN_MODEL / "weights" / "*.npz")))
    return Ensemble(ps)


def incumbent_predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_log_ratio, raw_unwidened_sigma_log_ratio) from released EnergyNet v2.1 5-MLP ensemble."""
    ens = _incumbent_ensemble()
    X2 = np.asarray(X, dtype=np.float64)
    # Ensemble.predict(X, E_phys=1) returns (exp(mean_m), sqrt(mean(var) + var(m)), comp)
    ms, vs = [], []
    for m in ens.m:
        mu, sd, _ = m.forward(X2)
        ms.append(mu)
        vs.append(sd ** 2)
    M = np.stack(ms, axis=0)
    V = np.stack(vs, axis=0)
    mean_lr = np.mean(M, axis=0)
    raw_sd = np.sqrt(np.mean(V, axis=0) + np.var(M, axis=0))
    return mean_lr.astype(np.float64), raw_sd.astype(np.float64)


def headline_metric(y_true: np.ndarray, y_pred: np.ndarray, strata: np.ndarray | None = None) -> float:
    """Mean per-trip MdAPE (%) over ICE, HEV, PHEV (lower is better; matches EnergyNet card)."""
    ape = 100.0 * np.abs(np.exp(np.asarray(y_pred, float) - np.asarray(y_true, float)) - 1.0)
    if strata is None:
        return float(np.median(ape))
    st = np.asarray(strata, dtype=str)
    vals = [float(np.median(ape[st == g])) for g in HEADLINE_GROUPS if np.any(st == g)]
    return float(np.mean(vals)) if vals else float(np.median(ape))


def metric_details(y_true: np.ndarray, y_pred: np.ndarray, strata: np.ndarray) -> dict:
    ape = 100.0 * np.abs(np.exp(np.asarray(y_pred, float) - np.asarray(y_true, float)) - 1.0)
    out = {
        "headline_mdape_pct": round(headline_metric(y_true, y_pred, strata), 2),
        "pooled_mdape_pct": round(float(np.median(ape)), 2),
        "log_mae": round(float(np.mean(np.abs(y_pred - y_true))), 4),
        "strata": {},
    }
    for g in sorted(set(strata.tolist())):
        m = strata == g
        out["strata"][g] = {
            "n": int(np.sum(m)),
            "mdape_pct": round(float(np.median(ape[m])), 2),
            "log_mae": round(float(np.mean(np.abs(y_pred[m] - y_true[m]))), 4),
        }
    return out


if __name__ == "__main__":
    build_banks_and_real()
    d = real_dev()
    print(f"[energynet] dev trips={len(d['X'])}, cars={len(set(d['unit_ids']))}, strata={dict(zip(*np.unique(d['strata'], return_counts=True)))}")
