"""Unified cross-platform harness for TreeTwin + ConformalRatchet.

Stages (strictly separated; exam refuses to run until PREREG.md is frozen with freeze.py):
  1. ``train_tune`` : Train & tune GBDT-Plain and GBDT-Mono on physics simulation banks ONLY,
                      export to pure-numpy mlratchet.gbm.TreeEnsemble, verify parity & TreeSHAP.
  2. ``dev_twin``   : Evaluate on Real Dev ONLY; run exact TreeSHAP real-vs-twin diagnostics and
                      fit pooled widen-only, raw Mondrian, and shrunk Mondrian calibrators.
  3. ``exam``       : Locked behind PREREG.md SHA-256. Scores Incumbent P, GBDT-Plain, GBDT-Mono,
                      stratum-level 68% conformal coverage, and GBDT-vs-MLP disagreement ONCE.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np

from mlratchet.calibrate import ONE_SIGMA, MondrianConformal, stratum_coverage, widen_factor
from mlratchet.gbm import TreeEnsemble, from_lightgbm
from mlratchet.twintest import disagreement_detector, shap_residual_gap, shap_twin_diff, twin_report

import freeze as FR
import tt_paths as P

PLATFORMS = ("buildingdetective", "airfoilnet", "energynet")
TUNE_GRID = [
    {"num_leaves": 15, "min_child_samples": 30, "learning_rate": 0.05},
    {"num_leaves": 31, "min_child_samples": 20, "learning_rate": 0.05},
    {"num_leaves": 31, "min_child_samples": 50, "learning_rate": 0.05},
    {"num_leaves": 63, "min_child_samples": 30, "learning_rate": 0.04},
]
SEEDS = (11, 23, 37)


def load_adapter(name: str):
    return importlib.import_module(f"adapters.{name}")


def _fit_lgb_reg(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    params: dict,
    seed: int,
    monotone_cst: list[int] | None = None,
    objective: str = "regression",
    alpha: float | None = None,
):
    kw = dict(
        n_estimators=800,
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        learning_rate=params["learning_rate"],
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
        objective=objective,
    )
    if alpha is not None:
        kw["alpha"] = alpha
    if monotone_cst is not None and any(v != 0 for v in monotone_cst):
        kw["monotone_constraints"] = list(monotone_cst)
    model = lgb.LGBMRegressor(**kw)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )
    return model


def _train_arm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    feature_names: list[str],
    monotone_cst: list[int] | None,
    out_dir: Path,
    arm_name: str,
) -> dict:
    """Tune on frozen sim-val only, train 3-seed mean + q16/q84 quantile forests, export to numpy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    best_cfg, best_mae, tune_log = None, float("inf"), []
    for cfg in TUNE_GRID:
        m = _fit_lgb_reg(X_tr, y_tr, X_va, y_va, cfg, seed=SEEDS[0], monotone_cst=monotone_cst)
        p = m.predict(X_va)
        mae = float(np.mean(np.abs(p - y_va)))
        tune_log.append({"cfg": cfg, "val_mae": round(mae, 6), "best_iter": int(m.best_iteration_)})
        if mae < best_mae:
            best_mae, best_cfg = mae, cfg

    # Fit 3-seed mean ensemble + q16 / q84 quantile forests on winning sim-val config
    mean_ensembles: list[TreeEnsemble] = []
    max_parity = 0.0
    max_eff = 0.0
    for k, s in enumerate(SEEDS):
        m = _fit_lgb_reg(X_tr, y_tr, X_va, y_va, best_cfg, seed=s, monotone_cst=monotone_cst)
        ens = from_lightgbm(m, feature_names=feature_names, monotone_cst=monotone_cst)
        ens.save_npz(str(out_dir / f"{arm_name}_mean_s{k}.npz"))
        mean_ensembles.append(ens)
        # Verify pure-numpy parity vs LightGBM on validation subset
        sub = X_va[: min(64, len(X_va))]
        max_parity = max(max_parity, float(np.max(np.abs(ens.predict(sub) - m.predict(sub)))))
        if k == 0:
            sub_s = X_va[: min(16, len(X_va))]
            phi, ev = ens.treeshap(sub_s)
            max_eff = float(np.max(np.abs(ev + phi.sum(axis=1) - ens.predict(sub_s))))

    # Quantile forests (16th and 84th percentiles for native heteroscedastic tree sigma;
    # LightGBM does not support monotone_constraints with objective='quantile', and quantiles
    # only set the predictive spread around the monotone mean ensemble).
    q16_m = _fit_lgb_reg(
        X_tr, y_tr, X_va, y_va, best_cfg, seed=101,
        monotone_cst=None, objective="quantile", alpha=0.158655,
    )
    q84_m = _fit_lgb_reg(
        X_tr, y_tr, X_va, y_va, best_cfg, seed=102,
        monotone_cst=None, objective="quantile", alpha=0.841345,
    )
    q16_e = from_lightgbm(q16_m, feature_names=feature_names)
    q84_e = from_lightgbm(q84_m, feature_names=feature_names)
    q16_e.save_npz(str(out_dir / f"{arm_name}_q16.npz"))
    q84_e.save_npz(str(out_dir / f"{arm_name}_q84.npz"))

    pred_va, sig_va = predict_gbdt_arm(out_dir, arm_name, X_va)
    n_params_total = sum(e.n_params for e in mean_ensembles) + q16_e.n_params + q84_e.n_params
    return {
        "arm": arm_name,
        "best_cfg": best_cfg,
        "tune_log": tune_log,
        "sim_val_mae": round(float(np.mean(np.abs(pred_va - y_va))), 6),
        "sim_val_cov68": round(float(np.mean(np.abs(pred_va - y_va) <= sig_va)), 4),
        "n_trees_mean0": mean_ensembles[0].n_trees,
        "n_params_mean0": mean_ensembles[0].n_params,
        "n_params_all": int(n_params_total),
        "numpy_vs_lgb_max_abs_diff": float(f"{max_parity:.3e}"),
        "treeshap_efficiency_max_err": float(f"{max_eff:.3e}"),
    }


def predict_gbdt_arm(out_dir: Path, arm_name: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy inference for a trained TreeTwin GBDT arm (zero lightgbm calls)."""
    preds = []
    for k in range(len(SEEDS)):
        ens = TreeEnsemble.load_npz(str(out_dir / f"{arm_name}_mean_s{k}.npz"))
        preds.append(ens.predict(X))
    P_arr = np.stack(preds, axis=0)
    mu = np.mean(P_arr, axis=0)
    ens_sd = np.std(P_arr, axis=0)
    q16 = TreeEnsemble.load_npz(str(out_dir / f"{arm_name}_q16.npz")).predict(X)
    q84 = TreeEnsemble.load_npz(str(out_dir / f"{arm_name}_q84.npz")).predict(X)
    q_sd = np.maximum(0.5 * (q84 - q16), 0.0)
    raw_sd = np.sqrt(q_sd ** 2 + ens_sd ** 2)
    raw_sd = np.maximum(raw_sd, 1e-3)
    return mu.astype(np.float64), raw_sd.astype(np.float64)


def stage_train_tune(platform: str) -> dict:
    ad = load_adapter(platform)
    t0 = time.time()
    X_tr, y_tr, st_tr, X_va, y_va, st_va = ad.sim_train_val()
    out_dir = P.work("models", platform)
    print(f"[{platform}] train_tune: X_tr={X_tr.shape}, X_va={X_va.shape}", flush=True)

    plain = _train_arm(X_tr, y_tr, X_va, y_va, ad.FEATURE_NAMES, None, out_dir, "gbdt_plain")
    mono = _train_arm(X_tr, y_tr, X_va, y_va, ad.FEATURE_NAMES, ad.MONOTONE_CST, out_dir, "gbdt_mono")

    # Also evaluate Incumbent P on the exact same frozen sim val set
    p_va, s_va = ad.incumbent_predict(X_va)
    inc_mae = float(np.mean(np.abs(p_va - y_va)))
    p_plain_va, _ = predict_gbdt_arm(out_dir, "gbdt_plain", X_va)
    p_mono_va, _ = predict_gbdt_arm(out_dir, "gbdt_mono", X_va)

    def _hm(yp):
        if platform == "energynet":
            return ad.headline_metric(y_va, yp, st_va)
        return ad.headline_metric(y_va, yp)

    summary = {
        "platform": platform,
        "title": ad.TITLE,
        "domain": ad.DOMAIN,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_features": int(X_tr.shape[1]),
        "incumbent_sim_val_mae": round(inc_mae, 6),
        "incumbent_sim_val_headline": round(_hm(p_va), 4),
        "plain_sim_val_headline": round(_hm(p_plain_va), 4),
        "mono_sim_val_headline": round(_hm(p_mono_va), 4),
        "gbdt_plain": plain,
        "gbdt_mono": mono,
        "wall_s": round(time.time() - t0, 1),
    }
    sp = out_dir / "train_summary.json"
    sp.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[{platform}] sim_val headline: Incumbent={summary['incumbent_sim_val_headline']} "
        f"Plain={summary['plain_sim_val_headline']} Mono={summary['mono_sim_val_headline']} "
        f"(parity={mono['numpy_vs_lgb_max_abs_diff']}, wall={summary['wall_s']}s)",
        flush=True,
    )
    return summary


def _fit_calibrators(y_dev: np.ndarray, pred_dev: np.ndarray, sig_dev: np.ndarray, strata_dev: np.ndarray) -> dict:
    resid = y_dev - pred_dev
    f_pool = widen_factor(resid, sig_dev, target=ONE_SIGMA)
    mc_raw = MondrianConformal.fit(resid, sig_dev, strata_dev, target=ONE_SIGMA, widen_only=True, shrink_k=0.0, min_n=5)
    mc_shr = MondrianConformal.fit(resid, sig_dev, strata_dev, target=ONE_SIGMA, widen_only=True, shrink_k=10.0, min_n=5)
    return {
        "pooled_widen": float(f_pool),
        "mondrian_raw": mc_raw.to_dict(),
        "mondrian_shrunk": mc_shr.to_dict(),
    }


def _eval_calibrators(y: np.ndarray, pred: np.ndarray, sig: np.ndarray, strata: np.ndarray, cals: dict) -> dict:
    resid = y - pred
    mc_raw = MondrianConformal.from_dict(cals["mondrian_raw"])
    mc_shr = MondrianConformal.from_dict(cals["mondrian_shrunk"])
    return {
        "raw": stratum_coverage(resid, sig, strata, target=ONE_SIGMA),
        "pooled_widen": stratum_coverage(resid, sig * cals["pooled_widen"], strata, target=ONE_SIGMA),
        "mondrian_raw": stratum_coverage(resid, mc_raw.halfwidth(sig, strata), strata, target=ONE_SIGMA),
        "mondrian_shrunk": stratum_coverage(resid, mc_shr.halfwidth(sig, strata), strata, target=ONE_SIGMA),
    }


def _fit_dev_gap_tree(X_dev: np.ndarray, resid: np.ndarray, feature_names: list[str]) -> TreeEnsemble:
    """Shallow dev-only diagnostic GBDT fitted on dev residuals (never used for exam prediction)."""
    m = lgb.LGBMRegressor(
        n_estimators=45,
        max_depth=3,
        num_leaves=7,
        min_child_samples=max(15, min(40, len(X_dev) // 25)),
        learning_rate=0.06,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=5.0,
        random_state=20261124,
        n_jobs=1,
        verbose=-1,
    )
    m.fit(X_dev, resid)
    return from_lightgbm(m, feature_names=feature_names)


def stage_dev_twin(platform: str) -> dict:
    ad = load_adapter(platform)
    out_dir = P.work("models", platform)
    dev = ad.real_dev()
    X, y, st, uid = dev["X"], dev["y"], dev["strata"], dev["unit_ids"]
    print(f"[{platform}] dev_twin: n_dev={len(X)}, units={len(set(uid.tolist()))}", flush=True)

    p_inc, s_inc = ad.incumbent_predict(X)
    p_pln, s_pln = predict_gbdt_arm(out_dir, "gbdt_plain", X)
    p_mon, s_mon = predict_gbdt_arm(out_dir, "gbdt_mono", X)

    # Fit calibrators on Dev only
    cals = {
        "incumbent": _fit_calibrators(y, p_inc, s_inc, st),
        "gbdt_mono": _fit_calibrators(y, p_mon, s_mon, st),
        "gbdt_plain": _fit_calibrators(y, p_pln, s_pln, st),
    }

    # Exact TreeSHAP real-vs-twin diagnostic on Dev only (using seed-0 pure-numpy GBDT-Mono)
    ens0 = TreeEnsemble.load_npz(str(out_dir / "gbdt_mono_mean_s0.npz"))
    shap_summary: dict = {}

    if "twin_X_real" in dev and "twin_X_sim" in dev:
        # Inverse model: true paired (X_real, X_twin) input-shift TreeSHAP (ZERO dev labels used!)
        sdiff = shap_twin_diff(
            ens0, dev["twin_X_real"], dev["twin_X_sim"],
            groups=ad.FEATURE_GROUPS, strata=dev["twin_strata"], scale=ad.SHAP_SCALE,
        )
        trep = twin_report(
            dev["twin_y_real"], dev["twin_y_sim"],
            ens0.predict(dev["twin_X_real"]), ens0.predict(dev["twin_X_sim"]),
            strata=dev["twin_strata"], unit_ids=dev["twin_unit_ids"], n_boot=500, seed=20261124,
        )
        shap_summary["mode"] = "input_shift_paired_twins"
        shap_summary["unit"] = ad.SHAP_UNIT
        shap_summary["n_pairs"] = sdiff["n"]
        shap_summary["ranked"] = sdiff["ranked"]
        shap_summary["by_stratum"] = sdiff["by_stratum"]
        shap_summary["twin_report"] = trep
    else:
        # Forward model: exact residual-gap TreeSHAP on Dev (diagnostic only!)
        sub_n = min(len(X), 1500)
        rng = np.random.default_rng(20261124)
        idx = np.sort(rng.choice(len(X), sub_n, replace=False)) if len(X) > sub_n else np.arange(len(X))
        resid_dev = y[idx] - ens0.predict(X[idx])
        gap_ens = _fit_dev_gap_tree(X[idx], resid_dev, ad.FEATURE_NAMES)
        sgap = shap_residual_gap(
            ens0, gap_ens, X[idx],
            groups=ad.FEATURE_GROUPS, strata=st[idx], scale=ad.SHAP_SCALE,
        )
        shap_summary["mode"] = "forward_residual_gap"
        shap_summary["unit"] = ad.SHAP_UNIT
        shap_summary["n_dev_diag"] = sgap["n"]
        shap_summary["ranked"] = sgap["ranked"]
        shap_summary["by_stratum"] = sgap["by_stratum"]

        # For EnergyNet: extract the 3 real measured spotlight regimes for model-families.astro Section 07!
        if platform == "energynet":
            T_c = dev["T_mean_C"][idx]
            sd_v = dev["stop_density"][idx]
            hwy = dev["hwy_frac"][idx]
            st_sub = st[idx]
            pr_g = sgap["phi_real"]
            pt_g = sgap["phi_twin"]
            cases = []
            regimes = [
                ("Cold-weather commute (VED dev, T < 0 °C)", T_c < 0.0),
                ("City hybrid (VED dev, HEV stop-and-go)", (st_sub == "HEV") & (sd_v >= np.median(sd_v[st_sub == "HEV"]))),
                ("Mild highway ICE (VED dev, twin agrees)", (st_sub == "ICE") & (T_c >= 10.0) & (hwy >= 0.20)),
            ]
            for label, mask in regimes:
                if not np.any(mask):
                    continue
                cases.append({
                    "name": label,
                    "n_trips": int(np.sum(mask)),
                    "n_cars": int(len(set(uid[idx][mask].tolist()))),
                    "features": sgap["names"],
                    "twin_mean": [round(float(v), 2) for v in np.mean(pt_g[mask], axis=0)],
                    "real_mean": [round(float(v), 2) for v in np.mean(pr_g[mask], axis=0)],
                    "twin_abs": [round(float(v), 2) for v in np.mean(np.abs(pt_g[mask]), axis=0)],
                    "real_abs": [round(float(v), 2) for v in np.mean(np.abs(pr_g[mask]), axis=0)],
                    "diff_mean": [round(float(v), 2) for v in np.mean(pr_g[mask] - pt_g[mask], axis=0)],
                    "mean_real_minus_sim_pct": round(float(np.mean(resid_dev[mask]) * 100.0), 2),
                })
            shap_summary["spotlight_cases"] = cases

    dev_out = {
        "platform": platform,
        "n_dev": int(len(X)),
        "n_dev_units": int(len(set(uid.tolist()))),
        "dev_metrics": {
            "incumbent": ad.metric_details(y, p_inc, st),
            "gbdt_plain": ad.metric_details(y, p_pln, st),
            "gbdt_mono": ad.metric_details(y, p_mon, st),
        },
        "calibrators": cals,
        "treeshap_twin": shap_summary,
    }
    dp = P.work("dev", f"{platform}.json", mkdir=True)
    dp.write_text(json.dumps(dev_out, indent=2) + "\n")
    top3 = [(r["feature"], r["mean_abs_diff"], r["mean_diff"]) for r in shap_summary["ranked"][:3]]
    print(f"[{platform}] dev_twin done. Top-3 TreeSHAP twin gaps ({ad.SHAP_UNIT}): {top3}", flush=True)
    return dev_out


def _paired_unit_boot(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    strata: np.ndarray,
    unit_ids: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 20261124,
) -> dict:
    """Paired unit-grouped bootstrap of metric(A) - metric(B) (negative = A beats B when lower is better)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(unit_ids, return_inverse=True)
    idx_by_u = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    n_u = len(uniq)
    point_a = metric_fn(y, pred_a, strata)
    point_b = metric_fn(y, pred_b, strata)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_u, size=n_u)
        rows = np.concatenate([idx_by_u[k] for k in pick])
        diffs[b] = metric_fn(y[rows], pred_a[rows], strata[rows]) - metric_fn(y[rows], pred_b[rows], strata[rows])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "a": round(float(point_a), 4),
        "b": round(float(point_b), 4),
        "diff_a_minus_b": round(float(point_a - point_b), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "significant_a_better": bool(hi < 0.0),
        "significant_b_better": bool(lo > 0.0),
    }


def stage_exam(platform: str) -> dict:
    prereg_sha = FR.verify_frozen()
    score_path = P.HERE / "scores" / f"{platform}.json"
    if score_path.exists():
        raise RuntimeError(f"Exam already scored once at {score_path}; refusing second run.")

    ad = load_adapter(platform)
    out_dir = P.work("models", platform)
    train_sum = json.loads((out_dir / "train_summary.json").read_text())
    dev_sum = json.loads(P.work("dev", f"{platform}.json").read_text())

    ex = ad.real_exam()
    X, y, st, uid = ex["X"], ex["y"], ex["strata"], ex["unit_ids"]
    print(f"[{platform}] ONE-SHOT EXAM: n_exam={len(X)}, units={len(set(uid.tolist()))} (PREREG sha={prereg_sha[:10]}...)", flush=True)

    p_inc, s_inc = ad.incumbent_predict(X)
    p_pln, s_pln = predict_gbdt_arm(out_dir, "gbdt_plain", X)
    p_mon, s_mon = predict_gbdt_arm(out_dir, "gbdt_mono", X)

    def _mfn(yt, yp, s_arr):
        if platform == "energynet":
            return ad.headline_metric(yt, yp, s_arr)
        return ad.headline_metric(yt, yp)

    boot_mono_vs_inc = _paired_unit_boot(y, p_mon, p_inc, st, uid, _mfn)
    boot_mono_vs_pln = _paired_unit_boot(y, p_mon, p_pln, st, uid, _mfn)
    boot_pln_vs_inc = _paired_unit_boot(y, p_pln, p_inc, st, uid, _mfn)

    # Coverage comparison across all 4 calibration methods on BOTH Incumbent P and GBDT-Mono
    cov_inc = _eval_calibrators(y, p_inc, s_inc, st, dev_sum["calibrators"]["incumbent"])
    cov_mon = _eval_calibrators(y, p_mon, s_mon, st, dev_sum["calibrators"]["gbdt_mono"])

    # Cross-family disagreement detector: |GBDT-Mono - Incumbent P| vs real exam error (zero labels at inference)
    dis_inc = disagreement_detector(p_mon, p_inc, abs_error=np.abs(p_inc - y), strata=st)
    dis_mon = disagreement_detector(p_mon, p_inc, abs_error=np.abs(p_mon - y), strata=st)

    result = {
        "platform": platform,
        "title": ad.TITLE,
        "domain": ad.DOMAIN,
        "target": ad.TARGET_NAME,
        "unit": ad.UNIT,
        "prereg_sha256": prereg_sha,
        "n_exam": int(len(X)),
        "n_exam_units": int(len(set(uid.tolist()))),
        "sim_val": {
            "incumbent": train_sum["incumbent_sim_val_headline"],
            "gbdt_plain": train_sum["plain_sim_val_headline"],
            "gbdt_mono": train_sum["mono_sim_val_headline"],
            "mono_parity": train_sum["gbdt_mono"]["numpy_vs_lgb_max_abs_diff"],
            "mono_treeshap_err": train_sum["gbdt_mono"]["treeshap_efficiency_max_err"],
            "mono_params_all": train_sum["gbdt_mono"]["n_params_all"],
        },
        "dev_metrics": dev_sum["dev_metrics"],
        "exam_metrics": {
            "incumbent": ad.metric_details(y, p_inc, st),
            "gbdt_plain": ad.metric_details(y, p_pln, st),
            "gbdt_mono": ad.metric_details(y, p_mon, st),
        },
        "paired_bootstrap": {
            "gbdt_mono_vs_incumbent": boot_mono_vs_inc,
            "gbdt_mono_vs_gbdt_plain": boot_mono_vs_pln,
            "gbdt_plain_vs_incumbent": boot_pln_vs_inc,
        },
        "conformal_coverage": {
            "incumbent": cov_inc,
            "gbdt_mono": cov_mon,
        },
        "calibrators_dev": dev_sum["calibrators"],
        "treeshap_twin_dev": dev_sum["treeshap_twin"],
        "disagreement_exam": {
            "vs_incumbent_error": dis_inc,
            "vs_gbdt_mono_error": dis_mon,
        },
    }

    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[{platform}] EXAM SAVED -> {score_path}\n"
        f"  Headline: Incumbent={boot_mono_vs_inc['b']} | GBDT-Plain={boot_pln_vs_inc['a']} | "
        f"GBDT-Mono={boot_mono_vs_inc['a']} (Mono-Inc={boot_mono_vs_inc['diff_a_minus_b']} CI={boot_mono_vs_inc['ci95']})\n"
        f"  Incumbent worst-stratum under-cov gap: pooled={cov_inc['pooled_widen']['worst_under_gap']:.3f} -> "
        f"mondrian_shrunk={cov_inc['mondrian_shrunk']['worst_under_gap']:.3f}\n"
        f"  Disagreement Spearman r vs Incumbent |err| = {dis_inc['overall']['spearman_r']} "
        f"(top-10% enrichment = {dis_inc['overall']['top10pct_enrichment']}x)",
        flush=True,
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("platform", choices=list(PLATFORMS) + ["all"])
    ap.add_argument("stage", choices=["train_tune", "dev_twin", "exam"])
    args = ap.parse_args()
    targets = list(PLATFORMS) if args.platform == "all" else [args.platform]
    for p in targets:
        if args.stage == "train_tune":
            stage_train_tune(p)
        elif args.stage == "dev_twin":
            stage_dev_twin(p)
        elif args.stage == "exam":
            stage_exam(p)


if __name__ == "__main__":
    main()
