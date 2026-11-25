"""Reusable twin-test and TreeSHAP real-vs-twin diagnostic harness.

The twin test asks the cleanest diagnostic question in physics-first ML: take a real dev unit,
simulate its closest twin under the forward physics model, and compare what happens on the real
unit against what happens on its twin.

- :func:`twin_report` quantifies the residual and prediction gap (with unit-grouped bootstrap CIs),
  overall and per stratum.
- :func:`shap_twin_diff` runs exact TreeSHAP (:mod:`mlratchet.gbm`) on matched ``(real, twin)``
  input pairs and ranks which physical features (or feature groups) account for the largest
  disagreement between reality and the simulator.
- :func:`disagreement_detector` tests whether cross-family disagreement ``|pred_a - pred_b|``
  (e.g. GBDT vs MLP trained on the same simulation bank) flags out-of-distribution real inputs
  without needing labels at inference.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .gbm import TreeEnsemble, group_shap


def _grouped_boot_ci(
    values: np.ndarray,
    unit_ids: np.ndarray | None,
    n_boot: int = 1000,
    seed: int = 0,
    stat: str = "mean",
) -> tuple[float, float]:
    """Paired unit-grouped bootstrap 95% CI."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    v = v[ok]
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    fn = np.median if stat == "median" else np.mean
    if unit_ids is None:
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        samples = fn(v[idx], axis=1)
    else:
        u = np.asarray(unit_ids)[ok]
        uniq, inv = np.unique(u, return_inverse=True)
        groups = [v[inv == k] for k in range(len(uniq))]
        samples = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            pick = rng.integers(0, len(groups), size=len(groups))
            cat = np.concatenate([groups[k] for k in pick])
            samples[b] = float(fn(cat))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return (float(lo), float(hi))


def twin_report(
    y_real,
    y_twin,
    pred_real=None,
    pred_twin=None,
    strata=None,
    unit_ids=None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Compare real dev units against their simulated twins.

    Returns a dictionary containing the simulator physics gap (``y_real - y_twin``), the learner
    twin gap (``|pred_real - y_real| - |pred_twin - y_twin|``) when predictions are supplied,
    unit-grouped 95% bootstrap CIs, and per-stratum summaries.
    """
    yr = np.asarray(y_real, dtype=np.float64)
    yt = np.asarray(y_twin, dtype=np.float64)
    pr = None if pred_real is None else np.asarray(pred_real, dtype=np.float64)
    pt = None if pred_twin is None else np.asarray(pred_twin, dtype=np.float64)
    uid = None if unit_ids is None else np.asarray(unit_ids)

    def _summarize(mask: np.ndarray) -> dict:
        m = mask & np.isfinite(yr) & np.isfinite(yt)
        if pr is not None and pt is not None:
            m = m & np.isfinite(pr) & np.isfinite(pt)
        n = int(np.sum(m))
        if n == 0:
            return {"n": 0}
        phys_diff = yr[m] - yt[m]
        u_sub = None if uid is None else uid[m]
        out: dict = {
            "n": n,
            "phys_bias_mean": round(float(np.mean(phys_diff)), 6),
            "phys_bias_median": round(float(np.median(phys_diff)), 6),
            "phys_mae": round(float(np.mean(np.abs(phys_diff))), 6),
            "phys_bias_ci95": [round(x, 6) for x in _grouped_boot_ci(phys_diff, u_sub, n_boot, seed)],
        }
        if pr is not None and pt is not None:
            ae_real = np.abs(pr[m] - yr[m])
            ae_twin = np.abs(pt[m] - yt[m])
            ae_gap = ae_real - ae_twin
            pred_shift = pr[m] - pt[m]
            out.update({
                "real_mae": round(float(np.mean(ae_real)), 6),
                "twin_mae": round(float(np.mean(ae_twin)), 6),
                "mae_gap_real_minus_twin": round(float(np.mean(ae_gap)), 6),
                "mae_gap_ci95": [round(x, 6) for x in _grouped_boot_ci(ae_gap, u_sub, n_boot, seed + 1)],
                "pred_shift_mean": round(float(np.mean(pred_shift)), 6),
                "pred_shift_mae": round(float(np.mean(np.abs(pred_shift))), 6),
            })
        return out

    all_mask = np.ones(yr.shape[0], dtype=bool)
    res: dict = {"overall": _summarize(all_mask)}
    if strata is not None:
        st = np.asarray(strata, dtype=str)
        res["strata"] = {g: _summarize(st == g) for g in sorted(set(st.tolist()))}
    return res


def shap_twin_diff(
    ensemble: TreeEnsemble,
    X_real: np.ndarray,
    X_twin: np.ndarray,
    *,
    groups: Mapping[str, Sequence[int]] | None = None,
    strata=None,
    output_idx: int = 0,
    scale: float = 1.0,
) -> dict:
    """Rank simulator-vs-reality feature gaps via exact TreeSHAP on matched ``(X_real, X_twin)``.

    Parameters
    ----------
    ensemble : TreeEnsemble
        Pure-numpy gradient-boosted tree ensemble trained on simulations.
    X_real, X_twin : np.ndarray of shape (n, n_features)
        Matched real dev inputs and their simulated twin inputs.
    groups : optional mapping ``{group_name: [feature_indices...]}``
        Rolls fine-grained features up into physical groups while preserving exact Shapley
        additivity.
    strata : optional array of shape (n,)
        If provided, computes per-stratum ranked gap tables in addition to the overall ranking.
    output_idx : int
        Output/class index for multi-output ensembles.
    scale : float
        Unit multiplier applied to SHAP values (for example ``100.0`` to read log-ratio
        corrections as approximate percent).

    Returns
    -------
    dict with keys: ``names``, ``expected_value``, ``ranked``, ``by_stratum``,
    ``phi_real``, ``phi_twin``, ``delta_phi``.
    """
    Xr = np.asarray(X_real, dtype=np.float64)
    Xt = np.asarray(X_twin, dtype=np.float64)
    if Xr.shape != Xt.shape:
        raise ValueError(f"X_real {Xr.shape} and X_twin {Xt.shape} must have the same shape")

    phi_r, ev = ensemble.treeshap(Xr, output_idx=output_idx)
    phi_t, _ = ensemble.treeshap(Xt, output_idx=output_idx)

    if groups is not None:
        names, gr = group_shap(phi_r, groups)
        _, gt = group_shap(phi_t, groups)
    else:
        names = (
            list(ensemble.feature_names)
            if ensemble.feature_names is not None
            else [f"f{j}" for j in range(ensemble.n_features)]
        )
        gr, gt = phi_r, phi_t

    gr = gr * scale
    gt = gt * scale
    diff = gr - gt

    def _rank(mask: np.ndarray) -> list[dict]:
        dr = gr[mask]
        dt = gt[mask]
        dd = diff[mask]
        mean_abs = np.mean(np.abs(dd), axis=0)
        tot_abs = float(np.sum(mean_abs))
        rows = []
        for j, name in enumerate(names):
            rows.append({
                "feature": name,
                "mean_real": round(float(np.mean(dr[:, j])), 6),
                "mean_twin": round(float(np.mean(dt[:, j])), 6),
                "mean_diff": round(float(np.mean(dd[:, j])), 6),
                "mean_abs_real": round(float(np.mean(np.abs(dr[:, j]))), 6),
                "mean_abs_twin": round(float(np.mean(np.abs(dt[:, j]))), 6),
                "mean_abs_diff": round(float(mean_abs[j]), 6),
                "share_of_gap": round(float(mean_abs[j] / tot_abs), 4) if tot_abs > 0 else 0.0,
            })
        rows.sort(key=lambda r: r["mean_abs_diff"], reverse=True)
        return rows

    all_mask = np.ones(Xr.shape[0], dtype=bool)
    by_stratum: dict[str, list[dict]] = {}
    if strata is not None:
        st = np.asarray(strata, dtype=str)
        for g in sorted(set(st.tolist())):
            m = st == g
            if np.any(m):
                by_stratum[g] = _rank(m)

    return {
        "names": names,
        "expected_value": float(ev * scale),
        "n": int(Xr.shape[0]),
        "scale": float(scale),
        "ranked": _rank(all_mask),
        "by_stratum": by_stratum,
        "phi_real": gr,
        "phi_twin": gt,
        "delta_phi": diff,
    }


def disagreement_detector(
    pred_a,
    pred_b,
    abs_error=None,
    strata=None,
) -> dict:
    """Evaluate cross-family disagreement ``|pred_a - pred_b|`` as a label-free gap detector.

    When two different learners (for example a GBDT and an MLP) trained on the same simulation
    bank disagree strongly on a real input, that input is likely in a region where the simulator
    was thin or mismatched.
    """
    pa = np.asarray(pred_a, dtype=np.float64)
    pb = np.asarray(pred_b, dtype=np.float64)
    d = np.abs(pa - pb)
    ae = None if abs_error is None else np.abs(np.asarray(abs_error, dtype=np.float64))

    def _stats(mask: np.ndarray) -> dict:
        m = mask & np.isfinite(d)
        if ae is not None:
            m = m & np.isfinite(ae)
        if not np.any(m):
            return {"n": 0}
        dm = d[m]
        out: dict = {
            "n": int(np.sum(m)),
            "mean_disagreement": round(float(np.mean(dm)), 6),
            "median_disagreement": round(float(np.median(dm)), 6),
            "p90_disagreement": round(float(np.quantile(dm, 0.90)), 6),
        }
        if ae is not None and dm.size >= 5:
            am = ae[m]
            rho = spearmanr(dm, am)
            r_val = float(rho.statistic) if np.isfinite(rho.statistic) else float("nan")
            p_val = float(rho.pvalue) if np.isfinite(rho.pvalue) else float("nan")
            q90 = np.quantile(dm, 0.90)
            q50 = np.quantile(dm, 0.50)
            top = am[dm >= q90]
            bot = am[dm <= q50]
            mean_all = float(np.mean(am))
            out.update({
                "spearman_r": round(r_val, 4),
                "spearman_p": round(p_val, 6),
                "top10pct_mae": round(float(np.mean(top)), 6) if top.size else float("nan"),
                "bottom50pct_mae": round(float(np.mean(bot)), 6) if bot.size else float("nan"),
                "top10pct_enrichment": round(float(np.mean(top) / mean_all), 3) if (top.size and mean_all > 0) else float("nan"),
            })
        return out

    all_mask = np.ones(d.shape[0], dtype=bool)
    res: dict = {"overall": _stats(all_mask)}
    if strata is not None:
        st = np.asarray(strata, dtype=str)
        res["strata"] = {g: _stats(st == g) for g in sorted(set(st.tolist()))}
    return res


def shap_residual_gap(
    sim_ensemble: TreeEnsemble,
    gap_ensemble: TreeEnsemble,
    X_dev: np.ndarray,
    *,
    groups: Mapping[str, Sequence[int]] | None = None,
    strata=None,
    output_idx: int = 0,
    scale: float = 1.0,
) -> dict:
    """Exact TreeSHAP decomposition of a forward model's real-vs-simulator gap on dev inputs.

    For forward models where real units and their simulated twins share the same driving inputs
    ``X_dev`` (e.g. EnergyNet routes or AirfoilNet shapes), ``sim_ensemble`` is trained strictly
    on physics simulations (predicting ``y_sim``) and ``gap_ensemble`` is a shallow dev-only
    diagnostic tree fitted on ``y_real - sim_ensemble.predict(X_dev)`` (never used for exam
    predictions). By Shapley linearity, ``phi_real = phi_sim + phi_gap`` and the exact per-feature
    simulator-vs-reality gap is ``delta_phi = phi_gap``.
    """
    Xd = np.asarray(X_dev, dtype=np.float64)
    phi_sim, ev_sim = sim_ensemble.treeshap(Xd, output_idx=output_idx)
    phi_gap, ev_gap = gap_ensemble.treeshap(Xd, output_idx=output_idx)

    if groups is not None:
        names, gt = group_shap(phi_sim, groups)
        _, gd = group_shap(phi_gap, groups)
    else:
        names = (
            list(sim_ensemble.feature_names)
            if sim_ensemble.feature_names is not None
            else [f"f{j}" for j in range(sim_ensemble.n_features)]
        )
        gt, gd = phi_sim, phi_gap

    gt = gt * scale
    gd = gd * scale
    gr = gt + gd

    def _rank(mask: np.ndarray) -> list[dict]:
        dr = gr[mask]
        dt = gt[mask]
        dd = gd[mask]
        mean_abs = np.mean(np.abs(dd), axis=0)
        tot_abs = float(np.sum(mean_abs))
        rows = []
        for j, name in enumerate(names):
            rows.append({
                "feature": name,
                "mean_real": round(float(np.mean(dr[:, j])), 6),
                "mean_twin": round(float(np.mean(dt[:, j])), 6),
                "mean_diff": round(float(np.mean(dd[:, j])), 6),
                "mean_abs_real": round(float(np.mean(np.abs(dr[:, j]))), 6),
                "mean_abs_twin": round(float(np.mean(np.abs(dt[:, j]))), 6),
                "mean_abs_diff": round(float(mean_abs[j]), 6),
                "share_of_gap": round(float(mean_abs[j] / tot_abs), 4) if tot_abs > 0 else 0.0,
            })
        rows.sort(key=lambda r: r["mean_abs_diff"], reverse=True)
        return rows

    all_mask = np.ones(Xd.shape[0], dtype=bool)
    by_stratum: dict[str, list[dict]] = {}
    if strata is not None:
        st = np.asarray(strata, dtype=str)
        for g in sorted(set(st.tolist())):
            m = st == g
            if np.any(m):
                by_stratum[g] = _rank(m)

    return {
        "names": names,
        "expected_value_sim": float(ev_sim * scale),
        "expected_value_gap": float(ev_gap * scale),
        "n": int(Xd.shape[0]),
        "scale": float(scale),
        "ranked": _rank(all_mask),
        "by_stratum": by_stratum,
        "phi_real": gr,
        "phi_twin": gt,
        "delta_phi": gd,
    }

