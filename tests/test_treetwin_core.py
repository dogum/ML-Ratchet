"""Unit tests for mlratchet.gbm, Mondrian conformal calibration, and mlratchet.twintest."""
from __future__ import annotations

import itertools
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlratchet.calibrate import (
    ONE_SIGMA,
    MondrianConformal,
    conformal_quantile,
    coverage,
    stratum_coverage,
    widen_factor,
)
from mlratchet.gbm import TreeEnsemble, from_lightgbm, group_shap
from mlratchet.twintest import disagreement_detector, shap_twin_diff, twin_report


def _brute_force_tree_shap(ens: TreeEnsemble, x: np.ndarray) -> np.ndarray:
    """Exact O(2^p) brute-force path-dependent Shapley values for a single tree ensemble."""
    p = ens.n_features
    phi = np.zeros(p, dtype=np.float64)

    def _eval_subset(tree_idx: int, S: set[int]) -> float:
        s = int(ens.tree_offsets[tree_idx])
        e = int(ens.tree_offsets[tree_idx + 1])
        f_t, th_t, dl_t = ens.feat[s:e], ens.thresh[s:e], ens.default_left[s:e]
        l_t, r_t, v_t, c_t = ens.left[s:e], ens.right[s:e], ens.value[s:e], ens.cover[s:e]

        def _rec(u: int) -> float:
            f = int(f_t[u])
            if f < 0:
                return float(v_t[u])
            lc, rc = int(l_t[u]), int(r_t[u])
            if f in S:
                xv = x[f]
                go_l = (dl_t[u] != 0) if np.isnan(xv) else (xv <= th_t[u])
                return _rec(lc if go_l else rc)
            ct = float(c_t[lc] + c_t[rc])
            return (float(c_t[lc]) * _rec(lc) + float(c_t[rc]) * _rec(rc)) / ct

        return _rec(0)

    def _v(S: set[int]) -> float:
        return float(ens.base_score[0]) + sum(_eval_subset(t, S) for t in range(ens.n_trees))

    fact = [math.factorial(k) for k in range(p + 1)]
    for j in range(p):
        others = [k for k in range(p) if k != j]
        for r in range(len(others) + 1):
            w = fact[r] * fact[p - r - 1] / fact[p]
            for comb in itertools.combinations(others, r):
                S = set(comb)
                phi[j] += w * (_v(S | {j}) - _v(S))
    return phi


def _hand_built_ensemble() -> TreeEnsemble:
    """Two hand-built depth-3 trees with repeated feature splits on paths."""
    # Tree 0: root splits f0 <= 0.5 -> left splits f1 <= 1.0 -> left splits f0 <= -0.5 (repeated f0!)
    # Nodes: 0(f0<=0.5), 1(f1<=1.0), 2(f0<=-0.5), 3(leaf 1.0), 4(leaf 2.5), 5(leaf -0.8), 6(leaf 0.4)
    t0_feat = [0, 1, 0, -1, -1, -1, -1]
    t0_thr = [0.5, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0]
    t0_dl = [1, 0, 1, 1, 1, 1, 1]
    t0_l = [1, 2, 3, -1, -1, -1, -1]
    t0_r = [6, 5, 4, -1, -1, -1, -1]
    t0_v = [0.0, 0.0, 0.0, 1.0, 2.5, -0.8, 0.4]
    t0_c = [100.0, 60.0, 40.0, 15.0, 25.0, 20.0, 40.0]

    # Tree 1: root splits f2 <= 0.0 -> left leaf -1.2, right splits f1 <= -0.2 -> leaves (0.7, 1.9)
    t1_feat = [2, -1, 1, -1, -1]
    t1_thr = [0.0, 0.0, -0.2, 0.0, 0.0]
    t1_dl = [1, 1, 1, 1, 1]
    t1_l = [1, -1, 3, -1, -1]
    t1_r = [2, -1, 4, -1, -1]
    t1_v = [0.0, -1.2, 0.0, 0.7, 1.9]
    t1_c = [100.0, 35.0, 65.0, 30.0, 35.0]

    return TreeEnsemble(
        tree_offsets=np.array([0, len(t0_feat), len(t0_feat) + len(t1_feat)], dtype=np.int32),
        feat=np.array(t0_feat + t1_feat, dtype=np.int32),
        thresh=np.array(t0_thr + t1_thr, dtype=np.float64),
        default_left=np.array(t0_dl + t1_dl, dtype=np.uint8),
        left=np.array(t0_l + t1_l, dtype=np.int32),
        right=np.array(t0_r + t1_r, dtype=np.int32),
        value=np.array(t0_v + t1_v, dtype=np.float64),
        cover=np.array(t0_c + t1_c, dtype=np.float64),
        base_score=np.array([0.25], dtype=np.float64),
        tree_output=np.array([0, 0], dtype=np.int32),
        n_features=4,
        n_outputs=1,
        feature_names=("x0", "x1", "x2", "x3"),
    )


def test_treeshap_matches_brute_force_and_efficiency():
    ens = _hand_built_ensemble()
    X = np.array([
        [-1.0, 0.5, -0.5, 3.0],
        [0.0, 0.5, 0.5, -1.0],
        [0.2, 2.0, 0.1, 0.0],
        [1.5, -1.0, 2.0, 0.0],
        [np.nan, np.nan, 0.2, 1.0],
    ])
    preds = ens.predict(X)
    phi, ev = ens.treeshap(X)

    # Exact efficiency axiom
    assert np.allclose(ev + phi.sum(axis=1), preds, atol=1e-12)
    # Unused feature x3 has identically zero attribution
    assert np.allclose(phi[:, 3], 0.0, atol=1e-15)
    # Exact match against 2^p brute-force Shapley enumeration (including repeated f0 & NaNs)
    for i in range(len(X)):
        bf = _brute_force_tree_shap(ens, X[i])
        assert np.allclose(phi[i], bf, atol=1e-12)


def test_lightgbm_export_predict_monotone_and_treeshap_parity():
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(20261124)
    X = rng.normal(size=(800, 6))
    X[rng.random(X.shape) < 0.05] = np.nan
    y = 2.5 * np.nan_to_num(X[:, 0]) - 1.5 * np.nan_to_num(X[:, 1]) + 0.8 * np.nan_to_num(X[:, 2]) ** 2 + rng.normal(0, 0.2, 800)

    mc = [1, -1, 0, 0, 0, 0]
    model = lgb.LGBMRegressor(
        n_estimators=25,
        max_depth=4,
        num_leaves=12,
        learning_rate=0.1,
        monotone_constraints=mc,
        random_state=7,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(X, y)

    ens = from_lightgbm(model, feature_names=[f"f{i}" for i in range(6)], monotone_cst=mc)
    X_test = rng.normal(size=(40, 6))
    X_test[0, 1] = np.nan
    X_test[1, 0] = np.nan

    # Prediction parity vs LightGBM
    p_lgb = model.predict(X_test)
    p_np = ens.predict(X_test)
    assert np.allclose(p_np, p_lgb, atol=1e-6)

    # Exact TreeSHAP parity vs LightGBM built-in pred_contrib
    contrib_lgb = model.predict(X_test, pred_contrib=True)
    phi_np, ev_np = ens.treeshap(X_test)
    assert np.allclose(ev_np, contrib_lgb[0, -1], atol=1e-5)
    assert np.allclose(phi_np, contrib_lgb[:, :-1], atol=1e-5)
    assert np.allclose(ev_np + phi_np.sum(axis=1), p_np, atol=1e-10)

    # Verify monotonicity along f0 (+1) and f1 (-1)
    grid = np.zeros((21, 6))
    grid[:, 0] = np.linspace(-3, 3, 21)
    assert np.all(np.diff(ens.predict(grid)) >= -1e-12)
    grid2 = np.zeros((21, 6))
    grid2[:, 1] = np.linspace(-3, 3, 21)
    assert np.all(np.diff(ens.predict(grid2)) <= 1e-12)

    # NPZ round-trip
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "ens.npz")
        ens.save_npz(p)
        ens2 = TreeEnsemble.load_npz(p)
        assert np.allclose(ens2.predict(X_test), p_np, atol=1e-12)
        assert ens2.monotone_cst == tuple(mc)
        assert ens2.feature_names == ens.feature_names

    # Group SHAP exact additivity
    names, gphi = group_shap(phi_np, {"linear": [0, 1], "quad": [2], "rest": [3, 4, 5]})
    assert names == ["linear", "quad", "rest"]
    assert np.allclose(gphi.sum(axis=1), phi_np.sum(axis=1), atol=1e-12)


def test_mondrian_conformal_calibration():
    rng = np.random.default_rng(42)
    # Two strata with very different true residual scales: A has sigma_true=1.0, B has sigma_true=3.0,
    # while model predicts sigma_hat=1.0 everywhere. Plus a tiny stratum C (n=2) testing fallback/shrinkage.
    n_a, n_b, n_c = 120, 120, 2
    r_dev = np.concatenate([rng.normal(0, 1.0, n_a), rng.normal(0, 3.0, n_b), rng.normal(0, 5.0, n_c)])
    s_dev = np.ones_like(r_dev)
    st_dev = np.array(["A"] * n_a + ["B"] * n_b + ["C"] * n_c)

    # Pooled widen-only vs Mondrian raw vs Mondrian shrunk
    f_pool = widen_factor(r_dev, s_dev, target=ONE_SIGMA)
    mc_raw = MondrianConformal.fit(r_dev, s_dev, st_dev, target=ONE_SIGMA, widen_only=True, shrink_k=0.0, min_n=5)
    mc_shrunk = MondrianConformal.fit(r_dev, s_dev, st_dev, target=ONE_SIGMA, widen_only=True, shrink_k=10.0, min_n=5)

    # Tiny stratum C (< min_n) falls back to global factor
    assert mc_raw.stratum_factors["C"] == mc_raw.global_factor
    # Stratum B gets a substantially wider factor than A
    assert mc_raw.stratum_factors["B"] > 2.0 * mc_raw.stratum_factors["A"]
    assert mc_shrunk.stratum_factors["B"] < mc_raw.stratum_factors["B"]
    assert mc_shrunk.stratum_factors["B"] > mc_shrunk.global_factor

    # Evaluate on held-out exam draws from A and B: Mondrian should dramatically cut worst-stratum gap
    r_ex = np.concatenate([rng.normal(0, 1.0, 2000), rng.normal(0, 3.0, 2000)])
    s_ex = np.ones_like(r_ex)
    st_ex = np.array(["A"] * 2000 + ["B"] * 2000)

    cov_pool = stratum_coverage(r_ex, s_ex * f_pool, st_ex, target=ONE_SIGMA)
    cov_mond = stratum_coverage(r_ex, mc_shrunk.halfwidth(s_ex, st_ex), st_ex, target=ONE_SIGMA)
    assert cov_mond["max_abs_stratum_gap"] < cov_pool["max_abs_stratum_gap"]
    assert cov_mond["worst_under_gap"] < 0.08

    # Serialization round-trip
    mc2 = MondrianConformal.from_dict(mc_shrunk.to_dict())
    assert np.allclose(mc2.halfwidth(s_ex, st_ex), mc_shrunk.halfwidth(s_ex, st_ex))
    # Conformal quantile finite-sample property
    # 3 scores cannot carry 90 % coverage: k = ceil(4 * 0.9) = 4 > 3, so the quantile is unbounded
    assert conformal_quantile([1.0, 2.0, 3.0], alpha=0.1) == float("inf")
    assert conformal_quantile([], alpha=0.1) == float("inf")
    # 9 scores at 90 %: k = ceil(10 * 0.9) = 9, the largest score
    assert conformal_quantile(np.arange(1.0, 10.0), alpha=0.1) == 9.0
    # 68.27 %: k = ceil(4 * 0.6827) = 3 <= 3
    assert conformal_quantile([1.0, 2.0, 3.0], alpha=1.0 - ONE_SIGMA) == 3.0
    assert coverage(r_ex, s_ex, f_pool) > 0.5


def test_twintest_harness_and_disagreement():
    ens = _hand_built_ensemble()
    # Twin matches x0, x2, x3, but real shifts x1 upward (+1.5)
    X_twin = np.array([
        [-0.8, 0.2, 0.5, 0.0],
        [0.1, 0.4, -0.5, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [-0.2, 0.1, 0.3, 0.0],
    ])
    X_real = X_twin.copy()
    X_real[:, 1] += 1.5  # huge shift on x1 only!
    strata = np.array(["cold", "cold", "mild", "mild"])

    diff = shap_twin_diff(ens, X_real, X_twin, strata=strata)
    assert diff["ranked"][0]["feature"] == "x1"
    assert diff["ranked"][0]["share_of_gap"] > 0.5
    assert "cold" in diff["by_stratum"] and "mild" in diff["by_stratum"]

    rep = twin_report(
        y_real=[1.2, 1.5, 0.8, 0.9],
        y_twin=[1.0, 1.0, 0.8, 0.9],
        pred_real=ens.predict(X_real),
        pred_twin=ens.predict(X_twin),
        strata=strata,
        unit_ids=["u1", "u1", "u2", "u2"],
        n_boot=200,
    )
    assert rep["overall"]["n"] == 4
    assert "cold" in rep["strata"]

    rng = np.random.default_rng(0)
    err = np.linspace(0.1, 5.0, 100)
    pa = np.zeros(100)
    pb = err + rng.normal(0, 0.2, 100)
    dis = disagreement_detector(pa, pb, abs_error=err)
    assert dis["overall"]["spearman_r"] > 0.85
    assert dis["overall"]["top10pct_enrichment"] > 1.5


def test_site_trees_reference_is_current():
    """site/tests/trees.test.ts checks the TypeScript tree runtime against a committed reference;
    it must match what mlratchet.gbm computes now."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "export_trees_reference", root / "experiments" / "treetwin" / "export_trees_reference.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    want = mod.build()
    have = json.loads(mod.OUT.read_text(encoding="utf-8"))
    assert have.keys() == want.keys()
    for k in ("preds", "phi", "ev", "value", "thresh"):
        assert np.allclose(np.asarray(have[k], dtype=float), np.asarray(want[k], dtype=float), rtol=0, atol=1e-15), k
