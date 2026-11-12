"""The shared building blocks: the ratchet rule, widen-only calibration, metrics, numpy layers, splits."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlratchet import Ratchet, gates  # noqa: E402
from mlratchet import calibrate, metrics, nn, splits  # noqa: E402


# ---- the ratchet ---------------------------------------------------------------------------
def quake_ladder():
    r = Ratchet("quakenet", margin=0.01, dev_gate=gates.no_worse_than(1.05), frozen_set="val8")
    r.judge("r1", "textbook", 0.556, {"dist": 10.0})
    r.judge("r2", "slow surface waves", 0.690, {"dist": 8.0})
    r.judge("r6", "tiny gain", 0.695, {"dist": 7.9})  # +0.005: below margin
    r.judge("r7", "big gain but dev worse", 0.80, {"dist": 8.5})  # dev +6%: gate fails
    r.judge("r8", "real gain", 0.80, {"dist": 8.2})  # dev +2.5%: passes
    return r


def test_first_rung_is_kept_and_margin_is_strict():
    r = quake_ladder()
    assert [x.decision for x in r.rungs] == ["kept", "kept", "rejected", "rejected", "kept"]
    assert r.incumbent.id == "r8"
    assert "needs more than +0.01" in r.rungs[2].why
    assert "dev gate" in r.rungs[3].why


def test_exactly_the_margin_is_not_enough():
    r = Ratchet("m", margin=0.01)
    r.judge("a", "", 0.50)
    assert r.judge("b", "", 0.51).decision == "rejected"
    assert r.judge("c", "", 0.5101).decision == "kept"


def test_lower_is_better_objective():
    r = Ratchet("m", margin=0.01, higher_is_better=False)
    r.judge("a", "", 1.0)
    assert r.judge("b", "", 0.995).decision == "rejected"
    assert r.judge("c", "", 0.9).decision == "kept"


def test_rescore_replays_decisions_on_a_new_frozen_set():
    # QuakeNet rung 5: rejected on val5 (0.002 short), kept when every decision was re-taken on val7.
    r = Ratchet("quakenet", margin=0.01)
    r.judge("r4", "", 0.861)
    r.judge("r5", "", 0.869)  # +0.008 -> rejected
    assert r.rungs[1].decision == "rejected"
    changed = r.rescore({"r4": 0.827, "r5": 0.838}, frozen_set="val7")
    assert r.rungs[1].decision == "kept" and changed == [("r5", "rejected", "kept")]
    assert r.frozen_set == "val7"


def test_release_and_rescore_keep_the_released_mark():
    r = Ratchet("m", margin=0.01)
    r.judge("a", "", 0.5)
    r.judge("b", "", 0.6)
    r.release()
    assert r.rungs[1].decision == "released"
    r.rescore({"a": 0.4, "b": 0.7}, "val2")
    assert r.rungs[1].decision == "released"


def test_starquake_and_beatnet_gates():
    g = gates.mean_ratio_at_most(1.10)
    assert g({"a": 1.1, "b": 1.0}, {"a": 1.0, "b": 1.0})[0]          # mean ratio 1.05
    assert not g({"a": 1.4, "b": 1.0}, {"a": 1.0, "b": 1.0})[0]      # 1.20
    g = gates.max_drop("sv_f1", 0.02)
    assert g({"sv_f1": 0.476}, {"sv_f1": 0.467})[0]
    assert not g({"sv_f1": 0.441}, {"sv_f1": 0.476})[0]             # BeatNet r4: -0.035


def test_save_matches_the_ladder_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    import json

    import yaml

    r = quake_ladder()
    r.release()
    p = tmp_path / "ratchet.yaml"
    r.save(p)
    schema = json.loads((Path(__file__).resolve().parents[1] / "mlratchet/schema/ratchet.schema.json").read_text())
    jsonschema.validate(yaml.safe_load(p.read_text(encoding="utf-8")), schema)


# ---- calibration ---------------------------------------------------------------------------
def test_widen_factor_is_widen_only_and_hits_target():
    rng = np.random.default_rng(0)
    res = rng.normal(0, 2.0, 20000)
    f = calibrate.widen_factor(res, np.ones_like(res))
    assert f == pytest.approx(2.0, rel=0.03)
    assert calibrate.coverage(res, np.ones_like(res), f) == pytest.approx(calibrate.ONE_SIGMA, abs=0.005)
    # over-wide bars are never narrowed
    assert calibrate.widen_factor(res, np.full_like(res, 10.0)) == 1.0


def test_wrap_degrees():
    assert calibrate.wrap_degrees(350, 10) == pytest.approx(-20)
    assert calibrate.wrap_degrees(10, 350) == pytest.approx(20)


# ---- metrics -------------------------------------------------------------------------------
def test_classification_metrics():
    t = ["N", "N", "V", "V", "S"]
    p = ["N", "V", "V", "V", "N"]
    assert metrics.balanced_accuracy(p, t, ["N", "S", "V"]) == pytest.approx((0.5 + 0 + 1) / 3)
    f1n, f1v = 2 * 1 / (2 + 1 + 1), 2 * 2 / (4 + 1 + 0)
    assert metrics.macro_f1(p, t, ["N", "S", "V"]) == pytest.approx((f1n + 0 + f1v) / 3)


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    s = np.round(rng.normal(size=300), 1)  # with ties
    y = rng.random(300) < 0.3
    pos, neg = s[y], s[~y]
    pair = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert metrics.roc_auc(s, y) == pytest.approx(pair)


def test_dumb_baselines_and_skill():
    assert metrics.median_guess([1, 2, 10])(3).tolist() == [2, 2, 2]
    assert metrics.majority_guess(["N", "N", "V"])(2).tolist() == ["N", "N"]
    assert metrics.skill(5.1, 13.0) == pytest.approx(1 - 5.1 / 13)
    assert metrics.within_factor([20, 100, 5], [10, 10, 10]) == pytest.approx(2 / 3)


# ---- numpy layers --------------------------------------------------------------------------
def _naive_conv1d(x, w, b, stride, pad, dil):
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    n, c, L = x.shape
    o, _, k = w.shape
    out = (L - dil * (k - 1) - 1) // stride + 1
    y = np.zeros((n, o, out))
    for i in range(out):
        seg = x[:, :, i * stride: i * stride + dil * (k - 1) + 1: dil]
        y[:, :, i] = np.einsum("nck,ock->no", seg, w)
    return y + b[None, :, None]


@pytest.mark.parametrize("stride,pad,dil", [(1, 0, 1), (2, 3, 1), (1, 2, 2)])
def test_conv1d_matches_naive(stride, pad, dil):
    rng = np.random.default_rng(2)
    x, w, b = rng.normal(size=(2, 3, 40)), rng.normal(size=(5, 3, 7)), rng.normal(size=5)
    assert nn.max_abs_diff(nn.conv1d(x, w, b, stride, pad, dil), _naive_conv1d(x, w, b, stride, pad, dil)) < 1e-10


def test_conv2d_matches_conv1d_on_a_row():
    rng = np.random.default_rng(3)
    x, w, b = rng.normal(size=(1, 2, 1, 30)), rng.normal(size=(4, 2, 1, 5)), rng.normal(size=4)
    y2 = nn.conv2d(x, w, b, stride=(1, 2), pad=(0, 1))[:, :, 0, :]
    y1 = nn.conv1d(x[:, :, 0, :], w[:, :, 0, :], b, stride=2, pad=1)
    assert nn.max_abs_diff(y1, y2) < 1e-12


def test_fold_batchnorm_is_exact():
    rng = np.random.default_rng(4)
    x, w, b = rng.normal(size=(2, 3, 20)), rng.normal(size=(4, 3, 3)), rng.normal(size=4)
    g, be, mu, var = rng.uniform(0.5, 2, 4), rng.normal(size=4), rng.normal(size=4), rng.uniform(0.5, 2, 4)
    y = nn.conv1d(x, w, b)
    bn = g[None, :, None] * (y - mu[None, :, None]) / np.sqrt(var[None, :, None] + 1e-5) + be[None, :, None]
    w2, b2 = nn.fold_batchnorm(w, b, g, be, mu, var)
    assert nn.max_abs_diff(nn.conv1d(x, w2, b2), bn) < 1e-12


def test_gelu_erf_is_accurate():
    from math import erf

    xs = np.linspace(-5, 5, 101)
    assert nn.max_abs_diff(nn.erf(xs), [erf(v) for v in xs]) < 2e-7


# ---- splits --------------------------------------------------------------------------------
def test_split_by_unit_is_reproducible_and_disjoint():
    units = [f"u{i}" for i in range(50) for _ in range(3)]  # 3 windows per unit
    dev, exam = splits.split_units(units, n_dev=5, seed=2026)
    assert len(dev) == 5 and len(exam) == 45 and not set(dev) & set(exam)
    assert (dev, exam) == splits.split_units(units, n_dev=5, seed=2026)
    strata = [int(u[1:]) % 3 for u in units]
    dev, exam = splits.split_units(units, n_dev=2, seed=1, strata=strata)
    assert len(dev) == 6
