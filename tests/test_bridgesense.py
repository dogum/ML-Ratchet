"""Protect BridgeSense's v1.1 fixes (Codex review of PR #14).

1. The per-mode thermal regression has an intercept fitted on that mode's valid readings.
2. HBTA modal bands come only from the UDS *_01 reference recordings.
3. Modes without valid reference readings stay out of the global anchor features.
4. Ensemble members keep their own configs.
Also checks that the copyable inference file and the training feature extractor agree.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / "models" / "bridgesense"
N = 32


def _load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mods(monkeypatch):
    # Training scripts import siblings by bare name; isolate them from other models' modules.
    out = {}
    for name in ("_paths", "simulator", "features", "extract_hbta_modes"):
        try:
            out[name] = _load(name, MODEL / "training" / f"{name}.py", monkeypatch)
        except ImportError:  # h5py is optional outside the training environment
            out[name] = None
    out["bridgesense"] = _load("bridgesense", MODEL / "bridgesense.py", monkeypatch)
    return out


def _T3(t):
    t = np.asarray(t, dtype=float)
    return np.stack([t, t, t], axis=1)


def _dropout_episode(M=2):
    """Codex's case: reference 0..31 C, only the last 16 readings valid, df = -0.001 (T - 23.5)."""
    T_ref = np.arange(N, dtype=float)
    df_ref = np.zeros((N, M))
    mask_ref = np.zeros((N, M))
    for m in range(M):
        df_ref[16:, m] = -0.001 * (T_ref[16:] - 23.5)
        mask_ref[16:, m] = 1.0
    T_q = np.full(N, 5.0)
    df_q = np.tile(-0.001 * (T_q - 23.5)[:, None], (1, M))
    return T_ref, df_ref, mask_ref, T_q, df_q, np.ones((N, M))


def test_thermal_fit_has_intercept_under_dropouts(mods):
    sim = mods["simulator"]
    T_ref, df_ref, mask_ref, T_q, df_q, mask_q = _dropout_episode()
    a = sim.compute_physics_anchor(T_ref, df_ref, mask_ref, T_q, df_q, mask_q)
    # v1.0 gave a slope near -0.000249 and a spurious +2.78 % anchor here
    np.testing.assert_allclose(a["slopes"], -0.001, rtol=1e-3)
    assert abs(a["anchor_dk_pct"]) < 0.01

    f = mods["bridgesense"].compute_anchor_and_features(_T3(T_ref), _T3(T_q), df_ref, df_q, mask_ref, mask_q)
    assert abs(float(f["anchor"][0])) < 0.01
    assert abs(float(f["g_feat"][0])) < 1e-3


def test_reference_less_mode_left_out_of_global_features(mods):
    sim = mods["simulator"]
    T_ref = np.linspace(2.0, 18.0, N)
    T_q = np.linspace(4.0, 12.0, N)
    df_ref = np.zeros((N, 2))
    mask_ref = np.zeros((N, 2))
    mask_ref[:, 0] = 1.0                 # mode 0 is fully supported, mode 1 never seen in the reference
    df_q = np.zeros((N, 2))
    df_q[:, 1] = 0.05                    # unsupported mode shows a large shift
    mask_q = np.ones((N, 2))
    a = sim.compute_physics_anchor(T_ref, df_ref, mask_ref, T_q, df_q, mask_q)
    assert list(a["eligible"]) == [True, False]
    assert a["anchor_dk_pct"] == pytest.approx(0.0, abs=1e-9)   # v1.0: +5 %
    assert a["max_abs_z"] == pytest.approx(0.0, abs=1e-9)

    f = mods["bridgesense"].compute_anchor_and_features(_T3(T_ref), _T3(T_q), df_ref, df_q, mask_ref, mask_q)
    assert f["m_mask"][:2].tolist() == [1.0, 0.0]
    assert float(f["anchor"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(f["g_feat"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(f["g_feat"][2]) == pytest.approx(0.0, abs=1e-6)


def test_inference_features_match_training_features(mods):
    """bridgesense.py duplicates training/features.py so the model folder copies on its own; keep them equal."""
    rng = np.random.default_rng(11)
    M = 14
    for _ in range(25):
        T_ref = _T3(rng.uniform(-5, 20, N))
        T_q = _T3(rng.uniform(-10, 15, N))
        df_ref = rng.normal(0, 0.004, (N, M))
        df_q = rng.normal(0.002, 0.004, (N, M))
        mask_ref = (rng.random((N, M)) < rng.uniform(0.05, 0.9)).astype(float)
        mask_q = (rng.random((N, M)) < rng.uniform(0.05, 0.9)).astype(float)
        df_ref *= mask_ref
        df_q *= mask_q
        a = mods["features"].extract_features(T_ref, T_q, df_ref, df_q, mask_ref, mask_q)
        b = mods["bridgesense"].compute_anchor_and_features(T_ref, T_q, df_ref, df_q, mask_ref, mask_q)
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=1e-5, atol=1e-6, err_msg=k)


def test_mixed_ensemble_keeps_each_members_config(mods, tmp_path):
    bs = mods["bridgesense"]
    raw = np.load(MODEL / "bridgesense_weights.npz", allow_pickle=False)
    meta = json.loads(str(raw["meta"].item()))
    w0 = {k[3:]: raw[k] for k in raw.files if k.startswith("m0.")}
    cfg = dict(meta.get("configs", [meta["config"]])[0])
    cfg_a, cfg_b = dict(cfg, use_anchor=True), dict(cfg, use_anchor=False)

    def write(path, cfgs):
        arrays = {f"m{k}.{n}": v for k in range(len(cfgs)) for n, v in w0.items()}
        m = dict(meta, n_members=len(cfgs), config=cfgs[-1], configs=cfgs, widen_dk=1.0, widen_env=1.0, prob_temp=1.0)
        arrays["meta"] = np.array(json.dumps(m))
        np.savez(path, **arrays)
        return bs.BridgeSense(path)

    ens = write(tmp_path / "ens.npz", [cfg_a, cfg_b])
    one_a = write(tmp_path / "a.npz", [cfg_a])
    one_b = write(tmp_path / "b.npz", [cfg_b])
    rng = np.random.default_rng(3)
    T_ref, T_q = _T3(rng.uniform(0, 15, N)), _T3(rng.uniform(-5, 10, N))
    df_ref, df_q = rng.normal(0, 0.003, (N, 6)), rng.normal(0.01, 0.003, (N, 6))
    args = (T_ref, T_q, df_ref, df_q, np.ones((N, 6)), np.ones((N, 6)))
    r, ra, rb = ens.predict(*args), one_a.predict(*args), one_b.predict(*args)
    assert ra["delta_k_pct"] != pytest.approx(rb["delta_k_pct"], abs=1e-3)  # the anchor matters here
    assert r["delta_k_pct"] == pytest.approx(0.5 * (ra["delta_k_pct"] + rb["delta_k_pct"]), abs=1e-5)


def test_hbta_bands_come_from_reference_recordings_only(mods):
    hb = mods["extract_hbta_modes"]
    if hb is not None:
        assert hb.is_reference_recording("UDS_01", "UDS") or hb.is_reference_recording("x_01", "UDS")
        assert not hb.is_reference_recording("x_02", "UDS")
        assert not hb.is_reference_recording("x_01", "DS5")
    bands = json.loads((MODEL / "training" / "hbta_bands.json").read_text(encoding="utf-8"))
    refs, held = bands["reference_recordings"], bands["held_out_uds"]
    assert refs and all(r.endswith("_01") for r in refs)
    assert held and not set(refs) & set(held) and all(h.endswith("_02") for h in held)
    assert bands["peak_freqs_hz"] == sorted(bands["peak_freqs_hz"])


def test_ratchet_cache_rescores_when_checkpoint_bytes_change(monkeypatch, tmp_path):
    """Codex PR #14 (27 Nov): a retrained checkpoint under the same rung id must not reuse old scores."""
    sc = _load("score_cache", MODEL / "training" / "score_cache.py", monkeypatch)
    ck, bank, code = tmp_path / "r6.pt", tmp_path / "val3.npz", tmp_path / "features.py"
    ck.write_bytes(b"weights v1")
    bank.write_bytes(b"bank")
    code.write_text("x = 1")
    entry = {"id": "r6", "ckpts": ["r6"]}
    calls = []

    def score():
        calls.append(1)
        return {"sim_score": 0.5 + len(calls)}

    def fp(e=entry):
        return sc.fingerprint(e, [str(ck)], [str(code)], [str(bank)])

    cache = {}
    s1, fresh1 = sc.cached(cache, "r6@val3", fp(), score)
    s2, fresh2 = sc.cached(cache, "r6@val3", fp(), score)
    assert fresh1 and not fresh2 and s2["sim_score"] == s1["sim_score"] and len(calls) == 1

    ck.write_bytes(b"weights v2 (retrained)")            # same rung id, new checkpoint contents
    _, fresh = sc.cached(cache, "r6@val3", fp(), score)
    assert fresh and len(calls) == 2
    code.write_text("x = 2")                              # feature extraction changed
    _, fresh = sc.cached(cache, "r6@val3", fp(), score)
    assert fresh and len(calls) == 3
    bank.write_bytes(b"regenerated bank")
    _, fresh = sc.cached(cache, "r6@val3", fp(), score)
    assert fresh and len(calls) == 4
    _, fresh = sc.cached(cache, "r6@val3", fp({"id": "r6", "ckpts": ["r6", "r6_s2"]}), score)
    assert fresh and len(calls) == 5                      # registry entry edited
    legacy = {"r6@val3": {"sim_score": 0.9}}              # v1.0 cache entries have no fingerprint
    _, fresh = sc.cached(legacy, "r6@val3", fp(), score)
    assert fresh and len(calls) == 6
