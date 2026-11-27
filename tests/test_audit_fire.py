"""Regression tests for the fire-platform audit findings fixed on 27 Nov 2026 (docs/audits/)."""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(model: str, module: str):
    """The model's module through the registry, which keeps sibling modules (pipeline, common, features) private."""
    import mlratchet
    return mlratchet.get(model).module(module)


def test_firedetective_refuses_an_empty_history():
    fd = _load("firedetective", "firedetective")
    model = fd.load()
    W = model.nets[0].W if hasattr(model, "nets") else model.W
    seq = np.zeros((1, 12, W["seq_mu"].shape[-1]))
    static = np.zeros((1, W["st_mu"].shape[-1])) if "st_mu" in W else np.zeros((1, 8))
    with pytest.raises(ValueError, match="no observed step"):
        model.predict(seq, seq, static)


def test_firesize_warns_on_inputs_it_never_saw():
    fs = _load("firesize", "firesize")
    m = fs.FireSize()
    x = {f: np.array([float(m.W["mu"][i])]) for i, f in enumerate(m.features)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.probs(x)                                   # the training mean is silent
    x["pt_SB"] = np.array([1.0])                     # slash-blowdown at the ignition
    with pytest.warns(UserWarning, match="beyond 8 sd"):
        p = m.probs(x)
    assert np.isfinite(p).all()


def test_firesize_clip_does_not_depend_on_the_batch():
    fs = _load("firesize", "firesize")
    m = fs.FireSize()
    m.z_clip = 4.0
    try:
        base = {f: float(m.W["mu"][i]) for i, f in enumerate(m.features)}
        j = 0
        six = dict(base, **{m.features[j]: base[m.features[j]] + 6 * float(m.W["sd"][j])})
        alone = m.probs({k: np.array([v]) for k, v in six.items()})
        far = dict(base, pt_SB=1.0)
        with pytest.warns(UserWarning):
            both = m.probs({k: np.array([six[k], far[k]]) for k in six})
        np.testing.assert_allclose(alone[0], both[0], atol=1e-12)
    finally:
        m.z_clip = None
