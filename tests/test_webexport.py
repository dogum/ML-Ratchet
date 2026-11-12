"""The tensor-pack format the website's instruments read."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlratchet.webexport import roundtrip, write_pack  # noqa: E402


def test_pack_layout_roundtrips(tmp_path):
    a = np.arange(7, dtype=np.float32).reshape(7) / 3
    b = np.random.default_rng(0).normal(size=(2, 3, 5)).astype(np.float32)
    lab = np.array([0, 2, 1])
    man = write_pack(tmp_path / "p", {"a": a, "b": b, "lab": lab}, dtype="f16", meta={"x": 1}, keep={"lab": "i32", "a": "f32"})
    raw = (tmp_path / "p.bin").read_bytes()
    m = json.loads((tmp_path / "p.json").read_text())
    assert m == json.loads(json.dumps(man)) and m["meta"] == {"x": 1}
    for name, want, dt in (("a", a, "<f4"), ("b", roundtrip({"b": b})["b"], "<f2"), ("lab", lab, "<i4")):
        d = m["tensors"][name]
        assert d["offset"] % 8 == 0
        got = np.frombuffer(raw[d["offset"]:d["offset"] + d["bytes"]], dtype=dt).reshape(d["shape"]).astype(np.float64)
        assert np.allclose(got, want, atol=1e-7)
