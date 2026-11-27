"""Content fingerprints for the ratchet's score cache (no torch; tested in tests/test_bridgesense.py).

A cached score is reused only if everything that produced it is byte-identical: the rung's registry
entry (its checkpoint list, in order), every checkpoint file, the scored bank (simulated scores) and
the scoring and feature code. Retraining a rung under the same name, editing its checkpoint list,
regenerating a bank or changing feature extraction therefore forces a rescore. (v1.0 keyed the cache
by rung id and bank name only, so a retrained checkpoint silently reused the old model's scores.)
"""
from __future__ import annotations

import hashlib
import json
import os

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)

# Code that changes what a checkpoint scores: features, the network, the KW51 dev loader and truth, scoring
SIM_CODE = ("features.py", "simulator.py", "model.py", "train.py", "dev_eval.py")
DEV_CODE = ("features.py", "simulator.py", "model.py", "dev_eval.py", "realdata.py", "calibrate.py")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(entry: dict, ckpt_paths: list[str], code_files: list[str], data_files: list[str] = ()) -> str:
    """sha256 over the registry entry, checkpoint contents, data files and code files (order matters)."""
    h = hashlib.sha256()
    h.update(json.dumps({"id": entry["id"], "ckpts": list(entry["ckpts"])}, sort_keys=True).encode())
    for group, paths in (("ckpt", ckpt_paths), ("data", data_files), ("code", code_files)):
        for p in paths:
            h.update(f"{group}:{os.path.basename(p)}:{file_sha256(p)}\n".encode())
    return h.hexdigest()


def cached(cache: dict, key: str, fp: str, compute) -> tuple[dict, bool]:
    """Return (scores, fresh). Reuse cache[key] only when its stored fingerprint equals fp."""
    hit = cache.get(key)
    if hit is not None and hit.get("fingerprint") == fp:
        return hit, False
    scores = dict(compute())
    scores["fingerprint"] = fp
    cache[key] = scores
    return scores, True
