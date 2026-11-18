"""v1.1 rung r5 (17 Nov 2026; plan in NOTEBOOK.md): the window's observed Dst as inputs, 256 random units.

Builds train2h / val2h (same draws as train2 / val2 plus the simulated window Dst), checks they match the plain
banks, trains r5 at seeds 103 / 104 / 105 and the r3 reference, and scores everything with ratchet.py's objective on
val2h and the dev gate. Writes v11_results.json; the decision is applied to ratchet.yaml by hand from it.
"""
import json

import numpy as np

from _paths import work
from noise import HERE, read_result
from ratchet import error_ratio
from simbank import load_bank, make_bank
from split import dev_cases
from stormnet import StormNet
from train import train

cal = read_result("calibration.json")
fp = {}
for frozen in (False, True):
    fp["val2h" if frozen else "train2h"] = make_bank(2, frozen, cal["double"], with_past=True)
for name in ("train2", "val2"):
    a, b = load_bank(name), load_bank(name + "h")
    assert all(np.array_equal(a[k], b[k]) for k in ("wind", "dst", "truth")), name
val = load_bank("val2h")
dev = dev_cases()


def score(path):
    m = StormNet(path)
    kw = (lambda d: dict(past_dst=d["past_dst"])) if m.meta["mode"] == "history_dst" else (lambda d: {})
    sim = 1 - error_ratio(val, m.predict(val["wind"], val["dst"], **kw(val))["min_dst"])
    gate = {yr: error_ratio(d, m.predict(d["wind"], d["dst"], **kw(d))["min_dst"]) for yr, d in dev.items()}
    return sim, gate


out = {"banks": fp, "released_r3": dict(zip(("sim", "dev"), score(str(HERE.parent / "stormnet_weights.npz"))))}
train("r3", "history", ("train2",), cal, seed=103, folder="ckpt_v11")
out["r3_rebuilt"] = dict(zip(("sim", "dev"), score(work("ckpt_v11", "r3.npz"))))
for seed in (103, 104, 105):
    rid = f"r5_s{seed}"
    train(rid, "history_dst", ("train2h",), cal, seed=seed, folder="ckpt_v11", units=256)
    out[rid] = dict(zip(("sim", "dev"), score(work("ckpt_v11", rid + ".npz"))))
(HERE / "v11_results.json").write_text(json.dumps(out, indent=2) + "\n")
for k, v in out.items():
    if k != "banks":
        print(k, round(v["sim"], 4), {y: round(r, 4) for y, r in v["dev"].items()})
