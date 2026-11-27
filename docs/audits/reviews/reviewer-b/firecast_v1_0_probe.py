"""Replay FireCast's three shipped fire-days. Run from any directory with NumPy, SciPy and CPU Torch."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL = Path(__file__).resolve().parents[4] / "models/firecast"
sys.path[:0] = [str(MODEL), str(MODEL / "training/src")]
from demo import load_days
import firecast as fc
from model import FireCastNet


def folded_model(p):
    channels = [len(p[key + ".b"]) for key in ("stem", "e1a", "e2a", "e3a", "e4a")]
    net = FireCastNet(
        p["stem.w"].shape[1], p["mlp0.w"].shape[1], channels, len(p["mlp0.b"])
    )

    def assign(module, key):
        for field, suffix in (("weight", ".w"), ("bias", ".b")):
            value = torch.from_numpy(p[key + suffix])
            param = getattr(module, field)
            setattr(
                module,
                field,
                nn.Parameter(value.reshape_as(param) if param is not None else value),
            )

    blocks = {"stem": net.stem, **{f"d{i}": getattr(net, f"d{i}") for i in range(4)}}
    for i in range(1, 5):
        convs = getattr(net, f"e{i}")[-2:]
        blocks.update(zip((f"e{i}a", f"e{i}b"), convs))
    for key, block in blocks.items():
        assign(block[0], key)
        block[1] = nn.Identity()  # Exported convolutions already incorporate BatchNorm.
    for i, layer in enumerate(net.film):
        assign(layer, f"film{i}")
    for key, layer in (
        ("mlp0", net.mlp[0]),
        ("mlp1", net.mlp[2]),
        ("head", net.head),
        ("head_s", net.head_s),
    ):
        assign(layer, key)
    return net.eval()


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    net = fc.FireCastNP()
    assert all(np.isfinite(a).all() for a in net.p.values())
    ref = folded_model(net.p)
    for day in load_days(MODEL / "examples/demo_days.npz"):
        now, nxt, wx = day["now"], day["nxt"], day["wx"]
        X, s = fc.make_inputs(
            now,
            day["prev"],
            day["fl"],
            day["pix_pm"],
            day["pix_am"],
            None,
            wx,
            day["has_prev"],
            day["first"],
        )
        # Centred deployment-sized feature crop, aligned to the encoder stride.
        cy, cx = np.array(now.shape) // 2
        X = X[:, cy - 32 : cy + 32, cx - 32 : cx + 32].copy()
        logits = ref(torch.from_numpy(X[None]), torch.from_numpy(s[None])).numpy()[0]
        print(
            day["meta"]["name"],
            "folded logit max error",
            np.max(np.abs(logits - net.forward(X, s))),
        )
        win = fc.score_window(now)
        padded = np.pad(
            now, 112
        )  # ceil(10000 / 90), enough for the entire nominal window
        nominal = (fc.score_window(padded) & ~padded).sum()
        cand = win & ~now
        print("candidate cells in grid / nominal", cand.sum(), nominal)
        p = fc.predict(net, day, land=False)
        p_control = fc.predict(net, dict(day, nxt=~nxt), land=False)
        assert np.array_equal(p, p_control)
        changed = fc.predict(net, dict(day, wx=np.r_[wx[:8], wx[:8]]), land=False)
        for label, pred in (("archived weather", p), ("repeat day zero", changed)):
            score = fc.nextday(now, nxt, pred, fc.CELL_KM2, net.thr, win)
            print(label, "AP F1", score["ap"], score["f1"])
        print("archived exam AP F1", day["meta"]["exam_ap"], day["meta"]["exam_f1"])
        print(
            "candidate decisions changed",
            np.count_nonzero((p[cand] >= net.thr) != (changed[cand] >= net.thr)),
        )
    log = json.loads((MODEL / "training/ratchet.json").read_text())
    for r in log["rungs"][1:]:
        delta = r["sim_ap_F"] - r["incumbent_sim"]
        ratio = r["dev_ap_F"] / r["incumbent_dev"]
        assert r["kept"] == (delta > 0.01 and ratio >= 0.95)
        print(r["rung"], "sim delta / dev ratio / kept", delta, ratio, r["kept"])
    cal = json.loads((MODEL / "training/results/cal_r2_F.json").read_text())
    np.testing.assert_allclose(
        [net.cal_T, net.cal_b, net.thr], [cal["T"], cal["b"], cal["thr"]]
    )
    for b in cal["reliability_dev"]:
        if b["lo"] >= 0.35 and b["n"]:
            print("dev reliability", b)


if __name__ == "__main__":
    main()
