"""Restore the released weights into the training graphs and replay the original network ladders."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from wave5_export_probe import folded, linear, health

MODELS = Path(__file__).resolve().parents[4] / "models"


def block(w, prefix, b):
    for suffix, conv, bn in (("c1", b.c1, b.b1), ("c2", b.c2, b.b2), ("sk", b.sk, b.bs)):
        folded(w, prefix + suffix, conv, bn)


def head(w, model):
    for i, layer in enumerate(m for m in model.head if isinstance(m, torch.nn.Linear)):
        linear(w, f"h{i}", layer)


def beatnet(root):
    import model
    import beatnet
    import pipeline
    api = beatnet.BeatNet()
    w = api.W
    net = model.BeatNet(cin=api.meta["cin"], ctr=api.meta["ctr"], nout=w["h2_b"].size,
                        widths=[w[f"b{i}_c1_w"].shape[0] for i in range(api.meta["n_blocks"])]).eval()
    for i, b in enumerate(net.body):
        block(w, f"b{i}_", b)
    with np.load(root / "examples/202.npz") as z:
        feat = pipeline.features(z["ecg"], json.loads(str(z["meta"]))["fs"], z["beats"])
    args = (feat["w"][:16, None], np.nan_to_num(feat["rr"][:16]))
    head(w, net)
    return w, net, args, api.forward


def transitnet(root):
    import model
    import transitnet
    api = transitnet.TransitNet()
    w = api.net.W
    net = model.TransitNet(width=w["loc0_w"].shape[0] / 32, cent=False).eval()
    for key in ("loc", "sec", "glo", "oe"):
        layers = getattr(net, key)
        weights = [op[1] for op in api.net.meta["branches"][key] if op[0] == "conv"]
        convs = [(mod, layers[i + 1]) for i, mod in enumerate(layers) if isinstance(mod, torch.nn.Conv1d)]
        for prefix, (conv, bn) in zip(weights, convs, strict=True):
            folded(w, prefix, conv, bn)
    head(w, net)
    with np.load(root / "examples/toi617.npz") as z:
        m = json.loads(str(z["meta"]))
        sectors = [{k: z[f"{k}{i}"] for k in ("t", "f", "c1", "c2")} for i in range(m["n_sectors"])]
    with patch.object(api.net, "forward", wraps=api.net.forward) as record:
        api.read(sectors, m["period"], m["epoch"], m["duration_hours"], star=m["star"])
    return w, net, record.call_args_list[0].args[:5], api.net.forward


def starquake(root):
    import model
    import pipeline
    import starquake
    api = starquake.StarQuake().net
    w = api.W
    net = model.StarQuake(det=True).eval()
    for key in ("s1", "s2"):
        layers = getattr(net, key)
        folded(w, key, layers[0], layers[1])
    for key in ("r1", "r2"):
        for i, b in enumerate(getattr(net, key)):
            block(w, f"{key}_{i}_", b)
    head(w, net)
    with np.load(root / "examples/2695975.npz") as z:
        seg = pipeline.load_kepler(z)[20000:24400]
    def outputs(a, b, scalars):
        out = api.forward(a[0], b[0], scalars[0])
        means = np.array([out[k] for k in ("ln_numax", "ln_dnu")])
        sigmas = np.array([out[k] for k in ("sig_numax", "sig_dnu")])
        p = out["p_det"]
        return np.r_[(means - api.meta["mu_y"]) / api.meta["sd_y"],
                     np.log(sigmas / api.meta["sd_y"]), np.log(p / (1 - p))][None]
    return w, net, [v[None] for v in pipeline.prepare(seg)[:3]], outputs


def replay(root):
    inc = None
    rows = json.loads((root / "training/ratchet.json").read_text())
    for row in rows:
        keep = True
        if inc is not None:
            d, prev = row["dev"], inc["dev"]
            if root.name == "beatnet":
                gate = d["sv_f1"] >= prev["sv_f1"] - .02
            elif root.name == "transitnet":
                gate = d["logloss"] <= prev["logloss"] + .05 and d["k_frac_med"] <= prev["k_frac_med"] + .01
            else:
                gate = np.mean(np.array(d["keys"]) / prev["keys"]) <= 1.10
            keep = row["objective"] > inc["objective"] + .01 and gate
        assert row["accepted"] == keep, row["ckpt"]
        if keep:
            inc = row
    print(len(rows), "network decisions matched", inc["ckpt"])


if __name__ == "__main__":
    funcs = dict(transitnet=transitnet, beatnet=beatnet, starquake=starquake)
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=funcs)
    name = parser.parse_args().model
    root = MODELS / name
    sys.path[:0] = [str(root), str(root / "training")]
    replay(root)
    with torch.no_grad():
        w, net, args, predict = funcs[name](root)
        health(w)
        rng = np.random.default_rng(7)
        for label, batch in (("real", args), ("random", [rng.normal(size=a.shape).astype("f") for a in args])):
            batch = [np.ascontiguousarray(a, dtype="f") for a in batch]
            out = predict(*batch)
            ref = net(*map(torch.from_numpy, batch)).numpy()
            print(label, "max absolute output difference", np.max(np.abs(out - ref)))
            np.testing.assert_allclose(out, ref, atol=3e-5, rtol=2e-5)
