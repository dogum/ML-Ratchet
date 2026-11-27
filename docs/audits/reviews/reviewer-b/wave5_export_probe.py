"""Compare the released NumPy graphs with the original training classes using folded weights."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

MODELS = Path(__file__).resolve().parents[4] / "models"


def assign(param, array):
    param.copy_(torch.as_tensor(array, dtype=param.dtype))


def linear(weights, prefix, mod):
    assign(mod.weight, weights[prefix + "_w"])
    assign(mod.bias, weights[prefix + "_b"])


def folded(weights, prefix, conv, bn):
    # Carry the exported bias in BN. The multiplicative part of eval-mode BN is identity.
    assign(conv.weight, weights[prefix + "_w"])
    if conv.bias is not None:
        conv.bias.zero_()
    bn.running_mean.zero_()
    bn.running_var.fill_(1)
    bn.weight.fill_((1 + bn.eps) ** .5)
    assign(bn.bias, weights[prefix + "_b"])


def health(weights):
    arrays = [v for v in weights.values() if np.issubdtype(v.dtype, np.number)]
    assert all(np.isfinite(v).all() for v in arrays)
    kernels = [w.reshape(w.shape[0], -1) for k, w in weights.items() if k.endswith("_w")]
    counts = [np.count_nonzero(np.linalg.norm(w, axis=1) < 1e-4) for w in kernels]
    print("finite weight arrays, near-zero kernel rows / rows", sum(counts), "/", sum(map(len, kernels)))


def diff(label, numpy, pytorch):
    delta = np.max(np.abs(numpy - pytorch.numpy()))
    print(label, f"{delta:.4g}")
    np.testing.assert_allclose(numpy, pytorch.numpy(), atol=2e-5, rtol=2e-5)


def sundoctor(root):
    from sundoctor import SunDoctor
    import model2
    from unittest.mock import patch

    api = SunDoctor()
    net = model2.SunDoctor().eval()
    w = api.net.W
    health(w)
    folded(w, "s", net.stem[0], net.stem[1])
    folded(w, "p", net.proj[0], net.proj[1])
    for i, block in enumerate(net.blocks):
        for key, conv, bn in (("c1", block.c1, block.b1), ("c2", block.c2, block.b2), ("sk", block.sk, block.bs)):
            folded(w, f"b{i}{key}", conv, bn)
    for i, block in enumerate(net.tblocks):
        folded(w, f"t{i}c1", block.c1, block.b1)
        folded(w, f"t{i}c2", block.c2, block.b2)
    for key, mod in (("d", net.day), ("g0", net.glob[0]), ("g1", net.glob[2])):
        linear(w, key, mod)
    samples = []

    def capture(x):
        if not samples:
            samples.append(x[:1].copy())
        return original(x)

    original = api.net.forward
    with patch.object(api.net, "forward", side_effect=capture):
        api.diagnose_csv(next((root / "examples").glob("pvdaq2107*")), lat=38.996, lon=-122.134,
                         tilt=25, azimuth=180, dc_kw=37.2, tz="US/Pacific", n_strings=6)
    for x in [samples[0], np.random.default_rng(5).normal(size=samples[0].shape).astype("f")]:
        a, b = original(x)
        at, bt = net(torch.tensor(x, dtype=torch.float32))
        diff("day", a, at)
        diff("global", b, bt)


def swellcast(root):
    import model
    import swellcast as sc

    api, net = sc.SwellCast(), model.SwellNet().eval()
    w = api.W
    health(w)
    folded(w, "stem", net.stem, net.bstem)
    folded(w, "red", net.red, net.bred)
    for i, block in enumerate(net.body):
        for key, conv, bn in (("c1", block.c1, block.b1), ("c2", block.c2, block.b2), ("sk", block.sk, block.bs)):
            folded(w, f"b{i}_{key}", conv, bn)
    for i, mod in enumerate(net.head[::2]):
        linear(w, f"h{i}", mod)
    windows = []
    for file in sorted((root / "examples").glob("*.npz")):
        with np.load(file) as z:
            windows.append(z["x"])
    x = np.stack(windows)
    f = sc.features(x)
    np.testing.assert_array_equal(f, model.features(x))
    for label, feats in (("real", f), ("random", np.random.default_rng(5).normal(size=f[:1].shape).astype("f"))):
        diff(label, api.forward(feats), net(torch.tensor(feats)))


def tripdetective(root):
    import model
    import tripdetective as td
    from demo import load_example

    for filename in sorted(root.glob("*weights.npz")):
        api = td.TripDetective(filename)
        w = api.W
        health(w)
        net = model.TDNet(feat=api.feat, phys=api.phys).eval()
        for i, block in enumerate([net.stem, *net.blocks]):
            folded(w, f"c{i}", block.c, block.bn)
        for key, seq in (("trip", net.trip), ("car", net.car), ("led", net.h_led)):
            for i, layer in enumerate(m for m in seq if isinstance(m, torch.nn.Linear)):
                linear(w, f"{key}{i}", layer)
        for key in ("arch", "strat", "reg"):
            linear(w, key, getattr(net, "h_" + key))
        if api.phys:
            assign(net.skip, w["skip"])
        if "bus" in filename.name:
            with np.load(root / "examples/ZTBus_exam_mission.npz") as z:
                leg = {k[5:]: z[k] for k in z if k.startswith("leg0_")}
            raw, flags = td.TripDetectiveBus().raw(leg)
        else:
            info, trips = load_example(next(iter(sorted((root / "examples").glob("VED*")))))
            flags = td.flags_from_trip(trips[0], info["trips"][0]["fuel_source"] != "obd_fuel_rate_Lph")
            raw = td.raw_from_log(trips[0], flags)
            flags = np.array([flags[k] for k in td.FLAGS], dtype="f")
        x, mask, scalars = td.net_input(raw[:, :td.L], flags, feat=api.feat)
        for label, inputs in (("real", x), ("random channels, zero padding", np.random.default_rng(5).normal(size=x.shape).astype("f") * mask[None])):
            h = api.heads([api.encode(inputs, mask, scalars)], [scalars])
            out = net(torch.tensor(inputs[None]), torch.tensor(mask[None]), torch.tensor([0]), 1, torch.tensor(scalars[None]))
            print(filename.name, label)
            diff("architecture", h["arch"], out["arch"][0].softmax(0))
            diff("strategy", h["strat"], out["strat"][0].softmax(0))
            diff("regression mean", h["mu"], out["mu"][0])
            diff("regression sigma", h["sig"], out["ls"][0].exp())
            diff("ledger mean (network units)", h["led"][0][0] / td.LED_SCALE, out["led_mu"][0])
            diff("ledger sigma (network units)", h["led"][0][1] / td.LED_SCALE, out["led_ls"][0].exp())


if __name__ == "__main__":
    runners = dict(tripdetective=tripdetective, sundoctor=sundoctor, swellcast=swellcast)
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=runners)
    name = parser.parse_args().model
    root = MODELS / name
    sys.path[:0] = [str(root), str(root / "training")]
    with torch.no_grad():
        runners[name](root)
