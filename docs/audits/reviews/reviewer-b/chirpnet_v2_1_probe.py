"""Audit BBH search, PE conditioning and four folded exports on event strain or a BNS control."""

import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "models/chirpnet"
sys.path[:0] = [str(MODEL), str(MODEL / "training")]
import chirpnet as ch
import model
import flow


def params(mod, w, b):
    mod.weight = nn.Parameter(torch.as_tensor(w))
    mod.bias = nn.Parameter(torch.as_tensor(b))


def trunk(dst, src):
    ls = iter(src.layers)

    def conv(m):
        w, b, *_ = next(ls)
        params(m, w, b)

    conv(dst.stem[0])
    dst.stem[1] = nn.Identity()
    for block in dst.blocks:
        for c, b in (("c1", "b1"), ("c2", "b2"), ("sk", "bs")):
            conv(getattr(block, c))
            setattr(block, b, nn.Identity())


def torch_detector(net):
    kws = dict(
        widths=[len(net.layers[i][1]) for i in [0, 1, 4, 7, 10, 13]],
        hidden=len(net.lin[0][1]),
    )
    if net.arch == "2s":
        dst = model.ChirpNet2S(**kws)
        params(dst.mix[0], *net.mix)
        dst.mix[1] = nn.Identity()
        for k, b, p in zip(("c1", "c2", "sk"), ("b1", "b2", "bs"), net.post):
            params(getattr(dst.post, k), *p)
            setattr(dst.post, b, nn.Identity())
        params(dst.aux, *net.aux)
    else:
        dst = model.ChirpNet(**kws, pos=net.red is not None)
        if dst.pos:
            params(dst.red, *net.red)
    trunk(dst, net)
    params(dst.head[0], *net.lin[0])
    params(dst.head[2], *net.lin[1])
    return dst.eval()


def torch_pe(pe):
    dst = flow.ChirpPE()
    trunk(dst.cnn, pe)
    params(dst.ctx[0], *pe.ctx[0])
    params(dst.ctx[2], *pe.ctx[1])
    for layer, (mask, weights) in zip(dst.flow.layers, pe.flow):
        layer.mask.copy_(torch.tensor(mask))
        for i, (w, b) in enumerate(weights):
            params(layer.net[2 * i], w, b)
    return dst.eval()


def bns_parity():
    # Reconstruct the folded TF branch using torch.stft and conv2d.
    net = ch.ChirpNetBNS(MODEL / "chirpnet_bns_weights.npz")
    x = np.random.default_rng(8).normal(size=(1, 2, net.meta["window"])).astype("f")
    t = torch.tensor(x).reshape(2, -1)
    stft = torch.stft(
        t, net.nfft, net.hop, window=torch.hann_window(net.nfft), return_complex=True
    )
    p = stft.abs().square()[:, :, : net.nfr] / net.pnorm
    h = torch.log1p(torch.tensor(net.M) @ p)[None]
    tf_err = np.abs(h.numpy() - net.tf_map(x)).max()

    def conv(a, i):
        w, b, stride, pad = net.layers[i]
        return F.conv2d(a, torch.tensor(w), torch.tensor(b), stride, pad)

    h = F.gelu(conv(h, 0))
    for i in range(1, len(net.layers), 3):
        h = F.gelu(conv(F.gelu(conv(h, i)), i + 1) + conv(h, i + 2))
    h = torch.cat([h.mean((-1, -2)), h.amax((-1, -2))], 1)
    for i, (w, b) in enumerate(net.lin):
        h = F.linear(h, torch.tensor(w), torch.tensor(b))
        if i == 0:
            h = F.gelu(h)
    print(
        "BNS synthetic TF and output max errors",
        tf_err,
        np.abs(h.numpy() - net.forward(x)).max(),
    )


def ladders():
    logs = MODEL / "training/logs"
    inc = None
    for r in json.loads((logs / "ratchet_log.json").read_text()):
        a = r["val"]
        obj = a["sens"] + r["sens_glitch"] - 2 * a["mc_relerr_med"]
        if inc is None:
            keep = True
        else:
            b = inc["val"]
            old = b["sens"] + inc["sens_glitch"] - 2 * b["mc_relerr_med"]
            rule = r.get("rule", 1)
            if rule == 1:
                safe = a["sens"] >= b["sens"] - 0.003
                keep = (
                    a["sens"] >= b["sens"] + 0.005
                    and a["mc_relerr_med"] <= 1.05 * b["mc_relerr_med"]
                ) or (
                    safe
                    and (
                        a["mc_relerr_med"] <= 0.95 * b["mc_relerr_med"]
                        or r["sens_glitch"] >= inc["sens_glitch"] + 0.02
                    )
                )
            elif rule == 2:
                keep = obj > old + 0.01
            else:

                def fa(row):
                    return np.mean(
                        [
                            v
                            for k, v in row["single_fa"].items()
                            if rule == 3 or int(k) >= 12
                        ]
                    )

                keep = obj > old - 0.01 and fa(a) < 0.5 * fa(b)
        assert keep == r["accepted"], r["ckpt"]
        if keep:
            inc = r
    inc = None
    for r in json.loads((logs / "bns_ratchet_log.json").read_text()):
        v = r["val"]
        obj = v["sens"] - 2 * v["mc_relerr_med"]
        keep = inc is None or (obj > inc + 0.01 and v["glitch_fa"] <= 0.05)
        assert keep == r["accepted"]
        if keep:
            inc = obj


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    for path in MODEL.glob("*weights.npz"):
        z = np.load(path)
        assert all(np.isfinite(z[k]).all() for k in z if k != "meta")
    net = ch.ChirpNet(MODEL / "chirpnet_weights.npz")
    pe = ch.ChirpPE(MODEL / "chirpnet_pe_weights.npz")
    finder, checker, ref = (
        torch_detector(net),
        torch_detector(net.checker),
        torch_pe(pe),
    )
    meta = json.loads(
        (ROOT / "site/public/instruments/chirpnet/examples.json").read_text()
    )["meta"]
    latent = np.random.default_rng(4).normal(size=(1024, 6)).astype("f")
    print(
        "BBH background n, exposure h, max, threshold, zero-count FAR",
        len(net.bg),
        net.bg_live_s / 3600,
        net.bg.max(),
        net.meta["threshold"],
        net.far_per_hour(net.meta["threshold"]),
    )
    assert np.ceil((net.bg.max() + 0.5) * 2) / 2 == net.meta["threshold"]
    o3 = json.loads((MODEL / "training/logs/o3_dev_background.json").read_text())
    print(
        "archived O3 windows above threshold, time-slide hours, maximum",
        o3["n_above_7"],
        o3["livetime_s"] / 3600,
        o3["max"],
    )
    for file in sorted((MODEL / "examples").glob("H-*.hdf5")):
        raw, t0 = ch.load_gwosc(file)
        other = next((MODEL / "examples").glob("L-*" + file.name.split("_")[-1]))
        l, start = ch.load_gwosc(other)
        assert start == t0
        hits = net.scan(raw, l, t0)
        print(
            file.name, "unrestricted hits", [(r["gps"] - t0, r["score"]) for r in hits]
        )
        rec = next(e for e in meta["events"] if t0 <= e["gps"] < t0 + 32)
        whitened = [ch.whiten(v) for v in (raw, l)]
        H, L = [ch.gate(v)[0] for v in whitened]
        rolled = [np.roll(v, 4 * ch.FS) for v in whitened]
        relocated = net.scan(*rolled, gps_start=t0, whitened=True)
        print(
            "joint +4 s shift, loudest trigger displacement",
            relocated[0]["gps"] - hits[0]["gps"] if hits and relocated else None,
        )
        asd = np.r_[ch.asd_features(raw), ch.asd_features(l)].astype("f")

        def win(gps, place):
            i = int(round((gps - t0 - place) * ch.FS))
            return np.stack([H[i : i + ch.WIN], L[i : i + ch.WIN]]).astype("f")

        x = win(rec["gps"], 1.35)
        rng = np.random.default_rng(7)
        # Preserve each detector's exact sample values, destroy their coherent time structure.
        scrambled = np.stack([rng.permutation(a) for a in x])
        for label, a in (
            ("catalogue", x),
            ("trigger", win(hits[0]["gps"], 1.35) if hits else None),
            ("placement 0.85", win(rec["gps"], 0.85)),
            ("placement 1.75", win(rec["gps"], 1.75)),
            ("permuted", scrambled),
            ("zeros", np.zeros_like(x)),
        ):
            if a is None:
                continue
            o = net.predict(a[None])
            s = pe.sample(a, asd, z=latent)
            print(
                label,
                "score / chirp mass median [5,95]",
                float(o["score"][0]),
                np.percentile(np.exp(s[:, 0]), [50, 5, 95]),
            )
        # Same latent draws and target standardization as NumPy.
        t = torch.tensor(x)[None]
        z = torch.tensor(latent)
        ctx = ref.context(t, torch.tensor(asd)[None]).expand(len(z), -1)
        for layer in reversed(ref.flow.layers):
            z = layer.inverse(z, ctx)
        s = (z * flow.SD + flow.MU).numpy()
        print(
            "finder/checker/PE max raw errors",
            np.abs(finder(t).numpy() - net.forward(x[None])).max(),
            np.abs(checker(t).numpy()[:, :7] - net.checker.forward(x[None])).max(),
            np.abs(pe.sample(x, asd, z=latent) - s).max(),
        )
    bns_parity()
    ladders()


if __name__ == "__main__":
    main()
