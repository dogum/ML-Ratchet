"""Export StarQuake's live instrument for the website: float16 weights, the three real Kepler red giants from the exam
(their whole light curves, ppm on the cadence grid with NaN gaps), the exam's own pieces of each (4 yr, 1 yr, 90, 54,
27 days, drawn exactly as training/realbank.pieces draws them), and reference outputs for the browser parity test.

The browser runs the whole reading chain itself — high-pass, power spectrum, the 1024-bin log spectrum and the 48 x 110
comb map (a line-for-line port of pipeline.py, as training/sqnet.js was) and then the network — so added noise is
applied to the light curve, exactly where a fainter star's photon noise would be.

    python models/starquake/web/export_instrument.py      # writes site/public/instruments/starquake/
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import pipeline as PL  # noqa: E402
from starquake import StarQuake, gelu, conv1d, conv2d, scaling  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("starquake")
LENS = [27, 54, 90, 365, 0]          # training/realbank.py LENS; 0 = the whole light curve


def pieces(y, kic):
    """The exam's pieces of one star: training/realbank.pieces, returning start indices."""
    rng = np.random.default_rng(kic); out = {}
    for L in LENS:
        if L == 0: out[0] = (0, len(y)); continue
        n = int(L * 86400 / PL.DT)
        for _ in range(50):
            s = int(rng.integers(0, len(y) - n + 1))
            if np.isfinite(y[s:s + n]).mean() > 0.8: out[L] = (s, n); break
    return out


def raw_forward(net, x1, x2, s):
    """starquake.Net.forward up to the head's raw outputs (5 numbers)."""
    W = net.W; m = net.meta
    h = gelu(conv1d(((np.asarray(x1, np.float64) - 2.5) / 2)[None], W['s1_w'], W['s1_b'], 2, 3))
    for i in range(m['n1']):
        a = gelu(conv1d(h, W[f'r1_{i}_c1_w'], W[f'r1_{i}_c1_b'], 2, 3)); a = conv1d(a, W[f'r1_{i}_c2_w'], W[f'r1_{i}_c2_b'], 1, 3)
        h = gelu(a + conv1d(h, W[f'r1_{i}_sk_w'], W[f'r1_{i}_sk_b'], 2, 0))
    h1 = h.ravel()
    h = gelu(conv2d((np.asarray(x2, np.float64) * 4)[None], W['s2_w'], W['s2_b'], (1, 2), 1))
    for i in range(m['n2']):
        a = gelu(conv2d(h, W[f'r2_{i}_c1_w'], W[f'r2_{i}_c1_b'], (2, 2), 1)); a = conv2d(a, W[f'r2_{i}_c2_w'], W[f'r2_{i}_c2_b'], (1, 1), 1)
        h = gelu(a + conv2d(h, W[f'r2_{i}_sk_w'], W[f'r2_{i}_sk_b'], (2, 2), 0))
    v = np.concatenate([h1, h.ravel(), np.asarray(s, np.float64)])
    for j in range(m['n_head']):
        v = W[f'h{j}_w'] @ v + W[f'h{j}_b']
        if j < m['n_head'] - 1: v = gelu(v)
    return v


def main():
    sq = StarQuake(); net = sq.net
    W = {k: v for k, v in net.W.items()}
    net.W = {k: v.astype(np.float64) for k, v in roundtrip(W, "f16").items()}   # what the browser sees
    write_pack(OUT / "weights", W, dtype="f16", meta=net.meta)

    truth = json.load(open(os.path.join(MODEL, "examples", "truth.json")))
    xs, ex, cases = {}, [], []
    for i, kic in enumerate(sorted(truth)):
        t0 = time.time()
        y = PL.load_kepler(np.load(os.path.join(MODEL, "examples", f"{kic}.npz")))
        y = y.astype(np.float32).astype(np.float64)               # the browser gets float32
        xs[f"y{i}"] = y.astype(np.float32)
        P = pieces(y, int(kic)); t = truth[kic]; py = {}
        F1all = PL.prepare(y)[0]        # the whole light curve's log spectrum: fixes the plot's power axis per star
        for L in LENS:
            s0, n = P[L]; seg = y[s0:s0 + n]
            F1, F2, s, _, T = PL.prepare(seg)
            v = raw_forward(net, F1, F2, s)
            r = sq.read_ppm(seg, teff=t["teff"], e_teff=t["e_teff"])
            py[str(L)] = {k: float(r[k]) for k in ("numax", "numax_err", "dnu", "dnu_err", "p_osc", "mass", "mass_err", "radius", "radius_err", "days")}
            cases.append({"example": i, "variant": f"{L or 'all'}d", "start": s0, "n": n, "outputs": v.tolist()})
            print(f"  KIC {kic} {L or 'all':>4} d ({T:6.1f} d): numax {r['numax']:7.2f} ± {r['numax_err']:.2f} ({t['numax']}), dnu {r['dnu']:6.3f} ± {r['dnu_err']:.3f} ({t['dnu']}), p {r['p_osc']:.3f}")
        ex.append({"kic": kic, "truth": t, "f1range": [float(F1all.min()), float(F1all.max())], "pieces": {str(L): list(P[L]) for L in LENS}, "python": py})
        print(f"KIC {kic}: {time.time() - t0:.1f} s")
    write_pack(OUT / "examples", xs, dtype="f32", meta={"stars": ex, "dt": PL.DT, "lens": LENS})
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    size = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {len(ex)} stars, {len(cases)} cases, {size / 1e6:.2f} MB total")


if __name__ == "__main__":
    main()
