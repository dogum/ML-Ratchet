"""Export TransitNet's live instrument for the website: float16 weights, the three real TESS stars as the pipeline leaves
them just before folding (every detrended 2-minute point: time from the nearest transit, relative flux, odd/even transit),
and reference outputs for the browser parity test.

The slow steps of the reading pipeline (detrending, ephemeris alignment and refinement, flare clipping) run here, in
Python, through the model's own pipeline.read. The browser repeats only the last step, pipeline.views (fold, bin,
measure the depth and noise, normalise) and the scalars, then runs the network both ways in time, as transitnet.py does.
That is what lets the instrument change the light curve itself (noise, third light, a secondary eclipse, odd/even
depths) and re-read it honestly.

    python models/transitnet/web/export_instrument.py      # writes site/public/instruments/transitnet/
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import pipeline as PL  # noqa: E402
import transitnet as tn  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("transitnet")
STARS = ("toi270c", "pimenc", "toi617")
TRUTH = {"planet": "planet", "FP": "blend"}  # TOI-617.01: TFOP nearby eclipsing binary (NEB) with a centroid offset
NT = PL.NL  # template bins, on the local view's grid (+-2.5 durations)


def capture(net, z, m):
    """Run the real reading pipeline; keep the per-point arrays it hands to pipeline.views."""
    got = {}
    orig = PL.views

    def spy(dt, n, f, P, dur):
        got.update(dt=np.array(dt), n=np.array(n), f=np.array(f), P=float(P), dur=float(dur))
        return orig(dt, n, f, P, dur)

    PL.views = spy
    try:
        secs = [dict(t=z[f"t{j}"], f=z[f"f{j}"], c1=z[f"c1{j}"], c2=z[f"c2{j}"]) for j in range(m["n_sectors"])]
        r = net.read(secs, m["period"], m["epoch"], m["duration_hours"], star=m["star"])
    finally:
        PL.views = orig
    return got, r


def template(dt, y, dur):
    """The star's own mean dip on the local view's grid (fraction, <= 0 in transit), lightly smoothed and tapered to
    zero outside the transit: what the instrument copies to make a secondary eclipse or unequal odd/even transits."""
    x = dt / dur
    mu, cnt = PL.binned(x, y, -2.5, 2.5, NT)
    mu = np.where(cnt > 0, mu, 0.0)
    sm = np.convolve(np.pad(mu, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    xc = -2.5 + 5 * (np.arange(NT) + 0.5) / NT
    w = np.clip((0.8 - np.abs(xc)) / 0.2, 0, 1)
    return np.minimum(sm * w, 0.0)


def tpl_at(T, x):
    """Linear interpolation of the template at x (durations from mid-transit); zero outside. Same arithmetic as net.ts."""
    u = (x + 2.5) * NT / 5 - 0.5
    i = np.floor(u).astype(int)
    f = u - i
    a = np.where((i >= 0) & (i < NT), T[np.clip(i, 0, NT - 1)], 0.0)
    b = np.where((i + 1 >= 0) & (i + 1 < NT), T[np.clip(i + 1, 0, NT - 1)], 0.0)
    return a * (1 - f) + b * f


def perturb(dt, y, odd, P, dur, T, sec=0.0, oe=0.0, depth=1.0):
    """Deterministic changes to the light curve (the instrument adds its noise on top; the parity cases have none).
    The star's mean dip T is the eclipse signal S: add a secondary eclipse `sec` x the primary's depth at phase 0.5,
    make odd transits deeper by `oe` and even ones shallower by `oe`. Then `depth` < 1 dilutes everything with third
    light (an unresolved star 1/depth - 1 times as bright sharing the pixels), and `depth` > 1 scales the eclipse
    signal up (a bigger body in front), leaving the rest of the light curve as it was. Same arithmetic as net.ts."""
    y = y.astype(np.float64)
    x = dt / dur
    t1 = tpl_at(T, x)
    S = t1 * (1 + oe * np.where(odd > 0, 1.0, -1.0))
    if sec:
        S = S + sec * tpl_at(T, (np.mod(dt, P) - 0.5 * P) / dur)
    base = y - t1 + S
    return base * depth if depth < 1 else base + (depth - 1) * S


def main():
    net = tn.TransitNet()
    W = {k: v for k, v in net.net.W.items()}
    net.net.W = roundtrip(W, "f16")  # the browser sees float16 weights; reference outputs use the same values
    meta = dict(net.net.meta, mu_y=net.net.meta["mu_y"], sd_y=net.net.meta["sd_y"])
    write_pack(OUT / "weights", W, dtype="f16", meta=meta)

    tens, keep, ex, cases = {}, {}, [], []
    for i, name in enumerate(STARS):
        z = np.load(os.path.join(MODEL, "examples", name + ".npz"))
        m = json.loads(str(z["meta"]))
        got, r = capture(net, z, m)
        P, dur = got["P"], got["dur"]
        dt32 = got["dt"].astype(np.float32)
        y16 = (got["f"] - 1).astype(np.float16)
        odd = (got["n"] % 2 == 1).astype(np.uint8)
        dt, y = dt32.astype(np.float64), y16.astype(np.float64)
        T = template(dt, y, dur)
        x = dt / dur
        ntr = int(len(np.unique(got["n"][np.abs(x) < 0.5])))
        st = PL.star_scalars(m["star"])
        tens[f"dt{i}"], tens[f"y{i}"], tens[f"odd{i}"], tens[f"tpl{i}"] = dt32, y16, odd, T.astype(np.float32)
        keep.update({f"dt{i}": "f32", f"y{i}": "f16", f"odd{i}": "u8", f"tpl{i}": "f32"})
        T = T.astype(np.float32).astype(np.float64)

        def views_of(yy):
            V, info = PL.views(dt, got["n"], 1 + yy, P, dur)
            s = PL.scalars(P, dur, info, st)
            return V, info, s

        for variant, kw in (("raw", {}), ("binary", dict(sec=0.3, oe=0.2, depth=10.0)), ("blend", dict(sec=0.6, oe=0.5, depth=0.5)), ("diluted", dict(depth=0.2))):
            V, info, s = views_of(perturb(dt, y, odd, P, dur, T, **kw))
            fl = lambda a: a[..., ::-1]  # noqa: E731
            o1 = net.net.forward(V["local"][None], V["secondary"][None], V["glob"][None], V["oddeven"][None], s[None])[0]
            o2 = net.net.forward(fl(V["local"])[None], fl(V["secondary"])[None], fl(V["glob"])[None], fl(V["oddeven"])[None], s[None])[0]
            cases.append({"example": i, "variant": variant, **kw, "outputs": np.concatenate([o1, o2]).tolist()})
            R = net.predict_views({k: v[None] for k, v in V.items()}, s[None])
            print(f"  {name:8s} {variant:8s} planet {R['p_planet'][0]:.3f} eb {R['p_eb'][0]:.3f} blend {R['p_blend'][0]:.3f} "
                  f"nothing {R['p_nothing'][0]:.3f}  k {10 ** R['log_k'][0]:.4f}  depth {1e6 * info['depth']:.0f} ppm")

        pub = m["published"]
        ex.append({
            "id": name, "name": m["name"], "kind": m["kind"], "truth": TRUTH[m["kind"]],
            "P": P, "dur": dur, "duration_hours": m["duration_hours"], "n_sectors": m["n_sectors"], "ntr": ntr,
            "n_points": int(len(dt)), "star": {k: (None if not np.isfinite(v) else float(v)) for k, v in st.items()},
            "tic": m["star"],
            "published": {k: v for k, v in pub.items()},
            "python": {k: r[k] for k in ("p_planet", "p_eb", "p_blend", "p_nothing", "k", "k_lo", "k_hi", "b", "b_lo", "b_hi",
                                         "aR", "aR_lo", "aR_hi", "T14_h", "depth_ppm", "snr")},
        })
    write_pack(OUT / "examples", tens, dtype="f32", meta=ex, keep=keep)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    sz = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {len(ex)} stars, {sum(e['n_points'] for e in ex)} points, total {sz / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
