"""Export ChirpNet's live instrument for the website: float16 weights for the finder, the coincidence checker and the
posterior flow; 12 s of whitened, gated H1 + L1 strain around each of the four real events (the reading pipeline —
Welch-PSD whitening, 20-900 Hz band, 2048 Hz, 25-sigma gating — runs here, in Python, with chirpnet.py's own code);
and reference outputs for the browser parity test.

    python models/chirpnet/web/export_instrument.py      # writes site/public/instruments/chirpnet/
"""
import glob
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import chirpnet as C  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("chirpnet")
FS, WIN = C.FS, C.WIN
PRE, SEG = 8.0, 12.0          # each segment runs from 8 s before LIGO's merger time to 4 s after
PLACE = 1.35                  # default window: merger 1.35 s into the 2 s window (where ChirpNet-PE expects it)
N_Z = 8                       # parity: posterior samples from fixed latent draws

# GWTC-1 (Table III source frame medians and 90% intervals, Table I network SNR); merger GPS times from GWOSC
EVENTS = [
    dict(id="GW150914", gps_file="1126259446", gps=1126259462.44, date="14 Sep 2015", mc=28.6, mc_p=1.7, mc_m=1.5,
         m1=35.6, m2=30.6, z=0.09, dl=440, dl_p=150, dl_m=170, chi=-0.01, snr=24.4),
    dict(id="GW151012", gps_file="1128678884", gps=1128678900.44, date="12 Oct 2015", mc=15.2, mc_p=2.1, mc_m=1.2,
         m1=23.2, m2=13.6, z=0.21, dl=1080, dl_p=550, dl_m=490, chi=0.05, snr=10.0),
    dict(id="GW151226", gps_file="1135136334", gps=1135136350.65, date="26 Dec 2015", mc=8.9, mc_p=0.3, mc_m=0.3,
         m1=13.7, m2=7.7, z=0.09, dl=450, dl_p=180, dl_m=190, chi=0.18, snr=13.1),
    dict(id="GW170104", gps_file="1167559920", gps=1167559936.6, date="4 Jan 2017", mc=21.4, mc_p=2.2, mc_m=1.8,
         m1=30.8, m2=20.0, z=0.20, dl=990, dl_p=440, dl_m=430, chi=-0.04, snr=13.0),
]


def rounded_models(tmp):
    """ChirpNet (finder + checker) and ChirpPE with float16-rounded weights: exactly what the browser runs."""
    packs = {}
    for name in ("chirpnet_weights.npz", "chirpnet_coinc_weights.npz", "chirpnet_pe_weights.npz"):
        z = np.load(os.path.join(MODEL, name), allow_pickle=False)
        arrs = {k: z[k] for k in z.files}
        w = {k: v for k, v in arrs.items() if k not in ("meta", "background") and v.dtype.kind == "f"}
        packs[name] = w
        r = roundtrip(w, "f16")
        np.savez(os.path.join(tmp, name), **{**arrs, **r})
    net = C.ChirpNet(os.path.join(tmp, "chirpnet_weights.npz"))
    pe = C.ChirpPE(os.path.join(tmp, "chirpnet_pe_weights.npz"))
    return net, pe, packs


def main():
    with tempfile.TemporaryDirectory() as tmp:
        net, pe, packs = rounded_models(tmp)
    W = {}
    for pre, name in (("f_", "chirpnet_weights.npz"), ("c_", "chirpnet_coinc_weights.npz"), ("p_", "chirpnet_pe_weights.npz")):
        for k, v in packs[name].items():
            if pre == "c_" and k.startswith("aux"):
                continue  # auxiliary training head, unused at inference
            W[pre + k] = v
    bg = np.sort(net.bg)[::-1]
    meta = {
        "conv": net.meta["conv"], "threshold": net.meta["threshold"], "veto": net.veto, "drop": net.drop,
        "pe_layers": pe.meta["n_layers"], "pe_mu": pe.mu.tolist(), "pe_sd": pe.sd.tolist(), "pe_params": pe.meta["params"],
        "bg_top": bg[:3000].round(3).tolist(), "bg_n": int(len(bg)), "bg_live_s": net.bg_live_s, "bg_max": float(bg[0]),
        "release": net.meta.get("release", {}),
    }
    write_pack(OUT / "weights", W, dtype="f16", meta=meta)

    xs, ex, cases = {}, [], []
    zfix = np.random.default_rng(2024).standard_normal((N_Z, 6)).astype(np.float32)
    xs["z"] = zfix
    for i, e in enumerate(EVENTS):
        h1, t0 = C.load_gwosc(glob.glob(os.path.join(MODEL, "examples", f"H-H1_*-{e['gps_file']}-32.hdf5"))[0])
        l1, _ = C.load_gwosc(glob.glob(os.path.join(MODEL, "examples", f"L-L1_*-{e['gps_file']}-32.hdf5"))[0])
        H, gH = C.gate(C.whiten(h1))
        L, gL = C.gate(C.whiten(l1))
        te = e["gps"] - t0
        a = int(round((te - PRE) * FS))
        n = int(SEG * FS)
        Hs = H[a:a + n].astype(np.float16).astype(np.float32)   # the browser receives float16 strain
        Ls = L[a:a + n].astype(np.float16).astype(np.float32)
        xs[f"h{i}"], xs[f"l{i}"] = Hs, Ls
        asd = np.concatenate([C.asd_features(h1), C.asd_features(l1)]).astype(np.float32)
        t_ev = te - a / FS                         # LIGO merger time, seconds into the segment
        s0 = int(round((t_ev - PLACE) * FS))       # default window start (sample)

        # parity cases: as recorded; L1 slid 200 ms; L1 replaced by its own data 5 s earlier (H1 alone)
        for variant, ls in (("raw", s0), ("slide200", s0 - int(0.2 * FS)), ("h1alone", s0 - 5 * FS)):
            x = np.stack([Hs[s0:s0 + WIN], Ls[ls:ls + WIN]]).astype(np.float32)
            o = net.forward(x[None])[0]
            c = net.checker.forward(x[None])[0]
            smp = pe.sample(x, asd, z=zfix).reshape(-1)
            cases.append({"example": i, "variant": variant, "hs": s0, "ls": int(ls),
                          "outputs": np.concatenate([o, c, smp]).astype(float).tolist()})

        # the full Python pipeline on this event, for the record (scan with the released stride)
        hit = min(net.scan(H, L, gps_start=t0, whitened=True), key=lambda h: abs(h["gps"] - e["gps"]))
        ex.append({
            "id": e["id"], "date": e["date"], "gps": e["gps"], "t_ev": t_ev, "s0": s0, "asd": asd.tolist(),
            "gates": {"H1": gH, "L1": gL},
            "truth": {"mc_src": e["mc"], "mc_src_lo": e["mc"] - e["mc_m"], "mc_src_hi": e["mc"] + e["mc_p"], "z": e["z"],
                      "mc_det": e["mc"] * (1 + e["z"]), "mc_det_lo": (e["mc"] - e["mc_m"]) * (1 + e["z"]),
                      "mc_det_hi": (e["mc"] + e["mc_p"]) * (1 + e["z"]), "m1": e["m1"], "m2": e["m2"], "q": e["m2"] / e["m1"],
                      "dl": e["dl"], "dl_lo": e["dl"] - e["dl_m"], "dl_hi": e["dl"] + e["dl_p"], "chi": e["chi"], "snr": e["snr"]},
            "python": {k: hit[k] for k in ("gps", "score", "coincidence", "chirp_mass", "chirp_mass_68", "h1_alone", "l1_alone")},
        })
    write_pack(OUT / "examples", xs, dtype="f16", meta={"fs": FS, "win": WIN, "pre": PRE, "seg": SEG, "place": PLACE, "events": ex},
               keep={"z": "f32"})
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    tot = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {len(ex)} events, weights {os.path.getsize(OUT / 'weights.bin') / 1e6:.2f} MB, total {tot / 1e6:.2f} MB")
    for e in ex:
        print(f"  {e['id']}: python scan score {e['python']['score']:.2f}, coincidence {e['python']['coincidence']:.2f}, "
              f"chirp mass {e['python']['chirp_mass']:.1f} (LIGO {e['truth']['mc_det']:.1f})")


if __name__ == "__main__":
    main()
