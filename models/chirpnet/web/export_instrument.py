"""Export ChirpNet's live instrument for the website: float16 weights for the finder, the coincidence checker and the
posterior flow; 12 s of whitened, gated H1 + L1 strain around each of the four real events (the reading pipeline —
Welch-PSD whitening, 20-900 Hz band, 2048 Hz, 25-sigma gating — runs here, in Python, with chirpnet.py's own code);
and reference outputs for the browser parity test.

GW170817 (binary neutron stars, ChirpNetBNS) is a fifth event: 46 s of whitened 1024 Hz strain around the merger from
the 2,048 s cleaned GWOSC files (LIGO-P1700349), Livingston's raw data with the famous glitch (LIGO-P1700337), and
both detectors' data from 211 s earlier (for "one detector alone"). Strain ships ungated; the browser gates it (25 sigma,
chirpnet.gate), so the gating toggle is live. Download the three files into models/chirpnet/work/data_bns/ (git-ignored):

    https://dcc.ligo.org/public/0146/P1700349/001/H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5
    https://dcc.ligo.org/public/0146/P1700349/001/L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5
    https://dcc.ligo.org/public/0146/P1700337/001/L-L1_LOSC_C00_4_V1-1187006834-4096.hdf5

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

# GW170817 / ChirpNetBNS
BNS_DIR = os.path.join(MODEL, "work", "data_bns")
BNS_FILES = {"H1": "H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5", "L1": "L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5",
             "L1_raw": "L-L1_LOSC_C00_4_V1-1187006834-4096.hdf5"}
BNS_GPS = 1187008882.43       # GWTC-1 merger time
BNS_MC_DET = 1.1975           # GWTC-1 detector-frame chirp mass (+0.0001 / -0.0001)
BNS_PRE, BNS_SEG = 34.0, 46.0 # span: 34 s before the merger to 12 s after (room for a 4 s slide of a 16 s window)
BNS_OFF = 211                 # "alone": the partner's own data this many seconds earlier (as bns_real.single_detector_test)
BNS_PLACE = 15.0              # default window: merger 15 s in (training placed it 8-15.8 s in)

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
    z = np.load(os.path.join(MODEL, "chirpnet_bns_weights.npz"), allow_pickle=False)
    arrs = {k: z[k] for k in z.files}
    w = {k: v for k, v in arrs.items() if k not in ("meta", "background", "band_matrix")}
    packs["bns"] = {**w, "band_matrix": arrs["band_matrix"]}
    np.savez(os.path.join(tmp, "bns.npz"), **{**arrs, **roundtrip(w, "f16")})
    net = C.ChirpNet(os.path.join(tmp, "chirpnet_weights.npz"))
    pe = C.ChirpPE(os.path.join(tmp, "chirpnet_pe_weights.npz"))
    bns = C.ChirpNetBNS(os.path.join(tmp, "bns.npz"))
    return net, pe, bns, packs


def main():
    with tempfile.TemporaryDirectory() as tmp:
        net, pe, bns, packs = rounded_models(tmp)
    W = {}
    for pre, name in (("f_", "chirpnet_weights.npz"), ("c_", "chirpnet_coinc_weights.npz"), ("p_", "chirpnet_pe_weights.npz")):
        for k, v in packs[name].items():
            if pre == "c_" and k.startswith("aux"):
                continue  # auxiliary training head, unused at inference
            W[pre + k] = v
    for k, v in packs["bns"].items():
        W["n_" + k] = v
    bg = np.sort(net.bg)[::-1]
    bgb = np.sort(bns.bg)[::-1]
    meta = {
        "conv": net.meta["conv"], "threshold": net.meta["threshold"], "veto": net.veto, "drop": net.drop,
        "pe_layers": pe.meta["n_layers"], "pe_mu": pe.mu.tolist(), "pe_sd": pe.sd.tolist(), "pe_params": pe.meta["params"],
        "bg_top": bg[:3000].round(3).tolist(), "bg_n": int(len(bg)), "bg_live_s": net.bg_live_s, "bg_max": float(bg[0]),
        "release": net.meta.get("release", {}),
        "bns": {"conv": bns.meta["conv"], "nfft": bns.nfft, "hop": bns.hop, "frames": bns.nfr, "pnorm": bns.pnorm,
                "threshold": bns.meta["threshold"], "bg_top": bgb[:3000].round(3).tolist(), "bg_n": int(len(bgb)),
                "bg_live_s": bns.bg_live_s, "bg_max": float(bgb[0]), "release": bns.meta.get("release")},
    }
    write_pack(OUT / "weights", W, dtype="f16", meta=meta, keep={"n_band_matrix": "f32"})

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
    bns_ev = export_bns(bns, xs, cases, len(ex))
    write_pack(OUT / "examples", xs, dtype="f16", meta={"fs": FS, "win": WIN, "pre": PRE, "seg": SEG, "place": PLACE, "events": ex,
                                                         "bns": bns_ev},
               keep={"z": "f32"})
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    tot = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {len(ex)} events, weights {os.path.getsize(OUT / 'weights.bin') / 1e6:.2f} MB, total {tot / 1e6:.2f} MB")
    for e in ex:
        print(f"  {e['id']}: python scan score {e['python']['score']:.2f}, coincidence {e['python']['coincidence']:.2f}, "
              f"chirp mass {e['python']['chirp_mass']:.1f} (LIGO {e['truth']['mc_det']:.1f})")
    p = bns_ev["python"]
    print(f"  GW170817: python scan score {p['score']:.2f}, chirp mass {p['chirp_mass']:.4f} (LIGO {BNS_MC_DET}); "
          f"raw L1 gated {p['raw_gated']:.2f}, ungated {p['raw_ungated']:.2f}")


def export_bns(bns, xs, cases, index):
    """GW170817: whiten the 2,048 s files (as bns_real.prepare), cut a 46 s span, write parity cases."""
    missing = [f for f in BNS_FILES.values() if not os.path.exists(os.path.join(BNS_DIR, f))]
    if missing:
        raise SystemExit(f"GW170817 data missing from {BNS_DIR}: {missing} (see this script's docstring for the URLs)")
    FB, WB = C.FSB, C.WINB
    h, t0 = C.load_gwosc(os.path.join(BNS_DIR, BNS_FILES["H1"]))
    l, _ = C.load_gwosc(os.path.join(BNS_DIR, BNS_FILES["L1"]))
    lr, tr = C.load_gwosc(os.path.join(BNS_DIR, BNS_FILES["L1_raw"]))
    i0 = int(round((t0 - tr) * 4096))
    lr = lr[i0:i0 + len(h)]                              # the raw file's same 2,048 s, so its PSD matches bns_real
    H, L, Lr = C.whiten_bns(h), C.whiten_bns(l), C.whiten_bns(lr)
    te = BNS_GPS - t0
    a = int(round((te - BNS_PRE) * FB))
    n = int(BNS_SEG * FB)
    o = BNS_OFF * FB
    f16 = lambda v: v.astype(np.float16).astype(np.float32)
    span = {"H": f16(H[a:a + n]), "L": f16(L[a:a + n]), "Lr": f16(Lr[a:a + n]),
            "Hoff": f16(H[a - o:a - o + n]), "Loff": f16(L[a - o:a - o + n])}
    for k, v in span.items():
        xs[f"n_{k}"] = v
    t_ev = te - a / FB
    s0 = int(round((t_ev - BNS_PLACE) * FB))
    gated = {k: C.gate(v, 25.0, fs=FB) for k, v in span.items()}
    # parity cases: as recorded (gated), L1 slid by 1 s, raw L1 with the glitch gated, and with gating off
    for variant, lkey, ls, gate in (("bns", "L", s0, True), ("bns_slide1s", "L", s0 - FB, True),
                                    ("bns_glitch", "Lr", s0, True), ("bns_glitch_nogate", "Lr", s0, False)):
        Hx = gated["H"][0] if gate else span["H"]
        Lx = gated[lkey][0] if gate else span[lkey]
        x = np.stack([Hx[s0:s0 + WB], Lx[ls:ls + WB]]).astype(np.float32)
        cases.append({"example": index, "variant": variant, "hs": s0, "ls": int(ls), "l": lkey, "gate": gate,
                      "outputs": bns.forward(x[None])[0].astype(float).tolist()})
    # the Python pipeline for the record: scan 2 minutes around the merger (stride 0.25 s, gating inside)
    k0 = int((te - 60) * FB)
    seg = lambda v: v[k0:k0 + 120 * FB]
    hits = bns.scan(seg(H), seg(L), gps_start=t0 + k0 / FB, whitened=True)
    hr = bns.scan(seg(H), seg(Lr), gps_start=t0 + k0 / FB, whitened=True)
    hu = bns.scan(seg(H), seg(Lr), gps_start=t0 + k0 / FB, whitened=True, gate_sigma=1e9)
    hit = hits[0]
    return {"id": "GW170817", "date": "17 Aug 2017", "gps": BNS_GPS, "t_ev": t_ev, "s0": s0, "fs": FB, "win": WB,
            "pre": BNS_PRE, "seg": BNS_SEG, "off": BNS_OFF, "place": BNS_PLACE,
            "gates": {k: [(round(g0, 3), round(g1, 3)) for g0, g1 in v[1]] for k, v in gated.items()},
            "truth": {"mc_det": BNS_MC_DET, "mc_det_lo": BNS_MC_DET - 0.0001, "mc_det_hi": BNS_MC_DET + 0.0001,
                      "m1": 1.46, "m2": 1.27, "dl": 40, "snr": 32.4},
            "python": {"gps": hit["gps"], "score": hit["score"], "chirp_mass": hit["chirp_mass"], "chirp_mass_68": hit["chirp_mass_68"],
                       "raw_gated": hr[0]["score"] if hr else None, "raw_ungated": hu[0]["score"] if hu else None}}


if __name__ == "__main__":
    main()
