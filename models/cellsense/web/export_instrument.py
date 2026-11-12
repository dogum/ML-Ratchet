"""Export CellSense's live instrument for the website: float16 weights of all three networks (health, lifetime,
partial discharge), the real exam cell b3c33 read at seven points through its life (Q(V) and can temperature on the
voltage grid, exactly as examples/b3c33.npz holds them), and reference outputs for the browser parity test.

    python models/cellsense/web/export_instrument.py      # writes site/public/instruments/cellsense/

The browser computes features() itself (NaN fill, mask, 24 mV smoothing), so the parity cases cover it too.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.dirname(os.path.dirname(MODEL)))

import cellsense as CS  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir("cellsense")
NETS = (("h", "health"), ("l", "life"), ("p", "partial"))


def cut(q, t, stop):
    """a partial discharge from full, stopped once stop Ah are out: everything after is thrown away (NaN), as demo.py"""
    q, t = q.copy(), t.copy()
    c = q > stop
    q[c] = np.nan
    t[c] = np.nan
    return q, t


def main():
    cs = CS.CellSense()
    W, meta = {}, {}
    for p, name in NETS:
        net = getattr(cs, name)
        for k, v in net.W.items():
            W[f"{p}_{k}"] = v
        meta[name] = net.meta
        net.W = roundtrip(net.W, "f16")  # the browser sees float16 weights; reference outputs use the same values
    write_pack(OUT / "weights", W, dtype="f16", meta=meta)

    d = np.load(os.path.join(MODEL, "examples", "b3c33.npz"))
    Q, T = d["Q"].astype(np.float32), d["T"].astype(np.float32)
    ct = float(d["charge_time"])
    n = len(d["cycle"])

    def raw(net, X, S=None):
        return net.forward(X, S if net.meta["extra"] else None)[0]

    def life_S(cycle):
        return np.array([[np.log10(max(cycle, 1)) / 3.5, ct / 15.0, 1.0]], np.float32)

    cases, python = [], []
    for i in range(1, n):
        cyc = int(d["cycle"][i])
        sm_l, sm_p = cs.life.meta.get("smooth", 0), cs.partial.meta.get("smooth", 0)
        # full discharge with the new-cell reference: health + lifetime networks
        X = CS.features(Q[i], T[i], Q[0], T[0])
        Xl = CS.features(Q[i], T[i], Q[0], T[0], smooth=sm_l)
        o = np.concatenate([raw(cs.health, X), raw(cs.life, Xl, life_S(cyc))])
        cases.append({"example": i, "variant": "full", "outputs": o.tolist()})
        # no reference curve: the same two networks reading the discharge on its own
        X = CS.features(Q[i], T[i])
        Xl = CS.features(Q[i], T[i], smooth=sm_l)
        o = np.concatenate([raw(cs.health, X), raw(cs.life, Xl, life_S(cyc))])
        cases.append({"example": i, "variant": "noref", "outputs": o.tolist()})
        # partial discharges stopped at 0.75 Ah and 0.45 Ah: the partial-discharge network
        for stop in (0.75, 0.45):
            q, t = cut(Q[i], T[i], stop)
            o = raw(cs.partial, CS.features(q, t, Q[0], T[0], smooth=sm_p))
            cases.append({"example": i, "variant": f"cut{int(round(stop * 100)):03d}", "stop": stop, "outputs": o.tolist()})
        r = cs.diagnose(Q[i], T[i], Q[0], T[0], cycle=cyc, charge_time_min=ct)
        python.append({k: float(r[k]) for k in ("q_total", "q_total_sig", "ir", "ir_sig", "lli", "lam_ne", "lam_pe", "life", "life_lo", "life_hi", "left")})

    ex = {
        "cell": "b3c33", "policy": str(d["policy"]), "charge_time": ct, "cycle_life": int(d["cycle_life"]),
        "cycles": [int(c) for c in d["cycle"]],
        "measured_q_total": [None if not np.isfinite(v) else float(v) for v in d["measured_q_total"]],
        "measured_ir_mohm": [None if not np.isfinite(v) else float(v) for v in d["measured_ir_mohm"]],
        "note": str(d["note"]), "python": python,
    }
    write_pack(OUT / "examples", {"Q": Q, "T": T, "vgrid": d["vgrid"]}, dtype="f32", meta=ex)
    write_json(OUT / "reference.json", {"cases": cases, "tolerance": 2e-3})
    tot = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"wrote {OUT}: {n - 1} readings, {len(cases)} parity cases, {tot / 1e6:.2f} MB in total")


if __name__ == "__main__":
    main()
