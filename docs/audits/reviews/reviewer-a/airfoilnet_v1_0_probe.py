"""Forensic audit probe for AirfoilNet v1.0 (Reviewer A, first pass).

Checks:
  1. Weight health, ensemble diversity (pairwise cosine), and shipped metadata vs module docstring.
  2. Torch (training/train.py Net) vs NumPy (airfoilnet.py AirfoilNet) parity across all 4 examples.
  3. Coordinate invariances on all 4 shipped UIUC exam airfoils:
     - scale + rotation + translation
     - reverse Selig ordering
     - raw Lednicer ordering passed to polar(coords=...) (documented as supported)
  4. Physical mirror symmetry (y -> -y, alpha -> -alpha) on NACA 0012 and all 4 shipped airfoils.
  5. Feature sensitivity (Re 5e4 vs 5e5) and out-of-range stability (Re=1e7, alpha=+-30 deg).
  6. Split disjointness (training/splits.json) and ratchet replay (ratchet.json, ratchet_val2.json).

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/airfoilnet_v1_0_probe.py
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / "models/airfoilnet"),
    str(ROOT / "models/airfoilnet/training"),
]
import airfoilnet as an
import train as an_tr


def naca0012(n=60):
    th = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(th))
    xu = x[::-1]
    xl = x[1:]
    yt = lambda s: 0.6 * (0.2969 * np.sqrt(s) - 0.1260 * s - 0.3516 * s**2 + 0.2843 * s**3 - 0.1036 * s**4)
    return np.column_stack([np.concatenate([xu, xl]), np.concatenate([yt(xu), -yt(xl)])])


def main():
    w_path = ROOT / "models/airfoilnet/airfoilnet_weights.npz"
    w = np.load(w_path)
    meta = json.loads(str(w["meta"]))
    net = an.AirfoilNet(str(w_path))

    # 1. Weight health & ensemble diversity
    for k in w.files:
        if k == "meta":
            continue
        arr = w[k]
        assert np.isfinite(arr).all(), f"non-finite {k}"
        if arr.ndim == 2:
            assert (np.max(np.abs(arr), axis=1) >= 1e-8).all(), f"dead row in {k}"
            assert (np.max(np.abs(arr), axis=0) >= 1e-8).all(), f"dead col in {k}"
    vecs = [w[f"m{m}/net.0.w"].ravel() for m in range(meta["n_members"])]
    cos = [
        float(np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j])))
        for i in range(len(vecs))
        for j in range(i + 1, len(vecs))
    ]
    print(
        f"meta: arch={meta['arch']}, anchor={meta['anchor']}, p_latent={meta['p_latent']}, "
        f"n_members={meta['n_members']}, widen={[round(x, 3) for x in meta['widen']]}"
    )
    print(f"weights: finite, 0 dead rows/cols | member net.0.w pairwise cosine: {[round(c, 4) for c in cos]}")

    # Build Torch ensemble mirror
    torch_members = []
    for m in range(meta["n_members"]):
        tm = an_tr.Net(meta["cfg"])
        sd = tm.state_dict()
        lin_keys = sorted([k for k in sd if k.endswith(".weight")], key=lambda s: int(s.split(".")[1]))
        with torch.no_grad():
            for i, k in enumerate(lin_keys):
                sd[k].copy_(torch.tensor(w[f"m{m}/net.{i}.w"], dtype=torch.float32))
                sd[k.replace("weight", "bias")].copy_(torch.tensor(w[f"m{m}/net.{i}.b"], dtype=torch.float32))
        tm.eval()
        torch_members.append(tm)

    # 2–4. Shipped examples: parity + geometric invariances + mirror symmetry
    ex_dir = ROOT / "models/airfoilnet/examples"
    ex_files = sorted(f for f in os.listdir(ex_dir) if f.endswith(".json"))
    max_torch_diff = 0.0
    print("\nShipped UIUC exam examples:")
    for fn in ex_files:
        ex = json.loads((ex_dir / fn).read_text())
        xy = np.array(ex["coords"], float)
        Re = float(ex["Re"])
        al = np.array([row[0] for row in ex["lift"]["alpha_cl_cm"]], float)
        p, _ = an.cst_fit(an.normalize(xy))
        out_np = net.predict(p, Re, al)

        # Torch parity
        F_in = np.repeat(an.features(p, Re)[None, :], len(al), axis=0)
        T_in = an.trunk_features(al)
        anc = an.anchors_none(al)
        with torch.no_grad():
            mus = [
                tm(torch.tensor(F_in, dtype=torch.float32), torch.tensor(T_in, dtype=torch.float32)).numpy()[:, :3]
                * an.OUT_SCALE
                + anc
                for tm in torch_members
            ]
        mu_t = np.mean(mus, axis=0)
        td = max(
            float(np.max(np.abs(mu_t[:, 0] - out_np["cl"]))),
            float(np.max(np.abs(mu_t[:, 1] - np.log(out_np["cd"])))),
            float(np.max(np.abs(mu_t[:, 2] - out_np["cm"]))),
        )
        max_torch_diff = max(max_torch_diff, td)

        # Scale x2.5 + rotate 37 deg + translate
        th = np.radians(37.0)
        Rmat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        xy_srt = (xy * 2.5) @ Rmat.T + np.array([10.0, -5.0])
        out_srt = net.polar(coords=xy_srt, Re=Re, alphas=al)
        srt_diff = max(
            float(np.max(np.abs(out_srt["cl"] - out_np["cl"]))),
            float(np.max(np.abs(np.log(out_srt["cd"]) - np.log(out_np["cd"])))),
        )

        # Reverse Selig order
        out_rev = net.polar(coords=xy[::-1], Re=Re, alphas=al)
        rev_diff = max(
            float(np.max(np.abs(out_rev["cl"] - out_np["cl"]))),
            float(np.max(np.abs(np.log(out_rev["cd"]) - np.log(out_np["cd"])))),
        )

        # Raw Lednicer order (LE->TE upper, LE->TE lower) passed to polar(coords=...)
        yn = an.normalize(xy)
        ile = int(np.argmin(yn[:, 0]))
        xy_led = np.vstack([yn[: ile + 1][::-1], yn[ile:]])
        out_led = net.polar(coords=xy_led, Re=Re, alphas=al)
        led_cl_diff = float(np.max(np.abs(out_led["cl"] - out_np["cl"])))
        led_lncd_diff = float(np.max(np.abs(np.log(out_led["cd"]) - np.log(out_np["cd"]))))

        # Mirror flip: y -> -y, alpha -> -alpha
        xy_mir = np.column_stack([xy[:, 0], -xy[:, 1]])
        out_mir = net.polar(coords=xy_mir, Re=Re, alphas=-al)
        mir_cl = float(np.max(np.abs(out_mir["cl"] + out_np["cl"])))
        mir_lncd = float(np.max(np.abs(np.log(out_mir["cd"]) - np.log(out_np["cd"]))))
        mir_cm = float(np.max(np.abs(out_mir["cm"] + out_np["cm"])))

        # Re sensitivity (5e4 vs 5e5)
        out_lo = net.predict(p, 5e4, al)
        out_hi = net.predict(p, 5e5, al)
        dln_re = float(np.mean(np.log(out_lo["cd"]) - np.log(out_hi["cd"])))

        print(
            f"  {fn:20s}: torch_diff={td:.2e} | srt={srt_diff:.2e} | rev={rev_diff:.2e} | "
            f"lednicer(Cl,lnCd)=({led_cl_diff:.2f},{led_lncd_diff:.2f}) | "
            f"mirror(Cl,lnCd,Cm)=({mir_cl:.3f},{mir_lncd:.3f},{mir_cm:.4f}) | dlnCd(5e4->5e5)={dln_re:+.3f}"
        )
    print(f"max Torch-vs-NumPy diff: {max_torch_diff:.3e}")

    # 4b. Symmetric NACA 0012 symmetry & OOD check
    xy0012 = naca0012()
    al_sym = np.array([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    out0012 = net.polar(coords=xy0012, Re=2e5, alphas=al_sym)
    print(
        f"\nNACA 0012 (Re=2e5): Cl(0)={out0012['cl'][3]:+.4f}, Cm(0)={out0012['cm'][3]:+.4f}, "
        f"Cl(-6,+6)=({out0012['cl'][0]:+.4f},{out0012['cl'][-1]:+.4f}), "
        f"Cd(-6,+6)=({out0012['cd'][0]:.5f},{out0012['cd'][-1]:.5f})"
    )
    out_ood = net.polar(coords=xy0012, Re=1e7, alphas=np.array([-30.0, 0.0, 30.0]))
    assert all(np.isfinite(out_ood[k]).all() for k in ("cl", "cd", "cm"))
    print(f"NACA 0012 OOD (Re=1e7, alpha=-30,0,30): Cl={np.round(out_ood['cl'], 3)}, Cd={np.round(out_ood['cd'], 5)}")

    # 6. Splits disjointness & ratchet replay
    sp = json.loads((ROOT / "models/airfoilnet/training/splits.json").read_text())
    overlap = set(sp["dev"]) & set(sp["exam"])
    assert len(overlap) == 0, f"split overlap: {overlap}"
    print(f"\nsplits.json: dev={len(sp['dev'])}, exam={len(sp['exam'])}, overlap=0")

    for rfn in ("ratchet.json", "ratchet_val2.json"):
        rj = json.loads((ROOT / "models/airfoilnet/training" / rfn).read_text())
        inc_s, inc_d, inc_id = None, None, None
        for row in rj["log"]:
            if inc_s is None:
                dec = "kept"
                inc_s, inc_d, inc_id = row["sim"], row["dev_D"], row["id"]
            else:
                ok = (row["sim"] > inc_s + 0.01) and (row["dev_D"] <= 1.05 * inc_d)
                dec = "kept" if ok else "rejected"
                if ok:
                    inc_s, inc_d, inc_id = row["sim"], row["dev_D"], row["id"]
            assert dec == row["decision"], f"{rfn} mismatch on {row['id']}"
        assert inc_id == rj["incumbent"]
        print(f"{rfn}: {len(rj['log'])}/{len(rj['log'])} rows match rule -> incumbent={inc_id}")


if __name__ == "__main__":
    main()
