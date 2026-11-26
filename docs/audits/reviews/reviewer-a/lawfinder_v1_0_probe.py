"""First-pass 10-point forensic audit probe for LawFinder v1.0 (all 6 domains).

Run:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/lawfinder_v1_0_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "models/lawfinder"))
import lawfinder as lf_mod


def used_vars(tree):
    if tree[0] == "x":
        return {tree[1]}
    if tree[0] == "c":
        return set()
    s = set()
    for ch in tree[1:]:
        s |= used_vars(ch)
    return s


def main():
    print("=== LAWFINDER v1.0 FORENSIC AUDIT PROBE (6 DOMAINS) ===")
    lf = lf_mod.LawFinder()

    # 1. Widen-only sigma check & declared-vs-used variable audit
    for name in lf.names():
        law = lf[name]
        assert law.sigma >= law.raw_sigma - 1e-12, f"{name} sigma narrowed!"
        uv = sorted(law.vars[i] for i in used_vars(law.tree))
        unused = [v for v in law.vars if v not in uv]
        print(f"  {name:14s}: sigma {law.raw_sigma:.4g}->{law.sigma:.4g} (widen={law.sigma/law.raw_sigma:.3f}x) | unused={unused}")

    # 2. CRITICAL: Road law (p1-c23) complete physical inversion for EVs (e=1)
    road = lf["road"]
    base_ice = dict(v=20.0, K=0.1, G=0.0, U=0.0, M=1.6, D=0.7, R=0.01, T=20.0, h=0.0, e=0.0)
    base_ev = {**base_ice, "e": 1.0}
    ice_flat = float(road.predict(**base_ice)[0])
    ice_up = float(road.predict(**{**base_ice, "G": 6.0})[0])
    ev_flat = float(road.predict(**base_ev)[0])
    ev_up = float(road.predict(**{**base_ev, "G": 6.0})[0])
    ev_dn = float(road.predict(**{**base_ev, "G": -6.0})[0])
    ev_acc = float(road.predict(**{**base_ev, "K": 0.5})[0])
    print(
        f"\nRoad law EV inversion (v=20 m/s, M=1.6 t):\n"
        f"  ICE: flat={ice_flat:.4f} -> uphill +6%={ice_up:.4f} ({ice_up - ice_flat:+.4f} MJ/km)\n"
        f"  EV : flat={ev_flat:.4f} -> uphill +6%={ev_up:.4f} ({ev_up - ev_flat:+.4f} MJ/km, {(ev_up/ev_flat-1)*100:+.1f}%)\n"
        f"  EV : flat={ev_flat:.4f} -> downhill -6%={ev_dn:.4f} ({ev_dn - ev_flat:+.4f} MJ/km, {(ev_dn/ev_flat-1)*100:+.1f}%)\n"
        f"  EV : K 0.1->0.5 -> {ev_acc:.4f} ({ev_acc - ev_flat:+.4f} MJ/km, {(ev_acc/ev_flat-1)*100:+.1f}%)"
    )
    assert ice_up > ice_flat and ev_up < ev_flat * 0.6 and ev_dn > ev_flat * 1.9 and ev_acc < ev_flat * 0.4

    # Cross-check with real archived exam_road_reported.json EV strata
    er = json.loads((REPO / "models/lawfinder/training/results/exam_road_reported.json").read_text())
    ved_ev = er["strata"]["VED_EV"]["law"]
    epa_ev = er["strata"]["EPA_EV"]["law"]
    print(
        f"  Archived exam VED_EV (n={ved_ev['n']}): rho={ved_ev['rho']:.3f}, R2={ved_ev['r2']:.3f}, MdAPE={ved_ev['mdape']:.2f}%\n"
        f"  Archived exam EPA_EV (n={epa_ev['n']}): rho={epa_ev['rho']:.3f}, R2={epa_ev['r2']:.3f}, MdAPE={epa_ev['mdape']:.2f}%"
    )
    assert ved_ev["rho"] < -0.3 and ved_ev["r2"] < -1.0 and epa_ev["r2"] < -5.0

    # 3. CRITICAL: Bldg law (b2-c13) negative heating slope for Y >= 2022 + unused cooling C
    bldg = lf["bldg"]
    b_2022_h1 = float(bldg.predict(H=1.0, C=0.5, A=2.0, Y=2022.0)[0])
    b_2022_h5 = float(bldg.predict(H=5.0, C=0.5, A=2.0, Y=2022.0)[0])
    b_2025_h1 = float(bldg.predict(H=1.0, C=0.5, A=2.0, Y=2025.0)[0])
    b_2025_h5 = float(bldg.predict(H=5.0, C=0.5, A=2.0, Y=2025.0)[0])
    b_c5 = float(bldg.predict(H=1.0, C=5.0, A=2.0, Y=2025.0)[0])
    print(
        f"\nBldg law post-2022 inversion:\n"
        f"  Y=2022: dE/dH = {(b_2022_h5 - b_2022_h1)/4:+.2f} | Y=2025: dE/dH = {(b_2025_h5 - b_2025_h1)/4:+.2f} | C 0.5->5.0 diff={b_c5 - b_2025_h1:.4f}"
    )
    assert b_2022_h5 < b_2022_h1 and b_2025_h5 < b_2025_h1 and abs(b_c5 - b_2025_h1) == 0.0

    # 4. CRITICAL: Grid nadir NaN pole at E <= 66.89 GWs (triggered by LawFinder's own grid_inertia!)
    gi = lf["grid_inertia"]
    gn = lf["grid_nadir"]
    e_low = float(gi.predict(D=30.0, s=0.9)[0])
    nad_low = float(gn.predict(L=1.0, E=e_low, D=30.0)[0])
    nad_edge = float(gn.predict(L=1.0, E=66.89, D=30.0)[0])
    print(f"\nGrid composition: grid_inertia(D=30, s=0.9) = {e_low:.2f} GWs -> grid_nadir = {nad_low} (edge E=66.89 -> {nad_edge:.2f} Hz)")
    assert e_low < 66.89 and np.isnan(nad_low) and nad_edge > 3.0

    # 5. Fire law ignores W/M/R0 & P=0 -> -inf; Water law Q/P > 1 in humid regimes
    f0 = float(lf["fire"].predict(P=1.0, W=5.0, M=0.1, R0=10.0)[0])
    f1 = float(lf["fire"].predict(P=1.0, W=50.0, M=0.01, R0=100.0)[0])
    f_p0 = float(lf["fire"].predict(P=0.0, W=5.0, M=0.1, R0=10.0)[0])
    w_humid = float(lf["water"].predict(phi=0.1, fs=0.5, sync=0.0, S=0.2)[0])
    print(f"Fire law W/M/R0 diff={f1 - f0:.6f}, P=0 -> {f_p0} | Water humid phi=0.1 -> Q/P={w_humid:.4f} (> 1.0)")
    assert f0 == f1 and np.isinf(f_p0) and w_humid > 1.3

    # 6. Splits disjointness (water, fire, grid, bldg; note road absent from splits.json)
    sp = json.loads((REPO / "models/lawfinder/splits.json").read_text())
    for dom in ("water", "fire", "grid", "bldg"):
        ov = set(map(str, sp[dom]["dev"])) & set(map(str, sp[dom]["exam"]))
        assert len(ov) == 0, f"Overlap in {dom}!"
    assert "road" not in sp
    print("Splits check: 0 dev/exam overlap across water, fire, grid, bldg (road not listed in splits.json).")

    # 7. Exact ratchet arithmetic replay across all 6 *_ratchet.json files (238/238 decisions)
    res_dir = REPO / "models/lawfinder/training/results"
    total = matched = 0
    for rf in sorted(res_dir.glob("*_ratchet.json")):
        d = json.loads(rf.read_text())
        margin, dev_tol = float(d["margin"]), float(d["dev_tol"])
        by_id = {r["id"]: r for r in d["rungs"]}
        for h in d["history"]:
            ver = h["frozen_set"].replace("val_", "")
            inc_sim = inc_dev = None
            for dec in h["decisions"]:
                r = by_id[dec["id"]]
                sc = r["scores"][ver] if ver in r["scores"] else {"sim": r["sim"], "dev": r["dev"]}
                s_val, d_val = float(sc["sim"]), float(sc["dev"])
                calc = "kept" if (inc_sim is None or (s_val > inc_sim + margin and d_val <= inc_dev * dev_tol)) else "rejected"
                assert calc == dec["decision"], f"Mismatch in {rf.name} {h['frozen_set']} {dec['id']}"
                total += 1
                matched += 1
                if dec["decision"] == "kept":
                    inc_sim, inc_dev = s_val, d_val
    print(f"Ratchet replay: {matched}/{total} decisions match across all 6 domains and 11 physics passes.")
    print("LAWFINDER PROBE COMPLETE.")


if __name__ == "__main__":
    main()
