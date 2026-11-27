"""Second-pass independent cross-check probes for reviewer-b's 69873b7 audits:
BeatNet v1.0, TransitNet v1.0, and StarQuake v1.1.

Run all checks:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_69873b7_probe.py
Or a single target:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_69873b7_probe.py beatnet
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]


def check_beatnet():
    print("=== SECOND-PASS CHECK: BeatNet v1.0 ===")
    cal = json.loads((REPO / "models/beatnet/training/calib_v1.json").read_text())
    ex = json.loads((REPO / "models/beatnet/training/exam.json").read_text())
    dev_ids = cal["dev_records"]
    pr = ex["ds2"]["per_record"]
    ds2_ids = sorted(int(k) for k in pr.keys())
    assert 201 in dev_ids and 202 in ds2_ids and not (set(dev_ids) & set(ds2_ids))
    print(f"Dev IDs ({len(dev_ids)}): {dev_ids}; DS2 IDs ({len(ds2_ids)}): direct overlap=0; shared patient 201/202 confirmed.")

    def metrics_from_cm(cm):
        cm = np.asarray(cm, dtype=int)
        n = int(cm.sum())
        tp = np.diag(cm)
        fn = cm.sum(1) - tp
        fp = cm.sum(0) - tp
        se = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        ppv = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        f1 = np.where(se + ppv > 0, 2 * se * ppv / (se + ppv), 0.0)
        return dict(n=n, acc=float(tp.sum() / n), macro_f1=float(f1.mean()), sv_f1=float(0.5 * (f1[1] + f1[2])), f1=f1.tolist())

    cm_all = sum(np.asarray(v["cm"]) for v in pr.values())
    cm_no202 = sum(np.asarray(v["cm"]) for k, v in pr.items() if str(k) != "202")
    m_all = metrics_from_cm(cm_all)
    m_no = metrics_from_cm(cm_no202)
    assert abs(m_all["macro_f1"] - ex["ds2"]["beatnet"]["macro_f1"]) < 1e-6
    print(
        f"  DS2 all (22 rec, n={m_all['n']}): acc={m_all['acc']:.5f}, macro_f1={m_all['macro_f1']:.5f}, "
        f"sv_f1={m_all['sv_f1']:.5f}, S_f1={m_all['f1'][1]:.4f}, V_f1={m_all['f1'][2]:.4f}"
    )
    print(
        f"  DS2 excl 202 (21 rec, n={m_no['n']}): acc={m_no['acc']:.5f}, macro_f1={m_no['macro_f1']:.5f}, "
        f"sv_f1={m_no['sv_f1']:.5f}, S_f1={m_no['f1'][1]:.4f}, V_f1={m_no['f1'][2]:.4f}"
    )
    print(
        f"  Delta (excl 202 - all): acc={m_no['acc'] - m_all['acc']:+.5f}, "
        f"macro_f1={m_no['macro_f1'] - m_all['macro_f1']:+.5f}, sv_f1={m_no['sv_f1'] - m_all['sv_f1']:+.5f}"
    )
    assert m_no["macro_f1"] > m_all["macro_f1"] and m_no["sv_f1"] > m_all["sv_f1"]


def check_transitnet():
    print("\n=== SECOND-PASS CHECK: TransitNet v1.0 ===")
    sys.path.insert(0, str(REPO / "models/transitnet"))
    if "pipeline" in sys.modules:
        del sys.modules["pipeline"]
    from transitnet import TransitNet

    api = TransitNet()
    for f in sorted((REPO / "models/transitnet/examples").glob("*.npz")):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        secs = [{k: z[f"{k}{i}"] for k in ("t", "f", "c1", "c2")} for i in range(m["n_sectors"])]
        dur_d = m["duration_hours"] / 24.0
        print(f"  Candidate {f.stem}:")
        for frac in (0.0, 0.25, 0.50, 0.75, 0.90, 1.00):
            mod = []
            for s in secs:
                t = s["t"]
                ph = (t - m["epoch"] + m["period"] / 2) % m["period"] - m["period"] / 2
                keep = np.abs(ph) >= frac * dur_d
                mod.append({k: v[keep] for k, v in s.items()})
            out = api.read(mod, m["period"], m["epoch"], m["duration_hours"], star=m["star"])
            top = max(("planet", "eb", "blend", "nothing"), key=lambda c: out[f"p_{c}"])
            print(f"    excise |ph|<{frac:.2f}*dur: top={top:7s}({out['p_'+top]:.4f}) p_noth={out['p_nothing']:.4f} snr={out['snr']:8.2f}")
            if frac == 0.25:
                assert top != "nothing" and out["snr"] > 20.0
            if frac == 0.50:
                assert top == "nothing" and np.isfinite(out["snr"])
            if frac >= 0.90:
                assert top == "nothing" and out["snr"] == -np.inf

        # All-NaN flux raises ValueError cleanly
        raised = False
        try:
            api.read([{"t": s["t"], "f": np.full_like(s["f"], np.nan)} for s in secs], m["period"], m["epoch"], m["duration_hours"], star=m["star"])
        except ValueError:
            raised = True
        assert raised, "All-NaN flux should raise ValueError"
        print("    all_nan_flux: properly raised ValueError")


def check_starquake():
    print("\n=== SECOND-PASS CHECK: StarQuake v1.1 ===")
    sys.path.insert(0, str(REPO / "models/starquake"))
    if "pipeline" in sys.modules:
        del sys.modules["pipeline"]
    import pipeline as sq_pl
    from starquake import StarQuake

    sq = StarQuake()
    n90 = int(90 * 86400 / sq_pl.DT)
    z = dict(np.load(REPO / "models/starquake/examples/2695975.npz"))
    full = sq_pl.load_kepler(z)
    seg = full[20000 : 20000 + n90]
    ok_mask = np.isfinite(seg)

    # 1. Pure Gaussian white-noise amplitude sweep on exact Kepler cadence mask
    rng = np.random.default_rng(42)
    wn = rng.standard_normal(len(seg))
    print(f"  KIC 2695975 real 90d std={np.nanstd(seg):.1f} ppm; white-noise amplitude sweep:")
    for amp in (10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 3000.0):
        y = np.where(ok_mask, wn * amp, np.nan)
        r = sq.read_ppm(y, teff=5800.0)
        print(f"    white_noise {amp:6.1f} ppm: detected={str(r['detected']):5s} p_osc={r['p_osc']:.5f}")
        if amp <= 200.0:
            assert not r["detected"]
        if amp >= 500.0:
            assert r["detected"] and r["p_osc"] > 0.85

    # 2. Full-record load_kepler with 90d of nonzero quality-flagged (q=16) boundary rows
    last_cad = int(z["cad"][-1])
    z_qpad = {
        "cad": np.concatenate([z["cad"], np.arange(last_cad + 1, last_cad + 1 + n90, dtype=z["cad"].dtype)]),
        "flux": np.concatenate([z["flux"], np.full(n90, float(np.nanmedian(z["flux"])), dtype=z["flux"].dtype)]),
        "quarter": np.concatenate([z["quarter"], np.full(n90, z["quarter"][-1], dtype=z["quarter"].dtype)]),
        "q": np.concatenate([z["q"], np.full(n90, 16, dtype=z["q"].dtype)]),
    }
    full_qpad = sq_pl.load_kepler(z_qpad)
    r0 = sq.read_ppm(full, teff=4800.0)
    rq = sq.read_ppm(full_qpad, teff=4800.0)
    print(
        f"  Full KIC 2695975 load_kepler + 90d q=16 boundary: len {len(full)}->{len(full_qpad)}, "
        f"days {r0['days']:.2f}->{rq['days']:.2f}, dnu {r0['dnu']:.4f}->{rq['dnu']:.4f}, mass {r0['mass']:.3f}->{rq['mass']:.3f}"
    )
    assert len(full_qpad) > len(full) and abs(rq["days"] - r0["days"]) < 1e-6 and abs(rq["mass"] - r0["mass"]) > 0.05


def main():
    targets = set(sys.argv[1:]) or {"beatnet", "transitnet", "starquake"}
    if "beatnet" in targets:
        check_beatnet()
    if "transitnet" in targets:
        check_transitnet()
    if "starquake" in targets:
        check_starquake()
    print("\nALL SECOND-PASS CHECKS PASSED.")


if __name__ == "__main__":
    main()
