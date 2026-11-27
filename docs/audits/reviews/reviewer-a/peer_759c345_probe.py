"""Second-pass independent cross-check probes for reviewer-b's 759c345 audits:
TripDetective v2.1, SunDoctor v2.1, and SwellCast v1.1.

Run all checks:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_759c345_probe.py
Or a single target:
    OPENBLAS_NUM_THREADS=1 python3 -B docs/audits/reviews/reviewer-a/peer_759c345_probe.py sundoctor
"""
import gzip
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]


def check_sundoctor():
    print("=== SECOND-PASS CHECK: SunDoctor v2.1 ===")
    sys.path.insert(0, str(REPO / "models/sundoctor"))
    import sundoctor

    doc = sundoctor.SunDoctor()
    ca_cfg = dict(lat=38.996, lon=-122.134, tilt=25, azimuth=180, dc_kw=37.2, tz="US/Pacific", n_strings=6)
    co_cfg = dict(
        lat=40.3864, lon=-104.5512, mount="tracker", max_angle=60, gcr=0.35, backtrack=False, dc_kw=592.25, tz="US/Mountain"
    )

    for name, fname, cfg in [
        ("CA_2107", "pvdaq2107_inverter11_2019H2.csv", ca_cfg),
        ("CO_9068", "pvdaq9068_module2_2018winter.csv", co_cfg),
    ]:
        df_full = pd.read_csv(REPO / "models/sundoctor/examples" / fname)
        tcol = df_full.columns[0]
        ts = pd.to_datetime(df_full[tcol])
        dates = sorted(ts.dt.date.unique())[:30]
        df30 = df_full[ts.dt.date.isin(dates)].copy()
        ts30 = pd.to_datetime(df30[tcol])

        # 1. Single adjacent-day swap (day 14 <-> day 15) in an otherwise sorted month
        d14, d15 = dates[14], dates[15]
        df_swap = pd.concat(
            [df30[ts30.dt.date < d14], df30[ts30.dt.date == d15], df30[ts30.dt.date == d14], df30[ts30.dt.date > d15]],
            ignore_index=True,
        )
        # 2. Single late-arriving day moved to end (29 of 30 days sorted)
        df_late = pd.concat([df30[ts30.dt.date != dates[0]], df30[ts30.dt.date == dates[0]]], ignore_index=True)
        # 3. 5-hour midday gap (10:00-15:00 local) on ac_power_kw
        df_mid = df30.copy()
        hrs = ts30.dt.hour
        df_mid.loc[(hrs >= 10) & (hrs < 15), "ac_power_kw"] = np.nan
        # 4. All-NaN ac_power_kw
        df_nan = df30.copy()
        df_nan["ac_power_kw"] = np.nan

        with tempfile.TemporaryDirectory() as td:
            res = {}
            for label, dff in [
                ("orig", df30),
                ("swap_d14_d15", df_swap),
                ("day0_at_end", df_late),
                ("midday_10_15", df_mid),
                ("all_nan_P", df_nan),
            ]:
                p = Path(td) / f"{label}.csv"
                dff.to_csv(p, index=False)
                for home in (False, True):
                    r = doc.diagnose_csv(str(p), home=home, **cfg)
                    s = r["summary"]
                    losses = {k: round(float(v), 2) for k, v in s["loss_pct_of_possible"].items() if v >= 0.05}
                    res[(label, home)] = (s["days"], float(s["kwh_made"]), losses)
                    print(
                        f"  {name} home={str(home):5s} {label:13s}: days={s['days']:2d}, "
                        f"kWh={s['kwh_made']:9.2f}, losses={losses}"
                    )

            # Assert key second-pass findings
            assert res[("day0_at_end", False)][0] == 0 and res[("day0_at_end", True)][0] == 0
            assert res[("swap_d14_d15", False)][0] == 29 and res[("swap_d14_d15", True)][0] == 29
            assert res[("all_nan_P", False)][0] == 0 and res[("all_nan_P", True)][0] == 0
            if name == "CA_2107":
                assert res[("midday_10_15", False)][2]["outage"] > 25.0
                assert res[("midday_10_15", True)][2]["outage"] > 30.0
                assert res[("swap_d14_d15", True)][2]["outage"] > 3.0
            else:
                assert res[("midday_10_15", False)][2]["shading"] > 10.0
                assert res[("midday_10_15", True)][2]["shading"] > 14.0
                assert res[("swap_d14_d15", True)][2]["snow"] > 1.0


def check_tripdetective():
    print("\n=== SECOND-PASS CHECK: TripDetective v2.1 ===")
    sys.path.insert(0, str(REPO / "models/tripdetective"))
    import tripdetective as td
    from demo import load_example

    # 1. Partial power dropout progression on longest ZTBus leg (index 8)
    bus = td.TripDetectiveBus()
    with np.load(REPO / "models/tripdetective/examples/ZTBus_exam_mission.npz") as z:
        n = len(json.loads(str(z["info"]))["legs"])
        legs = [{k.removeprefix(f"leg{i}_"): z[k] for k in z.files if k.startswith(f"leg{i}_")} for i in range(n)]
    k = int(np.argmax([len(l["v"]) for l in legs]))
    rest = [j for j in range(n) if j != k]
    base = np.array([r["mass_kg"] for r in bus.read_mission(legs)])

    prev_other = 0.0
    for frac in (0.10, 0.25, 0.50, 0.75, 1.00):
        mod = [dict(l) for l in legs]
        pw = mod[k]["power_W"].astype(float).copy()
        pw[: int(len(pw) * frac)] = np.nan
        mod[k]["power_W"] = pw
        m = np.array([r["mass_kg"] for r in bus.read_mission(mod)])
        d_leg = float(m[k] - base[k])
        d_other = float(m[rest[0]] - base[rest[0]])
        print(f"  ZTBus leg {k} {int(frac*100):3d}% NaN power: d_leg={d_leg:+8.2f} kg | d_other13={d_other:+7.2f} kg")
        assert d_other > prev_other
        prev_other = d_other

    # Verify training/busbank.py uses the same np.nan_to_num(elec_power) zero-fill
    bb_txt = (REPO / "models/tripdetective/training/busbank.py").read_text()
    assert "np.nan_to_num(np.asarray(a['elec_power'], float))" in bb_txt
    print("  Verified training/busbank.py:169 zero-fills missing elec_power via np.nan_to_num")

    # 2. Car reader empty / all-NaN guardrails
    car = td.TripDetective()
    _, trips = load_example(str(REPO / "models/tripdetective/examples/VED_436.npz"))
    r0 = car.read_car(trips)
    r_nan_v = car.read_car([{**tr, "v": np.full_like(tr["v"], np.nan, dtype=float)} for tr in trips])
    r_zero_v = car.read_car([{**tr, "v": np.zeros_like(tr["v"], dtype=float)} for tr in trips])
    r_all_nan = car.read_car(
        [
            {kk: np.full_like(vv, np.nan, dtype=float) if isinstance(vv, np.ndarray) and vv.dtype.kind == "f" else vv for kk, vv in tr.items()}
            for tr in trips
        ]
    )
    print(
        f"  Car VED_436: base={r0['curb_kg']:.2f} kg ({r0['architecture']}) | "
        f"all_nan_v={r_nan_v['curb_kg']:.2f} kg ({r_nan_v['architecture']}) | "
        f"all_zero_v={r_zero_v['curb_kg']:.2f} kg ({r_zero_v['architecture']}) | "
        f"all_nan_f={r_all_nan['curb_kg']:.2f} kg ({r_all_nan['architecture']})"
    )
    assert abs(r_nan_v["curb_kg"] - r_zero_v["curb_kg"]) < 1e-6 and np.isfinite(r_all_nan["curb_kg"])


def check_swellcast():
    print("\n=== SECOND-PASS CHECK: SwellCast v1.1 ===")
    sys.path.insert(0, str(REPO / "models/swellcast"))
    import swellcast as sc

    # 1. Field-specific sentinel check across all 3 packaged raw NDBC 51004 (2018) files
    raw_dir = REPO / "models/swellcast/examples/raw"
    counts = {}
    for fn in ("51004w2018.txt.gz", "51004d2018.txt.gz", "51004j2018.txt.gz"):
        lines = gzip.open(raw_dir / fn, "rt", errors="ignore").read().splitlines()
        hdr = [l for l in lines if l.startswith("#") or l[:2] in ("YY", "YYYY")][0].replace("#", "").split()
        nt = 5 if hdr[4].lower() in ("mm", "mn") else 4
        n_99 = 0
        vmax = -1.0
        for l in lines:
            if not l or l.startswith("#") or l[0].isalpha():
                continue
            p = l.split()
            if len(p) != len(hdr):
                continue
            v = np.array(p[nt:], float)
            n_99 += int(((v >= 98.9) & (v <= 99.01)).sum())
            fin = v[v < 998.9]
            if fin.size:
                vmax = max(vmax, float(fin.max()))
        counts[fn] = (n_99, vmax)
        print(f"  {fn:20s}: cells_in_[98.9,99.01]={n_99:4d}, max_non999={vmax:.3f}")
    assert counts["51004d2018.txt.gz"][0] == 3445
    assert counts["51004j2018.txt.gz"] == (0, 98.0)

    # 2. All-NaN vs all-zero spectrogram guardrail on public SwellCast.read()
    sm = sc.SwellCast()
    x0 = np.load(REPO / "models/swellcast/examples/41047_danielle_2010.npz")["x"]
    r_nan = sm.read(np.full_like(x0, np.nan, dtype=np.float32))
    r_zero = sm.read(np.zeros_like(x0, dtype=np.float32))
    print(
        f"  read(all_nan) : detected={r_nan['detected']:.0f}, p_swell={r_nan['p_swell']:.4f}, "
        f"D_km={r_nan['D_km']:.1f}, hs={r_nan['hs']:.2f} m, dir={r_nan['dir']:.1f} deg"
    )
    print(
        f"  read(all_zero): detected={r_zero['detected']:.0f}, p_swell={r_zero['p_swell']:.4f}, "
        f"D_km={r_zero['D_km']:.1f}, hs={r_zero['hs']:.2f} m"
    )
    assert r_nan["detected"] == 1.0 and r_nan["p_swell"] > 0.50
    assert r_zero["detected"] == 0.0 and r_zero["p_swell"] < 0.01


def main():
    targets = set(sys.argv[1:]) or {"sundoctor", "tripdetective", "swellcast"}
    if "sundoctor" in targets:
        check_sundoctor()
    if "tripdetective" in targets:
        check_tripdetective()
    if "swellcast" in targets:
        check_swellcast()
    print("\nALL SECOND-PASS CHECKS PASSED.")


if __name__ == "__main__":
    main()
