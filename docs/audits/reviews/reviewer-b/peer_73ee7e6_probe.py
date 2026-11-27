"""Wave 7 independent cross-checks. Use one model argument per process."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

MODELS = Path(__file__).resolve().parents[4] / "models"


def lawfinder(root):
    import lawfinder as lf
    model = lf.LawFinder()
    road = model["road"]
    params = dict(v=20., K=.1, G=0., U=2., M=1.6, D=.7, R=10., T=20., h=0., e=1.)
    log = json.loads((root / "training/results/road_ratchet.json").read_text())
    counts = {}
    for rung in [r for r in log["rungs"] if r["id"] == log["released"] or r["version"] == "p2"]:
        law = lf.Law(road.d | {"tree": rung["tree"], "consts": rung["consts"]})
        grades = np.array([-2., 0., 2.])
        y = law(**(params | {"G": grades}))
        # Fixed total absolute grade, all three net grades are physically attainable.
        k = law(**(params | {"K": .3})) - law(**params)
        label = ("nonfinite" if not np.isfinite(y).all() else "flat" if np.allclose(y, y[0])
                 else "increases" if np.all(np.diff(y) >= 0)
                 else "decreases" if np.all(np.diff(y) <= 0) else "mixed")
        if rung["version"] == "p2":
            counts[label] = counts.get(label, 0) + 1
        print(rung["id"], label, y, "delta K", k)
    print("p2 response counts", counts, "textbook", road.textbook(**(params | {"G": grades})))
    for powertrain in (0., 1.):
        y = road(**(params | {"G": grades, "e": powertrain}))
        expected_sign = -1 if powertrain else 1
        assert np.all(expected_sign * np.diff(y) > 0)
    home = model["bldg"]
    for year in (2000, 2020, 2022, 2025):
        delta = home(H=2, C=.5, A=2., Y=year) - home(H=1, C=.5, A=2., Y=year)
        print("year / heating slope", year, delta)
    # Survey the intersection of the two published demand ranges, in 1 GW and .01 share steps.
    demand, share = np.meshgrid(np.arange(16., 47.), np.arange(5, 81) / 100)
    inertia = model["grid_inertia"](D=demand, s=share)
    nadir = model["grid_nadir"](L=1., E=inertia, D=demand)
    print("grid points, E below 90, nonfinite nadirs", inertia.size, (inertia < 90).sum(),
          (~np.isfinite(nadir)).sum(), "finite inside nadir E range", np.isfinite(nadir[(inertia >= 90) & (inertia <= 420)]).all())
    assert (~np.isfinite(nadir)).any()


def bearingnet(root):
    from bearingnet import BearingNet
    model = BearingNet()
    dev = set(json.loads((root / "training/bn_devprobe_r2.json").read_text())["dev"])
    exam_cwru, exam_pb = set(), set()
    dev_pb = {"K006", "KA01", "KI03"}  # source manifest in the model card and NOTEBOOK.md
    rows = []
    for f in sorted((root / "examples").glob("*.npz")):
        with np.load(f) as z:
            x, fs, m = z["x"].astype(float), float(z["fs"]), json.loads(str(z["meta"]))
        if "_CWRU_" in f.name:
            exam_cwru.add(int(f.stem.split("_")[2]))
        else:
            exam_pb.add(f.stem.split("_")[2])
        opts = dict(fs=fs, rpm_nominal=m["rpm_nominal"], bearing=m["bearing"])
        whole = model.read(x, **opts)
        one = model.read(x, max_win=1, **opts)
        x[len(x) // 2] = np.nan
        broken = model.read(x, **opts)
        rows.append((m["truth"], one["condition"], whole["condition"], broken["condition"],
                     np.isnan(list(broken["p"].values())).all()))
        print(f.stem, rows[-1], "windows", whole["n_windows"], "p(N) single/full",
              round(one["p"]["N"], 4), round(whole["p"]["N"], 4))
    print("accurate single/default", sum(t == a for t, a, *_ in rows), sum(t == b for t, _, b, *_ in rows))
    print("healthy corrupted with NaN probabilities", sum(d == "N" and bad for _, _, _, d, bad in rows))
    assert all(d == "N" and bad for _, _, _, d, bad in rows)
    print("CWRU dev and packaged exam file IDs", sorted(dev), sorted(exam_cwru), "overlap", dev & exam_cwru)
    print("Paderborn packaged bearings, dev overlap", sorted(exam_pb), dev_pb & exam_pb)
    assert not dev & exam_cwru and not dev_pb & exam_pb


def plantdetective(root):
    import plantdetective as pd
    model = pd.Detective()
    for file in sorted((root / "examples").glob("*.npz")):
        with np.load(file) as z:
            # One complete year isolates within-window changes from year-selection changes.
            year = pd._local(z["t"], 0)[2]
            complete = [year == yr for yr in np.unique(year) if (year == yr).sum() in (8760, 8784)]
            # Use the most complete measured calendar year (earliest breaks ties).
            use = max(complete, key=lambda sel: np.isfinite(z["x"][sel]).mean())
            x, t, pr, gas = [z[k][use].copy() for k in ("x", "t", "price", "gas")]
            print(file.stem, "year", year[use][0], "measured fraction", np.isfinite(x).mean())
            ctx = json.loads(str(z["meta"]))["context"]
        order = np.arange(len(t))
        moved = np.r_[order[24:], order[:24]]
        gap = x.astype(float).copy()
        # Remove one known, interior operating hour each day where both neighbours are also operating.
        candidates = np.flatnonzero((x[1:-1] > .5) & (x[:-2] > .5) & (x[2:] > .5)) + 1
        _, first = np.unique(candidates // 24, return_index=True)
        gap[candidates[first]] = np.nan
        single = x.astype(float).copy()
        if len(first):
            single[candidates[first[0]]] = np.nan
        for label, values, idx in (("base", x, order), ("day 0 at end", x, moved),
                                   ("stable resort", x, moved[np.argsort(t[moved], kind="stable")]),
                                   ("one missing on-hour", single, order), ("missing on-hours", gap, order)):
            pred = model.infer(values[idx], t[idx], pr[idx], gas[idx], ctx)
            X = pd.window_vector(values[idx], t[idx], pr[idx], gas[idx], ctx)[None]
            off = pd.count_offsets(X, model.Ws[0]["cskip"])[0]
            j = pd.REG.index("starts_yr")
            a = np.array([pred["estimates"][k]["value"] for k in ("cf", "starts_yr")])
            print(file.stem, label, "CF/starts", a, "count offset (log1p starts)", off[j], "hours removed", int(np.sum(np.isfinite(x) & ~np.isfinite(values))))
            if label == "base":
                base = a
            if label == "missing on-hours":
                assert a[1] > base[1]
            if label == "stable resort":
                np.testing.assert_array_equal(a, base)
    for filename in ("ratchet.json", "ratchet_v1_1.json"):
        log = json.loads((root / "training" / filename).read_text())
        inc = None
        for row in log["rungs"]:
            keep = inc is None or (row["val"] > row.get("incumbent_val_on_this_set", inc["val"]) + log["margin"]
                                   and row["dev"] >= inc["dev"] - log["dev_tol"])
            assert keep == row["kept"], row["rung"]
            if keep:
                inc = row
        print(filename, len(log["rungs"]), "decisions matched")


if __name__ == "__main__":
    funcs = dict(lawfinder=lawfinder, bearingnet=bearingnet, plantdetective=plantdetective)
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=funcs)
    name = parser.parse_args().model
    root = MODELS / name
    sys.path.insert(0, str(root))
    funcs[name](root)
