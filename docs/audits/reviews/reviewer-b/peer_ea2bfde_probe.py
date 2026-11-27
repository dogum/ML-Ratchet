"""Cross-check archived examples and public APIs. Run each mode in a fresh Python process."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODELS = Path(__file__).resolve().parents[4] / "models"


def firepilot(root):
    import firepilot as fp

    net = fp.ValueNet(str(root / "firepilot_weights.npz"))
    afternoon = [fp.work_clock(dt.datetime(2020, 1, 1, 12) + dt.timedelta(minutes=m), 0)[1][-1]
                 for m in range(181)]
    print("12:00–15:00 local, minute sweep", min(afternoon), max(afternoon))
    assert np.allclose(afternoon, 34)

    # MTBS IDs encode the nominal event longitude. Use it for both locations as a proxy.
    results = root / "training/results"
    exam_ids = json.loads((results / "exam_summary_p2.json").read_text())["stage_b_fires"]
    dev_cases = json.loads((results / "fidelity_dev_p2.json").read_text())["cases"]
    for label, ids in (("exam", exam_ids), ("dev fidelity", [case["fire_id"] for case in dev_cases])):
        work = []
        for fid in ids:
            lon = -int(fid[7:13]) / 1000
            assert -180 <= lon < -60, fid
            utc = pd.Timestamp("2020-01-01 13:30") - pd.Timedelta(hours=lon / 15)
            work.append(fp.work_clock(utc.floor("h"), lon)[1][-1])
        print(label, "ID-longitude approximation, count/min/max", len(ids), min(work), max(work))
        assert np.allclose(work, 34)

    def plan(ctx):
        chosen, info = fp.plan_value(ctx, net, np.random.default_rng(0))
        return np.array([info["pred"], ctx.plan_eval(chosen)[0]["s_proxy"]])

    for file in sorted((root / "examples").glob("*.npz")):
        with np.load(file) as z:
            store = dict(z)
        rec = json.loads(file.with_suffix(".json").read_text()) | {"rate": 300.0}
        base = fp.Context.from_store(store, rec)
        # Start at 06:00 local solar time. Context consumes the supplied relative clock.
        t, w = fp.work_clock(dt.datetime(2020, 1, 1, 6), 0)
        morning = fp.Context.from_store(store | {"clock_t": t, "clock_w": w}, rec)
        actual = plan(morning)
        morning._scn[6] = base._scn[6]
        restored = plan(morning)
        print(file.stem, "W", base.clock[1][-1], w[-1], "pred/proxy",
              "archived", plan(base), "morning", actual, "morning restoring feature 6", restored)
        assert np.isclose(w[-1], 36) and np.isclose(base.clock[1][-1], 34)
        assert abs(actual[0] - restored[0]) > 0.05


def cellsense(root):
    import cellsense as cs
    import realexam
    import realdata

    nmc = cs.CellSense(chemistry="nmc811")
    with np.load(root / "examples/m50t_W9.npz") as z:
        q, ref = z["Q"][[5, 0]]
    v = cs.VGRID_NMC
    for end in [2.50, 2.5009, 2.5011, *np.linspace(2.51, 2.60, 10)]:
        keep = v > end
        endpoint = np.interp(end, v[::-1], q[::-1])
        curve = cs.q_of_v_nmc(np.r_[v[keep], end], np.r_[q[keep], endpoint])
        pred = nmc.diagnose(curve, ref)
        print(f"{end:.4f} V", np.isfinite(curve[-1]), f"{pred['q_total']:.6f} +/- {pred['q_total_sig']:.6f} Ah")
        assert np.isfinite(curve[-1]) == (end < 2.501)
    # Exercise the real selector on eligible fixture rows including all calibration IDs.
    ids = set(realexam.CA.CAL)
    continuation = set(realdata.CONT) | {v[0] for v in realdata.CONT.values()}
    assert not ids & continuation
    others = set(realdata.CONT) | {"b3c33"}
    fixture = {k: dict(cyc_idx=np.arange(10), Qdlin=np.ones((10, 12)), Tdlin=np.full((10, 12), 25.),
                       summary=dict(QDischarge=np.ones(10), IR=np.full(10, .02)), batch=k[:2],
                       cycle_life_pub=100, policy="fixture") for k in ids | others}
    dev = {row["cell"] for row in realexam.readings(fixture, only=ids)}
    exam = {row["cell"] for row in realexam.readings(fixture)}
    print("selector fixture, dev/exam/overlap", len(dev), len(exam), len(dev & exam))
    assert dev == ids and exam == others


def basindetective(root, old_flow_dir):
    import basindetective as bd

    import torch
    from train import MLP
    from basindetective.features import EPSQ
    model = bd.load()
    real = []
    fields = ["P", "T_mean", "PET"]

    def predict(cal, tgt):
        changes = bd.predict(model, cal[fields], cal.Q_mm.values, tgt[fields])["changes"]
        return np.array([changes[k]["mean"] for k in ("d_logq", "d_rr")])

    for file in sorted((root / "examples").glob("*_cal.csv")):
        name = file.name.removesuffix("_cal.csv")
        late = pd.read_csv(file, index_col=0, parse_dates=True)
        early = pd.read_csv(file.with_name(name + "_tgt.csv"), index_col=0, parse_dates=True)
        print(name, "late to old", predict(late, early), "identical", predict(late, late))
        real.append(bd.featurize(late[fields], late.Q_mm.values, early[fields])[0])
        assert "Q_mm" not in early.columns
        # Equal length four-water-year windows drawn from the observed-flow period.
        a = late.loc["1999-10-01":"2003-09-30"]
        b = late.loc["2004-10-01":"2008-09-30"]
        fwd, rev = predict(a, b), predict(b, a)
        truth = np.log((b.Q_mm.mean() + EPSQ) / (a.Q_mm.mean() + EPSQ))
        print(" observed split, forward/reverse/sum (d_logq, d_rr)", fwd, rev, fwd + rev,
              "observed forward d_logq", truth)
        if old_flow_dir:
            q = pd.read_csv(old_flow_dir / (name + "_flow.csv"), index_col=0, parse_dates=True)
            assert q.index.is_unique
            early["Q_mm"] = q.Q_mm.reindex(early.index)
            assert early.Q_mm.notna().all()
            print(" observed old to late", predict(early, late))

    # Same mixed batch and seeded synthetic inputs as reviewer A. Isolate the source of the f32 maximum.
    first = model.members[0]
    random = first.W["x_med"] + np.random.default_rng(42).standard_normal((16, len(first.names))) * first.W["x_iqr"]
    x = np.vstack([real, random])
    worst = np.zeros(2)
    with torch.no_grad():
        for m in model.members:
            net = MLP(len(m.names), len(m.targets), m.W["w0"].shape[0], m.nl, 0.).eval()
            for i, layer in enumerate([*net.hs, net.out]):
                wk, bk = (f"w{i}", f"b{i}") if i < m.nl else ("wo", "bo")
                layer.weight.copy_(torch.as_tensor(m.W[wk]))
                layer.bias.copy_(torch.as_tensor(m.W[bk]))
            err = np.abs(net(torch.tensor(m.norm(x), dtype=torch.float32)).numpy() - m.raw(x))
            worst = np.maximum(worst, [err[:len(real)].max(), err[len(real):].max()])
    print("float32 raw max abs error, real/synthetic", worst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("firepilot", "cellsense", "basindetective"))
    parser.add_argument("--old-flow-dir", type=Path, help="dated <camels_id>_flow.csv files with daily Q_mm")
    args = parser.parse_args()
    root = MODELS / args.model
    sys.path[:0] = [str(root), str(root / "training")]
    if args.model == "firepilot":
        firepilot(root)
    elif args.model == "cellsense":
        cellsense(root)
    else:
        basindetective(root, args.old_flow_dir)
