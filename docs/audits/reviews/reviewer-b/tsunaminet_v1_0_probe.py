"""Audit source conditioning, target selection, arrival lead time and release defaults."""

import json
import os
import tempfile
import sys
from pathlib import Path
import numpy as np
import torch

MODEL = Path(__file__).resolve().parents[4] / "models/tsunaminet"
sys.path[:0] = [str(MODEL), str(MODEL / "training")]
scratch = tempfile.TemporaryDirectory()
os.environ.setdefault("ML_RATCHET_WORK", scratch.name)
import tsunaminet as tn
from model import TsunamiNet as TorchNet


def reference(net):
    z = np.load(MODEL / "tsunaminet_weights.npz")
    assert all(np.isfinite(z[k]).all() for k in z if k != "meta")
    m = TorchNet(w=[len(c[1]) for c in net.conv], extra=net.meta["extra"])
    state = {}
    for src, dst in [(f"c{i}", f"cnn.{2 * i}") for i in range(4)] + [
        (f"{p}{i}", f"{p}.{2 * i}") for p in ("st", "ctx") for i in range(2)
    ]:
        for key, field in (("w", "weight"), ("b", "bias")):
            state[f"{dst}.{field}"] = torch.from_numpy(z[src + key])
    for i in range(len(net.flow)):
        state[f"flow.layers.{i}.mask"] = torch.from_numpy(z[f"f{i}m"])
        for j in range(3):
            for key, field in (("w", "weight"), ("b", "bias")):
                state[f"flow.layers.{i}.net.{2 * j}.{field}"] = torch.from_numpy(
                    z[f"f{i}{key}{j}"]
                )
    m.load_state_dict(state)
    return m.eval()


def archived():
    rows = json.loads((MODEL / "training/logs/exam_rows.json").read_text())
    events = json.loads((MODEL / "training/logs/events.json").read_text())
    dev, exam = [{e["id"] for e in events if e["split"] == s} for s in ("dev", "exam")]
    assert not dev & exam and {r["id"] for r in rows} == exam
    leads = [(t["ta"] - max(r["input_ta"]) - 69, t) for r in rows for t in r["targets"]]
    print("dev/exam ids", len(dev), len(exam))
    print(
        "nominal leads min median max and nonpositive",
        np.percentile([l for l, _ in leads], [0, 50, 100]),
        sum(l <= 0 for l, _ in leads),
    )
    print(
        "target window already started",
        sum(l <= 10 for l, _ in leads),
        "of",
        len(leads),
    )
    log = json.loads((MODEL / "training/logs/ratchet.json").read_text())
    for frozen in sorted({d["frozen"] for d in log["decisions"]}):
        inc = None
        for d in [d for d in log["decisions"] if d["frozen"] == frozen]:
            scores = next(r for r in log["rungs"] if r["id"] == d["rung"])["scores"][
                frozen
            ]
            passed = inc is None or (
                scores["frozen"]["score"] > inc["frozen"]["score"] + 0.01
                and scores["dev"]["dev_error"] <= 1.05 * inc["dev"]["dev_error"]
            )
            assert passed == d["kept"]
            if passed:
                inc = scores
    print("archived rung decisions", len(log["decisions"]))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    net, gf = tn.TsunamiNet(), tn.GreensFunctions()
    ref = reference(net)
    latent = np.random.default_rng(71).normal(size=(128, 4)).astype("f")
    stats = []
    for path in sorted((MODEL / "examples").glob("*.npz")):
        z = np.load(path)
        m = json.loads(str(z["meta"]))
        rec = {s: (z["eta"][i], z["mask"][i]) for i, s in enumerate(m["sites"])}

        def event(records):
            return gf.event(m["lon"], m["lat"], records)

        e = event(rec)
        evargs = [e[k] for k in ("X", "S", "M")]
        if not e["inputs"]:
            continue
        # At origin only the first sample can have arrived. Check short arrays and
        # fixed-length records with future minutes marked unavailable.
        prefix = {k: (a[:1], b[:1]) for k, (a, b) in rec.items()}
        try:
            short = event(prefix)
            print("one-minute array inputs", short["inputs"])
        except ValueError as exc:
            print("one-minute array", type(exc).__name__, str(exc))
        no_inputs = event(
            {k: (np.zeros_like(a), np.zeros_like(b)) for k, (a, b) in rec.items()}
        )
        assert not no_inputs["inputs"]
        with np.errstate(all="ignore"):
            try:
                no_data = net.posterior(no_inputs)
                print(
                    "empty network Mw and 68 percent band",
                    no_data["mw"],
                    no_data["mw68"],
                )
            except (ValueError, OverflowError) as exc:
                print("empty posterior", type(exc).__name__, str(exc))
        p = net.posterior(e)
        x, s, mask = [torch.as_tensor(a)[None] for a in evargs]
        extra = torch.tensor(tn.nnls_features(e["X"], e["M"]))[None]
        pred = ref.sample(
            x, s, mask, n=len(latent), z=torch.tensor(latent), E=extra
        ).numpy()
        error = np.max(np.abs(pred - net.sample(*evargs, z=latent)))
        end = max(e["input_ta"]) + 69
        suffix = {
            k: (np.where(np.arange(len(a)) > end, a + 0.2, a), b)
            for k, (a, b) in rec.items()
        }
        assert np.array_equal(event(suffix)["X"], e["X"])
        empty = event({k: (np.zeros_like(a), b) for k, (a, b) in rec.items()})
        assert empty["inputs"] == e["inputs"]
        print(
            m["name"],
            "inputs, last input minute, Mw true/full/wave-zero, parity",
            e["inputs"],
            end,
            m["mw"],
            p["mw"],
            net.posterior(empty)["mw"],
            error,
        )
        # Rebuild the stage before outcome-based inclusion, with the archived processed series.
        center = [
            e["frame"]["units"][j]
            for j in (tn.HALF, tn.NCOL + tn.HALF)
            if e["frame"]["units"][j] >= 0
        ]
        kept = set()
        for site, (a, b) in rec.items():
            ta = gf.arrival(site, center)
            if site in e["inputs"] or ta is None or ta > 1250:
                continue
            lo, hi = max(ta - 10, 0), ta + 180
            valid = b[lo : hi + 1].astype(bool)
            pre = a[:lo][b[:lo].astype(bool)]
            if valid.mean() < 0.5 or len(pre) < 8:
                continue
            obs = np.max(np.abs(a[lo : hi + 1][valid]))
            sigma = 1.4826 * np.median(np.abs(pre - np.median(pre))) + 1e-4
            include = obs >= 5 * sigma
            if include:
                kept.add(site)
            gf.cal = net.cal
            f = gf.forecast(e, p, site, lo, hi, valid, widen=net.cal["widen"]["amp"])
            stats.append(
                (
                    include,
                    abs(np.log10(f["median"] / obs)),
                    f["lo68"] <= obs <= f["hi68"],
                )
            )
        assert kept == {t["site"] for t in m["targets"]}
        if m["targets"]:
            t = m["targets"][0]
            b = rec[t["site"]][1][t["a"] : t["b"] + 1].astype(bool)
            args = (e, p, t["site"], t["a"], t["b"], b)
            gf.cal = {}
            bare = gf.forecast(*args)
            gf.cal = net.cal
            calibrated = gf.forecast(*args, widen=net.cal["widen"]["amp"])
            print(
                "first target default / demo [median,lo,hi] m",
                list(bare.values()),
                list(calibrated.values()),
            )
    a = np.asarray(stats)
    for label, rows in (
        ("scored", a[a[:, 0] == 1]),
        ("below 5 sigma", a[a[:, 0] == 0]),
        ("all quality-eligible", a),
    ):
        print(label, len(rows), "mean abs log10 error / coverage", rows[:, 1:].mean(0))
    archived()


if __name__ == "__main__":
    main()
