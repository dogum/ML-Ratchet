"""Check the raw trace API, both shipped networks and the frozen GridPulse log."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

MODEL = Path(__file__).resolve().parents[4] / "models/gridpulse"
sys.path[:0] = [str(MODEL), str(MODEL / "training")]
from gridpulse import GridPulse, onset
from net import Net


def cutoff(gp):
    t, f = np.loadtxt(MODEL / "examples/trace_example.csv", delimiter=",", skiprows=1).T
    i = onset(f)
    n = 10
    for missing in (False, True):
        a = f.copy()
        if missing:
            a[i + n - 1 : i + n + 1] = np.nan
        b = a.copy()
        b[i + n + 1] += 0.05
        np.testing.assert_array_equal(a[: i + n + 1], b[: i + n + 1])
        before, after = [gp.read(t, x, N=n, onset_index=i) for x in (a, b)]
        print(
            "missing",
            missing,
            {
                k: (before[k], after[k])
                for k in ("loss_MW", "E_GWs", "nadir_Hz", "settle_Hz")
            },
        )
        if not missing:
            assert before == after
        else:
            assert abs(before["loss_MW"] - after["loss_MW"]) > 90
    late = f.copy()
    late[i + 25 :] -= 0.25
    print("automatic onset times", [gp.read(t, x, N=n)["onset_t"] for x in (f, late)])
    try:
        gp.read(t[: i + n + 1], f[: i + n + 1], N=n, onset_index=i)
    except ValueError as e:
        print("prefix only", e)


@torch.no_grad()
def parity(gp, z):
    w = gp.W
    hid = [
        w[f"b{i}_w"].shape[0]
        for i in range(sum(k.endswith("_w") and k.startswith("b") for k in w))
    ]
    net = Net(nin=len(w["xm"]), hid=hid).eval()
    state = {}
    for i in range(len(hid)):
        for key, field in (("w", "weight"), ("b", "bias")):
            state[f"body.{2 * i}.{field}"] = torch.from_numpy(w[f"b{i}_{key}"])
    for head in ("mu", "ls"):
        for key, field in (("w", "weight"), ("b", "bias")):
            state[f"{head}.{field}"] = torch.from_numpy(w[f"{head}_{key}"])
    net.load_state_dict(state)
    for n in (10, 60):
        args = (z["w"], np.full(6, n), np.ones(6), np.full(6, 50), z["demand"])
        x = (gp._features(*args) - w["xm"]) / w["xs"]
        mu, ls = net(torch.tensor(x, dtype=torch.float32))
        m, s = gp.forward(*args)
        print(
            n,
            "max |z| / target-space parity mean and widened SD",
            np.max(np.abs(x)),
            np.max(np.abs(m - (mu.numpy() * w["ys"] + w["ym"]))),
            np.max(np.abs(s - ls.exp().numpy() * w["ys"] * w["widen"])),
        )
    # Same 4-64-64-2 architecture and exact GELU as training/ercot_head.py.
    for drop, inertia, load, share, loss in (
        (0.11, 230000, 45000, 0.3, 600),
        (0.25, 180000, 55000, 0.5, 1200),
    ):
        public = gp.ercot_summary(drop, inertia, load, share, loss)
        for label, head, a, b in (
            ("loss_MW", "loss", drop, inertia / 1000),
            ("delta_f_Hz", "dfnad", loss, inertia / 1000),
            ("inertia_GWs", "E", drop, loss),
        ):
            weights = {
                k.split("__")[1]: v
                for k, v in gp.E.items()
                if k.startswith(head + "__")
            }
            net = nn.Sequential(
                nn.Linear(4, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )
            for layer, key in ((0, "b.0"), (2, "b.2"), (4, "o")):
                net[layer].load_state_dict(
                    {
                        f: torch.from_numpy(weights[key + "." + f])
                        for f in ("weight", "bias")
                    }
                )
            x = np.array([np.log(a), np.log(b), np.log(load / 1000), share], np.float32)
            m, ls = net(torch.from_numpy((x - weights["xm"]) / weights["xs"])).numpy()
            ref = [
                m * weights["ys"] + weights["ym"],
                np.exp(np.clip(ls, -6, 3)) * weights["ys"],
            ]
            print(
                head,
                "max log-space error (value and interval factor)",
                np.max(np.abs(np.log(public[label]) - ref)),
            )


def main():
    torch.set_num_threads(1)
    gp = GridPulse()
    assert all(np.isfinite(a).all() for a in [*gp.W.values(), *gp.E.values()])
    print(
        "waveform scale range",
        gp.W["xs"].min(),
        gp.W["xs"].max(),
        "widen",
        gp.W["widen"],
    )
    cutoff(gp)
    z = np.load(MODEL / "examples/gb_exam_examples.npz")
    parity(gp, z)
    for label, signal in (("waveform", z["w"]), ("flat", np.zeros_like(z["w"]))):
        out = gp.read_windows(signal, N=10, demand_MW=z["demand"])
        for key, truth in (
            ("loss_MW", "loss_label"),
            ("E_GWs", "E_true"),
            ("nadir_Hz", "nadir"),
        ):
            y = z[truth]
            valid = np.isfinite(y)
            print(
                label,
                key,
                "count / MAE",
                valid.sum(),
                np.mean(np.abs(out[key][valid] - y[valid])),
            )
    ladders = yaml.safe_load((MODEL / "ratchet.yaml").read_text())["ladders"]
    rebuilt = next(r["sim"] for r in ladders[1]["rungs"] if r["id"] == "r5_p2r")
    inc = None
    for rung in json.loads((MODEL / "training/ratchet.json").read_text()):
        objective = np.mean(
            [
                rung["skills"][k]
                for k in ("loss@60", "E@60", "beta@60", "nadir@10", "settle@10")
            ]
        )
        np.testing.assert_allclose(objective, rung["objective"], atol=1e-6)
        passed = True
        if inc is not None:
            base_score = rebuilt if rung["val"] == "p2r" else inc["objective"]
            dev_ratio = np.mean([v / inc["dev"][k] for k, v in rung["dev"].items()])
            passed = objective > base_score + 0.01 and dev_ratio <= 1.10
        assert passed == rung["accepted"]
        print(rung["rung"], rung["val"], objective, passed)
        if passed:
            inc = rung


if __name__ == "__main__":
    main()
