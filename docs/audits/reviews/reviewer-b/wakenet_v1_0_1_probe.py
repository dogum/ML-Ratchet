"""Compare WakeNet's shipped graph, physics anchors, and demo denominator."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

MODEL = Path(__file__).resolve().parents[4] / "models/wakenet"
sys.path[:0] = [str(MODEL), str(MODEL / "training")]
import wakenet as wn
from model import EdgeWakeGNN


def replay_rungs():
    ladder = yaml.safe_load((MODEL / "ratchet.yaml").read_text())["ladders"][0]
    ref = json.loads((MODEL / "training/v4_reference.json").read_text())
    inc = None
    for r in ladder["rungs"]:
        if inc is None:
            passed = True
        else:
            sim = ref["r6_on_val4"] if r["id"] == "r10" else inc["sim"]
            gates = ("waked_mae_h1", "waked_mae_h2", "farm_mae")
            ratios = [r["dev"][k] / inc["dev"][k] for k in gates]
            passed = r["sim"] > sim + 0.01 and max(ratios) <= 1.05
            print(r["id"], "sim delta, worst dev ratio", r["sim"] - sim, max(ratios))
        assert passed == (r["decision"] in ("released", "kept"))
        if passed:
            inc = r


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    net = wn.WakeNet()
    ref = EdgeWakeGNN(**net.meta["arch"]).eval()
    ref.load_state_dict({k: torch.from_numpy(v) for k, v in net.W.items()})
    assert all(np.isfinite(v).all() for v in net.W.values())
    print("version", net.meta["version"])
    rows = []
    max_parity = max_rotation = max_permutation = 0
    for path in sorted((MODEL / "examples").glob("*.npz")):
        z = np.load(path)
        m = json.loads(str(z["meta"]))
        vectors = ("east_m", "north_m", "hub_h_m", "yaw_deg", "online", "elev_m")
        args = {k: z[k] for k in vectors}
        args.update(
            turbine=m["turbine_model"],
            u_inf=m["u_inf_ms"],
            wind_dir_deg=m["wind_dir_deg"],
            ti_inf=m["ti_inf"],
            sigma_wd_deg=m["sigma_wd_deg"],
            air_density=m["air_density"],
        )
        r = net.read(**args, p_front_kw=m["p_front_kw"])
        rotate = net.read(
            **dict(
                args,
                east_m=-args["north_m"],
                north_m=args["east_m"],
                wind_dir_deg=(args["wind_dir_deg"] - 90) % 360,
            )
        )
        reverse = net.read(**dict(args, **{k: args[k][::-1] for k in vectors}))
        np.testing.assert_array_equal(rotate["is_free"], r["is_free"])
        np.testing.assert_array_equal(reverse["is_free"][::-1], r["is_free"])
        max_rotation = max(max_rotation, np.max(np.abs(rotate["eta"] - r["eta"])))
        max_permutation = max(
            max_permutation, np.max(np.abs(reverse["eta"][::-1] - r["eta"]))
        )
        graph_args = dict(args, turbine=wn.WP.TURBINES[args["turbine"]])
        g = wn.PL.build_graph(**graph_args, anchor=net.meta.get("anchor", "textbook"))
        keys = ("node_x", "edge_x", "edge_mask", "online", "is_free", "eta_anchor")
        inputs = [g[k] for k in keys]
        torch_out = ref(*(torch.as_tensor(x)[None] for x in inputs))
        np_out = net.forward_graph(*inputs)
        for k, v in torch_out.items():
            max_parity = max(max_parity, np.max(np.abs(np_out[k] - v.numpy()[0])))
        tuned = wn.PL.build_graph(**graph_args, anchor="tuned")["eta_anchor"]
        true = z["power_true_kw"] / m["p_front_kw"]
        idx = np.where(r["online"] & ~r["is_free"])[0]
        shown = idx[np.argsort(r["eta"][idx])][:6]
        err = np.abs(r["eta"] - true)
        rows.extend(
            (
                i in shown,
                err[i],
                err[i] <= r["eta_sig"][i],
                abs(tuned[i] - true[i]),
                abs(r["eta_anchor"][i] - true[i]),
            )
            for i in idx
        )
    print(
        "max raw parity, rotated eta, permuted eta",
        max_parity,
        max_rotation,
        max_permutation,
    )
    a = np.asarray(rows)
    for label, sample in (("all", a), ("demo", a[a[:, 0].astype(bool)])):
        print(
            label,
            len(sample),
            "eta MAE, coverage, tuned MAE, textbook MAE",
            sample[:, 1:].mean(0),
        )
    replay_rungs()


if __name__ == "__main__":
    main()
