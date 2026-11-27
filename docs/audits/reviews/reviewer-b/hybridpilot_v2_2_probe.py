"""Closed-loop preview endpoint replay on the three shipped VED traces, pinned RoadSim v3.1."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

MODEL = Path(__file__).resolve().parents[4] / "models/hybridpilot"
sys.path[:0] = [str(MODEL), str(MODEL / "training")]
import hybridpilot as hp
from pipeline import load_example, PHYSICS, WARM
from roadsim import simulate
from roadsim.sim import get_vehicle, normalize_trip
from train import Net


class BoundedContext(hp.TripContext):
    """Hold speed and grade at the last advertised sample for its final power interval."""
    def __init__(self, vp, trip):
        super().__init__(vp, trip)
        original = hp.road_power(vp, trip) / vp["P_eng_max"]
        held = original.copy()
        # Interleave independent endpoints to vectorize each two-sample calculation.
        for parity in (0, 1):
            idx = np.arange(parity, self.N - 1, 2)
            bounded = dict(trip)
            for key in ("v", "grade"):
                bounded[key] = trip[key].copy()
                bounded[key][idx + 1] = trip[key][idx]
            held[idx] = hp.road_power(vp, bounded)[idx] / vp["P_eng_max"]
        self.delta = np.stack((held.clip(min=0) - original.clip(min=0),
                               held.clip(max=0) - original.clip(max=0)), axis=1)

    def preview(self, k, length):
        out = super().preview(k, length)
        for i, (a, b) in enumerate(hp.BINS):
            if b == length and k + b < self.N:
                out[4*i:4*i+2] += self.delta[k+b] / (b-a+1)
        return out


def ladder():
    rows = json.loads((MODEL / "training/ratchet_v2_2.json").read_text())
    def pooled(r):
        return sum(r[k]["cs_rate"] * r[k]["n"] for k in ("frozen", "dev")) / sum(r[k]["n"] for k in ("frozen", "dev"))
    best = None
    for r in rows:
        keep = best is None
        if best:
            f, d = r["frozen"], r["dev"]
            prev, gate = best["frozen"], best["dev"]
            ok_dp = d["pct_dp_mean"] >= gate["pct_dp_mean"] - .002
            keep = f["skill"] > prev["skill"] + .01 and ok_dp and d["cs_rate"] >= gate["cs_rate"] - .03
            if r.get("rule") == "v2.2":
                keep |= (pooled(r) >= pooled(best) + .02 - 1e-9 and f["cs_rate"] >= prev["cs_rate"] - 1e-9
                         and d["cs_rate"] >= gate["cs_rate"] - 1e-9 and f["skill"] >= prev["skill"] - .01 and ok_dp)
        assert keep == r["kept"], r["rung"]
        if keep:
            best = r
    print("replayed", len(rows), "release ladder rows, incumbent", best["rung"])


def parity(features):
    for file in sorted((MODEL / "weights").glob("*.npz")):
        net = hp.MLP(file)
        w = net.p
        assert all(np.isfinite(a).all() for a in w.values()) and np.all(w["sd"] > 0)
        torch_net = Net(len(w["sd"]), w["W0"].shape[1], w["mu"], w["sd"], depth=net.nl - 1)
        with torch.no_grad():
            for i, layer in enumerate(m for m in torch_net.f if isinstance(m, torch.nn.Linear)):
                layer.weight.copy_(torch.tensor(w[f"W{i}"].T))
                layer.bias.copy_(torch.tensor(w[f"b{i}"]))
                assert (np.linalg.norm(w[f"W{i}"], axis=0) > 1e-4).all()
            y = torch_net(torch.tensor(features, dtype=torch.float32)).numpy()
        err = np.max(abs(y - net(features)))
        print(file.name, "parity / sd range", err, w["sd"].min(), w["sd"].max())
        z = (features - w["mu"]) / w["sd"]
        variable = features.std(axis=0) > 1e-7
        print("varying features / normalized std below .01", variable.sum(), np.flatnonzero(variable & (z.std(axis=0) < .01)))
        assert err < 2e-5


def grade_gap(spec, trip, soc0):
    # Start at a moving segment selected using only the original trace.
    start = 20 + int(np.argmax(trip["v"][20:140]))
    end = start + 10
    for gap in (False, True):
        runs = []
        for change in (0., .04):
            grade = trip["grade"].copy()
            if gap:
                grade[start:end] = np.nan
            grade[end:] += change
            route = dict(trip, grade=grade)
            strategy = hp.make_roadsim_strategy(preview_s=0, nav=False)
            out = simulate(spec, route, strategy=strategy, init_soc=soc0, init_temps=dict(WARM),
                           record=True, options=PHYSICS)
            runs.append(out)
        # Pinned physics records each one-second interval at its end.
        assert len(runs[0]["eng_power"]) == len(trip["v"])
        delta = {field: max(abs(runs[0][field][:end] - runs[1][field][:end]))
                 for field in ("eng_power", "batt_current", "eng_on")}
        if not gap:
            assert all(v == 0 for v in delta.values())
        print("grade gap", gap, "cutoff", end - 1, "max earlier differences W / A / engine state", delta)


def main():
    ladder()
    features = []
    for entry in json.loads((MODEL / "examples/index.json").read_text()):
        spec, trip = load_example(entry["trip"])
        soc = trip["soc"][np.isfinite(trip["soc"])]
        soc0 = float(soc[0]) if entry["arch"].startswith("phev") and len(soc) else None
        if soc0 is not None and soc0 > 1.5:
            soc0 /= 100
        grade_gap(spec, trip, soc0)
        vp = hp.vehicle_params_from_roadsim(get_vehicle(spec))
        route = normalize_trip(trip)
        original_context = hp.TripContext(vp, route)
        for k in (40, 80, 120):
            altered = dict(route, v=route["v"].copy(), grade=route["grade"].copy())
            altered["v"][k+2:] += 2.
            altered["grade"][k+2:] += .01
            other = hp.TripContext(vp, altered)
            for field in ("a", "grade", "ema30_v", "ema300_v", "ema60_P", "ema300_P", "ema120_stop"):
                assert getattr(original_context, field)[k] == getattr(other, field)[k]
            np.testing.assert_array_equal(original_context.preview(k, 0), other.preview(k, 0))
        for length in (0, 5, 30):
            def run(context):
                strategy = hp.make_roadsim_strategy(preview_s=length, nav=False)
                with patch.object(hp, "TripContext", context):
                    out = simulate(spec, trip, strategy=strategy, init_soc=soc0, init_temps=dict(WARM),
                                   record=True, options=PHYSICS)
                return out, np.asarray(strategy.trace)
            baseline, trace = run(hp.TripContext)
            bounded, trace_b = run(BoundedContext)
            a = baseline["summary"]
            b = bounded["summary"]
            eng = np.count_nonzero(baseline["eng_on"] != bounded["eng_on"])
            print(entry["trip"], length, "fuel g / bounded minus nominal / delta final SOC / different engine samples",
                  a["fuel_g"], b["fuel_g"] - a["fuel_g"], b["soc_end"] - a["soc_end"], eng)
            if length == 0:
                assert np.array_equal(trace, trace_b) and a["fuel_g"] == b["fuel_g"]
        original = hp.features
        def capture(*args, **kwargs):
            x = original(*args, **kwargs)
            features.append(x)
            return x
        with patch.object(hp, "features", capture):
            run(hp.TripContext)  # 30 s preview, no navigation
    parity(np.asarray(features))


if __name__ == "__main__":
    main()
