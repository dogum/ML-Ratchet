"""Audit released offers, replay three shipped node-years, and check promotion arithmetic."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

MODEL = Path(__file__).resolve().parents[4] / "models" / "batterypilot"
sys.path.insert(0, str(MODEL))
import batterypilot as bp
from demo import load


def perturbations(policy, battery):
    year = load(MODEL / "examples/ERCOT_HB_WEST_2024.csv")
    path = {key: np.array(values[:144]) for key, values in year.items()}
    decision = 44
    assert path["hour"][decision] == 14
    assert path["hour"][54] == 0 and path["day"][54] == path["day"][decision] + 1

    def offers(data):
        return bp.run_policy(policy, data, battery, battery.c_lin(data["T"]))[3]

    original = offers(path)
    for key, hours, change in [
        ("T", slice(45, 57), 5), ("rt", slice(45, 57), 100),
        ("da", slice(54, 78), 100), ("rt", slice(0, 1), 100),
    ]:
        altered = dict(path)
        altered[key] = path[key].copy()
        altered[key][hours] += change
        delta = np.max(np.abs(offers(altered) - original), axis=1)
        print(f"{key} [{hours.start}:{hours.stop}] +{change}: max knot change at "
              f"hours 0, 43, 44 = {delta[0]:.4f}, {delta[43]:.4f}, {delta[44]:.4f}")


def net_revenues(path, policy, battery, persistence):
    temp = path["T"]
    wear = battery.c_lin(np.r_[25., temp[:-1]] if persistence else temp)
    if persistence:
        # The released planners pass the known-price horizon as a slice. Hold
        # wear at its first supplied (lagged) value throughout each such horizon.
        backward_pass = bp.backward_pass
        lp_schedule = bp.lp_schedule

        def fixed_back(price, cost, *args, **kwargs):
            return backward_pass(price, np.full_like(cost, cost[0]), *args, **kwargs)

        def fixed_lp(price, bat, cost, *args, **kwargs):
            return lp_schedule(price, bat, np.full_like(cost, cost[0]), *args, **kwargs)

        with patch.object(bp, "backward_pass", fixed_back), patch.object(bp, "lp_schedule", fixed_lp):
            return dispatch_and_settle(path, policy, battery, wear)
    return dispatch_and_settle(path, policy, battery, wear)


def dispatch_and_settle(path, policy, battery, wear):
    charge, discharge, _, _ = bp.run_policy(policy, path, battery, wear)
    rolling = bp.rolling_da_lp(path["da"], path["hour"], path["day"], battery, wear)
    annualize = 8760 / len(path["rt"]) / (battery.P * 1000)
    return [bp.settle(battery, path["rt"], ch, dis, 0., path["T"])["net"] * annualize
            for ch, dis in [(charge, discharge), rolling]]


def main():
    log = json.loads((MODEL / "training/ratchet_v1_3.json").read_text())
    incumbent, *candidates = log["rungs"]
    decisions = 0
    for row in candidates:
        if row["name"] == "bb":  # R5 is the predeclared classical reference.
            continue
        keep = (row["frozen_capture"] - incumbent["frozen_capture"] > 0.01
                and row["dev_capture"] - incumbent["dev_capture"] >= -0.01)
        assert keep == row["kept"], row["rung"]
        if keep:
            incumbent = row
        decisions += 1
    print(f"Promotion decisions matching log: {decisions}; final incumbent {incumbent['rung']}")
    policy, battery = bp.PolicyNet(), bp.Battery(P=1., duration_h=2.)
    perturbations(policy, battery)
    print("Node-year, released BP/rolling, persistence BP/rolling ($/kW-year)")
    for example in sorted((MODEL / "examples").glob("*.csv")):
        path = load(example)
        released = net_revenues(path, policy, battery, persistence=False)
        replay = net_revenues(path, policy, battery, persistence=True)
        print(example.stem, *(f"{v:.6f}" for v in released + replay))


if __name__ == "__main__":
    main()
