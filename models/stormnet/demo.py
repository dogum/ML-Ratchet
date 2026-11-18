"""Run the released network on the preselected Kyoto/OMNI primary-exam issues."""
import json
from pathlib import Path

import numpy as np
from stormnet import StormNet

if __name__ == "__main__":
    base = Path(__file__).with_name("examples")
    net = StormNet()
    print("StormNet · next three-hour minimum of hourly Dst (nT)")
    print("Issue time (UTC)           Forecast    68% half-width   Observed   Persistence")
    for record in json.loads((base / "records.json").read_text()):
        with np.load(base / record["file"]) as z:
            r = net.predict(z["wind"][-1], z["last_dst"][-1])
            print(f"{record['issues'][-1]:24s} {r['min_dst']:8.1f} {r['sigma']:16.1f}"
                  f" {z['future_dst'][-1].min():10.0f} {z['last_dst'][-1]:13.0f}")
    print("Four distinct blocks, one issue per severity band. The browser replays seven issues per block.")
