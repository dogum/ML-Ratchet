"""Day selection, reference dating and folded-network replay on packaged trend logs."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

MODEL = Path(__file__).resolve().parents[4] / "models/faultfinder"
sys.path[:0] = [str(MODEL), str(MODEL / "training")]
import faultfinder as ff
from train8 import Folded
import labsim
from buildingsim.weather import clear_cache


def wind_cache(model):
    times = pd.date_range("2019-01-01", periods=31 * 24, freq="h", tz="UTC")
    h = np.arange(len(times))
    sun = 350 * np.maximum(0, np.sin(np.pi * (h % 24 - 6) / 12))
    weather = pd.DataFrame(dict(T_out=-3 + 6 * np.sin(2*np.pi*h/24), RH=65., GHI=sun,
                                DNI=sun, DHI=.3*sun, wind_dir=180., pressure=101325., cloud=.4), index=times)
    meta = dict(lat=41.8, lon=-87.6, elev=180., tz="UTC")
    def day(speed):
        with patch.object(labsim, "weather", return_value=(weather.assign(wind_speed=speed), meta)):
            return labsim.sim_day("2019-01-23", None)
    clear_cache()
    low, warm = day(2), day(8)
    clear_cache()
    high = day(8)
    np.testing.assert_equal(low, warm)
    assert np.nanmax(abs(high[ff.R["T_sa"]] - low[ff.R["T_sa"]])) > 0
    print("VAV winter day, fresh 2 / cached 8 / fresh 8 m/s, probabilities",
          [row["p_fault"] for row in model.predict_raw([low, warm, high])],
          "max supply temperature change (C)", np.nanmax(abs(high[ff.R["T_sa"]]-low[ff.R["T_sa"]])))


def main():
    model = ff.FaultFinder()
    wind_cache(model)
    assert all(np.isfinite(a).all() for a in model.W.values())
    for name, a in model.W.items():
        if name.endswith("_W"):
            print(name, "low-norm outputs", np.sum(np.linalg.norm(a.reshape(len(a), -1), axis=1) < 1e-4))
    base = dict(model.W)
    base["c1_W"] = base["c1_W"][:, :ff.NF]
    torch_net = Folded(base).eval()
    with torch.no_grad():
        torch_net.c1_new.copy_(torch.from_numpy(model.W["c1_W"][:, ff.NF:]))
    ex = MODEL / "examples"
    ref_files = json.loads((ex / "references.json").read_text())
    for name, names in ref_files.items():
        day = pd.read_csv(ex / name)
        refs = [pd.read_csv(ex / n) for n in names]
        date = pd.to_datetime(day.time_local.iloc[0]).normalize()
        for ref in refs:
            age = date - pd.to_datetime(ref.time_local.iloc[0]).normalize()
            assert pd.Timedelta(days=1) <= age <= pd.Timedelta(days=7)
        def score(frame, ref_days=()):
            return model.read_day(frame, day=date, ref_days=ref_days)["p_fault"]
        raw = ff.raw_day(day)
        x = model.inputs([raw], [[ff.raw_day(ref) for ref in refs]])
        with torch.no_grad():
            heads = torch_net(torch.from_numpy(x))
        err = max(np.max(abs(a.numpy() - b)) for a, b in zip(heads, ff.forward(model.W, x)))
        assert err < 1e-4
        print(name, "folded parity", err, "single / with refs", score(day), score(day, refs))
        for ref in refs:
            first = pd.to_datetime(ref.time_local.iloc[0]).normalize()
            single = model.read_day(ref, day=first)["p_fault"]
            joined = model.read_day(pd.concat([ref, day]), day=first)["p_fault"]
            print("real earlier day, unchanged / appended real later day", first.date(), single, joined)
        # Identical values and masks in reference days, but timestamps moved to the future.
        moved = [ref.assign(time_local=pd.to_datetime(ref.time_local) + pd.Timedelta(days=14)) for ref in refs]
        assert score(day, refs) == score(day, moved)
        # Same complete target day. Only the following day's temperature varies.
        future = day.assign(time_local=pd.to_datetime(day.time_local) + pd.Timedelta(days=1))
        flat = day.assign(T_sa=18.)
        alone = score(flat)
        flat_control = score(pd.concat([flat, future.assign(T_sa=18.)]))
        extended = score(pd.concat([flat, future.assign(T_sa=19.)]))
        assert alone == flat_control
        if name.startswith("flexlab"):
            assert alone > .5 > extended
        print("flat supply temperature: day alone / same future / changed future", alone, flat_control, extended,
              "unmodified target day / appended duplicate", score(day), score(pd.concat([day, future])))
    empty = model.read_day(pd.DataFrame(index=pd.date_range("2025-01-01", periods=96, freq="15min")))
    print("no sensor readings", empty["p_fault"], empty["diagnosis"][0])

    log = json.loads((MODEL / "training/ratchet.json").read_text())
    best = None
    for row in log["rungs"]:
        scores = row["scores"]
        s, d = scores["v4"]["skill"], scores["dev"]["dev_auc"]
        keep = best is None or (s > best[0] + .01 and d >= max(.6, best[1] - .05))
        assert keep == (row["decision"] == "kept"), row["name"]
        if keep:
            best = (s, d)
    v5 = json.loads((MODEL / "training/ratchet_r7.json").read_text())
    decisions = v5["decisions"]
    r6 = v5["scores"]["r6"]
    best = (r6["v5"]["skill"], r6["dev"]["dev_auc"])
    for key in ("r7_context", "r8"):
        r = decisions[key]
        assert (np.mean(r["skill_seeds"]) > best[0] + .01 and r["dev_auc_mean"] >= max(.6, best[1] - .05)) == (key == "r8")
    print("v4 and newer scored candidates", len(log["rungs"]), best, list(decisions))


if __name__ == "__main__":
    main()
