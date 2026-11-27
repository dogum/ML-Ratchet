"""Regression tests for the water-platform audit findings fixed on 27 Nov 2026 (docs/audits/)."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(model: str, module: str):
    """The model's module through the registry, which keeps sibling modules (pipeline, common, features) private."""
    import mlratchet
    return mlratchet.get(model).module(module)


@pytest.mark.parametrize("cut", ["2022-01-15", "2022-04-17", "2022-04-18"])
def test_snowsense_partial_year_stops_at_the_last_input(cut):
    ss = _load("snowsense", "snowsense")
    st = json.loads((ROOT / "models/snowsense/examples/stations.json").read_text())
    st = st["497_WY_SNTL"] if isinstance(st, dict) and "497_WY_SNTL" in st else next(
        s for s in (st if isinstance(st, list) else st.values()) if str(s.get("id", "")).startswith("497"))
    df = pd.read_csv(ROOT / "models/snowsense/examples/497_WY_SNTL.csv", index_col=0, parse_dates=True)
    out = ss.SnowSense().predict(df.loc[:cut], st["lat"], st["elev_m"], wys=[2022])
    daily, annual = out["daily"], out["annual"]
    assert len(daily) > 0                                   # was empty before day 200
    assert daily.index.max() <= pd.Timestamp(cut)           # was 165 dates past the last input on 18 Apr
    assert pd.isna(annual.loc[2022, "melt_out_date"])


@pytest.mark.parametrize("bad", ["nan", "short", "zero"])
def test_basindetective_refuses_unusable_calibration_flow(bad):
    bd = _load("basindetective", "basindetective")
    import glob
    f = sorted(glob.glob(str(ROOT / "models/basindetective/examples/*_cal.csv")))[0]
    cal = pd.read_csv(f, index_col=0, parse_dates=True)
    tgt = pd.read_csv(f.replace("_cal.csv", "_tgt.csv"), index_col=0, parse_dates=True)
    q = cal["Q_mm"].to_numpy(float).copy()
    if bad == "nan":
        q[:] = np.nan
    elif bad == "short":
        q[300:] = np.nan
    else:
        q[:] = 0.0
    with pytest.raises(ValueError):
        bd.predict(bd.load(), cal[["P", "T_mean", "PET"]], q, tgt)
