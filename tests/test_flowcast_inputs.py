"""FlowCast reads forcing by position; its predict() must check the dates (PR #6 review, 18 Nov 2026)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1] / "models" / "flowcast"


@pytest.fixture(scope="module")
def case():
    sys.path.insert(0, str(HERE))
    import flowcast as fc
    e = json.loads((HERE / "examples" / "examples.json").read_text(encoding="utf-8"))[0]
    f = pd.read_csv(HERE / "examples" / e["forcing_csv"], index_col=0, parse_dates=True)
    return fc.FlowCast(), f, e


def test_forcing_one_day_short_of_the_warm_up_is_rejected(case):
    m, f, e = case
    s0 = pd.Timestamp(e["start"]) - pd.Timedelta(days=365)
    with pytest.raises(ValueError, match="365 days before"):
        m.predict(f.loc[s0 + pd.Timedelta(days=1):], e["attrs"], e["start"], e["end"])


def test_a_missing_day_is_restored_not_shifted(case):
    m, f, e = case
    full = m.predict(f, e["attrs"], e["start"], e["end"])
    gap_day = pd.Timestamp(e["start"]) + pd.Timedelta(days=200)
    gappy = m.predict(f.drop(gap_day), e["attrs"], e["start"], e["end"])
    assert gappy.index.equals(full.index)
    # days after the gap stay aligned: median relative difference 0.4 % (one interpolated day moves the climate
    # statistics a little); when the gap shifted every later day by one, it was 15 %
    full_q, gap_q = full["Q"].values, gappy["Q"].values
    rel = np.abs(gap_q - full_q) / np.maximum(full_q, 0.1)
    assert np.median(rel[220:]) < 0.02
