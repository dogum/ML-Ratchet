"""FireSize must accept missing weather, as training did (PR #7 review, 21 Nov 2026)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "models" / "firesize"


def test_missing_weather_uses_the_training_fallbacks():
    sys.path.insert(0, str(HERE))
    import firesize as F
    fuel = np.full((13, 13), 102)
    zero = np.zeros((13, 13))
    nan_wx = {k: np.nan for k in F.WXF}
    f = F.make_features(fuel, zero, zero, zero, nan_wx, nan_wx, F.CAUSES[0], 0, None, 200, F.REGIONS[0])
    # training/src/physfeat.py: 1-h moisture 10 %, 10 m wind 3 m/s when the weather is missing
    expect = float(np.log1p(F.ros_mmin(102, 10.0, 3.0)))
    assert np.isclose(f["ros_pt"], expect) and np.isclose(f["ros_pt_m1"], expect)
