"""FireCast must drop detections whose footprint misses the grid (PR #7 review, 21 Nov 2026)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "models" / "firecast"


def test_off_grid_detection_adds_no_pixels():
    sys.path.insert(0, str(HERE))
    import firecast as fc
    det = {"row": np.array([5.0, -20.0, 5.0]), "col": np.array([-100.0, 5.0, 5.0]), "ps": np.array([0.375] * 3),
           "pt": np.array([0.375] * 3), "frp": np.array([10.0, 10.0, 10.0])}
    out = fc.pixel_raster(det, (32, 32))
    assert out[:, 0].max() == 0                              # nothing pulled onto column 0 from col = -100
    assert out[0, :].max() == 0                              # nothing pulled onto row 0 from row = -20
    assert out[5, 5] > 0                                     # the on-grid detection is still drawn
