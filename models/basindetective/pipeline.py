"""BasinDetective pipeline: from CSV files, or from a CAMELS-US basin id when the Watershed `waterdata` package is
installed.

python3 pipeline.py cal.csv tgt.csv          # cal.csv: date,P,T_mean,PET,Q_mm ; tgt.csv: date,P,T_mean,PET
python3 pipeline.py camels_01022500          # needs the Watershed waterdata package and its Caravan data
                                             # (ML Ratchet: shared/water, data from `mlratchet data water`)
Units: P and PET mm/day, T_mean degC, Q_mm mm/day over the basin. Calibration window ideally 9-10 water years
(the model was trained on 9); target window ideally 10 water years (trained on 10).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import basindetective as bd   # noqa: E402


def from_csv(cal_csv, tgt_csv, det=None):
    det = det or bd.load()
    cal = pd.read_csv(cal_csv, index_col=0, parse_dates=True)
    tgt = pd.read_csv(tgt_csv, index_col=0, parse_dates=True)
    return bd.predict(det, cal[['P', 'T_mean', 'PET']], cal['Q_mm'].values, tgt[['P', 'T_mean', 'PET']])


def from_camels(basin_id, cal=('1999-10-01', '2008-09-30'), tgt=('1989-10-01', '1999-09-30'), det=None):
    # optional: only this CAMELS-id mode imports waterdata (the repository's shared/water; its data root is
    # $ML_RATCHET_WATER or shared/water). The CSV mode and the package need numpy, scipy and pandas only.
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'shared', 'water')))
    import waterdata as wd
    det = det or bd.load()
    f = wd.forcing(basin_id)[['P', 'T_mean', 'PET']]
    q = wd.streamflow(basin_id, 'caravan').Q_mm
    c = f.loc[cal[0]:cal[1]]
    return bd.predict(det, c, q.reindex(c.index).values, f.loc[tgt[0]:tgt[1]])


if __name__ == '__main__':
    r = from_csv(*sys.argv[1:3]) if len(sys.argv) > 2 else from_camels(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k != 'calibration_signatures'}, indent=1, default=float))
