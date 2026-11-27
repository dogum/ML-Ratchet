"""Run ReservoirPilot on any daily reservoir record, with WaterData's task construction.

    python pipeline.py record.csv CAPACITY_HM3 [--weights reservoirpilot_weights.npz] [--out result.csv]

record.csv: columns date, storage_hm3, inflow_cms, release_cms (daily). Like WaterData's ResOpsUS exam, the record
from 1980 is split in half: the first half defines demand D(doy) (median release), Qmax (99th percentile release)
and the inflow climatology; the second half (longest gap-free block, at most 10 years) is operated by the policy
from the observed start storage and its real inflow. Prints J and its parts for the pilot, SOP and the record.

v1.0.1 (27 Nov 2026): the causal task, as in the dated re-score: the record is split first and each half is
interpolated (gaps up to 5 days) on its own, the split day is in the scored half only, and capacity is the larger of
CAPACITY_HM3 and the highest storage in the calibration half. --exam-definition restores v1.0's task.
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reservoirpilot as rp  # noqa: E402


def _interp(d):
    d = d.copy()
    d[['inflow_cms', 'release_cms']] = d[['inflow_cms', 'release_cms']].interpolate(limit=5)
    d['storage_hm3'] = d.storage_hm3.interpolate(limit=5)
    return d


def split_record(df, causal=True):
    d = df[['storage_hm3', 'inflow_cms', 'release_cms']].loc['1980-01-01':].dropna().asfreq('D')
    half = d.index[len(d) // 2]
    if causal:
        cal, ev = _interp(d.loc[:half - pd.Timedelta('1D')]), _interp(d.loc[half:]).dropna()
    else:
        d = _interp(d)
        cal, ev = d.loc[:half], d.loc[half:].dropna()
    blk = (ev.index.to_series().diff() != pd.Timedelta('1D')).cumsum()
    ev = ev[blk == blk.value_counts().idxmax()].iloc[:3653]
    return cal, ev, d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('csv'); ap.add_argument('capacity', type=float)
    ap.add_argument('--weights', default=rp.WEIGHTS)
    ap.add_argument('--out', default=None)
    ap.add_argument('--exam-definition', action='store_true', help="v1.0's task (capacity over the whole record)")
    a = ap.parse_args()
    df = pd.read_csv(a.csv, parse_dates=['date']).set_index('date')
    causal = not a.exam_definition
    cal, ev, d = split_record(df, causal)
    cap = a.capacity if causal else max(a.capacity, float(d.storage_hm3.max()))
    task = rp.make_task(cal, ev, cap, causal_capacity=causal)
    pol = rp.Pilot.load(a.weights)
    R, S = pol.run(task); Rs, Ss = rp.sop(task)
    for name, (r, s) in (('ReservoirPilot', (R, S)), ('SOP', (Rs, Ss)), ('record', (ev.release_cms.values, ev.storage_hm3.values))):
        c = rp.cost_components(r, s, task)
        print(f'{name:15s} ' + '  '.join(f'{k} {v:.4f}' for k, v in c.items()))
    if a.out:
        pd.DataFrame({'inflow_cms': task['I'], 'demand_cms': task['D'], 'pilot_release_cms': R, 'pilot_storage_hm3': S,
                      'sop_release_cms': Rs, 'sop_storage_hm3': Ss}, index=ev.index).to_csv(a.out)


if __name__ == '__main__':
    main()
