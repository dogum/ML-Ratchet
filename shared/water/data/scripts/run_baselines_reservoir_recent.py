"""Recent-years reservoir exam (WY2021-2025) on the California reservoirs that CDEC extends past ResOpsUS.
Same task definition and policies as run_baselines_reservoir.py; demand, Qmax and rule curve come from the reservoir's
whole ResOpsUS record (1980..2020); scored on CDEC 2020-10-01..2025-09-30. Out: proc/baselines/reservoir_recent_scores.parquet
"""
import os, sys
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.dirname(__file__))
import waterdata as wd
from run_baselines_reservoir import doy_curve, cost, simulate, dp

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')


def main():
    sp = wd.splits()['reservoirs']
    xw = pd.read_csv(f'{ROOT}/proc/reservoirs/cdec/crosswalk.csv')
    meta = wd.reservoir_meta().set_index('res_id')
    rows = []
    for r in xw.itertuples():
        split = 'dev' if r.res_id in sp['dev'] else ('exam' if r.res_id in sp['exam'] else 'aux')
        h = wd.reservoir(r.res_id).loc['1980-01-01':'2020-09-30'][['storage_hm3', 'inflow_cms', 'release_cms']].dropna()
        if len(h) < 3650:
            continue
        ev = wd.reservoir_cdec(r.cdec_id, '2020-10-01', '2025-09-30').asfreq('D').interpolate(limit=5).dropna()
        if len(ev) < 1000:
            continue
        blk = (ev.index.to_series().diff() != pd.Timedelta('1D')).cumsum()
        ev = ev[blk == blk.value_counts().idxmax()]
        cap = max(float(meta.loc[r.res_id, 'cap_hm3']), float(h.storage_hm3.max()))
        Dc = doy_curve(h.release_cms); Sc = doy_curve(h.storage_hm3); Qmax = float(h.release_cms.quantile(0.99))
        I = ev.inflow_cms.values.astype(float); D = Dc.reindex(ev.index.dayofyear).values
        Ss = Sc.reindex(ev.index.dayofyear).values; S0 = float(ev.storage_hm3.iloc[0])
        res = {'historical': ev.release_cms.values, 'sop': simulate('sop', I, D, Qmax, cap, S0)[0],
               'rule': simulate('rule', I, D, Qmax, cap, S0, Ss)[0], 'dp': dp(I, D, Qmax, cap, S0)}
        for pol, R in res.items():
            rows.append({'res_id': r.res_id, 'cdec_id': r.cdec_id, 'split': split, 'policy': pol, 'n_days': len(ev),
                         'J': float(np.mean(cost(R, D, Qmax))),
                         'shortage_ratio': float(np.sum(np.maximum(0, D - R)) / np.sum(D)),
                         'flood_days_per_yr': float(np.mean(R > Qmax) * 365.25)})
        print(r.cdec_id, split, {p: round(float(np.mean(cost(v, D, Qmax))), 4) for p, v in res.items()}, flush=True)
    pd.DataFrame(rows).to_parquet(f'{ROOT}/proc/baselines/reservoir_recent_scores.parquet')


if __name__ == '__main__':
    main()
