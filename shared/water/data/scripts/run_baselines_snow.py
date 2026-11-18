"""SnowSense classical baselines on SNOTEL (dev + exam stations of splits.json['snotel']).

Per station: calibration WY2001-2010, scoring WY2011-2025 (complete water years only: >=330 SWE days, >=300 T and P days).
  clim  day-of-year median SWE of the calibration water years
  dd    degree-day model (waterdata.models.degree_day: tt, ddf, pmult, tmelt) calibrated per station on the calibration
        years (RMSE), driven by the station's own P and T_mean (T gaps <= 3 d interpolated, other T gaps -> calibration
        mask; P gaps -> 0).
Timing: SNOTEL SWE is start-of-day; the model's SWE is end-of-day, so obs SWE(D+1) is compared with sim(D).
Out: proc/baselines/snow_scores.parquet, snow_summary.json
"""
import json, os, sys
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root
sys.path.insert(0, _CODE)
import waterdata as wd
from waterdata import metrics as M
from waterdata.models import calibrate_dd, degree_day

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')


def wy(idx):
    return idx.year + (idx.month >= 10).astype(int)


def main():
    sp = wd.splits()['snotel']
    meta = wd.snotel_meta().set_index('station_id')
    rows = []
    for k, sid in enumerate(sp['dev'] + sp['exam']):
        d = wd.snotel(sid, '2000-10-01', '2025-09-30')
        if len(d) < 3650:
            continue
        d = d.asfreq('D')
        T = d.T_mean.interpolate(limit=3)
        P = d.P.fillna(0)
        obs = d.SWE.shift(-1)                     # end of day D = start of day D+1
        w = wy(d.index)
        okwy = set(meta.loc[sid, 'wy_complete'])
        cal = np.isin(w, list(range(2001, 2011))) & np.isin(w, list(okwy))
        ev = np.isin(w, list(range(2011, 2026))) & np.isin(w, list(okwy))
        if cal.sum() < 5 * 330 or ev.sum() < 5 * 330:
            continue
        # run from 2000-10-01 continuously; calibrate on cal mask (T must be finite)
        Tf = T.fillna(0).values
        mcal = cal & T.notna().values & obs.notna().values
        o_cal = np.where(mcal, obs.values, np.nan)
        th, rmse = calibrate_dd(P.values, Tf, o_cal, seed=20261118)
        sim = pd.Series(degree_day(th, P.values, Tf), index=d.index)
        clim = obs[cal].groupby(obs[cal].index.dayofyear).median().reindex(range(1, 367)).interpolate(limit_direction='both')
        cl = pd.Series(clim.reindex(d.index.dayofyear).values, index=d.index)
        oe = obs.where(ev)
        for m, s in [('clim', cl), ('dd', sim)]:
            r = M.swe_metrics(oe.dropna(), s)
            r.update({'station_id': sid, 'method': m, 'split': 'dev' if sid in sp['dev'] else 'exam',
                      'state': meta.loc[sid, 'state'], 'elev_m': meta.loc[sid, 'elev_m']})
            if m == 'dd':
                r.update({'tt': th[0], 'ddf': th[1], 'pmult': th[2], 'tmelt': th[3], 'rmse_cal': rmse})
            rows.append(r)
        if k % 50 == 0:
            print(k, sid, round(rows[-1]['nse'], 3), flush=True)
    sc = pd.DataFrame(rows)
    sc.to_parquet(f'{ROOT}/proc/baselines/snow_scores.parquet')
    summ = {}
    for m, g in sc[sc.split == 'exam'].groupby('method'):
        summ[m] = {'n': int(len(g)), 'nse_median': float(g.nse.median()), 'rmse_mm_median': float(g.rmse_mm.median()),
                   'peak_swe_abs_err_pct_median': float(g.peak_swe_abs_err_pct.median()),
                   'peak_swe_bias_pct_median': float(g.peak_swe_bias_pct.median()),
                   'meltout_mae_days_median': float(g.meltout_mae_days.median())}
    json.dump(summ, open(f'{ROOT}/proc/baselines/snow_summary.json', 'w'), indent=1)
    print(pd.DataFrame(summ).T.round(3))


if __name__ == '__main__':
    main()
