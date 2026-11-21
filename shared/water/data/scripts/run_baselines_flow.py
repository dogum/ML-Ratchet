"""Streamflow exam baselines on CAMELS-US (run after run_gr4j.py, build_nwis.py, build_nwm.py).

Scored periods (obs never used for calibration):
  val    1989-10-01..1999-09-30  obs = Caravan/CAMELS Q (mm/day)
  recent 2015-10-01..2023-09-30  obs = USGS NWIS daily values (mm/day over NWIS drainage area)
  NWM is scored on val and on recent clipped to 2015-10-01..2022-09-30 (NWM v3.0 retrospective ends 2023-01-31).
Methods:
  clim        day-of-year mean of the basin's own obs in the calibration period (31-day circular smoothing)  [gauged]
  persist1    Q(t) = Q_obs(t-1): 1-day-lead persistence                                                      [gauged, forecast]
  gr4j        GR4J+snow calibrated on the basin's own calibration period                                     [gauged]
  nn_dev      ungauged: mean of the 5 most similar DEV basins' calibrated GR4J run on the target forcing      [U-dev]
  nn_k12      ungauged: same with donors from the other 11 folds of ungauged_folds_k12 (all basins)          [U-k12]
  nwm         NOAA NWM v3.0 retrospective (uncalibrated to this split; NOAA calibrated it on its own basins)  [operational]
Out: proc/baselines/flow_scores.parquet (one row per basin x method x period), flow_summary.json, sims/nn_*.parquet
"""
import json, os, sys
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root
sys.path.insert(0, _CODE)
import waterdata as wd
from waterdata import metrics as M
from waterdata.models import gr4j_snow

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
B = f'{ROOT}/proc/baselines'
VAL = ('1989-10-01', '1999-09-30'); REC = ('2015-10-01', '2023-09-30'); REC_NWM = ('2015-10-01', '2022-09-30')
CAL = ('1999-10-01', '2008-09-30')
SIM_ATTRS = ['p_mean', 'aridity_FAO_PM', 'frac_snow', 'high_prec_freq', 'seasonality_FAO_PM', 'ele_mt_sav', 'slope_deg',
             'log_area', 'for_pc_sse', 'snd_pc_sav', 'cly_pc_sav', 'kar_pc_sse']


def clim(qcal, index):
    d = qcal.groupby(qcal.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction='both')
    v = np.concatenate([d.values[-15:], d.values, d.values[:15]])
    sm = pd.Series(np.convolve(v, np.ones(31) / 31, mode='valid'), index=range(1, 367))
    return pd.Series(sm.reindex(index.dayofyear).values, index=index)


def main():
    sp = wd.splits()['camels_us']
    a = wd.attributes('camels')
    a['log_area'] = np.log10(a['nwis_drain_area_km2'])
    Z = (a[SIM_ATTRS] - a[SIM_ATTRS].mean()) / a[SIM_ATTRS].std()
    ids = list(a.index)
    params = {b: json.load(open(f'{B}/gr4j/{b}.json'))['params'] for b in ids}
    gr = pd.read_parquet(f'{B}/sims/gr4j_gauged.parquet')
    nwm_l = pd.read_parquet(f'{ROOT}/proc/nwm/daily_local.parquet')
    folds = sp['ungauged_folds_k12']
    dev = set(sp['dev'])
    rows, nn = [], {'nn_dev': {}, 'nn_k12': {}}
    for k, b in enumerate(ids):
        f = wd.forcing(b, '1980-10-01', '2023-12-30')
        qc = wd.streamflow(b, 'caravan').Q_mm
        try:
            qn = wd.streamflow(b, 'nwis').Q_mm
        except FileNotFoundError:
            qn = pd.Series(dtype=float)
        sims = {'gr4j': gr[b]}
        d = ((Z - Z.loc[b]) ** 2).sum(axis=1)
        for name, pool in [('nn_dev', [x for x in ids if x in dev and x != b]),
                           ('nn_k12', [x for x in ids if folds[x] != folds[b]])]:
            donors = d.loc[pool].nsmallest(5).index
            s = np.mean([gr4j_snow(params[x], f.P.values, f.T_mean.values, f.PET.values) for x in donors], axis=0)
            sims[name] = pd.Series(s.astype(np.float32), index=f.index)
            nn[name][b] = sims[name]
        g = b.split('_')[1]
        if g in nwm_l.columns:
            sims['nwm'] = nwm_l[g] * 86.4 / wd.spec(b)['area_km2']
        qcal = qc.loc[CAL[0]:CAL[1]]
        for per, (s0, s1), obs in [('val', VAL, qc), ('recent', REC, qn)]:
            o = obs.loc[s0:s1]
            if o.notna().sum() < 365:
                continue
            cand = dict(sims)
            cand['clim'] = clim(qcal, o.index)
            cand['persist1'] = obs.shift(1, freq='D').reindex(o.index)
            for m, s in cand.items():
                oo = o
                if m == 'nwm' and per == 'recent':
                    oo = o.loc[REC_NWM[0]:REC_NWM[1]]
                r = M.summary(oo, s.reindex(oo.index))
                r.update({'basin': b, 'method': m, 'period': per, 'split': 'dev' if b in dev else 'exam',
                          'in_531': bool(a.loc[b, 'in_531'])})
                rows.append(r)
        if k % 50 == 0:
            print(k, b, flush=True)
    sc = pd.DataFrame(rows)
    sc.to_parquet(f'{B}/flow_scores.parquet')
    for name, dct in nn.items():
        pd.DataFrame(dct).to_parquet(f'{B}/sims/{name}.parquet', compression='zstd')
    summ = {}
    for (per, m), g in sc[sc.split == 'exam'].groupby(['period', 'method']):
        g531 = g[g.in_531]
        summ[f'{per}/{m}'] = {
            'n': int(len(g)), 'nse_median': float(g.nse.median()), 'kge_median': float(g.kge.median()),
            'nse_median_531': float(g531.nse.median()) if len(g531) else None,
            'kge_median_531': float(g531.kge.median()) if len(g531) else None,
            'fhv_median': float(g.fhv.median()), 'flv_median': float(g.flv.median()),
            'peak_bias_pct_median': float(g.peak_bias_pct.median()),
            'peak_abs_err_pct_median': float(g.peak_abs_err_pct_median.median()),
            'peak_timing_days_median': float(g.peak_timing_days.median()),
            'frac_nse_below_0': float((g.nse < 0).mean())}
    json.dump(summ, open(f'{B}/flow_summary.json', 'w'), indent=1)
    print(pd.DataFrame(summ).T[['n', 'nse_median', 'kge_median', 'nse_median_531', 'fhv_median', 'peak_abs_err_pct_median']].round(3))


if __name__ == '__main__':
    main()
