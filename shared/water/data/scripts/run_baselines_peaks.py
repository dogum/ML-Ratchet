"""Flood-peak exam baselines: USGS annual INSTANTANEOUS peaks of exam basins, water years 1990-1999 (validation) and
2016-2023 (recent). Peaks with codes 5/6/C (regulation, diversion, urbanisation) or 7 (historic) or partial dates are
excluded. Estimators (m3/s):
  cal_median   median annual instantaneous peak of the basin's own calibration years WY2000-2008 (gauged, dumb)
  gr4j_ratio   annual max of the GR4J daily simulation x R, R = median over DEV basins and WY2000-2008 of
               (instantaneous peak / annual max daily mean flow)
  nwm_hourly   annual max of NWM hourly flow (recent only, WY2016-2022: hourly kept only for 2015-10..2023-01)
  nwm_ratio    annual max of NWM daily-mean x R (both periods; WY<=2022)
Scores: median over exam basins of the per-basin median |% error| and of the median % bias; and pooled Spearman.
Out: proc/baselines/peak_scores.parquet, peak_summary.json
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root
sys.path.insert(0, _CODE)
import waterdata as wd

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
B = f'{ROOT}/proc/baselines'


def wyear(idx):
    return idx.year + (idx.month >= 10).astype(int)


def ann_max(s, minn=330):
    s = s.dropna(); w = wyear(s.index); g = s.groupby(w)
    c = g.count()
    return g.max()[c >= minn]


def main():
    sp = wd.splits()['camels_us']
    pk = pd.read_parquet(f'{ROOT}/proc/nwis/peaks.parquet')
    bad = pk.peak_cd.str.contains(r'(^|,)(5|6|7|C)(,|$)', regex=True) | pk.date_partial | pk.peak_cms.isna()
    pk = pk[~bad]
    gr = pd.read_parquet(f'{B}/sims/gr4j_gauged.parquet')
    nwm = pd.read_parquet(f'{ROOT}/proc/nwm/daily_local.parquet')
    import glob
    hmax = []
    for f in sorted(glob.glob(f'{ROOT}/raw/nwm/chanobs_hourly/*.parquet')):
        h = pd.read_parquet(f)
        hmax.append(h.groupby(wyear(h.index)).max())
    hmax = pd.concat(hmax).groupby(level=0).max().loc[2016:2022]
    # ratio inst peak / daily max on dev basins, WY2000-2008
    rat = []
    for b in sp['dev']:
        q = wd.streamflow(b, 'nwis').Q_cms.loc['1999-10-01':'2008-09-30']
        am = ann_max(q); p = pk[pk.gauge_id == b].set_index('wy').peak_cms
        j = pd.concat([am, p], axis=1, keys=['d', 'i']).dropna()
        rat += list((j.i / j.d).values)
    rat = [x for x in rat if np.isfinite(x) and x > 0]
    R = float(np.median(rat))
    rows = []
    for b in sp['exam']:
        p = pk[pk.gauge_id == b].groupby('wy').peak_cms.max()
        if len(p) == 0:
            continue
        area = wd.spec(b)['area_km2']
        cal = p.loc[2000:2008]
        g = b.split('_')[1]
        est = {'gr4j_ratio': ann_max(gr[b] * area / 86.4) * R}
        if len(cal) >= 5:
            est['cal_median'] = pd.Series(cal.median(), index=p.index)
        if g in nwm:
            est['nwm_ratio'] = ann_max(nwm[g]) * R
        for per, (y0, y1) in {'val': (1990, 1999), 'recent': (2016, 2023)}.items():
            obs = p.loc[y0:y1]
            if per == 'recent' and g in hmax:
                est['nwm_hourly'] = hmax[g]
            for m, e in est.items():
                if m == 'nwm_hourly' and per != 'recent':
                    continue
                j = pd.concat([obs, e], axis=1, keys=['o', 'e']).dropna()
                j = j[j.o > 0]
                if per == 'recent' and m.startswith('nwm'):
                    j = j.loc[:2022]
                for y, r in j.iterrows():
                    rows.append({'basin': b, 'wy': int(y), 'period': per, 'method': m, 'obs': r.o, 'est': r.e,
                                 'pct_err': 100 * (r.e - r.o) / r.o})
    sc = pd.DataFrame(rows)
    sc.to_parquet(f'{B}/peak_scores.parquet')
    summ = {'ratio_inst_over_daily_dev': R, 'n_ratio_samples': len(rat)}
    for (per, m), g in sc.groupby(['period', 'method']):
        pb = g.groupby('basin').pct_err
        summ[f'{per}/{m}'] = {'n_basins': int(g.basin.nunique()), 'n_peaks': int(len(g)),
                              'median_abs_pct_err': float(pb.apply(lambda x: x.abs().median()).median()),
                              'median_bias_pct': float(pb.median().median()),
                              'spearman_pooled_log': float(spearmanr(np.log(g.est), np.log(g.obs)).statistic)}
    json.dump(summ, open(f'{B}/peak_summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
