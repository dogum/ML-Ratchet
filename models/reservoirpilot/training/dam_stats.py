"""Dimensionless statistics of real reservoir tasks, used ONLY to set the simulated-task generator's ranges.
Dev dams (B's task exactly) and aux dams (B: 'usable for single-process calibration, never scored'; same task
construction with a relaxed 5-year minimum). Exam dams are never read here.
Out: data/dam_stats.parquet"""
import sys
import numpy as np, pandas as pd
import _paths  # noqa: E402  (repository paths; puts shared/water on sys.path)
from task import wd, doy_curve, K  # noqa: E402

def stats_from(cal, ev, cap):
    I = ev.inflow_cms.values; R = ev.release_cms.values
    mI = float(np.nanmean(cal.inflow_cms))
    Dc = doy_curve(cal.release_cms.dropna()); Ic = doy_curve(cal.inflow_cms.dropna())
    Qmax = float(cal.release_cms.quantile(0.99))
    lI = np.log(np.maximum(I, 1e-3 * max(mI, 1e-6)))
    return {'cap_years': cap / max(mI * 365.25 * K, 1e-9), 'qmax_over_I': Qmax / max(mI, 1e-9),
            'D_over_I': float(Dc.mean()) / max(mI, 1e-9), 'D_seas': float((Dc.max() - Dc.min()) / max(Dc.mean(), 1e-9)),
            'I_seas': float((Ic.max() - Ic.min()) / max(Ic.mean(), 1e-9)), 'I_cv': float(np.std(I) / max(np.mean(I), 1e-9)),
            'I_ac1': float(np.corrcoef(I[:-1], I[1:])[0, 1]), 'logI_ac1': float(np.corrcoef(lI[:-1], lI[1:])[0, 1]),
            'neg_frac': float(np.mean(I < 0)), 'I_p99_over_mean': float(np.quantile(I, 0.99) / max(np.mean(I), 1e-9)),
            'S0_frac': float(ev.storage_hm3.iloc[0] / cap), 'S_mean_frac': float(ev.storage_hm3.mean() / cap),
            'R_over_I': float(np.mean(R) / max(np.mean(I), 1e-9)), 'Dc_I_corr': float(np.corrcoef(Dc.values, Ic.values)[0, 1]),
            'n_eval': len(ev), 'mean_I_cms': mI}


def main():
    sp = wd.splits()['reservoirs']; meta = wd.reservoir_meta().set_index('res_id')
    rows = []
    for split in ('dev', 'aux'):
        for rid in sp[split]:
            d = wd.reservoir(rid).loc['1980-01-01':][['storage_hm3', 'inflow_cms', 'release_cms']].dropna()
            minlen = 3650 if split == 'dev' else 1826
            if len(d) < minlen:
                continue
            d = d.asfreq('D')
            d[['inflow_cms', 'release_cms']] = d[['inflow_cms', 'release_cms']].interpolate(limit=5)
            d['storage_hm3'] = d.storage_hm3.interpolate(limit=5)
            half = d.index[len(d) // 2]
            cal, ev = d.loc[:half].dropna(), d.loc[half:].dropna()
            blk = (ev.index.to_series().diff() != pd.Timedelta('1D')).cumsum()
            ev = ev[blk == blk.value_counts().idxmax()]
            if len(ev) < 700 or cal.inflow_cms.mean() <= 0:
                continue
            cap = max(float(meta.loc[rid, 'cap_hm3']), float(d.storage_hm3.max()))
            r = {'res_id': rid, 'split': split, 'main_use': meta.loc[rid, 'main_use'], 'mb_corr': meta.loc[rid, 'mb_corr']}
            r.update(stats_from(cal, ev.iloc[:3653], cap)); rows.append(r)
    df = pd.DataFrame(rows)
    df.to_parquet(_paths.WORK + '/data/dam_stats.parquet')
    q = df.drop(columns=['res_id', 'split', 'main_use']).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T
    print(df.split.value_counts()); print(q.round(3).to_string())


if __name__ == '__main__':
    main()
