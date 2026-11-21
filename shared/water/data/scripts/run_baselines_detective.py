"""BasinDetective classical baselines on CAMELS-US (run after run_gr4j.py).

Signatures (waterdata.signatures) are computed per basin from observed Q in the calibration (1999-10..2008-09) and
validation (1989-10..1999-09) periods -> proc/baselines/signatures.parquet.
Exam target = the validation-period signature of each EXAM basin. Estimators:
  dev_median  median over dev basins (dumb)
  knn_attr    mean of the 5 most similar DEV basins (catchment attributes only; ungauged)
  budyko_fu   runoff ratio only: 1 - Fu(aridity, w=2.6) from the basin's own val-period forcing (no Q)
  gr4j_sim    signature of the basin's GR4J simulation in the val period (gauged model, calibrated on cal period)
  cal_period  the same signature from the basin's own calibration-period Q (gauged; how stable signatures are)
Scores per signature over exam basins: Spearman rho, R2 (1 - SSE/SST), median |error|.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root
sys.path.insert(0, _CODE)
import waterdata as wd
from waterdata.signatures import signatures, budyko_fu

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
B = f'{ROOT}/proc/baselines'
PER = {'cal': ('1999-10-01', '2008-09-30'), 'val': ('1989-10-01', '1999-09-30')}
SIM_ATTRS = ['p_mean', 'aridity_FAO_PM', 'frac_snow', 'high_prec_freq', 'seasonality_FAO_PM', 'ele_mt_sav', 'slope_deg',
             'log_area', 'for_pc_sse', 'snd_pc_sav', 'cly_pc_sav', 'kar_pc_sse']
TARGETS = ['q_mean', 'runoff_ratio', 'bfi_lh', 'bfi_eckhardt', 'rec_k_days', 'rec_b', 'slope_fdc', 'hfd_mean',
           'high_q_freq', 'low_q_freq', 'q5', 'q95', 'elasticity_p']


def main():
    sp = wd.splits()['camels_us']
    a = wd.attributes('camels'); a['log_area'] = np.log10(a.nwis_drain_area_km2)
    gr = pd.read_parquet(f'{B}/sims/gr4j_gauged.parquet')
    rows = []
    for b in a.index:
        f = wd.forcing(b, '1980-10-01', '2014-12-31')
        q = wd.streamflow(b, 'caravan').Q_mm
        for per, (s0, s1) in PER.items():
            ff = f.loc[s0:s1]
            r = signatures(q.loc[s0:s1], ff.P, ff.PET, ff.T_mean); r.update({'basin': b, 'period': per, 'source': 'obs'})
            rows.append(r)
            if per == 'val':
                r = signatures(gr[b].loc[s0:s1].where(q.loc[s0:s1].notna()), ff.P, ff.PET, ff.T_mean)
                r.update({'basin': b, 'period': per, 'source': 'gr4j_sim'}); rows.append(r)
    sig = pd.DataFrame(rows)
    sig.to_parquet(f'{B}/signatures.parquet')
    obs_val = sig[(sig.source == 'obs') & (sig.period == 'val')].set_index('basin')
    obs_cal = sig[(sig.source == 'obs') & (sig.period == 'cal')].set_index('basin')
    sim_val = sig[(sig.source == 'gr4j_sim')].set_index('basin')
    dev, exam = sp['dev'], sp['exam']
    Z = (a[SIM_ATTRS] - a[SIM_ATTRS].mean()) / a[SIM_ATTRS].std()
    est = {'dev_median': pd.DataFrame({t: obs_cal.loc[dev, t].median() for t in TARGETS}, index=exam),
           'cal_period': obs_cal.loc[exam, TARGETS], 'gr4j_sim': sim_val.loc[exam, TARGETS]}
    knn = {}
    for b in exam:
        d = ((Z.loc[dev] - Z.loc[b]) ** 2).sum(axis=1).nsmallest(5).index
        knn[b] = obs_cal.loc[d, TARGETS].mean()
    est['knn_attr'] = pd.DataFrame(knn).T
    est['budyko_fu'] = pd.DataFrame({'runoff_ratio': 1 - budyko_fu(obs_val.loc[exam, 'aridity'].values)}, index=exam)
    out = {}
    for m, e in est.items():
        for t in e.columns:
            y = obs_val.loc[exam, t]; x = e[t].reindex(exam)
            ok = y.notna() & x.notna() & np.isfinite(y) & np.isfinite(x)
            if ok.sum() < 20:
                continue
            y, x = y[ok], x[ok]
            out[f'{t}/{m}'] = {'n': int(ok.sum()), 'spearman': float(spearmanr(x, y).statistic),
                               'r2': float(1 - ((x - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                               'median_abs_err': float((x - y).abs().median())}
    json.dump(out, open(f'{B}/detective_summary.json', 'w'), indent=1)
    df = pd.DataFrame(out).T
    print(df.round(3).to_string())
    print('basins with val runoff_ratio > 1:', int((obs_val.runoff_ratio > 1).sum()))


if __name__ == '__main__':
    main()
