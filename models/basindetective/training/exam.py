"""Final exam on the 619 CAMELS-US exam basins (run ONCE, with the released model; nothing is chosen here).

Protocol (gauged): inputs = the basin's own calibration-period (1999-10..2008-09) forcing and Q, plus the forcing of
the validation decade (1989-10..1999-09). Truth = signatures of the basin's observed validation-decade Q (Caravan, as
B's table) and HydroATLAS regulation (dor_frac). Classical baselines on the same targets (see realfeats.py).

python3 exam.py <release_model.npz>   -> exam.json, feats/exam_pred.parquet
"""
import common  # noqa: F401
from _paths import WATER_PROC
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import D, DELTA_T, EPSQ
from net import Detective
import evaluate as E

LEVEL = {'d_logq': 'q_mean', 'd_rr': 'runoff_ratio', 'd_bfi': 'bfi_lh', 'd_logq5': 'q5', 'd_logq95': 'q95',
         'd_hfd': 'hfd_mean'}
LOGT = {'d_logq', 'd_logq5', 'd_logq95'}


def stats(y, x, sd=None):
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    r = {'n': int(ok.sum()), 'rho': float(spearmanr(x, y).statistic) if np.std(x) > 0 else None,
         'mae': float(np.median(np.abs(x - y))), 'r2': E.r2(y, x)}
    if sd is not None:
        s = sd[ok]
        r['cov68'] = float((np.abs(y - x) <= s).mean())
        r['cov95'] = float((np.abs(y - x) <= 1.96 * s).mean())
    return r


def level_from_delta(t, cal, d):
    if t in LOGT:
        return np.exp(np.log(cal + EPSQ) + d) - EPSQ
    return cal + d


def delta_vs_obs(t, meta):
    lv = LEVEL[t]
    g, c = meta[f'gr4jlvl_{lv}'].values, meta[f'cal_{lv}'].values
    if t in LOGT:
        return np.log(g + EPSQ) - np.log(c + EPSQ)
    return g - c


def main(model_path, split='exam'):
    if split == 'exam' and not os.path.exists(f'{D}/feats/real_exam.npz'):
        import realfeats
        realfeats.build('exam')
    m = Detective(model_path)
    X, meta, names = E.real_set(split)
    assert names == m.names
    mu, sd, p = m.forward(X)
    out = {'model': os.path.basename(model_path), 'n_basins': int(len(meta)), 'delta': {}, 'level': {},
           'regulation': {}, 'strata': {}, 'params_plausibility': {}}
    pred = meta[['basin']].copy()
    for j, t in enumerate(m.targets):
        pred[f'mu_{t}'] = mu[:, j]
        pred[f'sd_{t}'] = sd[:, j]
    pred['p_reg'] = p
    # --- out-of-period change
    for t in DELTA_T:
        j = m.targets.index(t)
        y = meta[f'true_{t}'].values
        r = {'BasinDetective': stats(y, mu[:, j], sd[:, j]), 'GR4J (gauged, model-implied change)':
             stats(y, meta[f'gr4j_{t}'].values),
             'GR4J (gauged, sim target period minus observed cal)': stats(y, delta_vs_obs(t, meta)),
             'no change (persistence)': {'mae': float(np.nanmedian(np.abs(y)))}}
        if f'budyko_{t}' in meta:
            r['Budyko-Fu (w fitted on cal)'] = stats(y, meta[f'budyko_{t}'].values)
        out['delta'][t] = r
    # --- validation-decade levels (= calibration signature + change)
    sig = pd.read_parquet(f'{WATER_PROC}/baselines/signatures.parquet')  # read but unused, as run
    for t, lv in LEVEL.items():
        j = m.targets.index(t)
        cal = meta[f'cal_{lv}'].values
        y = meta[f'tgtobs_{lv}'].values
        r = {'BasinDetective': stats(y, level_from_delta(t, cal, mu[:, j])),
             'own calibration-period value': stats(y, cal),
             'GR4J sim (gauged)': stats(y, meta[f'gr4jlvl_{lv}'].values)}
        out['level'][lv] = r
    # --- regulation
    dor = meta['dor'].values
    for name, y in (('dor>0.1 (mid+high)', dor > 0.1), ('dor>0.5 (high)', dor > 0.5), ('dor>0 (any)', dor > 0),
                    ('NWIS peak code 5/6 in >=1/3 of WY1990-2008 peaks', meta['peak_reg_frac'].values >= 1 / 3)):
        y = y.astype(float)
        out['regulation'][name] = {'n_pos': int(y.sum()), 'BasinDetective_auc': E.auc(y, p),
                                   'GR4J misfit (1-NSE_cal)_auc': E.auc(y, 1 - meta['gr4j_nse_cal'].values),
                                   'seasonal damping index_auc': E.auc(y, -meta['seasdamp'].values),
                                   'BasinDetective_precision_at_n': float(y[np.argsort(-p)[:int(y.sum())]].mean())}
    jd = m.targets.index('log_dor')
    rg = dor > 0
    out['regulation']['log10 dor among regulated basins'] = stats(np.log10(dor[rg]), mu[rg, jd], sd[rg, jd])
    pr = (p > 0.5).astype(float)
    yb = (dor > 0.1).astype(float)
    out['regulation']['confusion_p>0.5'] = {'tp': int((pr * yb).sum()), 'fp': int((pr * (1 - yb)).sum()),
                                            'fn': int(((1 - pr) * yb).sum()), 'tn': int(((1 - pr) * (1 - yb)).sum())}
    # --- strata (change targets, network vs GR4J rho)
    fs = X[:, names.index('c_fsnow')]
    ar = X[:, names.index('c_arid')]
    strata = {'snow (frac_snow>=0.15)': fs >= 0.15, 'rain (frac_snow<0.15)': fs < 0.15, 'arid (PET/P>1)': ar > 1,
              'humid (PET/P<=1)': ar <= 1, 'regulated (dor>0.1)': dor > 0.1,
              'runoff ratio > 1 in cal': meta['cal_runoff_ratio'].values > 1}
    for sn, sel in strata.items():
        d = {'n': int(sel.sum())}
        for t in DELTA_T:
            j = m.targets.index(t)
            y = meta[f'true_{t}'].values[sel]
            d[t] = {'net': stats(y, mu[sel, j], sd[sel, j])['rho'], 'gr4j': stats(y, meta[f'gr4j_{t}'].values[sel])['rho']}
        out['strata'][sn] = d
    # --- hidden parameters: no real truth; plausibility only
    jp = m.targets.index('log_pmult')
    out['params_plausibility']['log_pmult vs calibration runoff ratio (rho)'] = float(
        spearmanr(mu[:, jp], meta['cal_runoff_ratio'].values, nan_policy='omit').statistic)
    out['params_plausibility']['median predicted P multiplier, cal RR>1 basins'] = float(
        np.exp(np.median(mu[meta['cal_runoff_ratio'].values > 1, jp])))
    out['params_plausibility']['median predicted P multiplier, all'] = float(np.exp(np.median(mu[:, jp])))
    jk = m.targets.index('log_kslow')
    out['params_plausibility']['log_kslow vs observed recession K (rho)'] = float(
        spearmanr(mu[:, jk], np.log(meta['cal_rec_k_days'].values), nan_policy='omit').statistic)
    jg = m.targets.index('gw_share')
    out['params_plausibility']['gw_share vs observed BFI (rho)'] = float(
        spearmanr(mu[:, jg], meta['cal_bfi_lh'].values, nan_policy='omit').statistic)
    tag = '' if split == 'exam' else f'_{split}'
    pred.to_parquet(f'{D}/feats/exam_pred{tag}.parquet')
    json.dump(out, open(f'{D}/exam{tag}.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'exam')
