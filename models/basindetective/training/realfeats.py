"""Real CAMELS-US features, truths and classical baselines for BasinDetective.

python3 realfeats.py dev    -> feats/real_dev.npz   (input: cal 1999-2008 forcing+Q; target window EARLY 1980-89)
python3 realfeats.py exam   -> feats/real_exam.npz  (input: cal 1999-2008 forcing+Q; target window VAL 1989-99)
                                (run once, at the end, for the final report only)

Truth for the out-of-period targets: signatures of the basin's own observed Q (Caravan, as B's tables) in the target
window minus the same signatures in the calibration window.
Classical baselines, same targets:
  gr4j      B's per-basin GR4J+snow calibrated on 1999-2008 Q (sims/gr4j_gauged.parquet), signature change between
            its own simulations of the two windows (model-implied change)
  budyko    Fu curve with w fitted to the calibration-period runoff ratio, re-evaluated at the target aridity
            (runoff ratio and mean flow only)
  persist   no change (zero; Spearman undefined, scored by MAE only)
Regulation truth: HydroATLAS upstream reservoir volume / natural annual flow (B's dor_frac, reg_class).
Classical regulation scores: GR4J calibration misfit (1 - NSE_cal) and the seasonal-damping index.
"""
import common  # noqa: F401
from _paths import WATER_PROC
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import waterdata as wd
from common import CAL, VAL, EARLY, D, EPSQ, lq
from features import featurize, sigs, delta_labels

B = f'{WATER_PROC}/baselines'


def fu_evap(phi, w):
    return 1 + phi - (1 + phi ** w) ** (1 / w)


def fit_w(rr, phi):
    target = 1 - rr
    lo, hi = 1.0001, 30.0
    f = lambda w: fu_evap(phi, w) - target
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return hi
    return brentq(f, lo, hi)


def budyko_delta(s_cal, P_cal, PET_cal, P_t, PET_t):
    phi_c = PET_cal / P_cal
    phi_t = PET_t / P_t
    w = fit_w(s_cal['runoff_ratio'], phi_c)
    rr_c = 1 - fu_evap(phi_c, w)
    rr_t = 1 - fu_evap(phi_t, w)
    q_t = max(s_cal['runoff_ratio'] + (rr_t - rr_c), 0) * P_t
    return {'d_rr': rr_t - rr_c, 'd_logq': float(np.log(q_t + EPSQ) - np.log(s_cal['q_mean'] + EPSQ)), 'w': w}


def build(split):
    tw = EARLY if split == 'dev' else VAL
    ids = wd.basins('camels', split)
    a = wd.attributes('camels')
    gr = pd.read_parquet(f'{B}/sims/gr4j_gauged.parquet', columns=[i for i in ids])
    X, rows = [], []
    names = None
    for b in ids:
        f = wd.forcing(b, tw[0], CAL[1])
        q = wd.streamflow(b, 'caravan').Q_mm
        cal = f.loc[CAL[0]:CAL[1], ['P', 'T_mean', 'PET']]
        tgt = f.loc[tw[0]:tw[1], ['P', 'T_mean', 'PET']]
        qc = q.reindex(cal.index)
        qt = q.reindex(tgt.index)
        x, s_cal, names = featurize(cal, qc.values, tgt)
        s_t = sigs(qt, tgt.P, tgt.PET, tgt.T_mean)
        r = {'basin': b, 'n_q_cal': int(qc.notna().sum()), 'n_q_tgt': int(qt.notna().sum())}
        if len(s_t) and r['n_q_tgt'] >= 3 * 365:
            r.update({f'true_{k}': v for k, v in delta_labels(s_cal, s_t).items()})
        for k in ('q_mean', 'runoff_ratio', 'bfi_lh', 'q5', 'q95', 'hfd_mean', 'rec_k_days', 'slope_fdc'):
            r[f'cal_{k}'] = s_cal.get(k, np.nan)
            r[f'tgtobs_{k}'] = s_t.get(k, np.nan) if len(s_t) else np.nan
        # GR4J model-implied change (masked like the observations)
        g = gr[b]
        gc = sigs(g.reindex(cal.index).where(qc.notna()), cal.P, cal.PET, cal.T_mean)
        gt = sigs(g.reindex(tgt.index).where(qt.notna()) if qt.notna().sum() > 365 else g.reindex(tgt.index),
                  tgt.P, tgt.PET, tgt.T_mean)
        r.update({f'gr4j_{k}': v for k, v in delta_labels(gc, gt).items()})
        for k in ('q_mean', 'runoff_ratio', 'bfi_lh', 'q5', 'q95', 'hfd_mean'):
            r[f'gr4jlvl_{k}'] = gt.get(k, np.nan)
        # Budyko-Fu, w fitted on the calibration period
        bd = budyko_delta(s_cal, cal.P.mean(), cal.PET.mean(), tgt.P.mean(), tgt.PET.mean())
        r['budyko_d_rr'] = bd['d_rr']
        r['budyko_d_logq'] = bd['d_logq']
        r['budyko_w'] = bd['w']
        # regulation truth + classical scores
        r['dor'] = float(a.loc[b, 'dor_frac'])
        r['reg_class'] = a.loc[b, 'reg_class']
        gj = json.load(open(f'{B}/gr4j/{b}.json'))
        r['gr4j_nse_cal'] = gj['nse_cal']
        r['seasdamp'] = float(x[names.index('q_seasdamp')])
        X.append(x)
        rows.append(r)
    meta = pd.DataFrame(rows)
    # NWIS peak codes 5 (regulation) / 6 (diversion) in WY1990-2008: an independent regulation flag
    pk = pd.read_parquet(f'{WATER_PROC}/nwis/peaks.parquet')
    pk = pk[(pk.wy >= 1990) & (pk.wy <= 2008) & pk.gauge_id.isin(ids)]
    cd = pk.peak_cd.astype(str)
    flag = pk.assign(r56=cd.str.contains('5') | cd.str.contains('6')).groupby('gauge_id').r56.mean()
    meta['peak_reg_frac'] = meta.basin.map(flag).fillna(0.0).values
    out = f'{D}/feats/real_{split}.npz'
    np.savez_compressed(out, X=np.array(X, np.float32), names=np.array(names))
    meta.to_parquet(f'{D}/feats/real_{split}_meta.parquet')
    print(split, len(ids), 'basins ->', out)


if __name__ == '__main__':
    build(sys.argv[1])
