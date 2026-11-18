"""Twin test on the 52 dev basins (non-exam windows only).

build:   python3 twin.py build
  For every dev basin and each of the five dev-calibrated model+snow pairs (water/checks/dev_calibrated_params.json),
  simulate the basin's twin with its real ERA5-Land forcing in the calibration window (1999-10..2008-09; warm-up on
  its own first year) and in the early window (1980-10..1989-09; spin-up 1979-80). Gauge noise as in the bank; the
  real record's gaps are copied onto the twin. Features + labels exactly as for the bank (true P multiplier = 1).
report:  python3 twin.py report <model.npz>
  1) parameter recovery on twins (known calibrated parameters): R2 / Spearman / coverage
  2) out-of-period change on twins (twin truth) vs on real (real truth)
  3) feature gap real vs twin (median standardised difference): where the simulator and reality differ
"""
import common  # noqa: F401
from _paths import DEV_PARAMS
import json
import sys

import numpy as np
import pandas as pd

import watersim as ws
import waterdata as wd
from watersim.labels import CAPACITY_LABEL
from common import CAL, EARLY, D
from features import featurize, sigs, delta_labels

PAIRS = ['hbv+degree_day', 'hbv+energy_balance', 'gr4j+degree_day', 'sacsma+degree_day', 'vic_lite+degree_day']


def kslow(model, pm, gw_mean):
    if model == 'hbv':
        return pm['K2']
    if model == 'sacsma':
        return pm['LZPK']
    if model == 'gr4j':
        R = max(gw_mean, 1e-6)
        X3 = pm['X3']
        return float(1 - (1 + (R / X3) ** 4) ** -0.25 + (R / X3) ** 4 * (1 + (R / X3) ** 4) ** -1.25)
    return pm['DS'] * pm['DSMAX'] / (pm['WS'] * pm['W2M'])


def noisy(q, rng, area):
    """Gauge noise of the physics version in env PHYS (v1: watersim default; v2+: dev-fitted, calib_gauge.py)."""
    import os
    if os.environ.get('PHYS', 'v1') == 'v1':
        return ws.noise.streamflow_obs(q, rng)
    from calib_gauge import gauge
    gp = json.load(open(f'{D}/models/gauge_v2.json'))
    return gauge(q, rng, area, gp['rho'], gp['cv'], gp['sig'])


def build():
    P = json.load(open(DEV_PARAMS))['params']
    rng = np.random.default_rng(20261118)
    X, rows = [], []
    names = None
    for b in wd.basins('camels', 'dev'):
        s = wd.spec(b)
        meta = wd.forcing_meta(b)
        fe = wd.forcing(b, '1979-10-01', EARLY[1])
        fc = wd.forcing(b, *CAL)
        q = wd.streamflow(b, 'caravan').Q_mm
        for pair in PAIRS:
            if pair not in P[b]:
                continue
            pr = P[b][pair]
            pr = eval(pr) if isinstance(pr, str) else pr
            model, snow = pair.split('+')
            oe = ws.simulate(s, (fe, meta), EARLY[0], EARLY[1], model=model, snow=snow, params=pr, labels=False)
            oc = ws.simulate(s, (fc, meta), CAL[0], CAL[1], model=model, snow=snow, params=pr, labels=False)
            qe = pd.Series(noisy(oe['ts']['Q'].values, rng, s['area_km2']), index=oe['ts'].index)
            qc = pd.Series(noisy(oc['ts']['Q'].values, rng, s['area_km2']), index=oc['ts'].index)
            qe = qe.where(q.reindex(qe.index).notna())
            qc = qc.where(q.reindex(qc.index).notna())
            cal = fc[['P', 'T_mean', 'PET']]
            tgt = fe.loc[EARLY[0]:EARLY[1], ['P', 'T_mean', 'PET']]
            x, s_cal, names = featurize(cal, qc.values, tgt)
            s_t = sigs(qe, tgt.P, tgt.PET, tgt.T_mean)
            r = {'basin': b, 'pair': pair}
            r.update({f'twin_{k}': v for k, v in delta_labels(s_cal, s_t).items()})
            ts = oc['ts']
            pm = oc['params']['model']
            r['log_kslow'] = float(np.log(1 / max(kslow(model, pm, float(ts['groundwater'].mean())), 1e-5)))
            r['gw_share'] = float(ts['Q_base_routed'].sum() / max(ts['Q_natural'].sum(), 1e-9))
            r['log_soilcap'] = float(np.log(max(CAPACITY_LABEL[model](pm), 1.0)))
            r['log_pmult'] = 0.0
            X.append(x)
            rows.append(r)
        print(b, flush=True)
    np.savez_compressed(f'{D}/feats/twin_dev.npz', X=np.array(X, np.float32), names=np.array(names))
    pd.DataFrame(rows).to_parquet(f'{D}/feats/twin_dev_meta.parquet')


def report(model_path, out_json=None):
    from scipy.stats import spearmanr
    from net import Detective
    import evaluate as E
    m = Detective(model_path)
    z = np.load(f'{D}/feats/twin_dev.npz')
    Xt = z['X']
    tm = pd.read_parquet(f'{D}/feats/twin_dev_meta.parquet')
    Xr, rm, names = E.real_set('dev')
    mu_t, sd_t, p_t = m.forward(Xt)
    mu_r, sd_r, p_r = m.forward(Xr)
    res = {'param_recovery': {}, 'delta_twin': {}, 'delta_real': {}, 'twin_minus_real_pred': {}, 'feature_gap': {}}
    for t in ('log_kslow', 'gw_share', 'log_soilcap', 'log_pmult'):
        j = m.targets.index(t)
        y = tm[t].values
        res['param_recovery'][t] = {
            'rho': float(spearmanr(mu_t[:, j], y).statistic) if np.std(y) > 0 else None,
            'r2': E.r2(y, mu_t[:, j]), 'median_abs_err': float(np.median(np.abs(mu_t[:, j] - y))),
            'bias': float(np.median(mu_t[:, j] - y)), 'cov68': E.coverage(y, mu_t[:, j], sd_t[:, j]),
            'median_sd': float(np.median(sd_t[:, j]))}
    ridx = {b: i for i, b in enumerate(rm.basin)}
    ri = np.array([ridx[b] for b in tm.basin])
    for t in ('d_logq', 'd_rr', 'd_bfi', 'd_logq5', 'd_logq95', 'd_hfd'):
        j = m.targets.index(t)
        y = tm[f'twin_{t}'].values
        ok = np.isfinite(y)
        res['delta_twin'][t] = {'rho': float(spearmanr(mu_t[ok, j], y[ok]).statistic),
                                'cov68': E.coverage(y, mu_t[:, j], sd_t[:, j])}
        yr = rm[f'true_{t}'].values
        okr = np.isfinite(yr)
        res['delta_real'][t] = {'rho': float(spearmanr(mu_r[okr, j], yr[okr]).statistic),
                                'cov68': E.coverage(yr, mu_r[:, j], sd_r[:, j])}
        # does the twin's own change resemble the real change? (simulator skill, not the network)
        res['delta_real'][t]['twin_truth_vs_real_truth_rho'] = float(
            pd.Series(y).groupby(tm.basin.values).mean().reindex(rm.basin).corr(
                pd.Series(yr, index=rm.basin), method='spearman'))
        res['twin_minus_real_pred'][t] = float(np.median(mu_t[:, j] - mu_r[ri, j]))
    res['p_regulated'] = {'twin_median': float(np.median(p_t)), 'real_median': float(np.median(p_r)),
                          'real_unregulated_median': float(np.median(p_r[rm.dor.values <= 0.1]))}
    Zt, Zr = m.norm(Xt), m.norm(Xr)
    gap = np.median(Zr[ri] - Zt, axis=0)
    order = np.argsort(-np.abs(gap))[:15]
    res['feature_gap'] = {names[i]: float(gap[i]) for i in order}
    if out_json:
        json.dump(res, open(out_json, 'w'), indent=1)
    return res


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build()
    else:
        print(json.dumps(report(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None), indent=1))
