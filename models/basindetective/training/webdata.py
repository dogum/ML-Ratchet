"""Data for the BasinDetective page (web/data.json) and the JS == Python check.

Examples: a fixed seeded draw of exam basins within strata (snow / rain x humid / arid, plus regulated), never
picked by score. For each: weekly log flow over the calibration and validation decades, monthly regimes, the
precomputed feature vector, the validation-decade daily forcing (quantised int16) so the page can recompute the
target-period features under the lab sliders, the truth, the network and the GR4J answers.
python3 webdata.py <release_model.npz>
"""
import common  # noqa: F401
import base64
import json
import subprocess
import sys

import numpy as np
import pandas as pd

import waterdata as wd
from common import CAL, VAL, D, DELTA_T
from features import featurize
from net import Detective
import evaluate as E

LOGT = {'d_logq', 'd_logq5', 'd_logq95'}


def b64(a, dt=np.float32):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


def model_json(m):
    mem = m.members if m.members else [m]
    out = {'targets': m.targets, 'sig_scale': [float(v) for v in np.broadcast_to(m.sig_scale, (len(m.targets),))],
           'cls_temp': m.cls_temp, 'members': []}
    for w in mem:
        W = w.W
        layers = [{'nin': int(W[f'w{i}'].shape[1]), 'nout': int(W[f'w{i}'].shape[0]), 'w': b64(W[f'w{i}']),
                   'b': b64(W[f'b{i}'])} for i in range(w.nl)]
        out['members'].append({'x_med': b64(W['x_med']), 'x_iqr': b64(np.minimum(W['x_iqr'], 3e38)),
                               'y_mu': [float(v) for v in W['y_mu']], 'y_sd': [float(v) for v in W['y_sd']],
                               'layers': layers, 'out': {'nin': int(W['wo'].shape[1]), 'nout': int(W['wo'].shape[0]),
                                                         'w': b64(W['wo']), 'b': b64(W['bo'])}})
    return out


def quantise(tgt):
    P = np.clip(np.round(tgt.P.values * 100), 0, 32767).astype(np.int16)
    T = np.clip(np.round(tgt.T_mean.values * 100), -32768, 32767).astype(np.int16)
    E_ = np.clip(np.round(tgt.PET.values * 100), 0, 32767).astype(np.int16)
    q = pd.DataFrame({'P': P / 100.0, 'T_mean': T / 100.0, 'PET': E_ / 100.0}, index=tgt.index)
    return q, P, T, E_


def pick(meta, X, names, seed=20261118):
    fs = X[:, names.index('c_fsnow')]
    ar = X[:, names.index('c_arid')]
    rng = np.random.default_rng(seed)
    strata = {'snow, humid': (fs >= 0.15) & (ar <= 1), 'snow, arid': (fs >= 0.15) & (ar > 1),
              'rain, humid': (fs < 0.15) & (ar <= 1), 'rain, arid': (fs < 0.15) & (ar > 1),
              'regulated': meta.dor.values > 0.1}
    n = {'snow, humid': 6, 'snow, arid': 4, 'rain, humid': 8, 'rain, arid': 6, 'regulated': 6}
    out = []
    ok = meta['true_d_logq'].notna().values
    for s, sel in strata.items():
        cand = np.where(sel & ok & ~meta.basin.isin([o[0] for o in out]).values)[0]
        for i in rng.choice(cand, size=min(n[s], cand.size), replace=False):
            out.append((meta.basin.iloc[i], s, int(i)))
    return out


def main(model_path):
    m = Detective(model_path)
    X, meta, names = E.real_set('exam')
    mu, sd, p = m.forward(X)
    a = wd.attributes('camels')
    ex = json.load(open(f'{D}/exam.json'))
    basins = []
    for b, stratum, i in pick(meta, X, names):
        f = wd.forcing(b, VAL[0], CAL[1])
        q = wd.streamflow(b, 'caravan').Q_mm.reindex(f.index)
        cal = f.loc[CAL[0]:CAL[1], ['P', 'T_mean', 'PET']]
        tgt, Pq, Tq, Eq = quantise(f.loc[VAL[0]:VAL[1], ['P', 'T_mean', 'PET']])
        x, s_cal, nm = featurize(cal, q.reindex(cal.index).values, tgt)
        wk = q.resample('7D').mean()
        rec = {'id': b, 'name': str(a.loc[b, 'gauge_name']).title(), 'stratum': stratum,
               'lat': float(a.loc[b, 'gauge_lat']), 'lon': float(a.loc[b, 'gauge_lon']),
               'area_km2': float(wd.spec(b)['area_km2']), 'dor': float(meta.dor.iloc[i]),
               'reg_class': str(meta.reg_class.iloc[i]),
               'week0': str(wk.index[0].date()), 'weekly_logq': [None if not np.isfinite(v) else round(float(np.log10(v + 0.01)), 3) for v in wk.values],
               'x': [float(v) if np.isfinite(v) else None for v in x],
               'cal_clim': None, 'tgt_P': b64(Pq, np.int16), 'tgt_T': b64(Tq, np.int16), 'tgt_E': b64(Eq, np.int16),
               'tgt_start': str(tgt.index[0].date()), 'n_tgt': int(len(tgt)),
               'cal': {k: float(meta[f'cal_{k}'].iloc[i]) for k in ('q_mean', 'runoff_ratio', 'bfi_lh', 'q5', 'q95', 'hfd_mean')},
               'truth': {t: (None if not np.isfinite(meta[f'true_{t}'].iloc[i]) else float(meta[f'true_{t}'].iloc[i])) for t in DELTA_T},
               'gr4j': {t: (None if not np.isfinite(meta[f'gr4j_{t}'].iloc[i]) else float(meta[f'gr4j_{t}'].iloc[i])) for t in DELTA_T},
               'pred': {t: [float(mu[i, j]), float(sd[i, j])] for j, t in enumerate(m.targets)},
               'p_reg': float(p[i])}
        from features import _climate
        cc, Pm, Em, Tm = _climate(cal)
        rec['cal_clim'] = {'c': {k: float(v) for k, v in cc.items()}, 'Pm': [float(v) for v in Pm],
                           'Tm': [float(v) for v in Tm]}
        # python reference for the lab at a few slider settings (JS check)
        rec['check'] = []
        for (aa, dd) in ((1.0, 0.0), (0.85, 1.5), (1.15, -2.0)):
            t2 = tgt.copy()
            t2['P'] = t2.P * aa
            t2['T_mean'] = t2.T_mean + dd
            t2['PET'] = t2.PET * (1 + 0.025 * dd)
            x2, _, _ = featurize(cal, q.reindex(cal.index).values, t2)
            mu2, sd2, p2 = m.forward(x2[None])
            rec['check'].append({'a': aa, 'd': dd, 'mu': [float(v) for v in mu2[0]], 'sd': [float(v) for v in sd2[0]],
                                 'p': float(p2[0])})
        basins.append(rec)
        print(b, stratum, flush=True)
    data = {'names': names, 'model': model_json(m), 'basins': basins, 'exam': ex,
            'ratchet': json.load(open(f'{D}/ratchet.json')),
            'twin': json.load(open(f'{D}/logs/twin_final.json')),
            'sigma': json.load(open(model_path.replace('.npz', '_sigma.json')))}
    json.dump(data, open(f'{D}/web/data.json', 'w'), default=float, separators=(',', ':'))
    print('web/data.json written')


if __name__ == '__main__':
    main(sys.argv[1])
