"""Re-score the released v1.0 weights on the 2023-2025 exam with exam.py's own metrics (numpy, no torch).
Added in the repository (14 Nov 2026); exam.py itself is kept as run.

    python wxprep.py && python renew.py                      # weather bundles, fleets, wind/solar priors -> work/data
    python calib_demand.py dev2019 && python calib_demand.py dev2021
    python calib_renew.py dev2019 && python calib_renew.py dev2021   # per-year fits for the cross-year dev bank
    python realbank.py dev && python realbank.py exam        # real banks (exam uses the released calib/ fits)
    python rescore_release.py                                # -> work/data/exam/release.json + comparison

exam.py imports evaluate.py, which loads the torch checkpoint. Here a numpy stand-in for evaluate (load / predict /
sigma_scale) runs demandcast.py on demandcast_weights.npz with the same arithmetic as evaluate.predict, then exam.main()
runs unchanged: sigma scales are re-fitted on the dev bank, and every exam metric is computed by exam.block(). The
released numbers are in training/exam_r8_wide.json; this script prints both side by side and writes
work/data/exam/release_vs_released.json.
"""
import json
import os
import sys
import types

import numpy as np

import common as C  # noqa: F401  (paths)
from common import DATA, SRC, BAS, jdump

MODEL = os.path.dirname(SRC)
sys.path.insert(0, MODEL)
import demandcast as DC  # noqa: E402

XK = {'d': 'xd', 'w': 'xw', 's': 'xs'}
PK = {'w': 'pw', 's': 'ps'}


def masks(B, kind):   # train.masks, verbatim (train.py imports torch)
    if kind == 'd':
        y = B['yd']; m = np.isfinite(y)
    elif kind == 'w':
        y = B['yw']; m = np.isfinite(y)
        if 'hasw' in B: m &= B['hasw'][:, None]
    else:
        y = B['ys']; m = np.isfinite(y) & (B['night'] < 0.5)
        if 'hass' in B: m &= B['hass'][:, None]
    return np.nan_to_num(y).astype(np.float32), m.astype(np.float32)


def load(name):
    return DC.DemandCast().W          # the released weights, float64 (stored float32)


def predict(W, B):
    """evaluate.predict with the numpy forward: raw (unscaled) sigmas, CF clipped to 0-1.2, solar 0 at night."""
    out = {}
    for k in ('d', 'w', 's'):
        X = np.nan_to_num(B[XK[k]].astype(np.float32))[:, :W[k]['xm'].shape[0]]
        P = B[PK[k]].astype(np.float32) if k != 'd' else None
        mu, sig = DC._forward(W[k], X, k, P)
        if k != 'd':
            mu = np.clip(mu, 0, 1.2)
        if k == 's':
            night = B['night'] > 0.5
            mu = np.where(night, 0.0, mu); sig = np.where(night, DC.FLOOR['s'], sig)
        out[k] = (mu.astype(np.float32), sig.astype(np.float32))
    return out


def sigma_scale(B, pred, kind, rows=None):   # evaluate.sigma_scale, verbatim
    y, m = masks(B, kind)
    m = m.astype(bool)
    if rows is not None: m &= rows[:, None]
    mu, sig = pred[kind]
    z = np.abs(y - mu)[m] / sig[m]
    return float(max(1.0, np.quantile(z, 0.683)))


shim = types.ModuleType('evaluate')
shim.load, shim.predict, shim.sigma_scale, shim.masks = load, predict, sigma_scale, masks
sys.modules['evaluate'] = shim
import exam  # noqa: E402


def compare():
    new = json.load(open(f'{DATA}/exam/release.json'))
    old = json.load(open(os.path.join(SRC, 'exam_r8_wide.json')))
    rows = []

    def add(name, a, b, d):
        status = 'reproduced' if round(a, d) == round(b, d) else ('close' if abs(a - b) <= max(0.05 * abs(b), 10 ** -d) else 'not reproduced')
        rows.append(dict(metric=name, released=round(b, d + 2), rescored=round(a, d + 2), status=status))

    keys = {'demand': ('model', 'ba_day_ahead', 'persistence_24h', 'same_hour_last_week', 'physics_prior', 'cov68', 'cov95'),
            'wind': ('model', 'physics_prior', 'persistence_24h', 'cf_climatology', 'cov68', 'cov95'),
            'solar': ('model', 'physics_prior', 'persistence_24h', 'cf_climatology', 'cov68', 'cov95')}
    for out in ('demand', 'wind', 'solar'):
        for k in keys[out]:
            a, b = new['overall']['actual'][out][k], old['overall']['actual'][out][k]
            add(f'pooled {out} {k}', 100 * a if k.startswith('cov') else a, 100 * b if k.startswith('cov') else b, 0 if k.startswith('cov') else (2 if out == 'demand' else 1))
        add(f'pooled {out} hours', new['overall']['actual'][out]['hours'], old['overall']['actual'][out]['hours'], 0)
    for ba in BAS:
        for out in ('demand', 'wind', 'solar'):
            a, b = new['table'][ba]['all']['actual'][out], old['table'][ba]['all']['actual'][out]
            if b['model'] is None or not np.isfinite(b['model']):
                continue
            for k in ('model', 'ba_day_ahead', 'physics_prior', 'persistence_24h') if out == 'demand' else ('model', 'physics_prior', 'persistence_24h'):
                add(f'{ba} {out} {k}', a[k], b[k], 2 if out == 'demand' else 1)
            add(f'{ba} {out} cov68', 100 * a['cov68'], 100 * b['cov68'], 0)
    for k in ('d', 'w', 's'):
        add(f'sigma scale {k} (dev bank)', new['sigma_scale_dev'][k], old['sigma_scale_dev'][k], 3)
    jdump(rows, f'{DATA}/exam/release_vs_released.json')
    n = {s: sum(r['status'] == s for r in rows) for s in ('reproduced', 'close', 'not reproduced')}
    for r in rows:
        print(f"{r['metric']:42s} released {r['released']:>10} rescored {r['rescored']:>10}  {r['status']}")
    print(n)


if __name__ == '__main__':
    exam.main('release')
    compare()
