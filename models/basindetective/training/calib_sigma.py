"""Widen-only error-bar calibration on dev (one factor per output), then pack the release model.

Out-of-period change targets: the 52 real dev basins (calibration 1999-2008 -> early window 1980-89, real truth).
Hidden parameters: the dev twins (calibrated parameters are known there).
log_dor: no dev truth with enough regulated basins (5) -> factor 1 (stated in the model card).
Factor f = max(1, q68(|y - mu| / sd)) so that 68 % of dev errors fall inside the 1-sigma bar; never below 1.
'regulated' probability: temperature fitted on dev by log loss, kept >= 1 (only softening allowed).

python3 calib_sigma.py <out.npz> <member.npz> [<member.npz> ...]
"""
import common  # noqa: F401
import json
import sys

import numpy as np
import pandas as pd

from common import D, DELTA_T
from net import Detective, pack_ensemble, REG_TARGETS
import evaluate as E


def main(out, members):
    tmp = f'{D}/models/_tmp_ens.npz'
    pack_ensemble(members, tmp)
    m = Detective(tmp)
    Xr, meta, _ = E.real_set('dev')
    mu, sd, p = m.forward(Xr)
    z = np.load(f'{D}/feats/twin_dev.npz')
    tm = pd.read_parquet(f'{D}/feats/twin_dev_meta.parquet')
    mut, sdt, _ = m.forward(z['X'])
    fac = np.ones(len(REG_TARGETS))
    info = {}
    for j, t in enumerate(REG_TARGETS):
        if t in DELTA_T:
            y, a, s = meta[f'true_{t}'].values, mu[:, j], sd[:, j]
            src = 'dev real (52)'
        elif t in tm:
            y, a, s = tm[t].values, mut[:, j], sdt[:, j]
            src = 'dev twins (260)'
            if np.std(y) == 0:   # log_pmult: twins all have m = 1 -> still a valid error sample
                pass
        else:
            info[t] = {'factor': 1.0, 'source': 'none (no dev truth)'}
            continue
        ok = np.isfinite(y)
        zz = np.abs(y[ok] - a[ok]) / s[ok]
        f = float(max(1.0, np.quantile(zz, 0.6827)))
        fac[j] = f
        info[t] = {'factor': f, 'source': src, 'cov68_before': float((zz <= 1).mean()),
                   'cov68_after': float((zz <= f).mean())}
    # temperature for the regulation probability
    y = (meta['dor'].values > 0.1).astype(float)
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    best, bt = None, 1.0
    for T in np.linspace(1, 5, 41):
        q = 1 / (1 + np.exp(-lg / T))
        ll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if best is None or ll < best - 1e-9:
            best, bt = ll, float(T)
    info['reg_temperature'] = bt
    pack_ensemble(members, out, sig_scale=fac, cls_temp=bt)
    json.dump(info, open(out.replace('.npz', '_sigma.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
