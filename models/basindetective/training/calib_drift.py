"""Physics rung v3 (dev only): decadal drift of the forcing precipitation.

Twin test finding (v1): calibrated twins of the 52 dev basins, driven by ERA5-Land, say the early decade (1980-89) had
+21 % more streamflow than 1999-2008 (median log ratio +0.21; GR4J: +0.17); the gauges say +6 %. ERA5-Land shows
+8.6 % more precipitation in the 1980s. The forcing product's precipitation level is not stationary between decades
(a known reanalysis issue: the observing system changes), so a network trusting the target-period forcing too much
will over-predict changes.

Measurement: for every dev basin and calibrated model, the twin's precipitation elasticity of the change (re-simulated
with the early-decade P x exp(-0.1)) converts the twin-minus-real change of log mean flow into an implied spurious
precipitation shift delta_b = (twin d_logq - real d_logq) / elasticity. Basin median over models.
Bank prior (v3): the TRUE target-period precipitation = SEEN / exp(delta), delta ~ N(mu, sd), with mu and sd scaled
by the time separation (dev: 19 years between window midpoints; bank / exam layout: 9.5 years -> factor 0.5, a
linear drift). Writes models/drift_v3.json, logs/calib_drift.json.
"""
import common  # noqa: F401
from _paths import DEV_PARAMS
import json

import numpy as np
import pandas as pd

import watersim as ws
import waterdata as wd
from common import CAL, EARLY, D, EPSQ
from features import sigs

PAIRS = ['hbv+degree_day', 'gr4j+degree_day', 'sacsma+degree_day', 'vic_lite+degree_day']


def main():
    P = json.load(open(DEV_PARAMS))['params']
    real = pd.read_parquet(f'{D}/feats/real_dev_meta.parquet').set_index('basin')
    rows = []
    for b in wd.basins('camels', 'dev'):
        s = wd.spec(b)
        meta = wd.forcing_meta(b)
        fe = wd.forcing(b, '1979-10-01', EARLY[1])
        fc = wd.forcing(b, *CAL)
        q = wd.streamflow(b, 'caravan').Q_mm
        for pair in PAIRS:
            pr = P[b][pair]
            pr = eval(pr) if isinstance(pr, str) else pr
            m, sn = pair.split('+')
            oc = ws.simulate(s, (fc, meta), *CAL, model=m, snow=sn, params=pr, labels=False)['ts']['Q']
            qm_c = oc.where(q.reindex(oc.index).notna()).mean()
            lq = []
            for k in (0.0, -0.1):
                f2 = fe.copy()
                f2['P'] = f2['P'] * np.exp(k)
                oe = ws.simulate(s, (f2, meta), EARLY[0], EARLY[1], model=m, snow=sn, params=pr, labels=False)['ts']['Q']
                lq.append(np.log(oe.where(q.reindex(oe.index).notna()).mean() + EPSQ) - np.log(qm_c + EPSQ))
            el = (lq[0] - lq[1]) / 0.1
            rows.append({'basin': b, 'pair': pair, 'twin_d_logq': lq[0], 'elasticity': el,
                         'real_d_logq': real.loc[b, 'true_d_logq']})
    df = pd.DataFrame(rows)
    df['delta'] = (df.twin_d_logq - df.real_d_logq) / df.elasticity.clip(lower=0.5)
    db = df.groupby('basin').delta.median().dropna()
    mu, sd = float(db.median()), float(1.4826 * np.median(np.abs(db - db.median())))
    fac = 9.5 / 19.0
    out = {'dev_median_delta_19y': mu, 'dev_robust_sd_19y': sd, 'median_elasticity': float(df.elasticity.median()),
           'n_basins': int(db.size), 'scale_to_bank': fac, 'mu': mu * fac, 'sd': sd * fac,
           'share_basins_delta_pos': float((db > 0).mean())}
    json.dump({'mu': out['mu'], 'sd': out['sd']}, open(f'{D}/models/drift_v3.json', 'w'))
    json.dump({'summary': out, 'rows': rows}, open(f'{D}/logs/calib_drift.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
