"""v1.1 (15 Nov 2026): the same network weights, with rung 7's prior written into the weights' meta. The prior is fitted
on all 29 dev arrivals (the leave-one-storm-out version of it was what the ratchet judged), and the error bars after the
blend get their own widen-only factors from dev. Rewrites ../swellcast_weights.npz in place; the arrays do not change."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as _P
sys.path.insert(0, _P.MODEL)
import events as EV, swellcast as SC, prior_dev as PD

if __name__ == '__main__':
    W = os.path.join(_P.MODEL, 'swellcast_weights.npz'); z = np.load(W); meta = json.loads(str(z['meta'])); arrays = {k: z[k] for k in z.files if k != 'meta'}
    meta.pop('prior', None); net = SC.SwellCast(); net.meta = meta
    ev = EV.load('dev', 'confirmed'); X = EV.windows(ev); P = net.predict(X, prior=False)
    D = np.array([e['D'] / 1e3 for e in ev]); t0 = np.array([e['t0_rel'] for e in ev]); dr = np.array([e['dir'] for e in ev])
    pr = PD.fit_prior(D); pr.update(cg06_kmh=PD.CG06_KMH, jitter_h=36.0)
    Q = SC.apply_prior(P, pr); f = PD.sigma_factors(Q, D, t0, dr); pr['sigma_factor'] = dict(logD=f['logD'], t0=f['t0'])
    meta.update(prior=pr, version='1.1', rung='r7'); print(json.dumps(pr))
    np.savez_compressed(W, meta=json.dumps(meta), **arrays)
