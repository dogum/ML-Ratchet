"""Error-bar factors (calib.py). Rung r12 chose 'simv1' on the release candidate (leave-one-car-out on the dev cars):
sigma x a per-architecture factor that brings the simulated val2 set to 68% (simulation only), then v1's widen-only
factor per output group fitted on DEV only.
python sigma_dev.py <rung>   -> models/energynet/sigma_factors.json (what energynet.py reads: SIM_<arch> and the groups)"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json
sys.path.insert(0, _P.SRC)
import numpy as np
import evaluate as EV, calib as CB


def epa_vehicle(M):
    r, o = EV.epa_metrics(M)
    return o.join(M.groupby(['vehicle_id', 'test']).sig.mean(), on=['vehicle_id', 'test'])


def dev_sigma(m, bus_spec=False, epa_hyb=False, method='v1'):
    """-> {output group: (k, tau)} from the dev sets only."""
    S = {}
    dv = EV.predict_real(m, 'ved_dev'); dv = dv[dv.y > dv.group.map(EV.FLOOR)]
    for g, d in dv.groupby('group'):
        S[f'VED_{g}'] = CB.fit(d, method)
    o = epa_vehicle(EV.epa_frame(m, 'dev', epa_hyb))
    for ar, d in o.groupby('arch'):
        S[f'EPA_{ar}'] = (CB.fit_k(d), 0.0)
    z = EV.predict_real(m, 'ztb', bus_spec); zl = z[(z.split == 'dev') & (z.kind == 'leg') & z.label_ok & (z.y > 0)]
    S['HD_ZTBus'] = (CB.fit_k(zl), 0.0)
    h = EV.predict_real(m, 'hd')
    ms = h[(h.set == 'Mersin') & (h.split == 'dev')].groupby('trip_id').agg(pred=('pred', 'mean'), y=('y', 'first'), sig=('sig', 'mean'), fuel=('fuel', 'first'))
    ms['pred'] = ms.pred * np.where(ms.fuel == 'cng', EV.LHV / EV.CNG_LHV, 1.0)
    S['HD_Mersin'] = (CB.fit_k(ms), 0.0)
    S['HD_ICCT'] = (CB.fit_k(h[(h.set == 'ICCT') & (h.split == 'dev')]), 0.0)
    return S


if __name__ == '__main__':
    from ratchet2 import RUNGS, load
    r = {x['name']: x for x in RUNGS}[sys.argv[1]]
    m = load(r); ks = CB.sim_factors(m, r['version'])
    S = dev_sigma(CB.SimScaled(m, ks), r.get('bus_spec', False), r.get('epa_hyb', False))
    out = {f'SIM_{a}': round(v, 4) for a, v in ks.items()}
    out.update({k: (round(v[0], 4) if v[1] == 0 else [round(v[0], 4), round(v[1], 4)]) for k, v in S.items()})
    json.dump(out, open(os.path.join(_P.MODEL, 'sigma_factors.json'), 'w'), indent=1)
    print(out)
