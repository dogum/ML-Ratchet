"""Twin test for the water law (dev basins only): simulate each dev basin's decade 1989-99 with WaterSim
  - 'ungauged' twin: the attribute prior's parameters (what the simulator believes about this basin), HBV + degree-day;
  - 'calibrated' twin: HBV + degree-day with WaterSim's dev-calibrated parameters (fitted on 1999-2008).
Then read each twin and the real basin with the law. If the law reads the twins well and the real basin badly, the gap
is in the simulator; if it reads both badly, the law misses something the simulator does.
"""
import json, os
import numpy as np, pandas as pd
import watersim as ws, waterdata as wd
import lawratchet as lr
import law_water as L
from water_real import VAL

ROOT = lr.ROOT


def main():
    from _paths import REPO
    cal = json.load(open(f'{REPO}/shared/water/checks/dev_calibrated_params.json'))['params']
    R = json.load(open(f'{ROOT}/results/water_ratchet.json'))
    dev = L.load_dev().set_index('basin')
    rows = []
    for b in dev.index:
        f = wd.forcing(b, '1988-10-01', VAL[1]); meta = wd.forcing_meta(b)
        spec = wd.spec(b)
        P = f['P'].loc[VAL[0]:VAL[1]].sum()
        out = {'basin': b, 'rr_real': dev.loc[b, 'rr']}
        for kind in ('ungauged', 'calibrated'):
            prm = cal.get(b, {}).get('hbv+degree_day') if kind == 'calibrated' else None
            if kind == 'calibrated' and prm is None:
                out[f'rr_{kind}'] = np.nan; continue
            o = ws.simulate(spec, (f, meta), VAL[0], VAL[1], model='hbv', snow='degree_day', routing='uh',
                            params=prm, labels=False)
            out[f'rr_{kind}'] = float(o['ts']['Q'].sum() / P)
        rows.append(out)
    T = pd.DataFrame(rows).set_index('basin')
    X = L.X_of(dev.loc[T.index].reset_index())
    res = {}
    for rid in [R['released'], lr.knee(R)['id'], 'r0']:
        r = next(g for g in R['rungs'] if g['id'] == rid)
        p = lr.law_eval(r, X)
        res[rid] = {k: float(np.nanmean(np.abs(p - T[f'rr_{k}'].values))) for k in ('real', 'ungauged', 'calibrated')}
    res['twin_vs_real_mae'] = {k: float(np.nanmean(np.abs(T[f'rr_{k}'] - T['rr_real']))) for k in ('ungauged', 'calibrated')}
    res['median_ratio_real_over_ungauged'] = float(np.nanmedian(T['rr_real'] / T['rr_ungauged']))
    T.to_parquet(f'{ROOT}/results/water_twin.parquet')
    json.dump(res, open(f'{ROOT}/results/water_twin.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
