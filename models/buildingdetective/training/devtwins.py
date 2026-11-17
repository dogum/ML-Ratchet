"""Simulated twins of the 52 BDG2 DEV buildings (twin test + sigma widening for the sim-only outputs).

Each dev building gets n twins: a generated building of its use type and climate zone with its real floor area,
year built (when known) and floors (when known); the system follows its meters (campus district heat and/or
chilled water as in W2; electricity-only buildings keep the generator's system); real 2017 site weather.
Features are made exactly as for the bank (same quirk emulator).
usage: python3 devtwins.py <tag> [n_twins=4]
"""
import json
import os
import sys
import warnings

import numpy as np

import common as C
import features as F
from simone import simulate_one
from wrappers import load_cal

warnings.filterwarnings('ignore')


def main(tag, n=4, w3=False):
    import buildingsim as bs
    import realdata as R
    cal = load_cal()
    meta = R.bdg2_buildings('dev')
    acc = {m: dict(X=[], Y=[], K=[], bid=[]) for m in ('hourly', 'monthly', 'annual')}
    rng = np.random.default_rng(777)
    for bi, row in enumerate(meta.itertuples()):
        ms = set(str(row.meters).split(','))
        chw, heat = 'chilledwater' in ms, ('steam' in ms or 'hotwater' in ms)
        kind = 'both' if (chw and heat) else ('heat' if heat else ('cool' if chw else False))
        for t in range(n):
            s = None
            for k in range(20):
                s = bs.generate_buildings(1, seed=int(rng.integers(1e9)), primary_use=row.primary_use,
                                          climate_zone=row.climate_zone)
                if s:
                    s = s[0]
                    break
            if not s:
                s = bs.generate_buildings(1, seed=int(rng.integers(1e9)), primary_use=row.primary_use)[0]
            s['floor_area_m2'] = float(row.floor_area_m2)
            if np.isfinite(row.year_built if row.year_built is not None else np.nan):
                s['year_built'] = int(row.year_built)
            if row.n_floors is not None and np.isfinite(row.n_floors):
                s['n_floors'] = int(row.n_floors)
            else:
                s['n_floors'] = max(1, int(round(np.sqrt(row.floor_area_m2) / 30)))
            s['climate_zone'] = row.climate_zone
            try:
                r = simulate_one(s, rng, cal, campus=kind, wx_key=('site', row.site_id, 2017))
            except Exception as e:
                print('ERR', row.id, e)
                continue
            met = r['meters']
            if w3:
                from wrappers import w3_base_loads
                met = w3_base_loads(met, np.random.default_rng(5000 + 10 * bi + t), commercial=True)
            Y = [r['y'][k] for k in C.REG]
            K = [C.USES.index(r['cls']['use']), C.HEAT.index(r['cls']['heat']), C.COOL.index(r['cls']['cool'])]
            loc = r['index'].tz_convert(r['tz'])
            T, G = r['T'].astype(float), r['GHI'].astype(float)
            mq = F.quirk_hourly(met, rng, len(T))[0] or met
            acc['hourly']['X'].append(F.feat_hourly(mq, T, G, loc, r['lat']))
            mm, Tm, Hm, Cm, nd = F.monthly_from_hourly(met, T, loc, shift_days=int(rng.integers(0, 16)))
            acc['monthly']['X'].append(F.feat_monthly(F.quirk_monthly(mm, rng), Tm, Hm, Cm, nd, r['lat']))
            d = np.asarray(loc.normalize().asi8)
            ud, inv = np.unique(d, return_inverse=True)
            Td = np.bincount(inv, weights=T) / np.bincount(inv)
            kwh = {c: float(np.nansum(v)) for c, v in met.items()}
            acc['annual']['X'].append(F.feat_annual(F.quirk_annual(kwh, rng) or kwh, float(np.maximum(F.HB - Td, 0).sum()),
                                                    float(np.maximum(Td - F.HB, 0).sum())))
            for m in acc:
                acc[m]['Y'].append(Y); acc[m]['K'].append(K); acc[m]['bid'].append(row.id)
        print(bi, row.id, kind, flush=True)
    od = os.path.join(C.ROOT, 'feats', 'real')
    for m, a in acc.items():
        np.savez(os.path.join(od, f'{tag}_{m}.npz'), X=np.array(a['X'], np.float32), Y=np.array(a['Y'], np.float32),
                 K=np.array(a['K'], np.int16), bid=np.array(a['bid']))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4, '--w3' in sys.argv)
