"""W1: fit one intensity factor per use type so generated site EUI medians match real survey medians.

Target: weighted median site EUI of CBECS 2018 (commercial) / RECS 2020 (homes) respondents in the PRIOR + DEV
splits (never exam). Generated buildings run on the TMYx city of their climate zone (as Phase 1's generator check).
Iterates f <- f * target / sim_median twice. Writes models/calibration.json['intensity'].
"""
import json
import sys
import time

import numpy as np

import common as C
from wrappers import load_cal, CAL
from simone import simulate_one


def wmedian(x, w):
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    return float(x[i][np.searchsorted(cw, cw[-1] / 2)])


def targets():
    import buildingdata as bd
    T = {}
    c = bd.cbecs()
    c = c[c.split != 'exam']
    site = c[['elec_kWh', 'gas_kWh', 'fueloil_kWh', 'district_heat_kWh']].fillna(0).sum(axis=1)
    ok = (site > 0) & (c.floor_area_m2 > 0)
    for u, g in c[ok].groupby('primary_use'):
        T[u] = wmedian((site[g.index] / g.floor_area_m2).to_numpy(), g.weight.to_numpy())
    r = bd.recs()
    r = r[r.split != 'exam']
    site = r[['elec_kWh', 'gas_kWh', 'fueloil_kWh', 'propane_kWh', 'wood_kWh']].fillna(0).sum(axis=1)
    ok = (site > 0) & (r.floor_area_m2 > 0)
    for u, g in r[ok].groupby('primary_use'):
        T[u] = wmedian((site[g.index] / g.floor_area_m2).to_numpy(), g.weight.to_numpy())
    return T


def main(n_per=20, iters=2):
    import buildingsim as bs
    cal = load_cal()
    T = targets()
    hist = []
    for it in range(iters):
        rng = np.random.default_rng(100 + it)
        res = {}
        t0 = time.time()
        for ui, u in enumerate(C.USES):
            specs = bs.generate_buildings(n_per, seed=500 + 13 * it + ui, primary_use=u)
            e = []
            for s in specs:
                from buildingsim.generator import weather_hint
                r = simulate_one(s, rng, cal, campus=False, wx_key=('tmy', weather_hint(s.get('climate_zone') or '4A'), None))
                e.append(r['info']['EUI'])
            res[u] = float(np.median(e))
            print(it, u, 'sim median EUI %.0f target %.0f factor %.3f' % (res[u], T[u], cal['intensity'][u]), flush=True)
        hist.append(dict(iter=it, factors=dict(cal['intensity']), sim_median=res, target=T,
                         ratio={u: res[u] / T[u] for u in C.USES}, wall_s=time.time() - t0))
        for u in C.USES:
            # damped update (EUI is not proportional to the factor: heating rises when gains fall)
            cal['intensity'][u] = float(np.clip(cal['intensity'][u] * (T[u] / res[u]) ** 0.8, 0.25, 2.5))
    cal['intensity_history'] = hist
    json.dump(cal, open(CAL, 'w'), indent=1)
    print(json.dumps({u: round(hist[-1]['ratio'][u], 2) for u in C.USES}))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, int(sys.argv[2]) if len(sys.argv) > 2 else 2)
