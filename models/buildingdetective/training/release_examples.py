"""Write release/examples/: 3 BDG2 exam buildings + 1 HUE home (fixed seed draws) and 4 RECS + 4 CBECS exam
respondents (fixed seed). BDG2-derived series are CC BY-SA 4.0 (share-alike), HUE CC BY 4.0, EIA public domain."""
import json
import os

import numpy as np

import common as C
import realdata as R

OD = os.path.join(C.ROOT, 'release', 'examples')


def main():
    os.makedirs(OD, exist_ok=True)
    rng = np.random.default_rng(31)
    meta = R.bdg2_buildings('exam')
    meta = meta[meta.meters.astype(str).str.contains('electricity')]
    uses = ['education', 'office', 'lodging']
    arrs, hourly = {}, []
    for k, u in enumerate(uses):
        g = meta[meta.primary_use == u]
        bid = g.id.to_numpy()[rng.integers(len(g))]
        c = R.bdg2_case(bid, 2017, meta.set_index('id').loc[bid])
        key = f'b{k}'
        hourly.append(dict(key=key, id=bid, tz=str(c['loc'].tz), lat=c['lat'], channels=sorted(c['meters']),
                           truth=dict(use=c['labels']['use'], floor_area_m2=c['labels']['A'], year_built=c['labels']['year'],
                                      meters=c['labels']['meters'])))
        arrs[key + '_utc'] = c['loc'].tz_convert('UTC').tz_localize(None).values.astype('datetime64[s]').astype(np.int64)
        arrs[key + '_T'] = c['T'].astype(np.float32); arrs[key + '_GHI'] = c['GHI'].astype(np.float32)
        for ch, x in c['meters'].items():
            arrs[key + '_' + ch] = x.astype(np.float32)
    hue = R.hue_cases()
    c = hue[rng.integers(len(hue))]
    key = 'h0'
    hourly.append(dict(key=key, id=c['id'], tz=str(c['loc'].tz), lat=c['lat'], channels=['elec'],
                       truth=dict(heat=c['labels']['heat'], codes=c['labels']['codes'])))
    arrs[key + '_utc'] = c['loc'].tz_convert('UTC').tz_localize(None).values.astype('datetime64[s]').astype(np.int64)
    arrs[key + '_T'] = c['T'].astype(np.float32); arrs[key + '_GHI'] = c['GHI'].astype(np.float32)
    arrs[key + '_elec'] = c['meters']['elec'].astype(np.float32)
    annual = []
    for name, fn in (('RECS', R.recs_rows), ('CBECS', R.cbecs_rows)):
        df, rows = fn('exam')
        for i in rng.choice(len(rows), 4, replace=False):
            r = rows[i]; x = r['row']
            kwh = {k: float(v) for k, v in r['kwh'].items() if v and np.isfinite(v) and v > 0}
            heat = R.recs_heat_label(x) if name == 'RECS' else R.cbecs_heat_label(x)
            annual.append(dict(id=f'{name} {r["id"]}', kwh=kwh, hdd=float(r['hdd']), cdd=float(r['cdd']),
                               truth=dict(use=x.primary_use, floor_area_m2=float(x.floor_area_m2), heat=heat,
                                          year_range=[x.year_built_lo, x.year_built_hi])))
    np.savez_compressed(os.path.join(OD, 'examples.npz'), **arrs)
    json.dump(dict(hourly=hourly, annual=annual), open(os.path.join(OD, 'examples.json'), 'w'), indent=1, default=float)
    print(len(hourly), len(annual))


if __name__ == '__main__':
    main()
