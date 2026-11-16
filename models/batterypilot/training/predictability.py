"""How much of the real-time spike value does the day-ahead price see?  Real dev years vs their simulated twins.

    python predictability.py            -> WORK/data/predictability_v<physics>.json (GridSim physics $GRIDSIM_PHYSICS,
                                           the scarcity calibration of that version, simbank.calib_file())

For ERCOT HB_HUBAVG 2019 and 2021 (dev; 2021 also without Winter Storm Uri, 8-24 Feb) and two twins per year (the
year's own real inputs, outage seeds 1 and 2): seen300 = sum over hours with RT > $300 of min(DA, RT) / sum of RT
(1 = the DA price had the whole spike), hours with RT - DA > $100 and > $300, hours with RT > $300 and > $1,000,
mean RT and DA, and the 2 h perfect-foresight value.  The v1.2 attempt found that a DA-anchored policy looks worse in
simulation than on real prices; this measures the cause on dev before the simulator is changed.
"""
import json

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from pricesim import Engine, load_inputs, fundamentals, prices, physics_version
from simbank import calibrated, calib_file
from realdata import node_frame, fill
from battery import Battery
from strategies import pf_daily

URI = ('2021-02-08', '2021-02-24')


def stats(rt, da, day):
    rt = np.asarray(rt, float); da = np.asarray(da, float)
    big = rt > 300
    b = Battery(1, 2, degradation=False)
    c, d = pf_daily(rt, day, b)
    s = rt - da
    return dict(seen300=round(float(np.minimum(da[big], rt[big]).sum() / rt[big].sum()), 3) if big.any() else None,
                da_median_in_rt300=round(float(np.median(da[big])), 1) if big.any() else None,
                sp100=int((s > 100).sum()), sp300=int((s > 300).sum()), h300=int(big.sum()),
                h1000=int((rt > 1000).sum()), rt_mean=round(float(rt.mean()), 2), da_mean=round(float(da.mean()), 2),
                pf2_kWyr=round(float((rt * (d - c)).sum() / 1000), 1))


def mask(index):
    return ~((index >= pd.Timestamp(URI[0], tz='UTC')) & (index < pd.Timestamp(URI[1], tz='UTC')))


def main():
    cal = calibrated('v2')
    v = physics_version()
    out = dict(physics=v, calibration=calib_file(), rows={})
    for year in (2019, 2021):
        f, _ = fill(node_frame('ERCOT', 'HB_HUBAVG', f'{year}-01-01', f'{year + 1}-01-01'))
        out['rows'][f'{year} real'] = stats(f.rt.values, f.da.values, f.day.values)
        if year == 2021:
            g = f[mask(f.index)]
            out['rows']['2021 real without Uri'] = stats(g.rt.values, g.da.values, g.day.values)
        eng = Engine('ERCO', year); inp = load_inputs('ERCO', year)
        for seed in (1, 2):
            o = prices(fundamentals(eng, inp, {}, cal, seed=seed), cal, {}, seed=seed)
            out['rows'][f'{year} twin{seed}'] = stats(o.rt.values, o.da.values, o.day.values)
            if year == 2021:
                g = o[mask(o.index)]
                out['rows'][f'2021 twin{seed} without Uri'] = stats(g.rt.values, g.da.values, g.day.values)
        for k, r in out['rows'].items():
            if k.startswith(str(year)):
                print(k, r, flush=True)
    fn = f'{_paths.WORK}/data/predictability_v{v.replace(".", "_")}.json'
    json.dump(out, open(fn, 'w'), indent=1)
    print(fn)


if __name__ == '__main__':
    main()
