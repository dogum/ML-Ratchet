"""Twin test on the dev years: simulate ERCOT 2019 and 2021 with their own real inputs and the calibrated simulator,
then run BatteryPilot and the rolling DA LP on the twins and on the real HB_HUBAVG prices of the same years.
If the policy reads twin and real alike, its capture rates should be close.

    python3 twin.py <rung>    -> data/twin_<rung>.json (+ data/twin_week.json for the page)
"""
import json, sys
import numpy as np, pandas as pd
import _paths  # noqa: E402  (this folder, GridSim and GridData on sys.path; outputs to WORK)
from pricesim import Engine, load_inputs, fundamentals, prices, physics_version
from simbank import calibrated
from evaluate import evaluate_path
from model import PolicyNet
from realdata import node_frame, fill

ROOT = _paths.WORK   # was the sandbox workspace grid/batterypilot
# v1.3 (GridSim physics 1.2): also the day-ahead backbone ('bb'); outputs get a _p<version> suffix
P12 = physics_version() == '1.2'
STRATS = ('pf', 'rolling', 'bp', 'bb') if P12 else ('pf', 'rolling', 'bp')
SUF = '_p1_2' if P12 else ''


def main(rung):
    net = PolicyNet(f'{ROOT}/ckpt/{rung}.npz')
    cal = calibrated('v2')
    out = {}
    weeks = {}
    for year in (2019, 2021):
        eng = Engine('ERCO', year); inp = load_inputs('ERCO', year)
        f, _ = fill(node_frame('ERCOT', 'HB_HUBAVG', f'{year}-01-01', f'{year + 1}-01-01'))
        real = {c: f[c].values for c in ['rt', 'da', 'df', 'T', 'hour', 'day', 'dow', 'doy']}
        r, _ = evaluate_path(real, (1, 2, 4), STRATS, net=net)
        out[f'{year}_real'] = r.groupby('strategy').net.sum().to_dict()
        for seed in (1, 2):
            o = prices(fundamentals(eng, inp, {}, cal, seed=seed), cal, {}, seed=seed)
            p = {c: o[c].values for c in ['rt', 'da', 'df', 'T', 'hour', 'day', 'dow', 'doy']}
            r, _ = evaluate_path(p, (1, 2, 4), STRATS, net=net)
            out[f'{year}_twin{seed}'] = r.groupby('strategy').net.sum().to_dict()
            if seed == 1:
                wk = slice(pd.Timestamp('2019-08-11 05:00', tz='UTC'), pd.Timestamp('2019-08-18 05:00', tz='UTC')) \
                    if year == 2019 else slice(pd.Timestamp('2021-02-10 06:00', tz='UTC'), pd.Timestamp('2021-02-21 06:00', tz='UTC'))
                weeks[year] = dict(t=[str(x) for x in o.loc[wk].index], sim_rt=o.loc[wk].rt.round(1).tolist(),
                                   sim_da=o.loc[wk].da.round(1).tolist(),
                                   real_rt=f.loc[wk].rt.round(1).tolist(), real_da=f.loc[wk].da.round(1).tolist())
        print(year, {k: v for k, v in out.items() if k.startswith(str(year))}, flush=True)
    for k, v in out.items():
        v['capture_bp'] = v['bp'] / v['pf']; v['capture_rolling'] = v['rolling'] / v['pf']
        if 'bb' in v:
            v['capture_bb'] = v['bb'] / v['pf']
        v['pf_per_kWyr_sum3dur'] = v['pf'] / 1000
    json.dump(out, open(f'{ROOT}/data/twin_{rung}{SUF}.json', 'w'), indent=1, default=float)
    json.dump(weeks, open(f'{ROOT}/data/twin_week{SUF}.json', 'w'))
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1])
