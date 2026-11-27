"""Final exam on REAL prices (never used for choosing anything).  Run once, with the released network.

    python3 exam.py <rung>

ERCOT 2023-2025 at 4 hubs + 8 load zones, CAISO 2024-2025 at 3 trading hubs + 3 DLAPs; batteries 1, 2, 4 h (1 MW).
Strategies: perfect foresight (annual LP, upper bound), rolling DA LP, daily spread, threshold (dev-period RT
quartiles of the node), BatteryPilot.  All settle at the hourly RT price; degradation by rainflow with the node's
BA temperature.  Writes data/exam/<rung>_rows.parquet (annual), _daily.parquet (daily net per strategy), and
_monthly.parquet.
"""
import os, sys, time
import numpy as np, pandas as pd
import _paths  # noqa: E402  (this folder, GridSim and GridData on sys.path; outputs to WORK)
from battery import Battery, settle
from evaluate import run_strategy
from model import PolicyNet
from realdata import node_frame, fill, ERCOT_NODES, CAISO_NODES

ROOT = _paths.WORK   # was the sandbox workspace grid/batterypilot
STRATS = ('pf', 'rolling', 'spread', 'threshold', 'bp')


def dev_thresholds(market, node):
    if market == 'ERCOT':
        x = pd.concat([node_frame('ERCOT', node, f'{y}-01-01', f'{y + 1}-01-01').rt for y in (2019, 2021)])
    else:
        x = node_frame('CAISO', node, '2023-08-01', '2024-01-01').rt
    return float(np.nanpercentile(x, 25)), float(np.nanpercentile(x, 75))


def exam_sets():
    s = [('ERCOT', n, y) for y in (2023, 2024, 2025) for n in ERCOT_NODES]
    s += [('CAISO', n, y) for y in (2024, 2025) for n in CAISO_NODES]
    return s


def main(rung, strats=STRATS, out=None, weights=None):
    """out: name of the output files (default the rung); weights: the .npz to load (default WORK/ckpt/<rung>.npz)."""
    os.makedirs(f'{ROOT}/data/exam', exist_ok=True)
    net = PolicyNet(weights or f'{ROOT}/ckpt/{rung}.npz')
    rung = out or rung
    rows, daily, monthly = [], [], []
    t0 = time.time()
    for m, n, y in exam_sets():
        f, miss = fill(node_frame(m, n, f'{y}-01-01', f'{y + 1}-01-01'))
        f = f.dropna(subset=['rt', 'da'])
        thr = dev_thresholds(m, n)
        p = {c: f[c].values for c in ['rt', 'da', 'df', 'T', 'hour', 'day', 'dow', 'doy']}
        for d in (1, 2, 4):
            bat = Battery(1.0, d)
            for st in strats:
                s = run_strategy(st, p, bat, net, thr)
                rows.append(dict(market=m, node=n, year=y, duration=d, strategy=st, gross=s['gross'], deg=s['deg'],
                                 net=s['net'], cycles=s['cycles'], hours=len(f), missing_rt=miss['rt'],
                                 missing_da=miss['da']))
                nt = pd.Series(s['gross_t'] - s['deg_t'], index=f.index)
                loc = f.index.tz_convert('America/Chicago' if m == 'ERCOT' else 'America/Los_Angeles')
                dd = nt.groupby(loc.normalize().tz_localize(None)).sum()
                daily.append(pd.DataFrame(dict(market=m, node=n, year=y, duration=d, strategy=st, day=dd.index,
                                               net=dd.values)))
                mm = nt.groupby(loc.month).sum()
                monthly.append(pd.DataFrame(dict(market=m, node=n, year=y, duration=d, strategy=st, month=mm.index,
                                                 net=mm.values)))
        print(m, n, y, 'done %.0fs' % (time.time() - t0), flush=True)
    pd.DataFrame(rows).to_parquet(f'{ROOT}/data/exam/{rung}_rows.parquet')
    pd.concat(daily).to_parquet(f'{ROOT}/data/exam/{rung}_daily.parquet')
    pd.concat(monthly).to_parquet(f'{ROOT}/data/exam/{rung}_monthly.parquet')


if __name__ == '__main__':
    main(sys.argv[1], tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 else STRATS)
