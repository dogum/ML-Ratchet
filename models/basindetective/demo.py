"""BasinDetective demo: four real CAMELS-US exam basins (fixed draw), a few seconds, numpy/scipy/pandas only.

python3 demo.py
Each basin: the network reads 1999-2008 forcing + streamflow and the 1989-1999 forcing, and says how six flow
signatures change in 1989-1999, whether a reservoir is upstream, and four hidden properties. The observed change
(from the real 1989-1999 gauge record, which the network never saw) and the calibrated GR4J model are printed beside.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import basindetective as bd   # noqa: E402

NICE = {'d_logq': ('mean flow', 'pct'), 'd_rr': ('runoff ratio', 'abs'), 'd_bfi': ('baseflow index', 'abs'),
        'd_logq5': ('low flow Q5', 'pct'), 'd_logq95': ('high flow Q95', 'pct'), 'd_hfd': ('half-flow date', 'days')}


def fmt(kind, v):
    if v is None or not np.isfinite(v):
        return '   n/a'
    if kind == 'pct':
        return f'{(np.exp(v) - 1) * 100:+5.0f}%'
    if kind == 'days':
        return f'{v:+5.0f}d'
    return f'{v:+6.3f}'


def main():
    det = bd.load()
    exp = json.load(open(os.path.join(HERE, 'examples', 'expected.json')))
    worst = 0.0
    for f in sorted(glob.glob(os.path.join(HERE, 'examples', '*_cal.csv'))):
        b = os.path.basename(f)[:-8]
        cal = pd.read_csv(f, index_col=0, parse_dates=True)
        tgt = pd.read_csv(f.replace('_cal.csv', '_tgt.csv'), index_col=0, parse_dates=True)
        tr = json.load(open(f.replace('_cal.csv', '_truth.json')))
        r = bd.predict(det, cal[['P', 'T_mean', 'PET']], cal['Q_mm'].values, tgt)
        print(f'\n{b}  {tr["name"]}')
        print(f'  {"change 1999-2008 -> 1989-1999":32s} {"network":>8s} {"+-1sd":>7s} {"observed":>9s} {"GR4J":>7s}')
        for t, (nm, kind) in NICE.items():
            c = r['changes'][t]
            sd = (f'{c["sd"] * 100:5.0f}%' if kind == 'pct' else (f'{c["sd"]:5.0f}d' if kind == 'days' else f'{c["sd"]:6.3f}'))
            print(f'  {nm:32s} {fmt(kind, c["mean"]):>8s} {sd:>7s} {fmt(kind, tr["observed_change"][t]):>9s} '
                  f'{fmt(kind, tr["gr4j_change"][t]):>7s}')
            worst = max(worst, abs(c['mean'] - exp[b][t]))
        h = r['hidden']
        print(f'  reservoir upstream: p = {r["regulation"]["p_regulated"]:.2f}   (HydroATLAS storage / annual flow = '
              f'{tr["reservoir_storage_over_annual_flow"]:.2f}; regulated if > 0.1)')
        print(f'  hidden (no real truth): slow store {h["slow_store_days"]:.0f} d, groundwater share '
              f'{h["gw_share"]["mean"]:.2f}, soil capacity {h["soil_capacity_mm"]:.0f} mm, precipitation x '
              f'{h["precip_multiplier"]:.2f}')
    print(f'\nself-check vs expected.json: max |difference| = {worst:.2e}', 'OK' if worst < 1e-6 else 'MISMATCH')


if __name__ == '__main__':
    main()
