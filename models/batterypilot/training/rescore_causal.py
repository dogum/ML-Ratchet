"""Dated supplementary re-score (27 Nov 2026): v1.3 with a causal day-ahead wear plan, on every exam node-year.

    python rescore_causal.py run       # exam.main with the released weights; strategies pf, rolling, spread,
                                       # threshold, bp (as released) and bpc (c_lin_plan = causal_wear(bat, T), 48 h)
    python rescore_causal.py report    # report.main on each variant -> data/exam/<name>_report.json + comparison

The audit (25 Nov 2026) found that the hybrid's day-ahead value curves used a wear cost from realized temperatures of
the coming hours (a perfect temperature forecast). `bpc` takes each hour's temperature from 48 h earlier, so every plan
(made at 14:00, running up to 33 h ahead) reads only temperatures known when it is made. Settlement and wear are
unchanged (realized temperatures). Nothing is chosen here: the fix was fixed by the audit before this run. `bp` is
the released behaviour with the hour-0 history fix of 27 Nov 2026, re-run on the same rows as the reference.
Outputs: WORK/data/exam/v1_3_rescore_2026-11-27_{bp,bpc}_*.parquet / _report.json, comparison json printed and saved.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import exam
import report

ROOT = _paths.WORK
NAME = 'v1_3_rescore_2026-11-27'
WEIGHTS = os.path.join(_paths.MODEL, 'batterypilot_weights.npz')     # the v1.3 release (= WORK/ckpt/r8_v6.npz)


def run():
    exam.main('r8_v6', ('pf', 'rolling', 'spread', 'threshold', 'bp', 'bpc'), out=NAME, weights=WEIGHTS)


def split():
    """One file set per variant, with that variant as 'bp', so report.main runs unchanged."""
    for kind in ('rows', 'daily', 'monthly'):
        df = pd.read_parquet(f'{ROOT}/data/exam/{NAME}_{kind}.parquet')
        for v in ('bp', 'bpc'):
            d = df[df.strategy != ('bpc' if v == 'bp' else 'bp')].copy()
            d.loc[d.strategy == v, 'strategy'] = 'bp'
            d.to_parquet(f'{ROOT}/data/exam/{NAME}_{v}_{kind}.parquet')


def compare():
    rows = pd.read_parquet(f'{ROOT}/data/exam/{NAME}_rows.parquet')
    rows['kWyr'] = rows.net / 1000 * 8760 / rows.hours
    pv = rows.pivot_table(index=['market', 'node', 'year', 'duration'], columns='strategy', values='kWyr')
    cyc = rows.pivot_table(index=['market', 'node', 'year', 'duration'], columns='strategy', values='cycles')
    out = {'units': '$ per kW-year net of wear; capture = net / perfect-hindsight net, mean over node-year-batteries'}
    for m, g in pv.groupby(level='market'):
        rec = dict(cases=int(len(g)))
        for s in ('bp', 'bpc', 'rolling', 'spread', 'threshold'):
            rec[f'{s}_capture'] = round(float((g[s] / g.pf).mean()), 4)
        d = g.bpc - g.bp
        rec.update(bpc_minus_bp_mean=round(float(d.mean()), 4), bpc_minus_bp_min=round(float(d.min()), 4),
                   bpc_minus_bp_max=round(float(d.max()), 4), bpc_below_bp=int((d < 0).sum()),
                   bp_beats_rolling=round(float((g.bp > g['rolling']).mean()), 3),
                   bpc_beats_rolling=round(float((g.bpc > g['rolling']).mean()), 3),
                   bp_cycles=round(float(cyc.loc[g.index].bp.mean()), 1), bpc_cycles=round(float(cyc.loc[g.index].bpc.mean()), 1))
        for dur, gd in g.groupby(level='duration'):
            rec[f'{dur}h'] = dict(bp=round(float((gd.bp / gd.pf).mean()), 4), bpc=round(float((gd.bpc / gd.pf).mean()), 4),
                                  rolling=round(float((gd['rolling'] / gd.pf).mean()), 4),
                                  bp_beats_rolling=int((gd.bp > gd['rolling']).sum()),
                                  bpc_beats_rolling=int((gd.bpc > gd['rolling']).sum()), n=int(len(gd)))
        out[m] = rec
    rel = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checks', 'exam_report_v1_3.json')))
    out['released_overall'] = rel['overall']
    json.dump(out, open(f'{ROOT}/data/exam/{NAME}_compare.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run()
    else:
        split()
        for v in ('bp', 'bpc'):
            report.main(f'{NAME}_{v}')
        compare()
