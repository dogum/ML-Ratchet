"""Dated supplementary re-score (27 Nov 2026): v1.1 with standardised inputs clipped to |z| <= 8 (z_clip=8).

    python rescore_zclip.py actual      # work/data/banks/real_exam.npz -> work/data/exam/v1.1_zclip8.json + rows
    python rescore_zclip.py forecast    # forecast weather 2024-2025 (work/data/forecast_full, built by rescore_v11)
    python rescore_zclip.py release2024 # the released 2024 forecast-weather row (work/data/forecast_release2024);
                                        # v1.0 arithmetic (demand and wind are the same in v1.1), -> release_zclip8.json

The released v1.1 weights and arithmetic (rescore_v11.predict_v11), with every _forward call clipping its standardised
inputs at +-8 (demandcast._forward(z_clip=8)); sigma scales re-fitted on the dev bank as exam.py does. Nothing is
chosen from these numbers: z_clip=8 was fixed by the audit (25 Nov 2026) before this run. Next to the pooled metrics it
reports the rows the audit named: solar rows that report output at an hour the physics calls dark (features 56-80 >
0.01 CF where 80-104 < 1e-6), and demand rows with missing history hours (feature 328, miss_frac > 0), each against the
same rows under v1.1 (work/data/exam/v1.1_pred.npz). The audit's 1,734 solar rows are those with output at 20:00 LST
(feature 66) over every row of the bank; any dark hour (16:00-07:00 LST) gives more rows.
"""
import os
import sys
import warnings

import numpy as np

from common import DATA, BAS, jdump
import rescore_v11 as V
import demandcast as DC

RR, exam = V.RR, V.exam
_fwd = DC._forward
ZC = 8.0


def _forward_clip(W, x, kind, prior=None, z_clip=None):
    return _fwd(W, x, kind, prior, z_clip=ZC)


def affected(B):
    xs = np.nan_to_num(B['xs'].astype(np.float64))
    dark = ((xs[:, 56:80] > 0.01) & (xs[:, 80:104] < 1e-6)).any(1) & B['hass']
    dark20 = (xs[:, 66] > 0.01) & (xs[:, 90] < 1e-6) & B['hass']     # 20:00 LST, the audit's count (1,734 bank rows)
    miss = B['xd'][:, 328].astype(np.float64) > 0
    return dark, dark20, miss


def rowstats(B, pred, rows):
    """Demand MAPE and solar / demand pooled errors on a row subset, with v1.1-style numpy arithmetic."""
    out = {}
    y = B['d_true']; p = B['pd_total'] * np.exp(pred['d_mu'])
    m = rows[:, None] & np.isfinite(y) & (y > 0)
    out['demand_mape'] = float(np.mean(np.abs(p - y)[m] / y[m]) * 100) if m.any() else None
    out['demand_hours'] = int(m.sum())
    y = B['ys']; w = np.broadcast_to(B['caps'][:, None], y.shape)
    m = rows[:, None] & B['hass'][:, None] & np.isfinite(y)
    out['solar_nmae'] = float(np.sum((np.abs(pred['s_mu'] - y) * w)[m]) / max(np.sum((y * w)[m]), 1e-9) * 100) if m.any() else None
    out['solar_hours'] = int(m.sum())
    return out


def run(data, tag, base='v1.1'):
    exam.DATA = data
    RR.shim.predict = V.predict_v11 if base == 'v1.1' else V.predict_v10
    DC._forward = _forward_clip
    V.DC._forward = _forward_clip
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exam.main(f'{base}_zclip8')
    DC._forward = _fwd
    B = dict(np.load(f'{data}/banks/real_exam.npz'))
    old = dict(np.load(f'{data}/exam/{base}_pred.npz')); new = dict(np.load(f'{data}/exam/{base}_zclip8_pred.npz'))
    dark, dark20, miss = affected(B)
    act = B['meta_fc'] == 0 if tag == 'actual' else ((B['meta_fc'] & 7) == 7)
    res = dict(tag=tag, z_clip=ZC, rows={})
    for name, sel in (('solar_dark_hour_rows', dark & act), ('solar_dark_2000LST_rows', dark20 & act), ('solar_other_rows', ~dark & B['hass'] & act),
                      ('demand_missing_history_rows', miss & act), ('demand_other_rows', ~miss & act)):
        res['rows'][name] = dict(n=int(sel.sum()), v11=rowstats(B, old, sel), zclip8=rowstats(B, new, sel))
        per = {}
        for ba in BAS:
            s = sel & (B['meta_ba'] == ba)
            if s.sum():
                per[ba] = dict(n=int(s.sum()), v11=rowstats(B, old, s), zclip8=rowstats(B, new, s))
        res['rows'][name]['by_ba'] = per
    jdump(res, f'{data}/exam/{base}_zclip8_rows.json')
    a = __import__('json').load(open(f'{data}/exam/{base}.json')); b = __import__('json').load(open(f'{data}/exam/{base}_zclip8.json'))
    mode = 'actual' if tag == 'actual' else 'forecast'
    for out in ('demand', 'wind', 'solar'):
        print(f"{tag} pooled {out}: {base} {a['overall'][mode][out]['model']:.3f} -> zclip8 {b['overall'][mode][out]['model']:.3f}"
              f"  cov68 {a['overall'][mode][out]['cov68']:.3f} -> {b['overall'][mode][out]['cov68']:.3f}")
    for k, v in res['rows'].items():
        print(k, v['n'], {kk: (v['v11'][kk], v['zclip8'][kk]) for kk in ('demand_mape', 'solar_nmae')})


if __name__ == '__main__':
    if sys.argv[1] == 'actual':
        run(DATA, 'actual')
    elif sys.argv[1] == 'forecast':
        run(f'{DATA}/forecast_full', 'forecast')
    else:
        run(f'{DATA}/forecast_release2024', 'forecast', base='release')
