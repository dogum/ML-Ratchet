"""The model.yaml exam rows from a grade file (exam.py) and its heat-input table: v1.0 and v1.1 side by side.

    python exam_rows.py R4_ens R6_cs      -> prints every row (median errors; fractions in points, others in %)
"""
from common import *
import numpy as np, pandas as pd
from report import table
from pdcore import TECHS


def rows(model):
    df = pd.read_pickle(f'{ROOT}/out/grade_exam_{model}.pkl')
    hi = pd.read_csv(f'{ROOT}/out/heatinput_exam_{model}.csv')
    t = table(df, 'ctx').set_index('kind')
    to = table(df, 'out').set_index('kind')
    r = {}
    for k in ('month', 'quarter', 'year', 'pooled'):
        for q in ('pmin_frac', 'ramp_up', 'starts_yr', 'med_up', 'min_up', 'min_dn', 'hr_full', 'hr_min', 'cf'):
            r[f'{q}|{k}'] = t.loc[k, q]
            for b in ('b_stats', 'b_default', 'b_923', 'b_dumb'):
                if f'{q}|{b}' in t.columns:
                    r[f'{q}|{k}|{b}'] = t.loc[k, f'{q}|{b}']
        r[f'tech_out|{k}'] = to.loc[k, 'tech_acc']; r[f'tech_rule|{k}'] = t.loc[k, 'tech_acc_rule']
    for tech in TECHS:
        tt = table(df, 'ctx', kinds=('pooled',), techs=[tech]).iloc[0]
        r[f'hr_full|{tech}'] = tt['hr_full']; r[f'hr_full|{tech}|n'] = tt['units']
        for b in ('b_default', 'b_923', 'b_dumb'):
            r[f'hr_full|{tech}|{b}'] = tt[f'hr_full|{b}']
        h = hi[(hi['mode'] == 'ctx') & (hi.tech == tech)]
        r[f'hi|{tech}'] = float(np.median(np.abs(h.err))); r[f'hi|{tech}|n'] = len(h)
        r[f'hi|{tech}|923'] = float(np.nanmedian(np.abs(h.err923)))
    h = hi[hi['mode'] == 'ctx']
    r['hi|all'] = float(np.median(np.abs(h.err))); r['hi|all|n'] = len(h)
    h2 = h[np.isfinite(h.err923)]
    r['hi|with923'] = float(np.median(np.abs(h2.err))); r['hi|with923|923'] = float(np.median(np.abs(h2.err923)))
    r['hi|with923|n'] = len(h2)
    return r


if __name__ == '__main__':
    R = {m: rows(m) for m in sys.argv[1:]}
    keys = list(R[sys.argv[1]].keys())
    print(pd.DataFrame(R).loc[keys].to_string(float_format=lambda v: f'{v:.4f}'))
