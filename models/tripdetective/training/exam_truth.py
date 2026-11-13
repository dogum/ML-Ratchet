"""What real truth exists, per VED car (written once; the exam is never used for choosing):
  - VED type (ICE/HEV/PHEV/EV): published -> graded architecture truth. Finer architecture is RoadData's inference
    from engine size (confidence column) -> reported separately, not as hard truth.
  - curb weight rounded to 500 lb (VED Generalized_Weight) -> mass truth with +-113 kg rounding.
  - engine displacement (VED static data) -> displacement truth.
  - road load: only for cars whose fueleconomy.gov match is 'high' AND whose candidate models exist in the EPA test-car
    list (target coefficients A, B, C from the coast-down): road-load force at 20/40/60 mph (median over candidate
    records, with the min-max spread as the truth's own uncertainty) and the EPA test weight (ETW).
usage: python3 exam_truth.py"""
from common import *
import re
import realdata as rd
import pandas as pd

LBF = 4.4482216; MPH = 0.44704
SPEEDS = [20, 40, 60]


def match_epa(example_models, epa):
    rows = []
    for em in str(example_models).split(';'):
        m = re.match(r'\s*(\d{4})\s+(\S+)\s+(.*)', em)
        if not m: continue
        y, make, model = int(m.group(1)), m.group(2).upper(), m.group(3).upper()
        model = model.replace('PLUG-IN HYBRID', '').replace('ENERGI', '').replace('2WD', '').replace('AWD', '').strip()
        toks = [t for t in re.split(r'\s+', model) if t]
        cand = epa[(epa.model_year == y) & (epa.make.str.upper() == make)]
        for t in toks:
            cand = cand[cand.model.str.upper().str.replace(' ', '').str.contains(re.escape(t.replace(' ', '')))]
        if 'PLUG' in em.upper() or 'ENERGI' in em.upper() or 'VOLT' in em.upper():
            cand = cand[cand.architecture == 'phev']
        rows.append(cand)
    return pd.concat(rows) if rows else pd.DataFrame()


def build():
    vt = rd.ved_vehicles(); epa = rd.epa_testcars()
    out = []
    for vid, r in vt[vt.match_confidence == 'high'].iterrows():
        if r.split not in ('dev', 'exam'): continue
        c = match_epa(r.example_models, epa)
        if r.ved_type == 'ICE' and len(c):
            c = c[(c.displacement_L - r.displacement_L).abs() < 0.15]
        if not len(c): continue
        F = np.array([[a * LBF + b * LBF * s + cc * LBF * s * s for s in SPEEDS] for a, b, cc in zip(c.A_lbf, c.B_lbf_mph, c.C_lbf_mph2)])
        out.append(dict(vid=vid, split=r.split, ved_type=r.ved_type, n_epa=len(c), models=';'.join(sorted(set(c.model_year.astype(str) + ' ' + c.model))),
                        etw_kg=float(np.median(c.etw_lb)) * LB, **{f'F{s}_N': float(np.median(F[:, i])) for i, s in enumerate(SPEEDS)},
                        **{f'F{s}_lo': float(F[:, i].min()) for i, s in enumerate(SPEEDS)}, **{f'F{s}_hi': float(F[:, i].max()) for i, s in enumerate(SPEEDS)}))
    d = pd.DataFrame(out)
    d.to_csv(f'{ROOT}/data/roadload_truth.csv', index=False)
    print(d[['vid', 'split', 'ved_type', 'n_epa', 'etw_kg', 'F20_N', 'F40_N', 'F60_N', 'F60_lo', 'F60_hi']].to_string())
    return d


if __name__ == '__main__':
    build()
