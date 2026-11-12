"""Baselines on the same real readings: the classical SYD-style pipeline (classical.py, run in realbank.py) and the
dumb baseline (always guess the dev stars' median numax and dnu). Writes baselines.json."""
import _paths as _P  # portable paths (see _paths.py)
import json, numpy as np
D = _P.WORK + '/data/'
meta = json.load(open(D + 'real_meta.json')); dev = [m for m in meta if m['split'] == 'dev' and m['L'] == 0]
med_nm, med_dn = np.median([m['numax'] for m in dev]), np.median([m['dnu'] for m in dev])
def sc(e1, e2):
    e1, e2 = np.nan_to_num(np.asarray(e1, float), nan=9.0), np.nan_to_num(np.asarray(e2, float), nan=9.0)   # a failed reading counts as a huge error
    return dict(n=len(e1), numax_med=float(np.median(np.abs(e1))), dnu_med=float(np.median(np.abs(e2))), numax_hit=float(np.mean(np.abs(e1) < 0.05)), dnu_hit=float(np.mean(np.abs(e2) < 0.02)),
                numax_out=float(np.mean(np.abs(e1) > 0.15)), dnu_out=float(np.mean(np.abs(e2) > 0.05)), numax_bias=float(np.median(e1)), dnu_bias=float(np.median(e2)))
OUT = {'dumb_guess': dict(numax=med_nm, dnu=med_dn)}
for split in ('dev', 'exam'):
    for L in (0, 365, 90, 54, 27):
        M = [m for m in meta if m['split'] == split and m['L'] == L]
        OUT[f'{split}_{L}'] = dict(classical=sc([np.log(m['cl_numax'] / m['numax']) for m in M], [np.log(m['cl_dnu'] / m['dnu']) for m in M]),
                                   dumb=sc([np.log(med_nm / m['numax']) for m in M], [np.log(med_dn / m['dnu']) for m in M]))
json.dump(OUT, open(_P.WORK + '/baselines.json', 'w'), indent=1)
for k, v in OUT.items():
    if k.startswith('exam'): print(k, 'classical', {a: round(b, 4) for a, b in v['classical'].items()}, '\n      dumb', {a: round(b, 3) for a, b in v['dumb'].items() if a in ('numax_med', 'dnu_med')})
