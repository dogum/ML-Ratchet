"""Summarise the dev calibration (data/dev_knobs.json) -> data/calib_summary.json (page + model card)."""
import json, collections
import numpy as np
import lc_common as C

kn = json.load(open(f'{C.ROOT}/data/dev_knobs.json'))
out = dict(n_buildings=len(kn), knobs={}, meters={})
for key in ('sched', 'oa', 'sys', 'vav_min'):
    out['knobs'][key] = dict(collections.Counter(str(v['best']['knobs'].get(key)) for v in kn.values()))
for m in C.METERS:
    g = [v['generic']['per'][m] for v in kn.values() if v.get('generic') and m in v['generic']['per']]
    b = [v['best']['per'][m] for v in kn.values() if m in v['best']['per']]
    if not b: continue
    out['meters'][m] = dict(
        n=len(b),
        generic_ratio_med=float(np.median([x['sim_mean'] / x['real_mean'] for x in g if x['real_mean'] > 0])) if g else None,
        generic_cv_med=float(np.median([x['cv_raw'] for x in g if np.isfinite(x['cv_raw'])])) if g else None,
        best_knobs_raw_ratio_med=float(np.median([x['sim_mean'] / x['real_mean'] for x in b if x['real_mean'] > 0])),
        a_med=float(np.median([x['a'] for x in b])), a_iqr=[float(np.percentile([x['a'] for x in b], q)) for q in (25, 75)],
        b_W_m2_med=float(np.median([x['b'] for x in b])), b_iqr=[float(np.percentile([x['b'] for x in b], q)) for q in (25, 75)],
        calibrated_cv_med=float(np.median([x['cv_fit'] for x in b])))
C.jdump(out, f'{C.ROOT}/data/calib_summary.json')
print(json.dumps(out, indent=1))
