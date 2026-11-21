"""Summarise checks/out/dev_calib/*.json -> dev_calibration.json (per-model medians and per-basin table) and
dev_calibrated_params.json (best parameter set per dev basin x model, for B's baselines and Phase 2)."""
import glob
import json

import numpy as np

from _common import write, CHECKS, CAL, VAL_EARLY, VAL_LATE

import os
OBJ = os.environ.get('OBJ', 'kge')
SUB = 'dev_calib' if OBJ == 'kge' else f'dev_calib_{OBJ}'
SUFFIX = '' if OBJ == 'kge' else f'_{OBJ}'
rows = [json.load(open(p)) for p in sorted(glob.glob(CHECKS + f'/out/{SUB}/*.json'))]
summary = {}
params = {}
for r in rows:
    key = f"{r['model']}+{r['snow']}"
    params.setdefault(r['basin'], {})[key] = r.get('params')
    for per in ('cal', 'val_early', 'val_late'):
        m = r.get(per)
        if not m:
            continue
        for metric in ('NSE', 'KGE', 'logNSE', 'PBIAS', 'FHV', 'FLV', 'peak_err_pct', 'peak_timing_days'):
            summary.setdefault(key, {}).setdefault(per, {}).setdefault(metric, []).append(m.get(metric))
table = {}
for key, d in summary.items():
    table[key] = {}
    for per, mm in d.items():
        table[key][per] = {'n_basins': len(mm['NSE'])}
        for metric, vals in mm.items():
            v = np.array([x for x in vals if x is not None], float)
            if len(v):
                table[key][per][metric] = {'median': float(np.median(v)), 'q25': float(np.percentile(v, 25)),
                                           'q75': float(np.percentile(v, 75))}
        nse = np.array([x for x in mm['NSE'] if x is not None])
        table[key][per]['frac_NSE_gt_0.5'] = float(np.mean(nse > 0.5)) if len(nse) else None
per_basin = {}
for r in rows:
    per_basin.setdefault(r['basin'], {})[f"{r['model']}+{r['snow']}"] = {
        per: (None if not r.get(per) else {k: r[per][k] for k in ('NSE', 'KGE')}) for per in
        ('cal', 'val_early', 'val_late')}
write(f'dev_calibration{SUFFIX}.json', {
    'what': f'per-basin calibration of each model on the CAMELS-US DEV basins (B splits), {OBJ.upper()} objective, '
            'JADE differential evolution 40 x 80 (3,240 runs), forcing Caravan ERA5-Land, PET = Caravan FAO-56 PM',
    'periods': {'calibration': CAL, 'validation_early': VAL_EARLY, 'validation_late': VAL_LATE,
                'note': 'the CAMELS validation years 1989-10..1999-09 are B\'s temporal EXAM period and were not used; '
                        'both validation windows here are non-exam'},
    'prior_skill': 'skill of the uncalibrated attribute-prior parameters: see prior_skill.json (the per-record '
                   '"prior_cal" entries in out/dev_calib* used transfer-function versions in development and are not reported)',
    'literature_context': 'Newman et al. (2017) SAC-SMA+Snow-17 CAMELS benchmark median NSE ~0.6-0.65 (validation); '
                          'Kratzert et al. (2019) LSTM median NSE ~0.74-0.76 (validation 1989-1999, Daymet/Maurer/NLDAS forcing)',
    'n_records': len(rows), 'summary': table, 'per_basin': per_basin})
write(f'dev_calibrated_params{SUFFIX}.json', {'what': f'best {OBJ.upper()}-calibrated parameters (flat dicts usable as simulate(params=...)) '
                                             'per dev basin and model+snow, calibration window ' + str(CAL),
                                     'params': params})
for key, d in sorted(table.items()):
    print(key, {per: (v.get('n_basins'), round(v['NSE']['median'], 3), round(v['KGE']['median'], 3)) for per, v in d.items()})
