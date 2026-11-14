"""Demo examples (examples/): the same five VED exam trips as v1 (one per powertrain; v1's seeded draw within strata,
never picked by result), with the vehicle record rebuilt from today's RoadData (VED mass no longer double-counts the
136 kg driver) and the Phase D model-year input, the measured energy, the baselines and the exam run's prediction.
python make_examples.py <rung>   (after exam2.py) -> examples/examples.json (+ the trips' .npz are unchanged)"""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json
sys.path.insert(0, _P.SRC)
import numpy as np, pandas as pd
import evaluate as EV, features as F

EX = os.path.join(_P.MODEL, 'examples')
old = json.load(open(os.path.join(EX, 'examples.json')))
B, T, X, E, M = EV.real_set('ved_exam')
idx = {t: i for i, t in enumerate(M.trip_id)}
ex = pd.read_parquet(f'{_P.WORK}/data/exam2_ved_v2.parquet').set_index('trip_id')
out = []
for o in old:
    i = idx[o['trip_id']]; arch = F.ARCHS[int(B['arch'][i])]; cls = F.CLASSES[int(B['cls'][i])]
    v = {'class': cls, 'architecture': arch}
    if T['m_level'][i] >= 1 and np.isfinite(B['m_rep'][i]):
        v['mass_kg'] = float(B['m_rep'][i]); v['mass_kind'] = 'reported'
    v['fuel'] = None if arch == 'bev' else ('diesel' if B['diesel'][i] > 0.5 else 'gasoline')
    v['thermal'] = 'warm' if (T['thermal'][i] and B['warm'][i] > 0.5) else ('cold' if (T['thermal'][i] and B['cold'][i] > 0.5) else None)
    v['obd_maf'] = bool(T['maf'][i])
    if T['disp'][i] and np.isfinite(B['disp'][i]): v['displacement_L'] = float(B['disp'][i])
    tr = int(B['trans'][i])
    if T['trans'][i] and tr >= 0: v['transmission'] = F.TRANS[tr]
    for k in ('label_city', 'label_hwy', 'label_city_kWh_per_100mi', 'label_hwy_kWh_per_100mi', 'battery_kWh'):
        if o['vehicle'].get(k) is not None: v[k] = o['vehicle'][k]          # window-sticker values, as v1 wrote them
    if T['soc'][i] and np.isfinite(B['soc'][i]): v['soc_start'] = float(B['soc'][i])
    if T['my'][i] and np.isfinite(B['my'][i]): v['model_year'] = float(B['my'][i])
    r = ex.loc[o['trip_id']]
    out.append(dict(trip_id=o['trip_id'], stratum=o['stratum'], vehicle=v, measured=float(r.y),
                    unit_note=o['unit_note'], exam_prediction_MJ=float(r.pred),
                    baselines_MJ=dict(label=float(r.b_label), moves_like=float(r.b_vsp), class_median=float(r.b_class))))
json.dump(out, open(os.path.join(EX, 'examples.json'), 'w'), indent=1)
print(json.dumps(out, indent=1)[:3000])
