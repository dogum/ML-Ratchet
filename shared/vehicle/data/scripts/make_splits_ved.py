"""Freeze the VED vehicle split (by vehicle, stratified by architecture, fixed seed).
Decided before any trip-level label was computed or looked at; uses only signal availability and trip counts."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import json, sys, hashlib, numpy as np, pandas as pd
sys.path.insert(0, _VEH)
from realdata.vedspec import ved_vehicle_table
ROOT = _VEH + '/data'
SEED = 20261112

av = pd.read_csv(f'{ROOT}/proc/ved_signal_availability.csv', index_col=0)
vt = ved_vehicle_table().set_index('VehId')
vt = vt.join(av[['trips', 'maf', 'absload', 'rpm', 'fuel_Lph', 'hv_A']], how='left')
has_fuel = (vt.maf > 0.3) | ((vt.absload > 0.3) & (vt.rpm > 0.3)) | (vt.fuel_Lph > 0.3)
has_elec = vt.hv_A > 0.3
vt['labelled'] = np.where(vt.ved_type.isin(['ICE', 'HEV']), has_fuel,
                          np.where(vt.ved_type == 'PHEV', has_fuel & has_elec, has_elec))
vt['labelled'] &= vt.trips.fillna(0) > 0

rng = np.random.default_rng(SEED)
split = {}
for v in vt.index:
    split[int(v)] = 'exam' if vt.labelled[v] else 'train_noise'
elig = vt[vt.labelled & (vt.trips >= 20)]


def pick(df, n):
    ids = sorted(df.index.tolist())
    return [int(x) for x in rng.choice(ids, size=n, replace=False)] if n else []


dev = []
dev += pick(elig[elig.ved_type == 'ICE'], 8)
dev += pick(elig[elig.ved_type == 'HEV'], 5)
dev += pick(elig[(elig.ved_type == 'PHEV') & (elig.architecture == 'phev_series')], 1)
dev += pick(elig[(elig.ved_type == 'PHEV') & (elig.architecture == 'phev_powersplit')], 1)
dev += pick(elig[elig.ved_type == 'EV'], 1)
for v in dev: split[v] = 'dev'

out = {'VED': {'unit': 'vehicle', 'seed': SEED,
               'rule': ('labelled = has a fuel signal (MAF, abs-load+rpm or OBD fuel rate; PHEV also needs HV current) and >=1 trip. '
                        'dev = seeded draw among labelled vehicles with >=20 trips: 8 ICE, 5 HEV, 1 PHEV series (Volt-type), '
                        '1 PHEV power-split, 1 EV. exam = all other labelled vehicles. train_noise = the vehicles with no fuel/energy '
                        'signal (no labels exist), usable only as route/speed/grade/weather inputs.'),
               'vehicles': {f'VED_{v}': s for v, s in sorted(split.items())}}}
vt['split'] = [split[int(v)] for v in vt.index]
print(vt.groupby(['ved_type', 'split']).size().unstack(fill_value=0))
print('dev', sorted(dev))
json.dump(out, open(f'{ROOT}/proc/splits_ved_part.json', 'w'), indent=1)
