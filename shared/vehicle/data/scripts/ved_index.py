"""Collect all VED trip meta JSONs into proc/ved/index.parquet; verify every npz opens and has the contract keys."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import glob, json, os, zipfile, numpy as np, pandas as pd
ROOT = _VEH + '/data'
KEYS = {'t', 'v', 'grade', 'elev', 'lat', 'lon', 'heading', 'T_amb', 'wind_speed', 'wind_dir', 'rho_air', 'surface',
        'fuel_rate', 'engine_rpm', 'batt_current', 'batt_voltage', 'soc', 'aux_power'}
rows, bad = [], []
for jf in sorted(glob.glob(f'{ROOT}/proc/ved/trips/*/*.json')):
    m = json.load(open(jf))
    nf = jf[:-5] + '.npz'
    try:
        with zipfile.ZipFile(nf) as z:
            names = {n[:-4] for n in z.namelist()}
            if z.testzip() is not None or not KEYS <= names: raise ValueError('keys/crc')
    except Exception as e:
        bad.append((jf, repr(e))); continue
    m['path'] = os.path.relpath(nf, f'{ROOT}/proc/ved')
    rows.append(m)
idx = pd.DataFrame(rows)
idx.to_parquet(f'{ROOT}/proc/ved/index.parquet', index=False)
print('trips', len(idx), 'bad', len(bad)); [print(b) for b in bad[:10]]
print(idx.groupby(['ved_type', 'split']).agg(trips=('trip_id', 'size'), vehicles=('vehicle_id', 'nunique'),
                                             label_ok=('q_label_ok', 'sum'), route_ok=('q_route_ok', 'sum'),
                                             km=('distance_m', lambda s: s.sum() / 1000)).round(0).to_string())
