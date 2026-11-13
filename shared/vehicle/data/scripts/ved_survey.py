"""Survey VED stage-1 data: per-vehicle signal availability and a timezone check (OAT vs ASOS)."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import glob, numpy as np, pandas as pd
ROOT = _VEH + '/data'
rows = []; oat_samples = []
for b in range(16):
    d = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{ROOT}/proc/ved_stage1/b{b:02d}/*.parquet')])
    g = d.groupby('VehId')
    a = g.agg(n=('ts_ms', 'size'), trips=('Trip', 'nunique'),
              **{c: (c, lambda s: float(s.notna().mean())) for c in
                 ['maf', 'rpm', 'absload', 'oat', 'fuel_Lph', 'ac_kW', 'ac_W', 'heater_W', 'hv_A', 'hv_V', 'hv_soc', 'stft1', 'stft2', 'ltft1', 'ltft2']})
    a['dt_med_ms'] = g.ts_ms.apply(lambda s: float(np.median(np.diff(s.values))) if len(s) > 2 else np.nan)
    rows.append(a)
    s = d[d.oat.notna()].sample(min(20000, int(d.oat.notna().sum())), random_state=0)
    s = s.assign(t_days=s.DayNum + s.ts_ms / 86400e3)[['VehId', 't_days', 'oat']]
    oat_samples.append(s)
    print('bucket', b, flush=True)
av = pd.concat(rows).sort_index()
av.to_csv(f'{ROOT}/proc/ved_signal_availability.csv')
pd.concat(oat_samples).to_parquet(f'{ROOT}/proc/ved_oat_sample.parquet')
print(av.describe().T.to_string())
