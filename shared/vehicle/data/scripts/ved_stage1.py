"""Stage 1: VED weekly CSVs -> parquet partitioned by vehicle bucket (VehId % NB).
Resumable: a weekly file listed in stage1_done.txt is skipped. Deletes each CSV after it is converted
(the .7z is kept, so this is reversible).
usage: python3 -I ved_stage1.py <extract_dir>
"""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, glob
import numpy as np, pandas as pd

ROOT = _VEH + '/data'
OUT = f'{ROOT}/proc/ved_stage1'
NB = 16
COLS = ['DayNum', 'VehId', 'Trip', 'ts_ms', 'lat', 'lon', 'speed_kmh', 'maf', 'rpm', 'absload', 'oat',
        'fuel_Lph', 'ac_kW', 'ac_W', 'heater_W', 'hv_A', 'hv_soc', 'hv_V', 'stft1', 'stft2', 'ltft1', 'ltft2']
DT = {c: 'float32' for c in COLS}
DT.update(DayNum='float64', VehId='int32', Trip='int32', ts_ms='int64', lat='float64', lon='float64')

def main(xdir):
    os.makedirs(OUT, exist_ok=True)
    donef = f'{OUT}/stage1_done.txt'
    done = set(open(donef).read().split()) if os.path.exists(donef) else set()
    for f in sorted(glob.glob(f'{xdir}/*.csv')):
        name = os.path.basename(f)[:-4]
        if name in done:
            os.remove(f); continue
        parts = {b: [] for b in range(NB)}
        for ch in pd.read_csv(f, header=0, names=COLS, dtype=DT, chunksize=500_000, na_values=['NaN', '']):
            b = ch.VehId.values % NB
            for k in range(NB):
                s = ch[b == k]
                if len(s): parts[k].append(s)
        for k in range(NB):
            if parts[k]:
                d = f'{OUT}/b{k:02d}'; os.makedirs(d, exist_ok=True)
                pd.concat(parts[k]).to_parquet(f'{d}/{name}.parquet', index=False)
        with open(donef, 'a') as fh: fh.write(name + '\n')
        os.remove(f)
        print('done', name, flush=True)

if __name__ == '__main__':
    main(sys.argv[1])
