"""Measure per-signal update cadence in raw VED rows of DEV vehicles only (for the log emulator)."""
import _paths as _P  # portable paths (see _paths.py)
import pandas as pd, glob, numpy as np, json, sys
sys.path.insert(0, _P.VEHICLE); import realdata as rd
dev = [int(v.split('_')[1]) for v, s in rd.splits()['VED']['vehicles'].items() if s == 'dev']
fs = sorted(glob.glob(_P.VEHICLE + '/data/proc/ved_stage1/*/*.parquet'))
rng = np.random.default_rng(0); fs = [fs[i] for i in sorted(rng.choice(len(fs), 60, replace=False))]
sig = ['speed_kmh', 'maf', 'rpm', 'absload', 'fuel_Lph', 'hv_A', 'hv_V', 'hv_soc', 'oat', 'ac_W', 'ac_kW', 'heater_W']
res = {}
for f in fs:
    d = pd.read_parquet(f); d = d[d.VehId.isin(dev)]
    for (veh, trip), g in d.groupby(['VehId', 'Trip']):
        t = g.ts_ms.values / 1000.
        res.setdefault(veh, {}).setdefault('row_dt', []).extend(np.diff(t).tolist())
        for s in sig:
            x = g[s].values
            ok = np.isfinite(x)
            if ok.sum() < 5: continue
            tt, xx = t[ok], x[ok]
            ch = np.r_[True, xx[1:] != xx[:-1]]
            # time between first appearance of a new value (only while moving, to avoid quiet stops)
            spd = g.speed_kmh.ffill().values[ok]
            tc = tt[ch]; mv = spd[ch] > 5
            dt = np.diff(tc)[mv[1:]]
            res[veh].setdefault(s, []).extend(dt.tolist())
out = {}
for veh, r in res.items():
    out[veh] = {s: dict(n=len(v), q=np.round(np.percentile(v, [10, 25, 50, 75, 90]), 2).tolist()) for s, v in r.items() if len(v) > 20}
    print(veh, json.dumps({s: o['q'] for s, o in out[veh].items()}))
json.dump(out, open(_P.WORK + '/data/dev_cadence.json', 'w'), indent=1)
