"""PVDAQ system 9068 (Kersey, Colorado): 4.7 MW of CdTe modules on single-axis trackers, 8 inverter modules,
224 tracker-angle channels, 24 DC combiners, POA and weather -> one 15-min npz (chunked, so the 870 MB tracker file fits)."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, pandas as pd, json, re
D = _P.WORK + '/data/'

def mean15(path, cols=None, rename=None, chunk=150000):
    hdr = pd.read_csv(path, nrows=0).columns; use = ['measured_on'] + [c for c in hdr if c != 'measured_on' and (cols is None or cols(c))]
    sums, cnts = [], []
    for ch in pd.read_csv(path, usecols=use, chunksize=chunk, parse_dates=['measured_on']):
        ch = ch.set_index('measured_on').astype('float32'); k = ch.index.floor('15min')
        sums.append(ch.groupby(k).sum(min_count=1)); cnts.append(ch.notna().groupby(k).sum())
    S = pd.concat(sums).groupby(level=0).sum(min_count=1); C = pd.concat(cnts).groupby(level=0).sum()
    M = S / C.replace(0, np.nan); return M

idx = pd.date_range('2017-09-01', '2023-11-16', freq='15min', inclusive='left')
ac = mean15(D + '9068_ac_power_data.csv', lambda c: c.startswith('inverter_module_') and 'ac_power' in c).reindex(idx)
irr = mean15(D + '9068_irradiance_data.csv', lambda c: 'temp_compensated' in c).reindex(idx)
env = mean15(D + '9068_environment_data.csv', lambda c: c.startswith('weather_station_ambient') or 'back-of-module' in c or c.startswith('wind_sensor_wind_speed_(m/s)')).reindex(idx)
trk = mean15(D + '9068_tracker_data.csv').reindex(idx)
dc = mean15(D + '9068_dc_combiner_data.csv').reindex(idx)
print('ac', ac.shape, 'irr', irr.shape, 'env', env.shape, 'trk', trk.shape, 'dc', dc.shape, flush=True)
mods = sorted({re.search(r'inverter_module_(\d\.\d)', c).group(1) for c in ac.columns})
pac = np.stack([ac[[c for c in ac.columns if f'inverter_module_{m}_' in c][0]].values for m in mods], 1).astype(np.float32)
poa = irr.mean(1).values.astype(np.float32); poa_pads = irr.values.astype(np.float32)
tamb = env[[c for c in env.columns if 'ambient' in c][0]].values.astype(np.float32)
tmod = env[[c for c in env.columns if 'back-of-module' in c]].mean(1).values.astype(np.float32)
wind = env[[c for c in env.columns if c.startswith('wind_sensor_wind_speed')][0]].values.astype(np.float32)
tnames = list(trk.columns); tmod_of = [re.search(r'tracker_(\d\.\d)-', c).group(1) for c in tnames]
ang = trk.values.astype(np.float32)
cur = [c for c in dc.columns if 'current' in c]; vol = [c for c in dc.columns if 'voltage' in c]
np.savez_compressed(D + 'sys9068_15min.npz', t=idx.values.astype('datetime64[m]').astype(np.int64), pac=pac, mods=np.array(mods), poa=poa, poa_pads=poa_pads,
                    tamb=tamb, tmod=tmod, wind=wind, ang=ang, tmod_of=np.array(tmod_of), tnames=np.array(tnames),
                    idc=dc[cur].values.astype(np.float32), vdc=dc[vol].values.astype(np.float32), dcnames=np.array(cur))
print('modules', mods, 'trackers per module', {m: tmod_of.count(m) for m in mods})
print('max ac per module', np.nanmax(pac, 0).round(0), 'nan frac', np.isnan(pac).mean().round(3), 'poa nan', np.isnan(poa).mean().round(3))
print('tracker angle range', np.nanpercentile(ang, [0.5, 50, 99.5]))
