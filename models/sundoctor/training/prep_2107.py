"""PVDAQ system 2107 (Farm Solar Array, Arbuckle CA): 24 identical string inverters + POA + weather -> one 15-min npz."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, pandas as pd
D = _P.WORK + '/data/'
hdr = pd.read_csv(D + '2107_electrical_data.csv', nrows=0).columns
cols = ['measured_on'] + [c for c in hdr if '_ac_power_' in c or '_dc_voltage_' in c or '_dc_current_' in c]
el = pd.read_csv(D + '2107_electrical_data.csv', usecols=cols, parse_dates=['measured_on'], dtype={c: 'float32' for c in cols[1:]})
el = el.set_index('measured_on').sort_index()
irr = pd.read_csv(D + '2107_irradiance_data.csv', parse_dates=['measured_on']).set_index('measured_on').sort_index()
env = pd.read_csv(D + '2107_environment_data.csv', parse_dates=['measured_on']).set_index('measured_on').sort_index()
print('electrical', el.index.min(), el.index.max(), len(el), ' irr', irr.index.min(), irr.index.max(), len(irr), ' env', env.index.min(), env.index.max())
idx = pd.date_range('2017-11-01', '2024-12-01', freq='15min', inclusive='left')
def r15(df): return df.resample('15min', label='left', closed='left').mean().reindex(idx)
E = r15(el); I = r15(irr); V = r15(env)
pac = np.stack([E[[c for c in E.columns if f'inv_{i:02d}_ac_power' in c][0]].values for i in range(1, 25)], 1).astype(np.float32)
def col(i, k):
    c = [c for c in E.columns if f'inv_{i:02d}_{k}' in c]
    return E[c[0]].values if c else np.full(len(E), np.nan)
vdc = np.stack([col(i, 'dc_voltage') for i in range(1, 25)], 1).astype(np.float32)   # inverter 5 has no DC voltage channel
idc = np.stack([E[[c for c in E.columns if f'inv_{i:02d}_dc_current' in c][0]].values for i in range(1, 25)], 1).astype(np.float32)
poa = I.iloc[:, 0].values.astype(np.float32); tamb = ((V.iloc[:, 0].values - 32) / 1.8).astype(np.float32); wind = (V.iloc[:, 1].values * 0.44704).astype(np.float32)
np.savez_compressed(D + 'sys2107_15min.npz', t=idx.values.astype('datetime64[m]').astype(np.int64), pac=pac, vdc=vdc, idc=idc, poa=poa, tamb=tamb, wind=wind)
print('saved', pac.shape, 'nan frac pac %.3f poa %.3f tamb %.3f' % (np.isnan(pac).mean(), np.isnan(poa).mean(), np.isnan(tamb).mean()))
print('max ac per inverter (kW)', np.nanmax(pac, 0).round(1))
