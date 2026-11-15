"""Consolidate EIA-930 six-month parts into per-BA hourly parquet files with quality flags.

proc/eia930/ba/<BA>.parquet   one row per hour (time_utc = hour-BEGINNING UTC), columns:
  d, df, ng, ti                      EIA 'Adjusted' demand / reported day-ahead forecast / net generation / total
                                     interchange (MW; ti > 0 = net export)
  ng_<FUEL>                          adjusted net generation by fuel (MW), harmonised codes (DATA.md section 2)
  d_rep, ng_rep, ti_rep              values as originally reported by the BA (before EIA screening)
  d_imp, ng_imp, ti_imp, fuel_imp    True where EIA imputed the value (EIA's anomaly screening replaced or filled it)
  q_*                                our own checks (see DATA.md)
Focus BAs also keep <fuel>_rep (reported) columns.
proc/eia930/interchange.parquet  (ba, diba, time_utc, interchange_MW), all BA pairs
proc/eia930/subregion.parquet    (ba, subregion, time_utc, demand_MW)
proc/eia930/quality_summary.json  per-BA coverage, imputed shares, closure statistics
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, glob, json
import numpy as np, pandas as pd
P = f'{GRID_ROOT}/data/proc/eia930'; PARTS = f'{P}/parts'; os.makedirs(f'{P}/ba', exist_ok=True)
FOCUS = ['ERCO', 'CISO', 'PJM', 'MISO', 'SWPP', 'ISNE', 'NYIS', 'BPAT']
NEW = {'HYDROPOWER_EXCLUDING_PUMPED_STORAGE': 'WAT_HY', 'PUMPED_STORAGE': 'WAT_PS',
       'SOLAR_WITHOUT_INTEGRATED_BATTERY_STORAGE': 'SUN_NOBAT', 'SOLAR_WITH_INTEGRATED_BATTERY_STORAGE': 'SUN_BAT',
       'SOLAR_WITHO_INTEGRATED_BATTERY_STORAGE': 'SUN_BAT',   # EIA typo in the Adjusted column header
       'WIND_WITHOUT_INTEGRATED_BATTERY_STORAGE': 'WND_NOBAT', 'WIND_WITH_INTEGRATED_BATTERY_STORAGE': 'WND_BAT',
       'BATTERY_STORAGE': 'BAT', 'OTHER_ENERGY_STORAGE': 'OES', 'UNKNOWN_ENERGY_STORAGE': 'UES', 'GEOTHERMAL': 'GEO'}
BASE = ['COL', 'NG', 'NUC', 'OIL', 'WAT', 'SUN', 'WND', 'OTH', 'UNK']


def rename(c):
    if c.startswith('ng_') and not c.startswith('ng__') and '__' in c:
        f, k = c[3:].split('__')
        f = NEW.get(f, f)
        return f'ng_{f}__{k}'
    return c


def load_part(f):
    d = pd.read_parquet(f)
    d.columns = [rename(c) for c in d.columns]
    if d.columns.duplicated().any(): d = d.loc[:, ~d.columns.duplicated()]
    for k in ('rep', 'imp', 'adj'):
        # harmonise the new (Jul 2024+) detailed categories back to the long-run ones
        for tot, subs in (('WAT', ['WAT_HY', 'WAT_PS']), ('SUN', ['SUN_NOBAT', 'SUN_BAT']), ('WND', ['WND_NOBAT', 'WND_BAT'])):
            sc = [f'ng_{s}__{k}' for s in subs if f'ng_{s}__{k}' in d]
            if sc and f'ng_{tot}__{k}' not in d:
                d[f'ng_{tot}__{k}'] = d[sc].astype('float32').sum(axis=1, min_count=1)
    return d


frames = []
for f in sorted(glob.glob(f'{PARTS}/BALANCE_*.parquet')):
    frames.append(load_part(f))
bal = pd.concat(frames, ignore_index=True)
del frames
bal['ba'] = bal.ba.astype(str)
bal = bal.drop_duplicates(['ba', 'time_utc'], keep='last').sort_values(['ba', 'time_utc'])
fuels = BASE + [c for c in ['WAT_HY', 'WAT_PS', 'SUN_NOBAT', 'SUN_BAT', 'WND_NOBAT', 'WND_BAT', 'BAT', 'OES', 'UES', 'GEO']
                if f'ng_{c}__adj' in bal]
summary = {}
for ba, d in bal.groupby('ba', sort=True):
    d = d.set_index('time_utc')
    full = pd.date_range(d.index.min(), d.index.max(), freq='h', tz='UTC')
    d = d.reindex(full); d.index.name = 'time_utc'
    o = pd.DataFrame(index=d.index)
    o['d'] = d['d__adj']; o['df'] = d['df__rep']; o['ng'] = d['ng__adj']; o['ti'] = d['ti__adj']
    o['d_rep'] = d['d__rep']; o['ng_rep'] = d['ng__rep']; o['ti_rep'] = d['ti__rep']; o['sum_diba'] = d['sum_diba__rep']
    for f in fuels:
        if f'ng_{f}__adj' in d: o[f'ng_{f}'] = d[f'ng_{f}__adj']
        if ba in FOCUS and f'ng_{f}__rep' in d: o[f'ng_{f}_rep'] = d[f'ng_{f}__rep']
    o['d_imp'] = d['d__imp'].notna(); o['ng_imp'] = d['ng__imp'].notna(); o['ti_imp'] = d['ti__imp'].notna()
    impc = [f'ng_{f}__imp' for f in fuels if f'ng_{f}__imp' in d]
    o['fuel_imp'] = d[impc].notna().any(axis=1) if impc else False
    # --- own checks ---
    o['q_missing_row'] = d['ba'].isna()                                      # hour absent from EIA files
    o['q_d_missing'] = o.d.isna()
    o['q_d_nonpos'] = o.d <= 0
    med = o.d.rolling(7, center=True, min_periods=4).median()
    o['q_d_spike'] = ((o.d - med).abs() > 0.2 * med.abs()) & ~o.d_imp     # >20% off the +-3 h median, not imputed
    resid = o.ng - o.ti - o.d                                               # balance: D = NG - TI
    o['closure_resid'] = resid.astype('float32')
    o['q_closure_bad'] = resid.abs() > np.maximum(0.02 * o.d.abs(), 50)
    fc = [f'ng_{f}' for f in BASE if f'ng_{f}' in o]
    fs = o[fc].sum(axis=1, min_count=1)
    o['fuel_resid'] = (o.ng - fs).astype('float32')
    o['q_fuel_sum_bad'] = (o.ng - fs).abs() > np.maximum(0.02 * o.ng.abs(), 50)
    o['q_df_missing'] = o.df.isna()
    # q_ok: demand usable as a label
    o['q_ok'] = ~(o.q_d_missing | o.q_d_nonpos | o.q_d_spike | o.d_imp)
    for c in o.columns:
        if o[c].dtype == 'float64' or o[c].dtype == object and c not in ('region',):
            try: o[c] = o[c].astype('float32')
            except Exception: pass
    for c in [c for c in o.columns if c.startswith('q_') or c.endswith('_imp')]:
        o[c] = o[c].fillna(False).astype(bool)
    o.reset_index().to_parquet(f'{P}/ba/{ba}.parquet', index=False, compression='zstd')
    s = {'start': str(o.index.min()), 'end': str(o.index.max()), 'hours': int(len(o)),
         'region': str(d.region.dropna().iloc[-1]) if d.region.notna().any() else None,
         'demand_coverage': float(o.d.notna().mean()), 'df_coverage': float(o.df.notna().mean()),
         'd_imputed_frac': float(o.d_imp.mean()), 'ng_imputed_frac': float(o.ng_imp.mean()),
         'fuel_imputed_frac': float(o.fuel_imp.mean()), 'q_ok_frac': float(o.q_ok.mean()),
         'd_spike_frac': float(o.q_d_spike.mean()), 'closure_bad_frac': float(o.q_closure_bad.mean()),
         'closure_resid_median_abs_MW': float(resid.abs().median()) if resid.notna().any() else None,
         'mean_demand_MW': float(o.d.mean()) if o.d.notna().any() else None}
    s['by_year'] = {int(y): {'d_imputed_frac': round(float(g.d_imp.mean()), 4), 'q_ok_frac': round(float(g.q_ok.mean()), 4),
                             'closure_bad_frac': round(float(g.q_closure_bad.mean()), 4),
                             'df_coverage': round(float(g.df.notna().mean()), 4)}
                    for y, g in o.groupby(o.index.year)} if ba in FOCUS else None
    summary[ba] = s
    if ba in FOCUS: print(ba, json.dumps({k: v for k, v in s.items() if k != 'by_year'}), flush=True)
json.dump(summary, open(f'{P}/quality_summary.json', 'w'), indent=1)
del bal

for kind, cols in (('INTERCHANGE', ['ba', 'diba', 'time_utc', 'interchange_MW']), ('SUBREGION', ['ba', 'subregion', 'time_utc', 'demand_MW'])):
    a = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{PARTS}/{kind}_*.parquet'))], ignore_index=True)
    for c in cols[:2]: a[c] = a[c].astype(str)
    a = a.drop_duplicates(cols[:3], keep='last').sort_values(cols[:3])
    for c in cols[:2]: a[c] = a[c].astype('category')
    a.to_parquet(f'{P}/{kind.lower()}.parquet', index=False, compression='zstd', row_group_size=1_000_000)
    print(kind, len(a), flush=True)
print('DONE')
