"""Cross-checks between sources -> proc/checks/qc_summary.json"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, json, glob
import numpy as np, pandas as pd
import sys; sys.path.insert(0, f'{GRID_CODE}')
import griddata as gd
PROC = f'{GRID_ROOT}/data/proc'; os.makedirs(f'{PROC}/checks', exist_ok=True)
out = {}

# 1. CAMPD Texas: annual totals; ERCOT coal+gas gross load vs EIA-930 ERCO net generation from coal / gas
u = pd.read_parquet(f'{PROC}/campd/units.parquet')[['plant_id_epa', 'unit_id_epa', 'ba', 'primary_fuel']]
camp = {}
for f in sorted(glob.glob(f'{PROC}/campd/hourly/tx_*.parquet')):
    y = int(f[-12:-8])
    d = pd.read_parquet(f, columns=['plant_id_epa', 'unit_id_epa', 'gross_load_MW', 'op_time', 'co2_short_tons', 'heat_input_MMBtu'])
    d = d.merge(u, on=['plant_id_epa', 'unit_id_epa'], how='left')
    d['fuel'] = np.where(d.primary_fuel.astype(str).str.contains('Coal', case=False), 'coal',
                         np.where(d.primary_fuel.astype(str).str.contains('Gas', case=False), 'gas', 'other'))
    gen = (d.gross_load_MW.fillna(0) * d.op_time.fillna(0))   # CAMPD gross load is the average while operating
    e = d[d.ba == 'ERCO'].assign(g=gen[d.ba == 'ERCO'])
    camp[y] = {'tx_co2_Mt': float(d.co2_short_tons.sum() * 0.90718 / 1e6), 'tx_heat_input_TBtu': float(d.heat_input_MMBtu.sum() / 1e6),
               'erco_coal_gross_TWh': float(e[e.fuel == 'coal'].g.sum() / 1e6), 'erco_gas_gross_TWh': float(e[e.fuel == 'gas'].g.sum() / 1e6)}
eia = gd.eia930('ERCO')
for y in camp:
    x = eia[eia.index.year == y]
    camp[y]['eia930_erco_coal_net_TWh'] = float(x.ng_COL.sum() / 1e6); camp[y]['eia930_erco_gas_net_TWh'] = float(x.ng_NG.sum() / 1e6)
    camp[y]['coal_net_over_gross'] = camp[y]['eia930_erco_coal_net_TWh'] / camp[y]['erco_coal_gross_TWh']
    camp[y]['gas_net_over_gross'] = camp[y]['eia930_erco_gas_net_TWh'] / camp[y]['erco_gas_gross_TWh']
out['campd_vs_eia930_erco'] = camp

# 2. Fleet capacity vs EIA-930 peak generation by fuel (ERCO, 2024)
fl = gd.fleet('ERCO', as_frame=True, existing_on='2024-07-01')
x = eia[eia.index.year == 2024]
out['fleet_vs_eia930_erco_2024'] = {
    'wind_GW_fleet': float(fl[fl.tech == 'wind'].pmax_MW.sum() / 1e3), 'wind_GW_max_hourly_eia930': float(x.ng_WND.max() / 1e3),
    'solar_GW_fleet': float(fl[fl.tech == 'solar_pv'].pmax_MW.sum() / 1e3), 'solar_GW_max_hourly_eia930': float(x.ng_SUN.max() / 1e3),
    'nuclear_GW_fleet': float(fl[fl.tech == 'nuclear'].pmax_MW.sum() / 1e3), 'nuclear_GW_max_hourly_eia930': float(x.ng_NUC.max() / 1e3),
    'coal_GW_fleet': float(fl[fl.tech == 'coal_steam'].pmax_MW.sum() / 1e3), 'coal_GW_max_hourly_eia930': float(x.ng_COL.max() / 1e3)}

# 3. Weather: NASA POWER vs ASOS population-weighted temperature
wq = {}
for ba in gd.FOCUS:
    try: w = gd.weather(ba)
    except Exception as e: continue
    if 'T_asos_C' not in w: continue
    m = w[['T2m_C', 'T_asos_C']].dropna()
    wq[ba] = {'n': int(len(m)), 'bias_power_minus_asos_C': float((m.T2m_C - m.T_asos_C).mean()),
              'mae_C': float((m.T2m_C - m.T_asos_C).abs().mean()), 'corr': float(m.corr().iloc[0, 1]),
              'ghi_coverage': float(w.ghi_Wm2.notna().mean()) if 'ghi_Wm2' in w else None,
              'asos_coverage': float(w.T_asos_C.notna().mean())}
out['weather_power_vs_asos'] = wq

# 4. EIA-930 balance closure and imputation, focus BAs, by split
q = {}
for ba in gd.FOCUS:
    d = gd.eia930(ba)
    for sp in ('dev', 'exam'):
        s = d[np.isin(d.index.year, gd.years(sp))]
        q.setdefault(ba, {})[sp] = {'q_ok_frac': float(s.q_ok.mean()), 'd_imputed_frac': float(s.d_imp.mean()),
                                    'closure_bad_frac': float(s.q_closure_bad.mean()), 'closure_resid_median_MW': float(s.closure_resid.median()),
                                    'ti_minus_sumdiba_median_abs_MW': float((s.ti - s.sum_diba).abs().median()),
                                    'df_missing_frac': float(s.df.isna().mean())}
out['eia930_quality_by_split'] = q
json.dump(out, open(f'{PROC}/checks/qc_summary.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:4000])
