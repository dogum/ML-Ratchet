"""Build per-unit generator fleets (GRID_BRIEF generator spec) for the focus BAs from PUDL EIA-860/923 tables,
CAMPD-derived unit characteristics (if built: proc/campd/unit_characteristics.parquet), and documented defaults.

Output: proc/fleet/fleet_<BA>.parquet (flat table, one row per EIA generator) and proc/fleet/fleet_<BA>.json
(list of spec dicts), plus proc/fleet/defaults.json (every default value with its source).
Every filled field has a companion source in the `src` dict (json) / `src_<field>` column (parquet).
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, json, sys
import numpy as np, pandas as pd

RAW = f'{GRID_ROOT}/data/raw'; PROC = f'{GRID_ROOT}/data/proc'
OUT = f'{PROC}/fleet'; os.makedirs(OUT, exist_ok=True)
FOCUS = ['ERCO', 'CISO', 'PJM', 'MISO', 'SWPP', 'ISNE', 'NYIS', 'BPAT']
BAS = sys.argv[1:] or FOCUS
FIRST_DATE = pd.Timestamp('2018-01-01')

TECH = {
    'Nuclear': 'nuclear', 'Conventional Steam Coal': 'coal_steam', 'Coal Integrated Gasification Combined Cycle': 'coal_steam',
    'Natural Gas Fired Combined Cycle': 'gas_cc', 'Natural Gas Fired Combustion Turbine': 'gas_ct',
    'Natural Gas Steam Turbine': 'gas_steam', 'Natural Gas Internal Combustion Engine': 'gas_ice',
    'Petroleum Liquids': 'oil', 'Petroleum Coke': 'oil', 'Conventional Hydroelectric': 'hydro_res',
    'Hydroelectric Pumped Storage': 'pumped', 'Onshore Wind Turbine': 'wind', 'Offshore Wind Turbine': 'wind',
    'Solar Photovoltaic': 'solar_pv', 'Batteries': 'battery', 'Wood/Wood Waste Biomass': 'biomass',
    'Other Waste Biomass': 'biomass', 'Landfill Gas': 'biomass', 'Municipal Solid Waste': 'biomass',
    'Geothermal': 'geothermal'}
IBR = {'wind', 'solar_pv', 'battery'}

# --- documented defaults -------------------------------------------------------------------------------------------
rts = pd.read_csv(f'{RAW}/rts_gmlc/gen.csv')
rts['pmin_frac'] = rts['PMin MW'] / rts['PMax MW']; rts['ramp_frac'] = rts['Ramp Rate MW/Min'] / rts['PMax MW']
rts['start_heat_warm_per_MW'] = rts['Start Heat Warm MBTU'] / rts['PMax MW']
RTS_ROW = {'nuclear': ('NUCLEAR', 'Nuclear'), 'coal_steam': ('STEAM', 'Coal'), 'gas_cc': ('CC', 'NG'), 'gas_ct': ('CT', 'NG'),
           'gas_steam': ('STEAM', 'Coal'), 'gas_ice': ('CT', 'Oil'), 'oil': ('STEAM', 'Oil'), 'biomass': ('STEAM', 'Coal'),
           'hydro_res': ('HYDRO', 'Hydro'), 'hydro_ror': ('ROR', 'Hydro'), 'pumped': ('HYDRO', 'Hydro')}
rts_def = {}
for tech, (ut, fu) in RTS_ROW.items():
    r = rts[(rts['Unit Type'] == ut) & (rts['Fuel'] == fu)]
    # RTS-GMLC heat-rate shape: avg HR at output_pct_0, incremental HRs to the following points
    pts = []
    for _, u in r.iterrows():
        pct = [u[f'Output_pct_{i}'] for i in range(5)]; pmax = u['PMax MW']
        if pd.isna(u['HR_avg_0']) or u['HR_avg_0'] <= 3412.01: continue
        mw = [p * pmax for p in pct if pd.notna(p)]
        fuel = [u['HR_avg_0'] * mw[0]]
        for i in range(1, len(mw)):
            inc = u[f'HR_incr_{i}']
            if pd.isna(inc): break
            fuel.append(fuel[-1] + inc * (mw[i] - mw[i - 1]))
        hr = [f / m for f, m in zip(fuel, mw)]
        pts.append((np.array(pct[:len(hr)], float), np.array(hr) / hr[-1]))   # shape normalised to HR at full load
    shape = None
    if pts:
        grid = np.array([0.4, 0.6, 0.8, 1.0]) if tech != 'nuclear' else np.array([0.99, 1.0])
        shape = [[float(x), float(np.median([np.interp(x, p, h) for p, h in pts]))] for x in grid]
    rts_def[tech] = dict(pmin_frac=float(r.pmin_frac.median()), ramp_frac_per_min=float(r.ramp_frac.median()),
                         min_up_h=float(r['Min Up Time Hr'].median()), min_down_h=float(r['Min Down Time Hr'].median()),
                         start_time_h=float(r['Start Time Warm Hr'].median()) if tech != 'nuclear' else None,
                         start_heat_MMBtu_per_MW=float(r.start_heat_warm_per_MW.median()),
                         inertia_H_s=float(r['Inertia MJ/MW'].median()), hr_shape=shape)
# hydro: RTS ramp 1.0 per min, min load 0 -> keep; storage/IBR: zero inertia
CO2 = {  # kg CO2 / MMBtu, 40 CFR Part 98 Subpart C Table C-1 (EPA GHG Reporting Program)
    'NG': 53.06, 'BIT': 93.28, 'SUB': 97.17, 'LIG': 97.72, 'RC': 93.28, 'WC': 93.28, 'SC': 97.17, 'DFO': 73.96, 'KER': 75.20,
    'JF': 72.22, 'RFO': 75.10, 'PC': 102.41, 'WO': 74.00, 'WDS': 93.80, 'LFG': 52.07, 'OBG': 52.07, 'AB': 118.17,
    'MSW': 90.70, 'BLQ': 94.40, 'OG': 59.00, 'BFG': 274.32, 'PG': 61.46, 'TDF': 85.97, 'SGC': 53.06, 'OBS': 105.51,
    'WDL': 81.60, 'OBL': 84.00, 'NUC': 0.0, 'WAT': 0.0, 'SUN': 0.0, 'WND': 0.0, 'GEO': 0.0, 'MWH': 0.0}
CO2_SRC = '40 CFR 98 Subpart C Table C-1 default CO2 emission factors (biogenic fuels listed but counted as emitted)'
GOV = {  # nominal governor parameters (see DATA.md section 4)
    'deadband_Hz_us': 0.036, 'droop_us': 0.05,
    'T_s': {'nuclear': 7.0, 'coal_steam': 7.0, 'gas_steam': 7.0, 'oil': 7.0, 'biomass': 7.0, 'gas_cc': 3.0,
            'gas_ct': 0.5, 'gas_ice': 0.5, 'hydro_res': 1.0, 'hydro_ror': 1.0, 'pumped': 1.0, 'geothermal': 7.0}}
H_KUNDUR_GEO = 3.0   # geothermal steam unit: low end of Kundur Table 3.2 thermal range (2.5-6 s)
DEFAULTS = {
    'rts_gmlc_by_tech': rts_def,
    'rts_gmlc_mapping': {k: list(v) for k, v in RTS_ROW.items()},
    'rts_gmlc_source': 'RTS-GMLC gen.csv (github.com/GridMod/RTS-GMLC, RTS_Data/SourceData/gen.csv; Barrows et al., '
                       '"The IEEE Reliability Test System: A Proposed 2019 Update", IEEE Trans. Power Syst. 35(1), 2020). '
                       'Medians over units of the mapped type. gas_steam and biomass use the coal STEAM row; gas_ice uses the '
                       'oil CT row; oil uses the oil STEAM row (choices, not data).',
    'co2_kg_per_MMBtu': CO2, 'co2_source': CO2_SRC,
    'governor': GOV,
    'governor_source': 'droop 5% and deadband +-0.036 Hz: NERC BAL-001-TRE-2 (Primary Frequency Response in the ERCOT Region) '
                       'and NERC Reliability Guideline "Primary Frequency Control" (2019) recommended values; T_s: nominal '
                       'turbine response time constants (steam reheat T_RH ~7 s, gas turbine ~0.5 s, hydro water starting '
                       'time ~1 s) from Kundur, Power System Stability and Control (1994), ch. 9. Nuclear: headroom 0 '
                       '(US nuclear units run at full output and provide no primary response in practice).',
    'inertia_source': 'RTS-GMLC "Inertia MJ/MW" by unit type (CC 5.0, CT 2.8, coal steam 3.0, nuclear 5.0, hydro 3.5 s on '
                      'the unit MVA base); inverter-based units 0; geothermal 3.0 (Kundur 1994 Table 3.2 thermal range '
                      '2.5-6 s, low end). mva_base = nameplate MW / EIA-860 nameplate power factor (default 0.85, '
                      'typical synchronous machine rating).',
    'pmin_source_order': ['EIA-860 minimum_load_mw', 'CAMPD (non-exam years) min stable gross load',
                          'RTS-GMLC PMin/PMax by type x pmax'],
    'ffr_battery': {'trigger_Hz': 59.85, 'delay_s': 0.25,
                    'source': 'ERCOT Nodal Protocols FFR (deploy within 15 cycles at 59.85 Hz); MW per unit unknown (None)'},
    'battery_eta_rt': 0.85, 'battery_eta_source': 'typical Li-ion AC round trip (NREL ATB 2023: 85%); assumption',
    'pf_default': 0.85,
}

# --- EIA-860 (PUDL) -----------------------------------------------------------------------------------------------
cols = ['plant_id_eia', 'generator_id', 'report_date', 'unit_id_pudl', 'plant_name_eia', 'balancing_authority_code_eia',
        'technology_description', 'prime_mover_code', 'energy_source_code_1', 'energy_source_code_2', 'generator_operating_date',
        'generator_retirement_date', 'operational_status', 'capacity_mw', 'summer_capacity_mw', 'winter_capacity_mw',
        'minimum_load_mw', 'nameplate_power_factor', 'energy_storage_capacity_mwh', 'net_capacity_mwdc', 'latitude',
        'longitude', 'state', 'county', 'time_cold_shutdown_full_load_code', 'fuel_cost_per_mmbtu',
        'unit_heat_rate_mmbtu_per_mwh', 'net_generation_mwh', 'capacity_factor', 'rto_iso_lmp_node_id', 'timezone',
        'turbines_num', 'carbon_capture', 'supercritical_tech', 'ultrasupercritical_tech', 'duct_burners']
g = pd.read_parquet(f'{RAW}/pudl/out_eia__yearly_generators.parquet', columns=cols)
g['report_date'] = pd.to_datetime(g.report_date); g['year'] = g.report_date.dt.year
for c in ['generator_operating_date', 'generator_retirement_date']: g[c] = pd.to_datetime(g[c], errors='coerce')
g = g[g.year >= 2017].sort_values(['plant_id_eia', 'generator_id', 'year'])
key = ['plant_id_eia', 'generator_id']
# latest non-null value of each attribute per generator (EIA-860 final years; 2026 rows are EIA-860M)
last = g.groupby(key).last()          # GroupBy.last skips nulls column-wise
last['last_report_year'] = g.groupby(key).year.max()
last['first_report_year'] = g.groupby(key).year.min()
ever_existing = g[g.operational_status == 'existing'].groupby(key).size().index
last = last.loc[last.index.isin(ever_existing)]
# drop generators retired before 2018
last = last[~(last.generator_retirement_date < FIRST_DATE)]
last = last.reset_index()
# status as of the latest report
st = g.groupby(key).apply(lambda d: d.operational_status.iloc[-1], include_groups=False).rename('status_latest')
last = last.merge(st.reset_index(), on=key, how='left')

# plant BA as of the latest year with a value
pl = pd.read_parquet(f'{RAW}/pudl/out_eia__yearly_plants.parquet', columns=['plant_id_eia', 'report_date', 'balancing_authority_code_eia'])
pl['year'] = pd.to_datetime(pl.report_date).dt.year
plba = pl.dropna(subset=['balancing_authority_code_eia']).sort_values('year').groupby('plant_id_eia').balancing_authority_code_eia.last()
last['ba'] = last.plant_id_eia.map(plba).fillna(last.balancing_authority_code_eia)

# EIA-923 annual heat rates by generator (fuel for electricity / net generation)
hr = pd.read_parquet(f'{RAW}/pudl/out_eia923__yearly_generation_fuel_by_generator.parquet')
hr['year'] = pd.to_datetime(hr.report_date).dt.year
hr = hr[(hr.net_generation_mwh > 0) & (hr.fuel_consumed_for_electricity_mmbtu > 0)]
hr['hr'] = hr.fuel_consumed_for_electricity_mmbtu / hr.net_generation_mwh
hr = hr[(hr.hr > 5.5) & (hr.hr < 25)]
# fleet value: generation-weighted over NON-EXAM years 2018-2022 (exam years 2023-2025 untouched)
h2 = hr[hr.year.between(2018, 2022)].groupby(key).apply(
    lambda d: pd.Series({'hr_eia923_2018_2022': (d.fuel_consumed_for_electricity_mmbtu.sum() / d.net_generation_mwh.sum()),
                         'gen_mwh_2018_2022': d.net_generation_mwh.sum()}), include_groups=False).reset_index()
last = last.merge(h2, on=key, how='left')

# wind / solar / storage details
def latest(tbl):
    d = pd.read_parquet(f'{RAW}/pudl/{tbl}.parquet'); d['report_date'] = pd.to_datetime(d.report_date)
    return d.sort_values('report_date').groupby(key).last().reset_index().drop(columns='report_date')
wind = latest('core_eia860__scd_generators_wind'); sol = latest('core_eia860__scd_generators_solar')
sto = latest('core_eia860__scd_generators_energy_storage')
last = last.merge(wind[key + ['turbine_hub_height_feet', 'predominant_turbine_manufacturer', 'predominant_turbine_model',
                              'design_wind_speed_mph', 'wind_quality_class']], on=key, how='left')
last = last.merge(sol[key + ['azimuth_angle_deg', 'tilt_angle_deg', 'uses_technology_single_axis_tracking',
                             'uses_technology_dual_axis_tracking', 'uses_technology_fixed_tilt', 'uses_bifacial_panels',
                             'standard_testing_conditions_capacity_mwdc']], on=key, how='left')
last = last.merge(sto[key + ['max_charge_rate_mw', 'max_discharge_rate_mw', 'served_arbitrage', 'served_frequency_regulation',
                             'served_ramping_spinning_reserve']], on=key, how='left')

# monthly delivered fuel cost by state/fuel group (EIA-923 receipts), for the default fuel price
fr = pd.read_parquet(f'{RAW}/pudl/out_eia923__monthly_fuel_receipts_costs.parquet',
                     columns=['report_date', 'state', 'fuel_type_code_pudl', 'fuel_cost_per_mmbtu', 'fuel_received_mmbtu'])
fr['year'] = pd.to_datetime(fr.report_date).dt.year
fr = fr[fr.year.between(2018, 2022) & fr.fuel_cost_per_mmbtu.between(0.1, 100)]
fr['w'] = fr.fuel_cost_per_mmbtu * fr.fuel_received_mmbtu
fstate = fr.groupby(['state', 'fuel_type_code_pudl']).apply(lambda d: d.w.sum() / d.fuel_received_mmbtu.sum(), include_groups=False)
fnat = fr.groupby('fuel_type_code_pudl').apply(lambda d: d.w.sum() / d.fuel_received_mmbtu.sum(), include_groups=False)
DEFAULTS['fuel_cost_state_2018_2022'] = {f'{s}|{f}': round(float(v), 3) for (s, f), v in fstate.items()}
DEFAULTS['fuel_cost_national_2018_2022'] = {f: round(float(v), 3) for f, v in fnat.items()}
DEFAULTS['fuel_cost_source'] = 'EIA-923 fuel receipts and costs (PUDL out_eia923__monthly_fuel_receipts_costs), receipt-weighted 2018-2022'
FUELGRP = {'NG': 'gas', 'BIT': 'coal', 'SUB': 'coal', 'LIG': 'coal', 'RC': 'coal', 'WC': 'coal', 'SC': 'coal', 'DFO': 'oil',
           'RFO': 'oil', 'KER': 'oil', 'JF': 'oil', 'PC': 'oil', 'WO': 'oil', 'NUC': 'nuclear'}
NUC_FUEL = 0.65   # $/MMBtu, approx. US nuclear fuel cost (EIA Electric Power Annual Table 8.4: ~0.6-0.7 $/MMBtu-equivalent)

# CAMPD-derived characteristics (non-exam years), if available
cc_path = f'{PROC}/campd/unit_characteristics_gen.parquet'
campd = pd.read_parquet(cc_path) if os.path.exists(cc_path) else None


def tech_of(r):
    t = TECH.get(r.technology_description, 'other')
    return t


def build(ba):
    d = last[last.ba == ba].copy()
    d['tech'] = d.apply(tech_of, axis=1)
    rows = []
    for _, r in d.iterrows():
        t = r.tech; src = {}
        cap = float(r.capacity_mw) if pd.notna(r.capacity_mw) else None
        pmax = r.summer_capacity_mw if pd.notna(r.summer_capacity_mw) and r.summer_capacity_mw > 0 else cap
        src['pmax_MW'] = 'EIA-860 summer capacity' if pd.notna(r.summer_capacity_mw) and r.summer_capacity_mw > 0 else 'EIA-860 nameplate'
        D = rts_def.get(t, {})
        u = dict(id=f'{int(r.plant_id_eia)}_{r.generator_id}', plant_id=int(r.plant_id_eia), generator_id=r.generator_id,
                 plant_name=r.plant_name_eia, region=ba, state=r.state, county=r.county,
                 lat=float(r.latitude) if pd.notna(r.latitude) else None, lon=float(r.longitude) if pd.notna(r.longitude) else None,
                 tech=t, eia_technology=r.technology_description, prime_mover=r.prime_mover_code,
                 cc_part=r.prime_mover_code if t == 'gas_cc' else None,
                 unit_id_pudl=int(r.unit_id_pudl) if pd.notna(r.unit_id_pudl) else None,
                 nameplate_MW=cap, pmax_MW=float(pmax) if pmax is not None and pd.notna(pmax) else None,
                 winter_MW=float(r.winter_capacity_mw) if pd.notna(r.winter_capacity_mw) else None,
                 online_date=str(r.generator_operating_date.date()) if pd.notna(r.generator_operating_date) else None,
                 retired_date=str(r.generator_retirement_date.date()) if pd.notna(r.generator_retirement_date) else None,
                 status_latest=r.status_latest, last_report_year=int(r.last_report_year),
                 fuel=r.energy_source_code_1, fuel2=r.energy_source_code_2 if pd.notna(r.energy_source_code_2) else None,
                 lmp_node=r.rto_iso_lmp_node_id if pd.notna(r.rto_iso_lmp_node_id) else None, ibr=t in IBR)
        P = u['pmax_MW'] or 0.0
        # pmin
        pmin = None
        if pd.notna(r.minimum_load_mw) and 0 <= r.minimum_load_mw <= (P or 1e9) * 1.01:
            pmin = float(r.minimum_load_mw); src['pmin_MW'] = 'EIA-860 minimum_load_mw'
        cu = None
        if campd is not None:
            m = campd[(campd.plant_id_eia == r.plant_id_eia) & (campd.generator_id == r.generator_id)]
            if len(m): cu = m.iloc[0]
        if pmin is None and cu is not None and pd.notna(cu.pmin_frac):
            pmin = float(cu.pmin_frac * P); src['pmin_MW'] = 'CAMPD 2018-2022 min stable gross load fraction x pmax'
        if pmin is None and 'pmin_frac' in D:
            pmin = float(D['pmin_frac'] * P); src['pmin_MW'] = f'RTS-GMLC {t} PMin/PMax x pmax'
        if t in ('wind', 'solar_pv'): pmin = 0.0; src['pmin_MW'] = 'IBR: 0'
        if t == 'battery': pmin = -float(r.max_charge_rate_mw) if pd.notna(r.max_charge_rate_mw) else -P; src['pmin_MW'] = 'battery: -max charge rate (EIA-860) or -pmax'
        u['pmin_MW'] = pmin
        # value without any CAMPD input (used by griddata.fleet(..., plant_safe=True) for PlantDetective exam plants)
        u['pmin_MW_nocampd'] = pmin if src.get('pmin_MW', '').find('CAMPD') < 0 else (float(D['pmin_frac'] * P) if 'pmin_frac' in D else None)
        # ramps
        if t in IBR or t in ('hydro_res', 'hydro_ror', 'pumped'):
            u['ramp_up_MW_per_min'] = u['ramp_down_MW_per_min'] = P; src['ramp'] = 'IBR/hydro: full range in 1 min (RTS-GMLC hydro ramp = PMax/min)'
        elif 'ramp_frac_per_min' in D:
            u['ramp_up_MW_per_min'] = u['ramp_down_MW_per_min'] = float(D['ramp_frac_per_min'] * P); src['ramp'] = f'RTS-GMLC {t}'
        else:
            u['ramp_up_MW_per_min'] = u['ramp_down_MW_per_min'] = None
        # observed hour-to-hour ramp (CAMPD, steady hours, p99): what the unit actually does, not its capability
        u['ramp_obs_p99_MW_per_h'] = float(60 * cu.ramp_up_frac_per_min * P) if cu is not None and pd.notna(cu.ramp_up_frac_per_min) else None
        u['starts_per_year_obs'] = float(cu.starts_per_year) if cu is not None and pd.notna(cu.starts_per_year) else None
        # up/down, start
        if cu is not None and pd.notna(cu.min_up_h):
            u['min_up_h'] = float(cu.min_up_h); u['min_down_h'] = float(cu.min_down_h); src['min_up_down'] = 'CAMPD 2018-2022 p5 of run / off spell lengths'
        elif 'min_up_h' in D:
            u['min_up_h'] = D['min_up_h']; u['min_down_h'] = D['min_down_h']; src['min_up_down'] = f'RTS-GMLC {t}'
        else:
            u['min_up_h'] = u['min_down_h'] = 0.0 if t in IBR else None
        u['start_time_h'] = D.get('start_time_h') if t not in IBR else 0.0
        u['cold_start_code_eia'] = r.time_cold_shutdown_full_load_code if pd.notna(r.time_cold_shutdown_full_load_code) else None
        # fuel cost
        fg = FUELGRP.get(r.energy_source_code_1)
        fc = None
        if t == 'nuclear': fc = NUC_FUEL; src['fuel_cost'] = 'EIA Electric Power Annual nuclear fuel cost (approx. 0.65 $/MMBtu)'
        elif fg is not None:
            fc = fstate.get((r.state, fg), fnat.get(fg)); src['fuel_cost'] = 'EIA-923 receipts 2018-2022, state x fuel group'
        elif t in ('wind', 'solar_pv', 'hydro_res', 'hydro_ror', 'pumped', 'battery', 'geothermal'): fc = 0.0
        u['fuel_cost_$_per_MMBtu'] = float(fc) if fc is not None and pd.notna(fc) else None
        u['co2_kg_per_MMBtu'] = CO2.get(r.energy_source_code_1)
        # heat rate
        hr923 = float(r.hr_eia923_2018_2022) if pd.notna(r.hr_eia923_2018_2022) else None
        u['hr_eia923_2018_2022_MMBtu_per_MWh'] = hr923
        curve = None
        if cu is not None and pd.notna(cu.hr_a):
            # CAMPD fit: heat input [MMBtu/h] = a + b P + c P^2 on GROSS load; expressed as average HR at grid points
            lo = max(pmin or 0.3 * P, 0.2 * P)
            curve = [[round(x, 2), round(float((cu.hr_a + cu.hr_b * x + cu.hr_c * x * x) / x), 3)] for x in np.linspace(lo, P, 4) if x > 0]
            src['heat_rate_curve'] = 'CAMPD 2018-2022 quadratic fit of hourly heat input vs gross load (gross basis)'
        elif D.get('hr_shape') and (hr923 or t in ('nuclear',)):
            base = hr923 if hr923 else 10.45
            # scale RTS shape so its value at 80% load equals the EIA-923 annual average
            shp = D['hr_shape']; s80 = np.interp(0.8, [p[0] for p in shp], [p[1] for p in shp]) if t != 'nuclear' else 1.0
            curve = [[round(p[0] * P, 2), round(base * p[1] / s80, 3)] for p in shp]
            src['heat_rate_curve'] = 'RTS-GMLC shape scaled to EIA-923 2018-2022 annual average (net basis)' if hr923 else 'nuclear 10.45 MMBtu/MWh (EIA typical)'
        u['heat_rate_curve'] = curve
        alt = curve
        if src.get('heat_rate_curve', '').startswith('CAMPD'):
            alt = None
            if D.get('hr_shape') and hr923:
                shp = D['hr_shape']; s80 = np.interp(0.8, [p[0] for p in shp], [p[1] for p in shp])
                alt = [[round(p[0] * P, 2), round(hr923 * p[1] / s80, 3)] for p in shp]
        u['heat_rate_curve_nocampd'] = alt
        u['min_up_h_nocampd'] = D.get('min_up_h', u['min_up_h']) if src.get('min_up_down', '').startswith('CAMPD') else u['min_up_h']
        u['min_down_h_nocampd'] = D.get('min_down_h', u['min_down_h']) if src.get('min_up_down', '').startswith('CAMPD') else u['min_down_h']
        # dynamics
        if t in IBR:
            u['inertia_H_s'] = 0.0; u['mva_base'] = P; src['inertia'] = 'IBR: 0'
        else:
            pf = float(r.nameplate_power_factor) if pd.notna(r.nameplate_power_factor) and 0.5 < r.nameplate_power_factor <= 1 else 0.85
            u['mva_base'] = round((cap or P) / pf, 2)
            u['inertia_H_s'] = D.get('inertia_H_s', H_KUNDUR_GEO if t == 'geothermal' else None)
            src['inertia'] = 'RTS-GMLC by type' if 'inertia_H_s' in D else ('Kundur Table 3.2 (geothermal)' if t == 'geothermal' else 'unknown')
        if t in GOV['T_s']:
            u['governor'] = dict(droop=GOV['droop_us'], deadband_Hz=GOV['deadband_Hz_us'], T_s=GOV['T_s'][t],
                                 headroom_frac=0.0 if t == 'nuclear' else None)
        else:
            u['governor'] = None
        u['ffr'] = dict(MW=None, trigger_Hz=59.85, delay_s=0.25) if (t == 'battery' and ba == 'ERCO') else None
        # storage
        if t in ('battery', 'pumped'):
            u['energy_MWh'] = float(r.energy_storage_capacity_mwh) if pd.notna(r.energy_storage_capacity_mwh) else None
            u['eta_rt'] = DEFAULTS['battery_eta_rt'] if t == 'battery' else 0.78
            src['eta_rt'] = DEFAULTS['battery_eta_source'] if t == 'battery' else 'pumped hydro typical 0.78 (assumption)'
            u['soc_min'] = 0.0; u['soc_max'] = 1.0
        # wind / solar
        if t == 'wind':
            u['hub_height_m'] = round(float(r.turbine_hub_height_feet) * 0.3048, 1) if pd.notna(r.turbine_hub_height_feet) else None
            u['power_curve'] = f'{r.predominant_turbine_manufacturer}|{r.predominant_turbine_model}' if pd.notna(r.predominant_turbine_model) else None
            u['design_wind_speed_mph'] = float(r.design_wind_speed_mph) if pd.notna(r.design_wind_speed_mph) else None
        if t == 'solar_pv':
            tr = 'dual' if r.uses_technology_dual_axis_tracking is True else ('single' if r.uses_technology_single_axis_tracking is True
                 else ('fixed' if r.uses_technology_fixed_tilt is True else None))
            u['tracking'] = tr
            dc = r.net_capacity_mwdc if pd.notna(r.net_capacity_mwdc) else r.standard_testing_conditions_capacity_mwdc
            u['dc_ac'] = round(float(dc) / cap, 3) if pd.notna(dc) and cap else None
            u['tilt'] = float(r.tilt_angle_deg) if pd.notna(r.tilt_angle_deg) else None
            u['azimuth'] = float(r.azimuth_angle_deg) if pd.notna(r.azimuth_angle_deg) else None
        # start cost: warm start fuel (RTS-GMLC MMBtu per MW) x fuel cost
        if 'start_heat_MMBtu_per_MW' in D and u['fuel_cost_$_per_MMBtu'] and t not in ('hydro_res', 'hydro_ror', 'pumped'):
            u['start_cost_$'] = round(D['start_heat_MMBtu_per_MW'] * P * u['fuel_cost_$_per_MMBtu'], 0); src['start_cost'] = 'RTS-GMLC warm start heat per MW x fuel cost'
        else:
            u['start_cost_$'] = 0.0 if t in IBR or t.startswith('hydro') or t == 'pumped' else None
        u['campd_linked'] = cu is not None
        u['src'] = src
        rows.append(u)
    return rows


for ba in BAS:
    rows = build(ba)
    json.dump(rows, open(f'{OUT}/fleet_{ba}.json', 'w'), indent=0, default=lambda o: None if o is pd.NA else str(o))
    flat = pd.DataFrame([{**{k: v for k, v in r.items() if k not in ('src', 'governor', 'ffr', 'heat_rate_curve', 'heat_rate_curve_nocampd')},
                          'heat_rate_curve': json.dumps(r['heat_rate_curve']), 'heat_rate_curve_nocampd': json.dumps(r['heat_rate_curve_nocampd']), 'governor': json.dumps(r['governor']),
                          'ffr': json.dumps(r['ffr']), 'src': json.dumps(r['src'])} for r in rows])
    flat.to_parquet(f'{OUT}/fleet_{ba}.parquet', index=False)
    live = flat[flat.retired_date.isna() & (flat.status_latest == 'existing')]
    print(ba, len(flat), 'units;', f'{live.pmax_MW.sum()/1e3:.1f} GW existing;',
          live.groupby('tech').pmax_MW.sum().div(1e3).round(1).to_dict(), flush=True)
json.dump(DEFAULTS, open(f'{OUT}/defaults.json', 'w'), indent=1, default=str)
