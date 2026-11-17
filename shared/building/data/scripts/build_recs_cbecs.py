"""EIA RECS 2020 (v7) and CBECS 2018 public microdata -> proc/eia/.

Writes, for each survey:
  <survey>_full.parquet : every public variable as published (codes, incl. replicate weights) - for anything not mapped
  <survey>.parquet      : one row per respondent, mapped to BuildingSim spec terms (SI units) + annual energy by fuel
Units: floor area m2 (1 ft2 = 0.09290304 m2); energy kWh (1 thousand Btu = 0.29307107 kWh); temperatures degC.
Annual totals by fuel are EIA's billing-based estimates (Energy Supplier Survey; PERIOD* says how many billing days);
end-use splits (space heating / cooling / water heating ...) are EIA MODEL estimates, not measurements.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import os, numpy as np, pandas as pd

D = f'{BLDG_ROOT}/data'
O = f'{D}/proc/eia'; os.makedirs(O, exist_ok=True)
FT2 = 0.09290304; KBTU = 0.29307107
F2C = lambda f: (f - 32) / 1.8


def yr_range(code, table):
    lo, hi = table.get(code, (np.nan, np.nan)); return lo, hi


def recs():
    r = pd.read_csv(f'{D}/raw/recs/recs2020_public_v7.csv', low_memory=False)
    r.to_parquet(f'{O}/recs2020_full.parquet', index=False, compression='zstd')
    na = lambda s: s.where(s >= 0)       # -2 / -4 etc. = not applicable / not asked
    out = pd.DataFrame({'id': 'RECS_' + r.DOEID.astype(str), 'doeid': r.DOEID, 'survey': 'RECS2020', 'data_year': 2020,
                        'state': r.state_postal, 'division': r.DIVISION, 'region': r.REGIONC,
                        'climate_zone': r.IECC_climate_code, 'ba_climate': r.BA_climate, 'urban_type': r.UATYP10,
                        'hdd65F_year': r.HDD65, 'cdd65F_year': r.CDD65, 'hdd65F_30yr': r.HDD30YR_PUB, 'cdd65F_30yr': r.CDD30YR_PUB})
    out['hdd18C_year'] = r.HDD65 / 1.8; out['cdd18C_year'] = r.CDD65 / 1.8
    out['housing_type'] = r.TYPEHUQ.map({1: 'mobile_home', 2: 'single_detached', 3: 'single_attached', 4: 'apartment_2_4', 5: 'apartment_5plus'})
    out['primary_use'] = r.TYPEHUQ.map({1: 'residential_single', 2: 'residential_single', 3: 'residential_single', 4: 'residential_multi', 5: 'residential_multi'})
    out['floor_area_m2'] = r.TOTSQFT_EN * FT2
    out['heated_area_m2'] = r.TOTHSQFT * FT2; out['cooled_area_m2'] = r.TOTCSQFT * FT2
    out['n_floors'] = na(r.STORIES).map({1: 1, 2: 2, 3: 3, 4: 4, 5: 2})
    yr = {1: (1900, 1949), 2: (1950, 1959), 3: (1960, 1969), 4: (1970, 1979), 5: (1980, 1989), 6: (1990, 1999), 7: (2000, 2009), 8: (2010, 2015), 9: (2016, 2020)}
    out['year_built_lo'] = r.YEARMADERANGE.map(lambda c: yr.get(c, (np.nan,))[0]); out['year_built_hi'] = r.YEARMADERANGE.map(lambda c: yr.get(c, (np.nan, np.nan))[1])
    out['year_built_code'] = r.YEARMADERANGE
    out['foundation'] = np.select([r.CELLAR == 1, r.CRAWL == 1, r.CONCRETE == 1], ['basement', 'crawlspace', 'slab'], None)
    out['attic'] = na(r.ATTIC)
    out['wall_material'] = r.WALLTYPE.map({1: 'brick', 2: 'wood', 3: 'siding', 4: 'stucco', 5: 'shingle', 6: 'stone', 7: 'concrete_block', 99: 'other'})
    out['glazing'] = r.TYPEGLASS.map({1: 'single', 2: 'double', 3: 'triple'})
    out['window_frame'] = r.WINFRAME.map({1: 'wood', 2: 'metal', 3: 'vinyl', 4: 'composite', 5: 'fiberglass'})
    out['n_windows_code'] = r.WINDOWS
    out['insulation_reported'] = r.ADQINSUL.map({1: 'well', 2: 'adequate', 3: 'poor', 4: 'none'})
    out['drafty'] = r.DRAFTY.map({1: 'always', 2: 'mostly', 3: 'sometimes', 4: 'never'})
    # HVAC -> contract system names (main heating equipment; cooling separately)
    fuel = r.FUELHEAT.map({5: 'electricity', 1: 'natural_gas', 2: 'propane', 3: 'fuel_oil', 7: 'wood', 99: 'other'})
    out['heating_fuel'] = fuel.where(r.HEATHOME == 1, 'none')
    eq = r.EQUIPM
    sysmap = np.select(
        [r.HEATHOME != 1, (eq == 4) & (r.GEOHP == 1), eq.isin([4, 13]), eq == 5, eq == 10, (eq == 3) & (r.FUELHEAT == 5),
         eq == 3, eq == 2, eq == 7, eq == 8],
        ['none', 'heat_pump_ground', 'heat_pump_air', 'baseboard_resistance', 'portable_electric', 'electric_furnace',
         'furnace', 'boiler_radiator', 'room_heater_fuel', 'wood_stove'], 'other')
    out['heating_equipment'] = sysmap
    out['cooling_equipment'] = np.where(r.AIRCOND != 1, 'none',
                                        r.ACEQUIPM_PUB.map({1: 'central_ac_or_hp', 3: 'mini_split_hp', 4: 'window_wall', 5: 'portable', 6: 'evaporative'}).fillna('other'))
    # contract HVAC system (best effort)
    out['system'] = np.select(
        [out.heating_equipment == 'heat_pump_ground', out.heating_equipment == 'heat_pump_air',
         out.heating_equipment.isin(['furnace', 'electric_furnace']) & (out.cooling_equipment == 'central_ac_or_hp'),
         out.heating_equipment.isin(['baseboard_resistance', 'portable_electric']), out.heating_equipment == 'boiler_radiator',
         (out.heating_equipment == 'none') & (out.cooling_equipment == 'none')],
        ['heat_pump_ground', 'heat_pump_air', 'split_ac_furnace', 'baseboard_resistance', 'radiant', 'none'], None)
    out['heating_equipment_age_code'] = na(r.EQUIPAGE); out['cooling_equipment_age_code'] = na(r.ACEQUIPAGE)
    out['secondary_heat'] = na(r.EQUIPAUX)
    out['water_heater_fuel'] = r.FUELH2O.map({5: 'electricity', 1: 'natural_gas', 2: 'propane', 3: 'fuel_oil', 7: 'wood', 8: 'solar', 99: 'other'})
    out['thermostat_control_heat'] = na(r.HEATCNTL); out['thermostat_control_cool'] = na(r.COOLCNTL)
    for a, b in [('T_heat_home_C', 'TEMPHOME'), ('T_heat_away_C', 'TEMPGONE'), ('T_heat_night_C', 'TEMPNITE'),
                 ('T_cool_home_C', 'TEMPHOMEAC'), ('T_cool_away_C', 'TEMPGONEAC'), ('T_cool_night_C', 'TEMPNITEAC')]:
        out[a] = F2C(na(r[b]))
    out['n_occupants'] = r.NHSLDMEM; out['pv'] = na(r.SOLAR)
    # energy (kWh/yr)
    out['elec_kWh'] = r.KWH; out['gas_kWh'] = r.BTUNG * KBTU; out['propane_kWh'] = r.BTULP * KBTU
    out['fueloil_kWh'] = r.BTUFO * KBTU; out['wood_kWh'] = r.BTUWD * KBTU
    out['site_kWh'] = r.TOTALBTU * KBTU if 'TOTALBTU' in r else np.nan
    out['elec_heat_kWh_model'] = r.KWHSPH; out['elec_cool_kWh_model'] = r.KWHCOL
    out['gas_heat_kWh_model'] = r.BTUNGSPH * KBTU
    out['heat_kWh_model'] = r.TOTALBTUSPH * KBTU; out['dhw_kWh_model'] = r.TOTALBTUWTH * KBTU
    out['cool_kWh_model'] = r.KWHCOL
    out['elec_billing_period_code'] = r.PERIODEL; out['gas_billing_period_code'] = r.PERIODNG
    out['elec_from_bills'] = r.PERIODEL.isin([1, 2])
    out['weight'] = r.NWEIGHT
    out.to_parquet(f'{O}/recs2020.parquet', index=False)
    print('RECS', out.shape, 'weighted homes (M)', round(out.weight.sum() / 1e6, 1))
    print(out.system.value_counts(dropna=False).to_string())


def cbecs():
    c = pd.read_csv(f'{D}/raw/cbecs/cbecs2018_final_public.csv', low_memory=False)
    c.to_parquet(f'{O}/cbecs2018_full.parquet', index=False, compression='zstd')
    pba = {1: 'vacant', 2: 'office', 4: 'laboratory', 5: 'warehouse_nonrefrigerated', 6: 'food_sales', 7: 'public_order_safety',
           8: 'outpatient_health', 11: 'warehouse_refrigerated', 12: 'religious_worship', 13: 'public_assembly', 14: 'education',
           15: 'food_service', 16: 'inpatient_health', 17: 'nursing', 18: 'lodging', 23: 'strip_mall', 24: 'enclosed_mall',
           25: 'retail_other', 26: 'service', 91: 'other'}
    use = {1: 'other', 2: 'office', 4: 'lab', 5: 'warehouse', 6: 'food', 7: 'other', 8: 'healthcare', 11: 'warehouse',
           12: 'assembly', 13: 'assembly', 14: 'education', 15: 'food', 16: 'healthcare', 17: 'healthcare', 18: 'lodging',
           23: 'retail', 24: 'retail', 25: 'retail', 26: 'other', 91: 'other'}
    out = pd.DataFrame({'id': 'CBECS_' + c.PUBID.astype(str), 'pubid': c.PUBID, 'survey': 'CBECS2018', 'data_year': 2018,
                        'region': c.REGION, 'division': c.CENDIV,
                        'climate_class': c.PUBCLIM.map({1: 'cold_or_very_cold', 2: 'cool', 3: 'mixed_mild', 4: 'warm', 5: 'hot_or_very_hot', 7: 'withheld'}),
                        'hdd65F_year': c.HDD65, 'cdd65F_year': c.CDD65})
    out['hdd18C_year'] = c.HDD65 / 1.8; out['cdd18C_year'] = c.CDD65 / 1.8
    out['pba'] = c.PBA.map(pba); out['pbaplus'] = c.PBAPLUS; out['primary_use'] = c.PBA.map(use)
    out['floor_area_m2'] = c.SQFT * FT2
    out['n_floors'] = c.NFLOOR.replace({994: 12, 995: 15}); out['n_floors_topcoded'] = c.NFLOOR.isin([994, 995])
    yr = {2: (1800, 1945), 3: (1946, 1959), 4: (1960, 1969), 5: (1970, 1979), 6: (1980, 1989), 7: (1990, 1999), 8: (2000, 2012), 9: (2013, 2018)}
    out['year_built_lo'] = c.YRCONC.map(lambda k: yr.get(k, (np.nan, np.nan))[0]); out['year_built_hi'] = c.YRCONC.map(lambda k: yr.get(k, (np.nan, np.nan))[1])
    out['year_built_code'] = c.YRCONC
    out['wall_material'] = c.WLCNS.map({1: 'masonry', 2: 'precast_concrete', 3: 'concrete', 4: 'siding_light', 5: 'sheet_metal', 6: 'vision_glass', 7: 'spandrel_glass', 8: 'other'})
    out['roof_material'] = c.RFCNS.map({1: 'built_up', 2: 'slate_tile', 3: 'wood', 4: 'shingles', 5: 'metal', 6: 'membrane', 7: 'concrete', 8: 'other'})
    out['glass_pct_code'] = c.GLSSPC
    out['window_wwr_mid'] = c.GLSSPC.map({1: 0.01, 2: 0.06, 3: 0.18, 4: 0.38, 5: 0.63, 6: 0.88})
    out['window_type'] = c.WINTYP.map({1: 'single', 2: 'multi', 3: 'mixed', 4: 'none'})
    out['cool_roof'] = c.RFCOOL.map({1: 1, 2: 0})
    mh = c.MAINHT; mc = c.MAINCL
    out['main_heating_code'] = mh; out['main_cooling_code'] = mc
    out['heating_fuel'] = np.select([mh.between(1, 8), mh.between(9, 15), mh.between(16, 19), mh.between(20, 26), mh.isin([27, 28]), mh.notna()],
                                    ['electricity', 'natural_gas', 'fuel_oil', 'propane', 'district', 'other'], 'none')
    out['system'] = np.select(
        [mh.isin([27, 28]) & (mc == 16), mh.isin([4, 12, 23]), mh.isin([2, 10, 21, 38]) | (mc == 2), mc.isin([3, 8, 10, 12, 14]) & mh.isin([3, 11, 17, 22, 27, 28, 34, 39]),
         mh.isin([1, 9, 16, 20]) & (mc == 1), mc == 5, mh.isin([5]) & mc.isna(), mh.isna() & mc.isna()],
        ['district_steam_chw', 'heat_pump_air', 'packaged_rtu', 'vav_chiller_boiler', 'split_ac_furnace', 'ptac', 'baseboard_resistance', 'none'], None)
    for v in ['BOILER', 'CHILLR', 'PKGCL', 'PKGHT', 'RCAC', 'HTPMPH', 'HTPMPC', 'FURNAC', 'ECN', 'EMCS', 'OCSN', 'ELEVTR', 'OPEN24']:
        if v in c: out[v.lower()] = c[v].map({1: 1, 2: 0})
    out['heated_pct'] = c.HEATP; out['cooled_pct'] = c.COOLP
    out['hours_per_week'] = c.WKHRS; out['n_workers'] = c.NWKER
    out['elec_kWh'] = c.ELCNS; out['gas_kWh'] = c.NGBTU * KBTU; out['fueloil_kWh'] = c.FKBTU * KBTU
    out['district_heat_kWh'] = c.DHBTU * KBTU; out['site_kWh'] = c.MFBTU * KBTU
    for a, b in [('elec_heat_kWh_model', 'ELHTBTU'), ('gas_heat_kWh_model', 'NGHTBTU'), ('elec_cool_kWh_model', 'ELCLBTU'),
                 ('heat_kWh_model', 'MFHTBTU'), ('elec_vent_kWh_model', 'ELVNBTU'), ('elec_light_kWh_model', 'ELLTBTU'), ('elec_dhw_kWh_model', 'ELWTBTU')]:
        if b in c: out[a] = c[b] * KBTU
    out['weight'] = c.FINALWT
    out.to_parquet(f'{O}/cbecs2018.parquet', index=False)
    print('CBECS', out.shape, 'weighted buildings (k)', round(out.weight.sum() / 1e3, 1))
    print(out.system.value_counts(dropna=False).to_string()); print(out.primary_use.value_counts().to_string())


if __name__ == '__main__':
    recs(); cbecs()
