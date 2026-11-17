"""Build compact generator priors from agent B's processed stock and survey tables (read-only inputs):
- ResStock 2025 R1 baseline (whole stock, modelled): 120,000 rows drawn with probability proportional to weight
- ComStock 2025 R3 baseline (whole stock, modelled): all rows, selected columns
- CBECS 2018 / RECS 2020: only the 'prior' split of shared/building/data/proc/splits.json (real survey respondents)
Output: refdata/priors/*.parquet (+ README in tools docstring)."""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import json
import numpy as np
import pandas as pd

SRC = f'{BLDG_ROOT}/data/proc'
OUT = f'{BLDG_ROOT}/buildingsim/refdata/priors'

RES_COLS = ['bldg_id', 'weight', 'in.sqft..ft2', 'in.ashrae_iecc_climate_zone_2004', 'in.state', 'in.county',
            'in.weather_file_latitude', 'in.weather_file_longitude', 'in.weather_file_city',
            'in.geometry_building_type_recs', 'in.geometry_foundation_type', 'in.geometry_attic_type',
            'in.geometry_wall_type', 'in.insulation_wall', 'in.insulation_ceiling', 'in.insulation_roof',
            'in.insulation_slab', 'in.insulation_foundation_wall', 'in.insulation_floor', 'in.windows',
            'in.window_areas', 'in.infiltration', 'in.heating_setpoint', 'in.cooling_setpoint',
            'in.heating_setpoint_offset_magnitude', 'in.heating_setpoint_offset_period',
            'in.cooling_setpoint_offset_magnitude', 'in.cooling_setpoint_offset_period',
            'in.hvac_heating_type_and_fuel', 'in.hvac_heating_efficiency', 'in.hvac_cooling_type',
            'in.hvac_cooling_efficiency', 'in.hvac_has_ducts', 'in.hvac_has_shared_system', 'in.water_heater_efficiency',
            'in.water_heater_fuel', 'in.lighting', 'in.plug_loads', 'in.usage_level', 'in.orientation', 'in.vintage',
            'in.geometry_stories', 'in.occupants', 'in.bedrooms', 'in.pv_system_size', 'in.pv_orientation',
            'in.cooking_range', 'in.clothes_dryer', 'in.geometry_building_number_units_mf',
            'in.geometry_building_level_mf', 'in.geometry_building_horizontal_location_mf', 'in.vacancy_status',
            'in.heating_fuel', 'in.duct_leakage_and_insulation', 'in.duct_location', 'in.air_leakage_to_outside_ach50',
            'in.interior_shading', 'in.misc_pool_pump', 'in.misc_hot_tub_spa']
COM_COLS = ['bldg_id', 'weight', 'in.sqft..ft2', 'in.as_simulated_ashrae_iecc_climate_zone_2006', 'in.state',
            'in.as_simulated_nhgis_county_gisjoin', 'in.as_simulated_weather_file_2018', 'in.comstock_building_type',
            'in.building_subtype', 'in.number_stories', 'in.aspect_ratio', 'in.rotation..degrees', 'in.year_built',
            'in.vintage', 'in.airtightness..m3_per_m2_h', 'in.hvac_system_type', 'in.heating_fuel',
            'in.service_water_heating_fuel', 'in.wall_construction_type', 'in.window_type',
            'in.window_to_wall_ratio_category', 'in.energy_code_followed_during_original_building_construction',
            'in.energy_code_followed_during_last_hvac_replacement', 'in.energy_code_followed_during_last_roof_replacement',
            'in.energy_code_followed_during_last_walls_replacement', 'in.interior_lighting_generation',
            'in.weekday_opening_time..hr', 'in.weekday_operating_hours..hr', 'in.weekend_opening_time..hr',
            'in.weekend_operating_hours..hr', 'in.tstat_htg_sp_f..f', 'in.tstat_clg_sp_f..f', 'in.tstat_htg_delta_f..delta_f',
            'in.tstat_clg_delta_f..delta_f', 'in.tstat_has_htg_setback_all', 'in.tstat_has_clg_setback_all',
            'in.hvac_night_variability', 'in.economizer_damper_stuck_fault_applicable']


def main():
    splits = json.load(open(f'{SRC}/splits.json'))
    rs = pd.read_parquet(f'{SRC}/stock/resstock_meta.parquet', columns=RES_COLS)
    rng = np.random.default_rng(20261116)
    p = rs['weight'].values / rs['weight'].sum()
    idx = rng.choice(len(rs), 120000, replace=False, p=p)
    rs = rs.iloc[np.sort(idx)].reset_index(drop=True)
    for c in rs.columns:
        if rs[c].dtype == object:
            rs[c] = rs[c].astype('category')
    rs.to_parquet(f'{OUT}/resstock_prior.parquet', index=False)
    del rs
    cs = pd.read_parquet(f'{SRC}/stock/comstock_meta.parquet', columns=COM_COLS)
    for c in cs.columns:
        if cs[c].dtype == object:
            cs[c] = cs[c].astype('category')
    cs.to_parquet(f'{OUT}/comstock_prior.parquet', index=False)
    cb = pd.read_parquet(f'{SRC}/eia/cbecs2018.parquet')
    cb = cb[cb['id'].isin(set(splits['cbecs2018']['prior']))]
    cb.to_parquet(f'{OUT}/cbecs_prior.parquet', index=False)
    rc = pd.read_parquet(f'{SRC}/eia/recs2020.parquet')
    rc = rc[rc['id'].isin(set(splits['recs2020']['prior']))]
    rc.to_parquet(f'{OUT}/recs_prior.parquet', index=False)
    cz = pd.read_parquet(f'{SRC}/stock/county_climate_zones.parquet')
    cz.to_parquet(f'{OUT}/county_climate_zones.parquet', index=False)
    print('cbecs prior', len(cb), 'recs prior', len(rc))


if __name__ == '__main__':
    main()
