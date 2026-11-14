"""Build refdata/hr/unit_heat_rates.parquet: EIA-923 annual operating heat rate per generator and per CC unit.

Source: PUDL out_eia923__yearly_generation_fuel_by_generator (Catalyst Cooperative, CC-BY-4.0; EIA-923 public domain).
heat rate = fuel consumed for electricity [MMBtu] / net generation [MWh]; combined-cycle parts (CT, CA, CS) are summed
per PUDL unit so the steam part's generation is credited to the fuel burnt in the gas turbines.
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import pandas as pd, numpy as np
src = f'{GRID_ROOT}/gridsim/refdata/hr/out_eia923__yearly_generation_fuel_by_generator.parquet'
g = pd.read_parquet(src, columns=['report_date', 'plant_id_eia', 'generator_id', 'unit_id_pudl',
                                  'fuel_consumed_for_electricity_mmbtu', 'net_generation_mwh'])
g['year'] = pd.to_datetime(g.report_date).dt.year
g = g[g.year >= 2015]
g = g.rename(columns={'plant_id_eia': 'plant_id', 'fuel_consumed_for_electricity_mmbtu': 'fuel_mmbtu',
                      'net_generation_mwh': 'netgen_mwh'})
g[['plant_id', 'generator_id', 'unit_id_pudl', 'year', 'fuel_mmbtu', 'netgen_mwh']].to_parquet(
    f'{GRID_ROOT}/gridsim/refdata/hr/unit_heat_rates.parquet', index=False)
print(len(g), g.year.min(), g.year.max())
