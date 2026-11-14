"""Argonne D3 chassis-dynamometer tests in the shared trip contract (numpy/pandas only).

    import sys; sys.path.insert(0, 'path/to/ML-ratchet/shared/vehicle')
    from realdata.d3 import d3_index, load_d3, d3_split, d3_specs
    idx = d3_index(split='dev', category='hybrid')     # one row per test: vehicle, cycle, ambient, start, totals, signals
    meta, arr = load_d3('D3_62201031')                 # 1 Hz contract arrays (+ *_10hz originals with tenhz=True)
    sp = d3_split()                                    # frozen vehicle split (data/proc/d3/splits.json), sha256 checked
    spec = d3_specs()['D3_2021_Hyundai_Sonata_Hybrid'] # RoadSim twin spec, with spec['provenance'] per field

Built by data/scripts/d3_index.py, d3_specs.py, d3_splits.py; documented in data/D3.md. Rule: calibration may use only
'dev' vehicles; 'exam' vehicles are for the final report.

Arrays (float32): t, v [m/s, dyno roll speed], grade (0 unless the dyno simulated one), elev, lat, lon (NaN), heading,
T_amb [C, test cell], wind_speed, wind_dir (0), rho_air (reference 1.1767: the dyno applies the target road load),
surface, solar [W/m2, 850 when the lamps were on], fuel_rate [g/s], engine_rpm, batt_current [A, + = discharge],
batt_voltage [V], soc [0-1], aux_power [W, DC/DC or alternator output where logged]; extras when logged: batt_power,
fuel_rate_modal, v_sched, rho_cell, T_coolant, T_oil, T_cabin, T_batt, T_trans_oil, P_heater, P_ac, bag, F_tractive.
Missing signals are NaN arrays. Attribution required: see ATTRIBUTION.
"""
import os, json, hashlib, functools
import numpy as np
import pandas as pd

_VEH = os.environ.get('ML_RATCHET_VEHICLE') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
D3 = os.path.join(_VEH, 'data', 'proc', 'd3')
ATTRIBUTION = ('This data is from the Downloadable Dynamometer Database and was generated at the Advanced Mobility '
               'Technology Laboratory (AMTL) at Argonne National Laboratory under the funding and guidance of the U.S. '
               'Department of Energy (DOE).')


@functools.lru_cache(maxsize=1)
def d3_split():
    """The frozen D3 vehicle split. Raises if the file's vehicle table no longer matches its recorded sha256."""
    s = json.load(open(os.path.join(D3, 'splits.json')))
    h = hashlib.sha256(json.dumps(s['vehicles'], sort_keys=True).encode()).hexdigest()
    if h != s['sha256']:
        raise ValueError(f'data/proc/d3/splits.json was edited: sha256 {h} != recorded {s["sha256"]}')
    return s


@functools.lru_cache(maxsize=1)
def _index():
    d = pd.read_parquet(os.path.join(D3, 'index.parquet'))
    sp = d3_split()['vehicles']
    d['split'] = d.vehicle_id.map(lambda v: sp.get(v, {}).get('split'))
    return d.set_index('test_id', drop=False)


def d3_index(split=None, category=None, vehicles=None, cycle=None, ok=True):
    """One row per D3 test. split: 'dev' | 'exam'; category: conventional | hybrid | plugin | electric | alternative;
    vehicles: names or ids; cycle: e.g. 'UDDS' (exact match on the recognised schedule)."""
    d = _index().copy()
    if ok is not None: d = d[d.ok == ok]
    if split: d = d[d.split == split]
    if category: d = d[d.category.isin([category] if isinstance(category, str) else list(category))]
    if vehicles is not None:
        v = [vehicles] if isinstance(vehicles, str) else list(vehicles)
        d = d[d.vehicle.isin(v) | d.vehicle_id.isin(v)]
    if cycle: d = d[d.cycle == cycle]
    return d


def load_d3(test_id, tenhz=False):
    """(meta, arrays) for one test. meta holds the index row ('row'), the raw channel behind each signal
    ('channels', 'units'), recognised schedule phases ('phases', 1 Hz sample indices), the summary-table entry and
    the TDMS test info. tenhz=True adds the 10 Hz originals (t_10hz, v_10hz, fuel_rate_10hz, ...)."""
    tid = test_id if str(test_id).startswith('D3_') else f'D3_{test_id}'
    r = _index().loc[tid]
    p = os.path.join(D3, r['path'])
    meta = json.load(open(p[:-4] + '.json'))
    meta['split'] = r['split']
    with np.load(p) as z:
        arr = {k: z[k] for k in z.files if tenhz or not k.endswith('_10hz')}
    return meta, arr


@functools.lru_cache(maxsize=1)
def _specs():
    return json.load(open(os.path.join(D3, 'specs.json')))


def d3_specs():
    """{vehicle_id: RoadSim twin spec}. Each spec carries 'provenance' (reported / derived / default per field),
    'epa_match' and 'd3_settings'; strip those keys before passing a spec to roadsim.simulate."""
    return json.loads(json.dumps(_specs()))
