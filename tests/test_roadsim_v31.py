"""RoadSim physics v3.1 (second hardware pass on hybrids, 14 Nov 2026): engine-era blocks limited to hybrid
architectures, the measured load-table mode of engine_era (an option, not in the preset) and the per-type plug-in heating
option (an option, not in the preset)."""
import copy, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'checks'))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'roadsim'))


def test_era_block_only_for_hybrids_on_conventional_maps():
    from roadsim import eras
    from roadsim.options import PRESETS
    E = PRESETS['roadsim-physics-v3.1']['engine_era']
    m = 'gas_na_toyota_a25a'
    mild = {'name': '2005 HONDA INSIGHT', 'model_year': 2005, 'architecture': 'hev_parallel_p2'}
    assert abs(eras.scale(mild, m, E) - 0.875) < 1e-12
    assert abs(eras.scale(dict(mild, model_year=2014), m, E) - 0.942) < 1e-12
    assert eras.scale(dict(mild, model_year=2019), m, E) == 1.0
    assert eras.scale(dict(mild, architecture='conv'), m, E) == 1.0                 # conventional cars stay at v2
    toy = {'id': 'D3_2010_Toyota_Prius', 'architecture': 'hev_powersplit'}
    assert abs(eras.scale(toy, 'gas_atk_toyota_tnga_hev_paper', E) - 0.385 / 0.41) < 1e-12   # v3 rows unchanged
    assert eras.table(toy, 'gas_atk_toyota_tnga_hev_paper', E) is None              # no load table in the preset


def _prius():
    from roadsim import load_vehicle
    s = load_vehicle({'class': 'car', 'architecture': 'hev_powersplit'})
    s['engine']['bsfc_map'] = 'gas_atk_toyota_tnga_hev_paper'
    s['name'] = '2010 Toyota Prius'
    return s


def _table_opt():
    from roadsim.options import PRESETS
    d = json.load(open(os.path.join(HERE, '..', 'shared', 'vehicle', 'roadsim', 'roadsim', 'data', 'engine_era_d3.json')))
    e = copy.deepcopy(PRESETS['roadsim-physics-v3']['engine_era'])
    e['tables'] = {'2zr_d3': d['table_2zr_estec']}; e['ref_bte'] = 0.40
    e['eras'] = [r + ['2zr_d3'] for r in e['eras']]
    return {'preset': 'roadsim-physics-v3', 'engine_era': e}


def test_era_table_changes_fuel_and_cached_engine_is_restored():
    from roadsim import simulate, make_trip
    s = _prius(); trip = make_trip('udds', T_amb=23.9)
    kw = dict(init_temps=dict(engine=88.0, tire_warm=1.0), init_soc=0.6, record=False)
    f3a = simulate(s, trip, options='roadsim-physics-v3', **kw)['summary']['fuel_g']
    ft = simulate(s, trip, options=_table_opt(), **kw)['summary']['fuel_g']
    f3b = simulate(s, trip, options='roadsim-physics-v3', **kw)['summary']['fuel_g']
    f1 = simulate(s, trip, options='roadsim-physics-v1', **kw)['summary']['fuel_g']
    f1b = simulate(copy.deepcopy(s) | {'id': 'fresh'}, trip, options='roadsim-physics-v1', **kw)['summary']['fuel_g']
    assert abs(ft - f3a) > 0.01 * f3a
    assert f3a == f3b and f1 == f1b                      # the cached vehicle's own map is back for every later run


def test_phev_type_and_heat_option():
    from roadsim import eras
    from roadsim.options import resolve
    assert eras.phev_type({'name': '2013 Chevrolet Volt', 'architecture': 'phev_series'}) == 'volt_gen1'
    assert eras.phev_type({'id': 'D3_2016_Chevrolet_Volt', 'architecture': 'phev_series'}) == 'volt_gen2'
    assert eras.phev_type({'name': '2013 Ford Cmax Energi', 'architecture': 'phev_powersplit'}) == 'ford_energi'
    assert eras.phev_type({'id': 'VED_11', 'architecture': 'phev_powersplit', 'phev_type': 'ford_energi'}) == 'ford_energi'
    assert eras.phev_type({'name': '2020 Toyota RAV4', 'architecture': 'hev_powersplit'}) is None
    assert resolve('roadsim-physics-v3.1')['phev_heat_by_type'] == {}


def test_per_type_heat_rule_starts_the_engine():
    from roadsim import simulate, make_trip, load_vehicle
    s = load_vehicle({'class': 'car', 'architecture': 'phev_powersplit'})
    s['name'] = '2013 Ford C-Max Energi'
    trip = make_trip('udds', T_amb=-7.0)
    kw = dict(init_temps=dict(engine=-7.0, oil=-7.0, cabin=-7.0, battery=-7.0, tire_warm=0.0), init_soc=0.9, record=True)
    base = simulate(s, trip, options='roadsim-physics-v3', **kw)
    opt = {'preset': 'roadsim-physics-v3', 'phev_heat_by_type': {'ford_energi': {'engine_below_C': 0.0, 'cold_start_only': True}}}
    heat = simulate(s, trip, options=opt, **kw)
    assert heat['summary']['fuel_g'] > base['summary']['fuel_g'] + 20.0
