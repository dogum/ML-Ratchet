"""RoadSim physics v3 (hardware pass on hybrids, 14 Nov 2026): the new options, the generator options and the moved
helpers (roadsim.epa, roadsim.hybrid_hw) with their old import paths (checks/epa_procedures.py, checks/hybrid_specs.py)."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'checks'))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'roadsim'))


def test_engine_era_scale_by_make_and_year():
    from roadsim import eras
    from roadsim.options import PRESETS
    E = PRESETS['roadsim-physics-v3']['engine_era']
    m = 'gas_atk_toyota_tnga_hev_paper'
    assert abs(eras.scale({'id': 'D3_2010_Toyota_Prius'}, m, E) - 0.385 / 0.41) < 1e-12
    assert abs(eras.scale({'model_year': 2006, 'name': '2006 TOYOTA HIGHLANDER HYBRID'}, m, E) - 0.37 / 0.41) < 1e-12
    assert eras.scale({'id': 'D3_2015_Honda_Accord_Hybrid'}, m, E) == 1.0          # only Toyota / Lexus are listed
    assert eras.scale({'name': '2024 TOYOTA RAV4 HYBRID'}, m, E) == 1.0            # TNGA era = the map itself
    assert eras.scale({'id': 'VED_241'}, m, E) == 1.0                              # no model year: modern
    assert eras.scale({'id': 'D3_2010_Toyota_Prius'}, 'gas_na_toyota_a25a', E) == 1.0   # conventional maps untouched
    assert PRESETS['roadsim-physics-v2.1'].get('engine_era') in ({}, None)


def test_engine_era_burns_more_fuel_only_when_on():
    from roadsim import simulate, make_trip, load_vehicle
    s = load_vehicle({'class': 'car', 'architecture': 'hev_powersplit'})
    s['engine']['bsfc_map'] = 'gas_atk_toyota_tnga_hev_paper'
    s['name'] = '2005 Toyota Prius'
    trip = make_trip('udds', T_amb=23.9)
    kw = dict(init_temps=dict(engine=88.0, tire_warm=1.0), init_soc=0.6, record=False)
    f21 = simulate(s, trip, options='roadsim-physics-v2.1', **kw)['summary']['fuel_g']
    f3 = simulate(s, trip, options='roadsim-physics-v3', **kw)['summary']['fuel_g']
    s['name'] = '2022 Toyota Prius'
    f3m = simulate(s, trip, options={'preset': 'roadsim-physics-v3', 'p2_tc_open_gears': 0}, **kw)['summary']['fuel_g']
    assert f3 > 1.03 * f21
    assert abs(f3m - f21) < 1e-9 * f21


def test_generator_options():
    from roadsim.generator import generate_vehicles
    a = generate_vehicles(40, seed=5)
    b = generate_vehicles(40, seed=5, options='roadsim-physics-v1')
    assert a == b                                           # options without generator keys: the v1 generator
    c = generate_vehicles(60, seed=5, options='roadsim-physics-v3')
    plug = [v for v in c if v['architecture'] in ('phev_powersplit', 'phev_series')]
    assert plug
    for v in plug:
        trac = max(m['max_power_kW'] for m in v['motors'] if m['role'] in ('MG2', 'traction', 'P2'))
        assert abs(v['battery']['max_power_kW'] - round(min(trac / 0.9, 9 * v['battery']['capacity_kWh']), 1)) < 1e-9
    for v in c:
        if v['architecture'] not in ('conv', 'bev'):
            assert v['regen']['blend'] == 'series'


def test_old_import_paths_and_procedures():
    import _opts
    import epa_procedures as EP
    import hybrid_specs as HS
    from roadsim import epa, hybrid_hw
    assert HS.lookup is hybrid_hw.lookup and HS.spec_of is hybrid_hw.spec_of
    assert HS.lookup('TOYOTA', 'PRIUS', 2010)[0] == 'ps'
    assert HS.lookup('HYUNDAI', 'IONIQ HYBRID', 2020)[1].get('tmed')
    from roadsim import load_vehicle
    s = EP.test_spec(load_vehicle({'class': 'car', 'architecture': 'conv'}))
    _opts.set('roadsim-physics-v2')
    a = EP.hot_cycle(s, 'hwfet')['FE']
    b = epa.hot_cycle(s, 'hwfet', options='roadsim-physics-v2')['FE']
    _opts.set(None)
    assert a == b
