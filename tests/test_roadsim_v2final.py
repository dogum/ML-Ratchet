"""RoadSim physics v2, calibration agent 2 (14 Nov 2026): the small options added for the hybrid / plug-in / HVAC / bus
calibration, and the default-physics switch. Every option is off in physics v1 (the fingerprint test covers that)."""
import copy, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'roadsim'))

import roadsim
from roadsim import simulate, load_vehicle, make_trip
from roadsim.options import DEFAULTS, PRESETS, DEFAULT_PHYSICS

W = dict(engine=88.0, driveline=80.0, tire_warm=1.0, cabin=21.0)


def _hev(arch):
    return load_vehicle({'class': 'car', 'architecture': arch})


def test_default_physics_is_v2_and_v1_is_reproducible_by_name():
    assert DEFAULT_PHYSICS == roadsim.PHYSICS_VERSION == 'roadsim-physics-v2'
    s = _hev('hev_powersplit'); t = make_trip('udds', T_amb=20.0)
    a = simulate(s, t, init_temps=W)['summary']['fuel_g']
    b = simulate(s, t, init_temps=W, options='roadsim-physics-v2')['summary']['fuel_g']
    c = simulate(s, t, init_temps=W, options='roadsim-physics-v1')['summary']['fuel_g']
    d = simulate(s, t, init_temps=W, options=None)['summary']['fuel_g']
    assert a == b and c == d and abs(a - c) > 1e-6


def test_new_options_default_off():
    for k, v in (('regen_share', {}), ('hybrid_chg_cap_frac', None), ('series_dd_prefer', False), ('phev_engine_heat', False),
                 ('phev_heat_cold_start_only', False), ('eng_fullload_enrich', 0.0)):
        assert DEFAULTS[k] == v
    assert PRESETS['roadsim-physics-v2']['regen_share'] == {'hev_parallel_p2': 0.75}


def test_regen_share_lowers_regen_and_ledger_closes():
    s = _hev('hev_parallel_p2'); t = make_trip('udds', T_amb=20.0)
    a = simulate(s, t, init_temps=W, options='roadsim-physics-v1')
    b = simulate(s, t, init_temps=W, options={'regen_share': {'hev_parallel_p2': 0.75}})
    assert b['ledger']['regen'] < 0.9 * a['ledger']['regen']
    assert b['summary']['fuel_g'] > a['summary']['fuel_g']
    assert b['ledger']['closure_rel'] < 1e-3


def test_series_dd_prefer_keeps_the_engine_locked_at_cruise():
    s = _hev('hev_series'); t = make_trip('hwfet', T_amb=20.0)
    o1 = {'series_direct_drive': True}
    a = simulate(s, t, init_temps=W, options=o1)
    b = simulate(s, t, init_temps=W, options=dict(o1, series_dd_prefer=True))
    assert np.isfinite(b['summary']['fuel_g']) and b['ledger']['closure_rel'] < 1e-3
    assert np.mean(b['eng_on']) >= np.mean(a['eng_on']) - 0.02


def test_charge_cap_limits_engine_charging():
    s = _hev('hev_parallel_p2'); t = make_trip('udds', T_amb=20.0)
    a = simulate(s, t, init_temps=W, options='roadsim-physics-v1')
    b = simulate(s, t, init_temps=W, options={'hybrid_chg_cap_frac': 0.05})
    pa = np.asarray(a['batt_current']) * np.asarray(a['batt_voltage'])
    pb = np.asarray(b['batt_current']) * np.asarray(b['batt_voltage'])
    ona, onb = np.asarray(a['eng_on']) > 0.5, np.asarray(b['eng_on']) > 0.5
    assert np.percentile(pb[onb], 5) >= np.percentile(pa[ona], 5)
    assert b['ledger']['closure_rel'] < 1e-3


def test_engine_on_moving_mild_hybrid():
    s = _hev('hev_parallel_p2'); s = copy.deepcopy(s)
    s['control'] = {'strategy': 'aecms', 'params': {'engine_on_moving': True}}
    o = simulate(s, make_trip('udds', T_amb=20.0), init_temps=W, options='roadsim-physics-v1')
    P = np.asarray(o['P_wheel']); v = np.asarray(o['v'])
    trac = (P > 2000.0) & (v > 1.0)
    assert np.mean(np.asarray(o['eng_on'])[trac] > 0.5) > 0.97


def test_phev_engine_heat_uses_the_warm_engine():
    s = load_vehicle({'class': 'car', 'architecture': 'phev_series'})
    t = make_trip('udds', T_amb=-7.0)
    base = {'preset': 'roadsim-physics-v2'}
    it = dict(engine=80.0, cabin=10.0, battery=10.0, tire_warm=1.0)
    a = simulate(s, t, init_temps=it, init_soc=0.2, options=dict(base, phev_engine_heat=False))
    b = simulate(s, t, init_temps=it, init_soc=0.2, options=dict(base, phev_engine_heat=True))
    assert np.sum(b['hvac_power']) < np.sum(a['hvac_power'])        # less electric heat in charge sustaining
    assert b['ledger']['closure_rel'] < 1e-3


def test_fullload_enrichment_costs_fuel_only_at_high_load():
    s = load_vehicle({'class': 'car', 'architecture': 'conv'})
    a = simulate(s, make_trip('us06', T_amb=20.0), init_temps=W, options={'eng_eta_scale': 1.0001})
    b = simulate(s, make_trip('us06', T_amb=20.0), init_temps=W, options={'eng_eta_scale': 1.0001, 'eng_fullload_enrich': 0.2,
                                                                          'eng_fullload_frac': 0.6})
    assert b['summary']['fuel_g'] > a['summary']['fuel_g']
    assert b['ledger']['closure_rel'] < 1e-3
