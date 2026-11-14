"""RoadSim physics v2, work package B2 (thermal, HVAC, accessories, engine efficiency, motor light load, battery aging).

Each option defaults to v1; switched on, it must move results the expected way by a sane amount, and the energy
ledger must still close."""
import copy, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'roadsim'))

import numpy as np
import pytest
from roadsim import simulate, make_trip, load_vehicle
from roadsim import thermal, aging
from roadsim.options import DEFAULTS, PRESETS, resolve

B2 = PRESETS['roadsim-physics-v2-b2']
TOL = 5e-4


def run(spec, cyc='udds', opt=None, it=None, **kw):
    o = simulate(spec, make_trip(cyc, **kw), options=opt, init_temps=it)
    assert o['ledger']['closure_rel'] < TOL, o['ledger']['closure_rel']
    return o


def energy_kWh(o):
    s = o['summary']
    return s['fuel_g'] * 42.6e3 / 3.6e6 + s['batt_kWh_out']


@pytest.fixture(scope='module')
def car():
    return load_vehicle({'class': 'car', 'architecture': 'conv'})


def test_defaults_are_v1(car):
    assert not thermal.active(resolve(None))
    assert all(k in DEFAULTS for k in B2)
    a = simulate(car, make_trip('udds', T_amb=-7.0), options='roadsim-physics-v1')
    b = simulate(car, make_trip('udds', T_amb=-7.0), options={k: DEFAULTS[k] for k in B2})
    assert a['summary']['fuel_g'] == b['summary']['fuel_g']
    assert np.array_equal(a['T_cabin'], b['T_cabin'])


def test_cold_engine_oil_node_costs_fuel_in_the_cold(car):
    # the cold set (oil node + Vogel oil + enrichment + axle) keeps the 24 C cold start of v1 (oil_fric_scale) and costs
    # more at -7 C
    cold = {'engine_thermal': 'v2', 'oil_viscosity': 'vogel', 'cold_enrich': 0.008, 'warmup_enrich': 0.004, 'axle_cold': True}
    off = copy.deepcopy(car); off['aux']['HVAC']['enabled'] = False
    d = {}
    for T in (23.9, -7.0):
        v1 = run(off, T_amb=T); v2 = run(off, opt=cold, T_amb=T)
        d[T] = v2['summary']['fuel_g'] / v1['summary']['fuel_g'] - 1
    assert abs(d[23.9]) < 0.03, d
    assert d[-7.0] > d[23.9] + 0.01, d
    assert 'oil' in v2['final_state'] and v2['final_state']['oil'] < v2['final_state']['engine'] + 15
    # a hot start is unaffected by the warm-up mechanisms
    hot = dict(engine=88.0, oil=95.0, driveline=80.0, tire_warm=1.0)
    h1 = run(car, it=hot, T_amb=20.0); h2 = run(car, it=hot, opt={'engine_thermal': 'v2', 'cold_enrich': 0.008}, T_amb=20.0)
    assert abs(h2['summary']['fuel_g'] / h1['summary']['fuel_g'] - 1) < 0.003


def test_cold_axle_and_tyres_raise_rolling_losses(car):
    soak = dict(engine=-7.0, cabin=-7.0, tire_warm=0.0)
    warm = dict(engine=88.0, driveline=80.0, axle=80.0, tire_warm=1.0)
    for opt in ({'axle_cold': True}, {'tire_thermal': True}):
        c = run(car, opt=opt, it=soak, T_amb=-7.0)['ledger']['rolling']
        w = run(car, opt=opt, it=warm, T_amb=-7.0)['ledger']['rolling']
        assert c > w * 1.01, (opt, c, w)
    # chassis dyno: only tire_dyno_frac of Crr follows tyre temperature
    dyno = copy.deepcopy(car); dyno['tire'] = {'K_t': 0.0, 'temp_effects': False}
    r0 = run(dyno, it=soak, T_amb=-7.0)['ledger']['rolling']
    r1 = run(dyno, opt={'tire_thermal': True}, it=soak, T_amb=-7.0)['ledger']['rolling']
    assert 1.02 < r1 / r0 < 1.4, r1 / r0


def test_belt_ac_pulldown_costs_more_than_v1(car):
    hot = dict(engine=88.0, driveline=80.0, tire_warm=1.0, cabin=50.0)
    v1 = run(car, 'sc03', it=hot, T_amb=35.0, solar=850.0)
    v2 = run(car, 'sc03', it=hot, opt={'hvac_model': 'v2'}, T_amb=35.0, solar=850.0)
    off = copy.deepcopy(car); off['aux']['HVAC']['enabled'] = False
    o = run(off, 'sc03', it=hot, T_amb=35.0, solar=850.0)
    f0, f1, f2 = o['summary']['fuel_g'], v1['summary']['fuel_g'], v2['summary']['fuel_g']
    assert f2 > f1 > f0
    assert 0.10 < f2 / f0 - 1 < 0.6, f2 / f0
    hv = v2['hvac_power'][1:]
    assert 1000 < hv[:120].mean() < 6000          # pull-down: belt compressor + condenser fan
    assert v2['T_cabin'][-1] < 30.0


def test_ptc_heater_capacity_and_seat_heat():
    leaf = load_vehicle({'class': 'car', 'architecture': 'bev', 'aux': {'HVAC': {'source': 'ptc', 'ptc_max_W': 5000}},
                         'battery': {'capacity_kWh': 24, 'nominal_V': 360, 'thermal': {'heater_W': 0}}})
    o = run(leaf, opt={'hvac_model': 'v2'}, it=dict(cabin=-7.0, battery=-7.0), T_amb=-7.0)
    hv = o['hvac_power'][1:]
    assert hv.max() < 5000 + 2 * 150 + 200        # PTC + blower (+ small 12 V loads)
    assert 1500 < hv[-600:].mean() < 5000 + 2 * 150 + 200
    s = run(leaf, opt={'hvac_model': 'v2', 'hvac_seat_heat': True}, T_amb=5.0, it=dict(cabin=5.0, battery=5.0))
    n = run(leaf, opt={'hvac_model': 'v2'}, T_amb=5.0, it=dict(cabin=5.0, battery=5.0))
    assert s['summary']['batt_kWh_out'] < n['summary']['batt_kWh_out']


def test_phev_electric_heat_avoids_engine_starts():
    phev = load_vehicle({'class': 'car', 'architecture': 'phev_powersplit', 'aux': {'HVAC': {'source': 'engine'}}})
    a = run(phev, opt={'hvac_model': 'v2'}, T_amb=-7.0)
    b = run(phev, opt={'hvac_model': 'v2', 'phev_electric_heat': True}, T_amb=-7.0)
    assert b['eng_on'].mean() < a['eng_on'].mean()
    assert b['summary']['fuel_g'] < a['summary']['fuel_g']


def test_engine_efficiency_two_parameters_tilt_city_more(car):
    hot = dict(engine=88.0, driveline=80.0, tire_warm=1.0)
    off = copy.deepcopy(car); off['aux']['HVAC']['enabled'] = False
    u0, h0 = run(off, 'udds', it=hot), run(off, 'hwfet', it=hot)
    u1, h1 = run(off, 'udds', it=hot, opt={'eng_dfmep_bar': 0.3}), run(off, 'hwfet', it=hot, opt={'eng_dfmep_bar': 0.3})
    du = u1['summary']['fuel_g'] / u0['summary']['fuel_g'] - 1; dh = h1['summary']['fuel_g'] / h0['summary']['fuel_g'] - 1
    assert du > dh > 0
    # the low-load adder is the city/highway tilt: several times more on UDDS than on HWFET
    ll = {'eng_lowload_W_per_L': 300.0}
    du = run(off, 'udds', it=hot, opt=ll)['summary']['fuel_g'] / u0['summary']['fuel_g'] - 1
    dh = run(off, 'hwfet', it=hot, opt=ll)['summary']['fuel_g'] / h0['summary']['fuel_g'] - 1
    assert du > 0.01 and du > 3 * max(dh, 0.0), (du, dh)
    u2 = run(off, 'udds', it=hot, opt={'eng_eta_scale': 1.05})
    assert abs(u2['summary']['fuel_g'] / u0['summary']['fuel_g'] - 1 / 1.05) < 0.01
    fam = run(off, 'udds', it=hot, opt={'eng_family_adj': {car['engine']['bsfc_map']: {'eng_eta_scale': 1.05}}})
    assert fam['summary']['fuel_g'] == pytest.approx(u2['summary']['fuel_g'], rel=1e-12)


def test_bus_accessories_and_doors():
    hyb = load_vehicle('ref_bus40_hybrid'); dsl = load_vehicle('ref_bus40_diesel')
    it = dict(engine=88.0, driveline=80.0, tire_warm=1.0, cabin=21.0)
    r = {}
    for opt in (None, {'bus_aux': True, 'hvac_model': 'v2'}):
        d = run(dsl, 'cbd14', opt=opt, it=it, T_amb=25.0, solar=400.0)['summary']['fuel_g']
        h = run(hyb, 'cbd14', opt=opt, it=it, T_amb=25.0, solar=400.0)['summary']['fuel_g']
        r[opt is None] = d / h - 1
    assert r[False] < r[True]                      # the hybrid's advantage shrinks


def test_large_motor_light_load():
    t8 = load_vehicle('ref_class8_bev')
    a = run(t8, 'hd_transient', it=dict(battery=20.0), T_amb=20.0)
    b = run(t8, 'hd_transient', it=dict(battery=20.0), T_amb=20.0, opt={'motor_staging': 2, 'motor_size_exp': 0.1})
    assert 0.0 < 1 - b['ledger']['mot_loss'] / a['ledger']['mot_loss'] < 0.6


def test_battery_aging_and_restore():
    s = aging.soh('NMC', years=5, efc=500, T_C=25)
    assert 0.75 < s['capacity'] < 0.97 and 1.0 < s['resistance'] < 1.6
    assert aging.soh('LFP', years=5, efc=500)['capacity'] > s['capacity']
    bev = load_vehicle({'class': 'car', 'architecture': 'bev'})
    new = run(bev, T_amb=20.0); old = run(bev, T_amb=20.0, opt={'batt_soh_capacity': 0.8, 'batt_soh_resistance': 1.5})
    d_new = 0.9 - new['summary']['soc_end']; d_old = 0.9 - old['summary']['soc_end']
    assert d_old / d_new == pytest.approx(1 / 0.8, rel=0.03)
    assert old['ledger']['batt_loss'] > new['ledger']['batt_loss']
    again = run(bev, T_amb=20.0)                    # the cached battery is restored to new
    assert again['summary']['soc_end'] == new['summary']['soc_end']
    from roadsim.sim import get_vehicle
    b = get_vehicle(bev).battery
    assert b.usable_kWh(-10.0) < b.usable_kWh(25.0) <= b.E_nom / 3.6e6
    # pack heater threshold: the v1 default BEV spec heats below 0 C; at -10 C threshold a -5 C trip needs no pack heat
    c1 = run(bev, T_amb=-5.0, it=dict(battery=-5.0)); c2 = run(bev, T_amb=-5.0, it=dict(battery=-5.0), opt={'batt_heat_T_on': -10.0})
    assert c2['summary']['batt_kWh_out'] < c1['summary']['batt_kWh_out'] - 0.3
    run(bev, T_amb=20.0)
    assert get_vehicle(bev).battery.T_heat_on == 0.0          # restored to the spec for a v1 run


def test_preset_ledger_closes_everywhere():
    for arch in ('conv', 'bev', 'hev_powersplit', 'hev_parallel_p2', 'hev_series', 'phev_powersplit', 'phev_series'):
        spec = load_vehicle({'class': 'suv', 'architecture': arch})
        for cyc, T in (('udds', -15.0), ('us06', 38.0)):
            run(spec, cyc, opt=dict(B2, hvac_defrost=T < 0, hvac_seat_heat=True), T_amb=T, solar=600.0 if T > 0 else 0.0)
