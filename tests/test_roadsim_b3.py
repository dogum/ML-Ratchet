"""RoadSim physics v2, work package B3 (vehicle dynamics): sub-steps, tyre model, human driver, wind.

Each option off = v1; on = changes results in the expected direction with a sane size; the ledger closes."""
import os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'roadsim'))

from roadsim import simulate, make_trip, OPTION_DEFAULTS, OPTION_PRESETS        # noqa: E402
from roadsim import tyre, wind, driver                                           # noqa: E402
from roadsim.sim import get_vehicle                                              # noqa: E402

CAR = {'class': 'car', 'architecture': 'conv'}
BEV = {'class': 'car', 'architecture': 'bev'}
HEV = {'class': 'car', 'architecture': 'hev_powersplit'}
WARM = {'engine': 85.0, 'driveline': 70.0, 'tire_warm': 1.0, 'cabin': 21.0}
TOL = 5e-4
B3_KEYS = ['substeps', 'substep_record', 'tyre_model', 'tyre_mu_scale', 'tyre_regen_mu_frac', 'tyre_ref_eu_class',
           'driver_model', 'driver_aggressiveness', 'driver_reaction_s', 'wind_profile', 'wind_z0_ref_m',
           'wind_z0_site_m', 'wind_shelter', 'wind_gusts', 'wind_turb_seed', 'drafting', 'draft_headway_s']


def energy_wh_per_km(r, hybrid=False):
    s = r['summary']
    return (s['fuel_g'] * 42.6 / 3.6 + s['batt_kWh_out'] * 1000.0 / (0.3 if hybrid else 1.0)) / (s['distance_m'] / 1000.0)


def run(spec, trip, **opt):
    r = simulate(spec, trip, options=opt or None, init_temps=WARM)
    assert r['ledger']['closure_rel'] < TOL, r['ledger']
    return r


def test_options_registered_with_v1_defaults():
    for k in B3_KEYS:
        assert k in OPTION_DEFAULTS
    assert OPTION_DEFAULTS['substeps'] == 1 and OPTION_DEFAULTS['tyre_model'] is False
    assert OPTION_DEFAULTS['driver_model'] == 'pi' and OPTION_DEFAULTS['wind_profile'] == 'none'
    assert OPTION_DEFAULTS['wind_gusts'] is False and OPTION_DEFAULTS['drafting'] is False
    assert set(OPTION_PRESETS['roadsim-physics-v2-b3']) <= set(B3_KEYS)


@pytest.mark.parametrize('spec', [CAR, HEV])
def test_explicit_v1_values_reproduce_v1(spec):
    T = make_trip('udds', wind_speed=4.0, wind_dir=30.0)
    a = simulate(spec, T, options='roadsim-physics-v1')
    b = simulate(spec, T, options={k: OPTION_DEFAULTS[k] for k in B3_KEYS})
    for k in ('fuel_g', 'batt_kWh_out', 'distance_m', 'trace_rms_mps'):
        assert a['summary'][k] == b['summary'][k]
    assert np.array_equal(a['fuel_rate'], b['fuel_rate'])
    assert a['ledger']['tyre_slip'] == 0.0


def test_substeps_10hz_close_to_1hz_and_outputs():
    T = make_trip('us06')
    a = run(CAR, T)
    b = run(CAR, T, substeps=10, substep_record=True)
    N = len(T['v'])
    assert len(b['v']) == N and len(b['fuel_rate']) == N
    assert len(b['hf']['v']) == (N - 1) * 10 + 1
    d = energy_wh_per_km(b) / energy_wh_per_km(a) - 1.0
    assert abs(d) < 0.01                     # 1 Hz integration error on US06 is small (measured +0.2 %)
    assert abs(b['summary']['distance_m'] / a['summary']['distance_m'] - 1) < 1e-3
    # per-second outputs are the means of the sub-steps
    hf = b['hf']['fuel_rate'][1:].reshape(N - 1, 10).mean(axis=1)
    assert np.allclose(b['fuel_rate'][1:], hf, rtol=1e-5, atol=1e-6)
    # shifts happen inside seconds at 10 Hz
    g = b['hf']['gear'][1:].reshape(N - 1, 10)
    assert (g.min(axis=1) != g.max(axis=1)).any()


def test_substeps_hybrid_start_fuel_kept():
    T = make_trip('udds')
    a = run(HEV, T)
    b = run(HEV, T, substeps=10)
    assert abs(energy_wh_per_km(b, True) / energy_wh_per_km(a, True) - 1) < 0.02


def test_tyre_model_slip_and_load_transfer():
    T = make_trip('udds')
    a = run(CAR, T); b = run(CAR, T, tyre_model=True)
    d = energy_wh_per_km(b) / energy_wh_per_km(a) - 1
    assert 0.002 < d < 0.02                   # slip losses: +0.5 % measured
    assert b['ledger']['tyre_slip'] > 10.0
    t = tyre.get(get_vehicle(CAR))
    Nf0, Nr0 = t.axle_loads(0.0, 0.0, 1.0)
    Nf1, Nr1 = t.axle_loads(3.0, 0.0, 1.0)    # accelerating loads the rear axle
    assert Nr1 > Nr0 and Nf1 < Nf0 and abs((Nf1 + Nr1) - (Nf0 + Nr0)) < 1e-6
    k1 = t.slip(1000.0, 8000.0, 0.9); k2 = t.slip(3000.0, 8000.0, 0.9)
    assert 0 < k1 < k2 < 0.2 and t.slip(-1000.0, 8000.0, 0.9) < 0


def test_tyre_snow_and_ice_limit_regen_and_traction():
    dry = run(BEV, make_trip('udds'), tyre_model=True)
    snow = run(BEV, make_trip('udds', surface=4), tyre_model=True)
    assert snow['ledger']['regen'] < 0.9 * dry['ledger']['regen']
    assert snow['ledger']['brake'] > 5 * dry['ledger']['brake']
    ice = run(CAR, make_trip('us06', surface=5), tyre_model=True)
    assert ice['summary']['trace_rms_mps'] > 2.0           # 0.1 g of grip cannot follow US06
    assert np.max(np.diff(ice['v'])) < 0.1 * 9.81 * 0.7    # acceleration within the driven axle's friction
    # regen cap: a front-drive car carries more braking on its driven axle than a rear-drive one
    f = tyre.get(get_vehicle(HEV)).limits(-3.0, 0.0, 1.0, 0.9, 0.0, 0.8)[2] / get_vehicle(HEV).mass
    r = tyre.get(get_vehicle(BEV)).limits(-3.0, 0.0, 1.0, 0.9, 0.0, 0.8)[2] / get_vehicle(BEV).mass
    assert f > r > 0.2 * 9.81


def test_tyre_eu_class():
    T = make_trip('hwfet')
    A = run(dict(CAR, tire={'eu_class': 'A'}), T, tyre_model=True)
    E = run(dict(CAR, tire={'eu_class': 'E'}), T, tyre_model=True)
    C = run(CAR, T, tyre_model=True)
    assert energy_wh_per_km(A) < energy_wh_per_km(C) < energy_wh_per_km(E)
    assert tyre.tyre_size('205/55R16') == pytest.approx((0.205, 0.3159), abs=1e-3)


def test_human_driver_span():
    T = make_trip('udds')
    base = run(CAR, T, substeps=10)
    res = {}
    for g in (0.0, 0.5, 1.0):
        r = run(CAR, T, substeps=10, driver_model='human', driver_aggressiveness=g)
        v = r['v'].astype(float)
        pke = np.sum(np.maximum(np.diff(v ** 2), 0)) / r['summary']['distance_m']
        res[g] = (pke, energy_wh_per_km(r) / energy_wh_per_km(base) - 1, r['summary']['trace_rms_mps'])
    assert res[0.0][0] < res[0.5][0] < res[1.0][0]
    assert res[0.0][1] < res[0.5][1] < res[1.0][1]
    assert -0.08 < res[0.0][1] < 0.0 and 0.05 < res[1.0][1] < 0.35
    assert all(x[2] < 1.0 for x in res.values())
    # step response: aggressive overshoots more than typical; eco does not overshoot
    v = np.concatenate([np.zeros(5), np.arange(0, 20, 2.0), np.full(30, 20.0)])
    over = {}
    for g in (0.0, 0.5, 1.0):
        r = run(CAR, make_trip(v), substeps=10, substep_record=True, driver_model='human', driver_aggressiveness=g)
        over[g] = r['hf']['v'].max() - 20.0
    assert over[0.0] < 0.05 and over[1.0] > over[0.5]


def test_wind_profile_gusts_drafting():
    assert 0.55 < wind.profile_factor(0.87, 0.03, 0.1) < 0.75
    assert wind.profile_factor(0.87, 0.03, 1.0) < wind.profile_factor(0.87, 0.03, 0.1)
    T = make_trip('hwfet', wind_speed=6.0, wind_dir=0.0)            # 6 m/s headwind at the station
    a = run(CAR, T)
    b = run(CAR, T, wind_profile='log')
    assert -0.12 < energy_wh_per_km(b) / energy_wh_per_km(a) - 1 < -0.02
    assert np.all(np.abs(b['headwind'][1:] / a['headwind'][1:] - wind.profile_factor(0.87, 0.03, 0.1)) < 1e-4)
    g1 = run(CAR, T, wind_gusts=True, substeps=10)
    g2 = run(CAR, T, wind_gusts=True, substeps=10)
    g3 = run(CAR, T, wind_gusts=True, substeps=10, wind_turb_seed=7)
    assert g1['summary']['fuel_g'] == g2['summary']['fuel_g'] != g3['summary']['fuel_g']
    assert 0.0 < energy_wh_per_km(g1) / energy_wh_per_km(a) - 1 < 0.02  # turbulence adds drag (sigma^2, yaw)
    assert 0.2 < np.std(g1['headwind'][1:] - a['headwind'][1:]) < 2.0
    d = run(CAR, make_trip('hwfet', gap_m=10.0), drafting=True)
    n = run(CAR, make_trip('hwfet'))
    assert -0.08 < energy_wh_per_km(d) / energy_wh_per_km(n) - 1 < -0.02
    assert run(CAR, make_trip('hwfet'), drafting=True)['summary']['fuel_g'] == n['summary']['fuel_g']   # no lead


def test_b3_preset_runs_and_closes():
    for spec, hyb in ((CAR, False), (BEV, False)):
        T = make_trip('udds', wind_speed=4.0, wind_dir=90.0)
        r = simulate(spec, T, options='roadsim-physics-v2-b3', init_temps=WARM)
        assert r['ledger']['closure_rel'] < TOL
        assert abs(energy_wh_per_km(r, hyb) / energy_wh_per_km(run(spec, T), hyb) - 1) < 0.08


def test_bad_option_values_raise():
    with pytest.raises(ValueError):
        simulate(CAR, make_trip('udds'), options={'driver_model': 'robot'})
    with pytest.raises(ValueError):
        simulate(CAR, make_trip('udds'), options={'wind_profile': 'cubic'})
    assert driver.params(0.25)['a_max'] == pytest.approx(0.5 * (1.5 + 2.6))
