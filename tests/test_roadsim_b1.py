"""RoadSim physics v2, work package B1 (powertrain and hybrid control): every option is off by default (= v1), moves
results in the expected direction with a sane magnitude when on, and keeps the energy ledger closed.
Equations and sources: shared/vehicle/roadsim/docs/v2_b1.md."""
import os, sys, json
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'roadsim'))

from roadsim import simulate, make_trip, load_vehicle, cycle            # noqa: E402
from roadsim.options import DEFAULTS, PRESETS                         # noqa: E402
from roadsim.generator import generate_vehicles                       # noqa: E402

B1 = [k for k in PRESETS['roadsim-physics-v2-b1']]
AT10 = {'type': 'AT', 'gears': [4.70, 2.99, 2.15, 1.80, 1.52, 1.28, 1.00, 0.85, 0.69, 0.64]}
ETA_EQ = 0.33 * 0.90          # battery energy -> fuel energy, as checks/epa_procedures.py
LHV = 43.0e6                  # J/kg gasoline (roadsim.constants FUELS)


def run(spec, cyc='udds', T=20.0, strat=None, opt=None, n=None, eng=90.0, soc=None, cab=None, **kw):
    v = cycle(cyc) if n is None else cycle(cyc)[:n]
    it = {'engine': eng} if cab is None else {'engine': eng, 'cabin': cab}
    o = simulate(spec, make_trip(v, T_amb=T, **kw), strategy=strat, options=opt, init_temps=it, init_soc=soc)
    assert o['ledger']['closure_rel'] < 5e-4, o['ledger']['closure_rel']
    return o


def fuel_eq_g(o):
    """fuel plus the net battery energy at a fixed equivalence (SOC-corrected fuel)."""
    return o['summary']['fuel_g'] + o['ledger']['batt_chem_out'] * 1e3 / ETA_EQ / LHV * 1e3


def starts(o):
    on = o['eng_on'] > 0.5
    return int(np.sum(on[1:] & ~on[:-1]))


def test_defaults_are_v1():
    for k in B1:
        assert k in DEFAULTS
    off = {k: DEFAULTS[k] for k in B1}
    assert off['p2_torque_converter'] is False and off['hybrid_start_energy'] is False
    assert off['series_direct_drive'] is False and off['powersplit_calibration'] is False and off['donor_model'] == 'v1'
    spec = load_vehicle({'class': 'car', 'architecture': 'hev_powersplit'})
    a = run(spec, n=400, eng=20.0)
    b = run(spec, n=400, eng=20.0, opt=off)
    assert a['summary']['fuel_g'] == b['summary']['fuel_g'] and a['summary']['soc_end'] == b['summary']['soc_end']


def test_p2_torque_converter():
    pick = load_vehicle({'class': 'pickup', 'architecture': 'hev_parallel_p2', 'transmission': AT10})
    a = run(pick, n=500, strat='parallel_rule')
    b = run(pick, n=500, strat='parallel_rule', opt={'p2_torque_converter': True})
    dl = (b['ledger']['trans_loss'] - a['ledger']['trans_loss']) / a['ledger']['trans_loss']
    assert 0.02 < dl < 1.0, dl                         # converter slip adds driveline loss at launches
    d = fuel_eq_g(b) / fuel_eq_g(a) - 1
    assert 0.001 < d < 0.10, d
    # a P2 with a dual-clutch box has no converter: the option changes nothing
    car = load_vehicle({'class': 'car', 'architecture': 'hev_parallel_p2'})
    c = run(car, n=300, strat='parallel_rule'); e = run(car, n=300, strat='parallel_rule', opt={'p2_torque_converter': True})
    assert c['summary']['fuel_g'] == e['summary']['fuel_g']


@pytest.mark.parametrize('arch', ['hev_parallel_p2', 'hev_series', 'hev_powersplit'])
def test_start_energy(arch):
    spec = load_vehicle({'class': 'car', 'architecture': arch})
    a = run(spec, n=800, strat='aecms')
    b = run(spec, n=800, strat='aecms', opt={'hybrid_start_energy': True})
    assert starts(b) <= starts(a)                      # starts cost more, so A-ECMS starts less often
    d = fuel_eq_g(b) / fuel_eq_g(a) - 1
    assert -0.01 < d < 0.08, d


def test_min_on_time_and_stay_on():
    spec = load_vehicle({'class': 'car', 'architecture': 'hev_parallel_p2'})
    a = run(spec, n=800, strat='aecms')
    b = run(spec, n=800, strat='aecms', opt={'hybrid_min_on_s': 8.0, 'hybrid_min_off_s': 3.0, 'hybrid_stay_on_P_frac': 0.15})
    assert starts(b) <= starts(a)
    on = b['eng_on'] > 0.5
    # an engine run shorter than 8 s may only end in braking or a stop (the engine is unfuelled when braking)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int), [0]])))
    for s0, s1 in zip(edges[::2], edges[1::2]):
        if s1 - s0 < 8 and s1 < len(on):
            assert b['F_wheel'][s1] <= 0.0 or b['v'][s1] < 0.5


def test_series_direct_drive():
    spec = load_vehicle({'class': 'car', 'architecture': 'hev_series'})
    a = run(spec, 'hwfet', strat='aecms')
    b = run(spec, 'hwfet', strat='aecms', opt={'series_direct_drive': True})
    d = fuel_eq_g(b) / fuel_eq_g(a) - 1
    assert -0.15 < d < -0.01, d                        # no double conversion at cruise
    kph = 0.5 * (b['v'][1:] + b['v'][:-1]) * 3.6      # mid-step speed, at which the step's engine speed is set
    rpm_dd = kph * DEFAULTS['series_dd_rpm_per_kph']
    cruise = (kph > 75) & (b['eng_on'][1:] > 0.5)
    lock = np.abs(b['eng_rpm'][1:] - rpm_dd) < 0.01 * rpm_dd
    assert lock[cruise].mean() > 0.5                   # the lock-up clutch is used at cruise
    # the rule strategy uses it too
    c = run(spec, 'hwfet', strat='series_power_follower', opt={'series_direct_drive': True})
    assert fuel_eq_g(c) < fuel_eq_g(run(spec, 'hwfet', strat='series_power_follower'))


def test_powersplit_calibration():
    spec = load_vehicle({'class': 'car', 'architecture': 'hev_powersplit'})
    a = run(spec, 'udds', strat='powersplit_rule', eng=20.0)
    b = run(spec, 'udds', strat='powersplit_rule', eng=20.0, opt={'powersplit_calibration': True})
    assert b['eng_on'].mean() > a['eng_on'].mean() + 0.02         # warm-up and catalyst holds
    # catalyst hold: the engine runs through the first 40 s of running (braking steps are unfuelled in RoadSim)
    assert (b['eng_on'][21:61] > 0.5).mean() > (a['eng_on'][21:61] > 0.5).mean() + 0.1, ((a['eng_on'][21:61] > 0.5).mean())
    h = run(spec, 'hwfet', strat='aecms', opt={'powersplit_calibration': True})
    fast = (h['v'][1:] * 3.6 > 75) & (h['F_wheel'][1:] > 0)
    assert h['eng_on'][1:][fast].mean() > 0.97                     # engine fuelled above ~72 km/h
    d = fuel_eq_g(b) / fuel_eq_g(a) - 1
    assert 0.0 < d < 0.15, d


def test_powersplit_heater_request():
    spec = load_vehicle({'class': 'car', 'architecture': 'hev_powersplit'})
    # 8 C, warm cabin, coolant at 30 C: about 1.5 kW of heat demand is below v1's heat trigger (30 % of 6 kW) but a
    # heater request for the production rule, which starts the engine below ps_heat_on_C and keeps it on up to
    # ps_heat_off_C; compared with the same rules without a heater request (speed and warm-up rules switched off)
    base = {'powersplit_calibration': True, 'ps_engine_on_kph': 200.0, 'ps_warmup_C': -50.0, 'ps_cat_hold_s': 0.0}
    a = run(spec, 'udds', T=8.0, strat='aecms', eng=30.0, cab=21.0, n=600, opt=dict(base, ps_heat_W=1e9))
    b = run(spec, 'udds', T=8.0, strat='aecms', eng=30.0, cab=21.0, n=600, opt=base)
    assert b['eng_on'].mean() > a['eng_on'].mean() + 0.05


def test_phev_cd_rules():
    ps = load_vehicle({'class': 'car', 'architecture': 'phev_powersplit'})
    a = run(ps, 'us06', strat='phev_cdcs', soc=0.9)
    b = run(ps, 'us06', strat='phev_cdcs', soc=0.9, opt={'phev_cd_rules': True, 'phev_cd_P_on_frac': 0.5})
    assert b['eng_on'].mean() > a['eng_on'].mean()                 # blended type: engine joins at high power
    erev = load_vehicle({'class': 'car', 'architecture': 'phev_series'})
    c = run(erev, 'udds', strat='phev_cdcs', soc=0.9, eng=20.0, opt={'phev_cd_rules': True})
    assert c['summary']['fuel_g'] == 0.0                          # EREV stays electric in CD
    d = run(erev, 'udds', T=-15.0, eng=-15.0, strat='phev_cdcs', soc=0.9, n=600,
            opt={'phev_cd_rules': True, 'phev_heat_engine_below_C': -10.0})
    assert d['summary']['fuel_g'] > 0.0                           # engine-assisted heating below the threshold


def test_dp_min_on_off():
    spec = load_vehicle({'class': 'car', 'architecture': 'hev_parallel_p2'})
    rules = {'hybrid_min_on_s': 6.0, 'hybrid_min_off_s': 3.0, 'start_penalty_J': 10e3, 'hybrid_start_energy': True}
    a = run(spec, n=700, strat='dp', opt=rules)
    b = run(spec, n=700, strat='dp', opt=dict(rules, dp_min_on_off=True))
    e = run(spec, n=700, strat='aecms', opt=rules)
    assert starts(b) <= starts(a)
    assert fuel_eq_g(b) <= fuel_eq_g(e) * 1.02                    # DP stays a reference for the rule-bound strategy
    assert fuel_eq_g(b) <= fuel_eq_g(a) * 1.01


def test_generator_donor_v2():
    v1 = generate_vehicles(6, seed=11)
    assert v1 == generate_vehicles(6, seed=11, donor_model='v1')
    from roadsim.generator import _donors
    D = _donors()['subsets']
    for v in generate_vehicles(25, seed=12, cls='car', arch='hev_powersplit', donor_model='v2'):
        g = v['generator']
        assert g['donor_model'] == 'v2' and g['donor_row'] in D[g['donor']]['hev']
    for v in generate_vehicles(25, seed=13, cls='car', arch='conv', donor_model='v2'):
        g = v['generator']
        assert g['donor_row'] not in set(D[g['donor']].get('hev', [])) | set(D[g['donor']].get('phev', []))


def test_ledger_closes_with_every_b1_option():
    for i, v in enumerate(generate_vehicles(10, seed=77, donor_model='v2')):
        o = simulate(v, make_trip(cycle(['udds', 'us06', 'hwfet'][i % 3])[:600], T_amb=[-10.0, 25.0][i % 2], grade=0.02 * (i % 3 - 1)),
                     options='roadsim-physics-v2-b1', init_temps={'engine': [20.0, 90.0][i % 2]})
        assert o['ledger']['closure_rel'] < 5e-4
