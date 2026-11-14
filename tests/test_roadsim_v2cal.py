"""RoadSim v2 calibration pieces: the roadsim-physics-v2 preset, the OBD measurement layer, the conventional
converter options and the corrected EPA procedures."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'roadsim'))
sys.path.insert(0, os.path.join(HERE, '..', 'shared', 'vehicle', 'checks'))

from roadsim import simulate, make_trip, load_vehicle  # noqa: E402
from roadsim.options import PRESETS, resolve  # noqa: E402
from roadsim import obd  # noqa: E402


def _car():
    return load_vehicle({'class': 'car', 'architecture': 'conv'})


def test_v2_preset_loads_and_resolves():
    assert 'roadsim-physics-v2' in PRESETS
    o = resolve('roadsim-physics-v2')
    assert o['engine_thermal'] == 'v2' and o['hvac_model'] == 'v2'


def test_conv_tc_options_default_is_v1_and_cost_fuel():
    s = _car(); s['transmission'] = {'type': 'AT', 'gears': [4.7, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.65]}
    t = make_trip('udds', T_amb=20.0)
    w = dict(engine=88.0, driveline=80.0, tire_warm=1.0)
    a = simulate(s, t, init_temps=w)['summary']['fuel_g']
    b = simulate(s, t, init_temps=w, options={'conv_tc_open_gears': 0, 'conv_tc_slip_rpm': 0.0})['summary']['fuel_g']
    assert a == b
    o = simulate(s, t, init_temps=w, options={'conv_tc_open_gears': 2, 'conv_tc_slip_rpm': 40.0})
    assert o['summary']['fuel_g'] > a * 1.005
    assert o['ledger']['closure_rel'] < 1e-3


def test_obd_label_model():
    s = _car()
    o = simulate(s, make_trip('udds', T_amb=20.0), init_temps=dict(engine=88.0, driveline=80.0, tire_warm=1.0))
    f = np.asarray(o['fuel_rate'], float)
    cut = (np.asarray(o['eng_rpm']) > 300) & (f <= 1e-9)
    assert cut.sum() > 20                                         # UDDS has fuel cut-off seconds
    m = obd.maf_fuel(o, s)
    D = s['engine'].get('displacement_L') or 2.0
    rpm = np.asarray(o['eng_rpm'], float)[cut]
    ref = np.exp(-1.578) * D ** 0.342 * np.maximum(rpm / 1000, 0.3) ** 0.226     # EnergyNet's fit, unchanged
    assert np.allclose(m[cut], ref)
    assert np.array_equal(obd.label_fuel(o, s, 'obd_fuel_rate_Lph'), np.nan_to_num(f))
    assert obd.label_fuel(o, s, 'maf_x_trims_over_afr').sum() > f.sum()
    rng = np.random.default_rng(3)
    log = obd.emulate(o, obd.logger_config('conv', rng, 'maf'), rng, spec=s)
    assert set(log) >= {'v', 'fuel_rate', 'engine_rpm', 'valid'} and len(log['v']) == len(o['v'])


def test_cold_ftp_procedure_heater_on():
    import epa_procedures as ep
    s = ep.test_spec(_car())
    s['tire'] = {'K_t': 0.0, 'temp_effects': False}
    r = ep.ftp(s, T=ep.T_COLD)
    assert np.isfinite(r['FE']) and r['FE'] > 5
