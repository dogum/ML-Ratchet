"""RoadSim physics options (v2 and later).

Every option defaults to the v1 behaviour, so RoadSim with no options reproduces physics v1 exactly
(tests/test_roadsim_fingerprint.py). A physics version is a named set of option values (PRESETS); the calibration
pass fits the numeric parameters on dev data only.

    simulate(spec, trip, options={'p2_torque_converter': True})       # per run
    spec['physics'] = {'cabin_model': 'v2'}                          # per vehicle
    simulate(spec, trip, options='roadsim-physics-v2')               # a whole preset

Precedence: DEFAULTS < preset < spec['physics'] < options dict. Unknown keys raise, so a typo never silently
falls back to v1.
"""
import copy

DEFAULTS = {
    # ---- powertrain and hybrid control (P2 torque converter and starts, e:HEV direct drive, production calibrations)
    # (docs/v2_b1.md has the equations and sources)
    'p2_torque_converter': False,       # P2 automatics with a torque converter (AT): traction slips through it below lock-up
    'p2_tc_lock_rpm': 1050.0,           # rpm, turbine speed above which the converter locks (1st gear needs 1.4x this)
    'p2_tc_open_gears': 0,              # gears (from 1st) in which the converter never locks; 0 = the lock-up rule only
    'hybrid_start_energy': False,       # engine starts also cost cranking energy (inertia + friction) and P2 K0 slip
    'start_fuel_idle_s': 1.5,           # s of warm idle fuel burnt per start (enrichment); 1.5 = the v1 start fuel
    'start_crank_rpm': 1000.0,          # rpm, speed the engine is spun to before first firing
    'start_crank_s': 0.3,               # s, cranking time against motoring friction
    'start_crank_eta': 0.85,            # -, motor+inverter efficiency while cranking
    'p2_k0_slip_frac': 1.0,             # -, share of the engine's sync energy also lost as K0 slip (1 = motor cranks via K0)
    'start_self_eta': 0.25,             # -, fuel-to-shaft efficiency when the engine spins itself up (battery at its limit)
    'hybrid_min_on_s': None,            # s, minimum engine-on time for every hybrid strategy (None = v1: 4 s, DP 0)
    'hybrid_min_off_s': None,           # s, minimum engine-off time (None = v1: 2 s, DP 0)
    'hybrid_stay_on_P_frac': None,      # engine stays on while wheel power > this x engine P_max (None = no rule)
    'start_penalty_J': 0.0,             # J, virtual drivability cost per start in cost-based choices (not fuel)
    'series_direct_drive': False,       # e:HEV/i-MMD lock-up clutch: engine drives the wheels directly at cruise
    'series_dd_rpm_per_kph': 20.0,      # rpm per km/h, engine speed in direct drive (~2000 rpm at 100 km/h)
    'series_dd_v_min_kph': 70.0,        # km/h, lowest direct-drive speed
    'series_dd_v_max_kph': 140.0,       # km/h, highest direct-drive speed
    'series_dd_eta': 0.96,              # -, engine-to-wheel gear efficiency in direct drive (two meshes + final drive)
    'powersplit_calibration': False,    # Toyota-type production rules on power-split strategies (below)
    'ps_engine_on_kph': 72.0,           # km/h, engine fuelled whenever driving above this (with positive demand)
    'ps_warmup_C': 40.0,                # C, engine held on (also at stops) until the coolant reaches this
    'ps_cat_hold_s': 40.0,              # s, engine kept running this long after a cold start (catalyst light-off)
    'ps_cat_cold_C': 35.0,              # C, a start below this coolant temperature is a cold start
    'ps_heat_on_C': 45.0,               # C, with heat demand the engine starts when the coolant is below this ...
    'ps_heat_off_C': 70.0,              # C, ... and runs until the coolant reaches this
    'ps_heat_W': 1000.0,                # W, cabin heat demand that counts as a heater request
    'phev_cd_rules': False,             # PHEV charge-depleting rules (power/speed thresholds, heat only if needed)
    'phev_cd_P_on_frac': 0.9,           # power-split PHEV: engine joins in CD above this x MG2 power
    'phev_cd_v_max_kph': 135.0,         # km/h, power-split PHEV: engine joins in CD above this speed
    'phev_heat_engine_below_C': None,   # C, PHEV with electric heat starts the engine for heat below this (None = never)
    'dp_min_on_off': False,             # DP honours hybrid_min_on_s / hybrid_min_off_s and start_penalty_J
    'donor_model': 'v1',                # generate_vehicles: 'v2' draws hybrid bodies from real hybrid records, vans from minivans
    # ---- end powertrain and hybrid control

    # ---- thermal: engine warm-up and cold, HVAC and cabin, A/C, per-engine efficiency, battery aging

    # ---- end thermal

    # ---- dynamics: sub-second steps, tyre model, driver model, wind

    # ---- end dynamics
}

PRESETS = {
    'roadsim-physics-v1': {},
    # work package B1 (powertrain and hybrid control): intended v2 values, before the dev calibration pass
    'roadsim-physics-v2-b1': {
        'p2_torque_converter': True, 'hybrid_start_energy': True, 'hybrid_min_on_s': 8.0, 'hybrid_min_off_s': 3.0,
        'hybrid_stay_on_P_frac': 0.15, 'start_penalty_J': 20e3, 'series_direct_drive': True,
        'powersplit_calibration': True, 'phev_cd_rules': True, 'dp_min_on_off': True, 'donor_model': 'v2',
    },
}


def resolve(options=None, spec=None):
    """-> a full option dict (a fresh copy)."""
    out = copy.deepcopy(DEFAULTS)
    layers = []
    if isinstance(options, str):
        layers.append(PRESETS[options]); options = None
    elif isinstance(options, dict) and isinstance(options.get('preset'), str):
        layers.append(PRESETS[options['preset']])
        options = {k: v for k, v in options.items() if k != 'preset'}
    if spec is not None and isinstance(spec.get('physics'), dict):
        layers.append(spec['physics'])
    if options:
        layers.append(options)
    for L in layers:
        for k, v in L.items():
            if k not in out:
                raise KeyError(f'unknown RoadSim option {k!r} (known: {sorted(out)})')
            out[k] = copy.deepcopy(v)
    return out
