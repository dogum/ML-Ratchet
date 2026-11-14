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

    # ---- end powertrain and hybrid control

    # ---- thermal: engine warm-up and cold, HVAC and cabin, A/C, per-engine efficiency, battery aging
    # (models in roadsim/thermal.py and aging.py; every other tunable number is in thermal.PARAMS, see docs/v2_b2.md)
    'engine_thermal': 'v1',        # 'v1' one lumped coolant node | 'v2' coolant + oil nodes, cold friction follows the oil
    'oil_viscosity': 'arrhenius',  # cold-friction viscosity law: 'arrhenius' (v1, 3776 K) | 'vogel' (multigrade oil, ATF)
    'cold_enrich': 0.0,            # after-start enrichment + catalyst heating: fuel fraction per K of coolant below 25 C at start (1/K)
    'warmup_enrich': 0.0,          # warm-up enrichment: fuel fraction per K of coolant below 20 C (1/K)
    'eng_dfmep_bar': 0.0,          # extra engine friction (bar FMEP): raises idle and light-load fuel most
    'eng_eta_scale': 1.0,          # engine efficiency scale (fuel divided by it): moves every load alike
    'eng_family_adj': {},          # per engine map {bsfc_map id: [dfmep_bar, eta_scale]}, replaces the two above for that map
    'fead_drag_Nm': 0.0,           # belt / accessory-drive drag on conventional engines (Nm at the crank)
    'axle_cold': False,            # final-drive oil + wheel-bearing drag rising in the cold (axle oil temperature node)
    'tire_thermal': False,         # tyre temperature node; rolling resistance follows it (replaces the v1 warm-up factor)
    'tire_temp_coef': 0.008,       # 1/K rolling-resistance change per K of tyre temperature (ISO 28580, passenger tyres)
    'tire_dyno_frac': 0.6,         # share of Crr that is the vehicle's own tyre loss on a chassis dynamometer
    'hvac_model': 'v1',            # 'v1' one-node cabin | 'v2' cabin air + interior mass, air flows, compressor, PTC, heat pump
    'hvac_T_set': 21.5,            # C cabin set point (v2)
    'hvac_recirc_heat': 0.3,       # recirculated share of HVAC air while heating (v2; mostly fresh air keeps windows clear)
    'hvac_recirc_cool': 0.8,       # ... while cooling (1.0 during pull-down, cabin > 3 K above set point)
    'hvac_seat_heat': False,       # seat + steering-wheel heaters; air set point 2 K lower (v2)
    'hvac_defrost': False,         # defrost / demist: fresh air, high blower, compressor dehumidifies above 3 C (v2)
    'ambient_rh': 0.4,             # ambient relative humidity when the trip has no 'rh' (v2)
    'phev_electric_heat': False,   # PHEVs with engine heat use an electric heater while the coolant is cold (no start for heat)
    'bus_aux': False,              # bus accessories: 24 V alternator, air compressor (stops, brakes), steering, door-open air
    'motor_staging': 0,            # machines above 300 kW act as N units; idle units shed at light load (0 = v1)
    'motor_size_exp': 0.0,         # machines above 150 kW: losses x (P_max / 150 kW)^-exp
    'batt_soh_capacity': 1.0,      # battery capacity / new (state of health)
    'batt_soh_resistance': 1.0,    # battery resistance / new
    'batt_age': None,              # {'years', 'efc', 'T_C', 'soc_mean', 'dod'}: SOH from aging.soh (overrides the two above)
    'batt_soc_window': None,       # [lo, hi] usable SOC window override
    'thermal_params': {},          # overrides for numbers in thermal.PARAMS
    # ---- end thermal

    # ---- dynamics: sub-second steps, tyre model, driver model, wind

    # ---- end dynamics
}

PRESETS = {
    'roadsim-physics-v1': {},
    'roadsim-physics-v2-b2': {
        'engine_thermal': 'v2', 'oil_viscosity': 'vogel', 'cold_enrich': 0.008, 'warmup_enrich': 0.004,
        'eng_dfmep_bar': 0.0, 'eng_eta_scale': 1.0, 'eng_family_adj': {}, 'fead_drag_Nm': 0.0,
        'axle_cold': True, 'tire_thermal': True, 'tire_temp_coef': 0.008, 'tire_dyno_frac': 0.6,
        'hvac_model': 'v2', 'hvac_T_set': 21.5, 'hvac_recirc_heat': 0.3, 'hvac_recirc_cool': 0.8,
        'hvac_seat_heat': False, 'hvac_defrost': False, 'ambient_rh': 0.4, 'phev_electric_heat': True,
        'bus_aux': True, 'motor_staging': 2, 'motor_size_exp': 0.1,
        'batt_soh_capacity': 1.0, 'batt_soh_resistance': 1.0, 'batt_age': None, 'batt_soc_window': None,
        'thermal_params': {},
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
