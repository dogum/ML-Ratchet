"""RoadSim physics options (v2 and later).

Every option's DEFAULTS value is the v1 behaviour: options=None, 'roadsim-physics-v1' or a dict without a 'preset'
key build on physics v1, which stays reproducible exactly (tests/test_roadsim_fingerprint.py). A physics version is a
named set of option values (PRESETS); the calibration passes fitted the numbers on dev data only. Since 13 Nov 2026
simulate() without an options argument runs DEFAULT_PHYSICS = 'roadsim-physics-v2' (loaded from
data/calibration_v2.json; decisions in CALIBRATION.md).

    simulate(spec, trip)                                              # physics v2 (the default)
    simulate(spec, trip, options='roadsim-physics-v1')               # physics v1, exactly
    simulate(spec, trip, options={'p2_torque_converter': True})       # v1 plus one option
    simulate(spec, trip, options={'preset': 'roadsim-physics-v2', 'regen_share': {}})   # v2 with one change
    spec['physics'] = {'hvac_model': 'v2'}                           # per vehicle

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
    'regen_share': {},                  # {architecture: s}: series (cooperative) brake blending gives regen at most s x the braking
                                        # force, friction brakes the rest (1 = v1: regen up to the motor/battery/stability caps)
    'hybrid_chg_cap_frac': None,        # ECMS family: battery charging from the engine while driving <= this x engine P_max (None = no cap)
    'series_direct_drive': False,       # e:HEV/i-MMD lock-up clutch: engine drives the wheels directly at cruise
    'series_dd_rpm_per_kph': 20.0,      # rpm per km/h, engine speed in direct drive (~2000 rpm at 100 km/h)
    'series_dd_v_min_kph': 70.0,        # km/h, lowest direct-drive speed
    'series_dd_v_max_kph': 140.0,       # km/h, highest direct-drive speed
    'series_dd_prefer': False,          # ECMS family: inside the lock-up window a running engine always drives through the clutch
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
    'phev_heat_cold_start_only': False, # ... only in a trip that began with a cold engine (coolant below ps_cat_cold_C)
    'dp_min_on_off': False,             # DP honours hybrid_min_on_s / hybrid_min_off_s and start_penalty_J
    'donor_model': 'v1',                # generate_vehicles: 'v2' draws hybrid bodies from real hybrid records, vans from minivans
    # ---- end powertrain and hybrid control

    # ---- thermal: engine warm-up and cold, HVAC and cabin, A/C, per-engine efficiency, battery aging
    # (models in roadsim/thermal.py and aging.py; every other tunable number is in thermal.PARAMS, see docs/v2_b2.md)
    'engine_thermal': 'v1',        # 'v1' one lumped coolant node | 'v2' coolant + oil nodes, cold friction follows the oil
    'oil_viscosity': 'arrhenius',  # cold-friction viscosity law: 'arrhenius' (v1, 3776 K) | 'vogel' (multigrade oil, ATF)
    'cold_enrich': 0.0,            # after-start enrichment + catalyst heating: fuel fraction per K of coolant below 25 C at start (1/K)
    'warmup_enrich': 0.0,          # warm-up enrichment: fuel fraction per K of coolant below 20 C (1/K)
    'eng_eta_scale': 1.0,          # engine efficiency scale (fuel divided by it): moves every load alike
    'eng_lowload_W_per_L': 0.0,    # extra fuel power per litre at idle, fading to zero at 4 kW/L brake: tilts city vs highway
    'eng_dfmep_bar': 0.0,          # extra engine friction (bar FMEP): moves FTP and HWFET nearly alike
    'eng_family_adj': {},          # per engine map {bsfc_map id: {'eng_eta_scale': .., 'eng_lowload_W_per_L': .., 'eng_dfmep_bar': ..}}
    'aux_base_W': None,            # light-duty 12 V base electrical load (W) replacing the class default (None = spec / v1)
    'onroad_aux_W': {},            # {architecture: W} light-duty on-road extra load (electrical-equivalent, 12 V supply): lights and
                                   # what real roads, tyres and loads add over the coast-down; only on road runs (tyre
                                   # temperature effects on), never on a dyno spec. Fitted on VED dev twins
    'conv_tc_open_gears': 0,       # light-duty conventional automatics: converter never locks in the lowest N gears (0 = v1 lock-up rule)
    'conv_tc_slip_rpm': 0.0,       # light-duty conventional automatics: controlled lock-up slip under drive in every locked gear but the top, rpm
    'fead_drag_Nm': 0.0,           # belt / accessory-drive drag on conventional engines (Nm at the crank)
    'eng_transient_fuel': 0.0,     # s: extra fuel fraction per (brake-torque rise per second / peak torque): tip-in
                                   # enrichment and port/wall wetting that a steady-state map lacks; light-duty gasoline
                                   # engines only (0 = v1)
    'eng_fullload_enrich': 0.0,    # fuel fraction added at full torque (component-protection enrichment), ramping from
    'eng_fullload_frac': 0.85,     # ... this share of the full-load torque; light-duty gasoline engines only (0 = v1)
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
    'phev_engine_heat': False,     # plug-ins with a PTC or heat pump also heat from a warm engine (>= 50 C coolant) first
    'phev_electric_heat': False,   # PHEVs with engine heat use an electric heater while the coolant is cold (no start for heat)
    'bus_aux': False,              # bus accessories: 24 V alternator, air compressor (stops, brakes), steering, door-open air
    'motor_staging': 0,            # machines above 300 kW act as N units; idle units shed at light load (0 = v1)
    'motor_size_exp': 0.0,         # machines above 150 kW: losses x (P_max / 150 kW)^-exp
    'batt_soh_capacity': 1.0,      # battery capacity / new (state of health)
    'batt_soh_resistance': 1.0,    # battery resistance / new
    'batt_age': None,              # {'years', 'efc', 'T_C', 'soc_mean', 'dod'}: SOH from aging.soh (overrides the two above)
    'batt_soc_window': None,       # [lo, hi] usable SOC window override
    'batt_heat_T_on': None,        # C: pack heater switches on below this (None = spec; v1 default spec 0 C with 3 kW on BEVs)
    'thermal_params': {},          # overrides for numbers in thermal.PARAMS
    # ---- end thermal

    # ---- dynamics: sub-second steps, tyre model, driver model, wind
    'substeps': 1,                  # integration sub-steps per trip second (1 = v1 at 1 Hz; 10 = 10 Hz), -
    'substep_record': False,        # also return the sub-step arrays in out['hf'] (needs substeps > 1)
    'tyre_model': False,            # brush-model slip, axle load transfer, friction-ellipse limits on traction and regen
    'tyre_mu_scale': 1.0,           # multiplier on the peak tyre-road friction table tyre.MU_PEAK, -
    'tyre_regen_mu_frac': 0.8,      # share of the driven axle's friction the brake controller lets regen use, -
    'tyre_ref_eu_class': 'C',       # EU-label class the spec's Crr stands for (rescaled when spec tire.eu_class is set)
    'driver_model': 'pi',           # 'pi' (v1 feed-forward + PI follower) or 'human' (lag, anticipation, overshoot)
    'driver_aggressiveness': 0.5,   # human driver: 0 eco, 0.5 typical, 1 aggressive, -
    'driver_reaction_s': 0.5,       # human driver: perception-reaction delay, s
    'wind_profile': 'none',         # 'none' (v1: trip wind acts at road level) or 'log' (10 m station wind to road height)
    'wind_z0_ref_m': 0.03,          # roughness length at the wind station (open airport), m
    'wind_z0_site_m': 0.1,          # roughness length around the road, m
    'wind_shelter': 1.0,            # terrain/roadside sheltering factor on the road-height mean wind, -
    'wind_gusts': False,            # add seeded Kaimal-spectrum turbulence (along and across the mean wind)
    'wind_turb_seed': 0,            # seed of the turbulence realisation, -
    'drafting': False,              # CdA reduction behind a lead vehicle (trip 'gap_m' [m] or draft_headway_s)
    'draft_headway_s': None,        # time gap to a lead vehicle when the trip has no 'gap_m' (None = no lead), s
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
    'roadsim-physics-v2-b3': {
        'substeps': 10, 'tyre_model': True, 'tyre_mu_scale': 1.0, 'tyre_regen_mu_frac': 0.8, 'tyre_ref_eu_class': 'C',
        'driver_model': 'pi', 'wind_profile': 'log', 'wind_z0_ref_m': 0.03, 'wind_z0_site_m': 0.1, 'wind_shelter': 1.0,
        'wind_gusts': True, 'drafting': True,
    },
    'roadsim-physics-v2-b2': {
        'engine_thermal': 'v2', 'oil_viscosity': 'vogel', 'cold_enrich': 0.008, 'warmup_enrich': 0.004,
        'eng_eta_scale': 1.0, 'eng_lowload_W_per_L': 0.0, 'eng_dfmep_bar': 0.0, 'eng_family_adj': {}, 'fead_drag_Nm': 0.0,
        'axle_cold': True, 'tire_thermal': True, 'tire_temp_coef': 0.008, 'tire_dyno_frac': 0.6,
        'hvac_model': 'v2', 'hvac_T_set': 21.5, 'hvac_recirc_heat': 0.3, 'hvac_recirc_cool': 0.8,
        'hvac_seat_heat': False, 'hvac_defrost': False, 'ambient_rh': 0.4, 'phev_electric_heat': True,
        'bus_aux': True, 'motor_staging': 2, 'motor_size_exp': 0.1,
        'batt_soh_capacity': 1.0, 'batt_soh_resistance': 1.0, 'batt_age': None, 'batt_soc_window': None, 'batt_heat_T_on': -10.0,
        'thermal_params': {},
    },
}


# the physics simulate() runs when no options are given (options=None is always physics v1)
DEFAULT_PHYSICS = 'roadsim-physics-v2'


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


def _load_v2():
    """the v2 preset: option values (fitted numbers included) from data/calibration_v2.json."""
    import json, os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'calibration_v2.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        pre = json.load(f)['preset']
    bad = sorted(set(pre) - set(DEFAULTS))
    if bad:
        raise KeyError(f'calibration_v2.json: unknown RoadSim options {bad}')
    return pre


_v2 = _load_v2()
if _v2 is not None:
    PRESETS['roadsim-physics-v2'] = _v2
