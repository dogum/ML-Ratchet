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
    'roadsim-physics-v2-b3': {
        'substeps': 10, 'tyre_model': True, 'tyre_mu_scale': 1.0, 'tyre_regen_mu_frac': 0.8, 'tyre_ref_eu_class': 'C',
        'driver_model': 'pi', 'wind_profile': 'log', 'wind_z0_ref_m': 0.03, 'wind_z0_site_m': 0.1, 'wind_shelter': 1.0,
        'wind_gusts': True, 'drafting': True,
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
