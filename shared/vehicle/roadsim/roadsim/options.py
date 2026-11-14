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

    # ---- end dynamics
}

PRESETS = {
    'roadsim-physics-v1': {},
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
