"""OBD log emulator: turns a RoadSim run into what a VED-style logger would have written, then processes it with the
same 1 Hz resampling RoadData used on the real logs.

Since the Phase D rebuild (14 Nov 2026) the emulator lives upstream in RoadSim (shared/vehicle/roadsim/roadsim/obd.py,
EMU_VERSION 1). It is this file's v1.0 code moved there unchanged (per-PID polling clocks measured on the 16 dev cars'
raw rows, held values, quantisation, MAF fuel estimate that misses enrichment and shows motoring air in fuel cut, trims,
dropouts, RoadData's resampler). This module only re-exports it, so the banks, the twin tests and the lab all use the
one upstream copy.

Added here (TripDetective-specific, a logger rung): logger_config_ved(), the VED channel mix measured on the 16 dev
cars (which channels a VED logger records for each powertrain type)."""
from common import *
from roadsim.obd import EMU_VERSION, EMU_CAL as CAL, held_interp, logger_config, emulate  # noqa: F401  (upstream copy)

# VED channel mix by powertrain, measured on the 16 dev cars' trips (realbank dev flags, 14 Nov 2026): the 12
# conventional cars and 5 HEVs log fuel (11 MAF, 1 absolute load) and rpm and never an HV battery; the 2 plug-ins log
# fuel (Energi: MAF; Volt-type: the OBD fuel-rate PID), rpm, HV battery and A/C / heater power; the Leaf logs battery
# and A/C / heater power. In VED a battery channel means 'plug-in'. A little slack is left on every share.
VED_MIX = {
    'conv': dict(fuel=(['maf', 'absload', 'pid', None], [0.85, 0.10, 0.02, 0.03]), rpm=0.97, batt=0.0, aux=0.0),
    'hev': dict(fuel=(['maf', 'absload', 'pid', None], [0.85, 0.10, 0.02, 0.03]), rpm=0.97, batt=0.03, aux=0.0),
    'phev': dict(fuel=(['maf', 'absload', 'pid', None], [0.47, 0.03, 0.47, 0.03]), rpm=0.95, batt=0.97, aux=0.9),
    'bev': dict(fuel=([None], [1.0]), rpm=0.0, batt=0.97, aux=0.9),
}


def logger_config_ved(arch, rng, fuel_src=None, p_generic=0.25):
    """channel set of a VED-style logger for this powertrain (VED_MIX); with probability p_generic the car keeps the
    generic v1 mix instead (logger_config), so the network still sees other channel sets."""
    if rng.random() < p_generic:
        return logger_config(arch, rng, fuel_src)
    c = logger_config(arch, rng, fuel_src)          # cadences, trims, dropouts as v1 (same random draws first)
    k = 'conv' if arch == 'conv' else ('bev' if arch == 'bev' else ('phev' if arch.startswith('phev') else 'hev'))
    mix = VED_MIX[k]
    fs, p = mix['fuel']
    c['fuel'] = fuel_src or fs[int(rng.choice(len(fs), p=p))]
    c['rpm'] = bool(rng.random() < mix['rpm']) or c['fuel'] in ('maf', 'absload')
    c['batt'] = bool(arch != 'conv' and rng.random() < mix['batt'])
    c['aux'] = bool(c['batt'] and rng.random() < mix['aux'])
    return c
