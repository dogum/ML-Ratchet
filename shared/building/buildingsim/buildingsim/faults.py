"""HVAC fault injection for FaultFinder.

`faults` is a list of dicts: {'type': <name>, 'start': timestamp or None, 'end': timestamp or None, **params}.
compile_faults() turns it into per-step arrays (NaN = inactive) that the systems read. Every step also gets a
label (out['ts']['fault'], '' = healthy) so trend logs come with ground truth.

Fault models (what each does to the physics):
- oa_damper_stuck(position)      OA damper fixed at `position` (0..1) whenever the supply fan runs.
- oa_damper_leak(leak)            closed/minimum damper still passes `leak` (fraction of supply) of outdoor air.
- economizer_disabled()           no free cooling.
- economizer_high_limit(high_limit) wrong changeover temperature (C).
- cooling_valve_stuck(position)   chilled-water valve fixed: coil load = position x design UA-like capacity;
                                  DX units: compressor stuck at runtime fraction `position` when the fan runs.
- cooling_valve_leak(leak)        valve passes `leak` x design capacity when commanded closed (heating coil
                                  compensates to hold supply temperature: simultaneous heating and cooling).
- heating_valve_stuck(position), heating_valve_leak(leak)   same for the heating / preheat coil.
- reheat_valve_leak(leak)         VAV box reheat passes `leak` x reheat capacity.
- coil_fouling(severity)          cooling-coil capacity x (1 - severity); supply airflow x (1 - severity/2)
                                  (air-side pressure drop), fan power x (1 + severity/2) for VAV/CV fans.
- refrigerant_charge(charge)      charge = -0.3 means 30 % undercharged, +0.2 overcharged. DX capacity and EIR
                                  multipliers approximate the trends in Kim & Braun (2012, Int. J. Refrig. 35):
                                  under u: cap x (1 - 0.25 u - 0.6 u^2), EIR x (1 + 0.15 u + 0.6 u^2);
                                  over o: cap x (1 - 0.1 o), EIR x (1 + 0.3 o).
- condenser_fouling(severity)     condenser airflow loss: condensing temperature + 8 x severity K (DX);
                                  chiller condenser approach + 5 x severity K.
- low_airflow(severity)           filter clogging / belt slip: supply airflow x (1 - severity).
- sensor_bias(sensor, bias)       sensor in T_zone, T_sup, T_mix, T_out, T_ret: control uses and the trend log
                                  reports the biased value.
- schedule_override(mode)         'hvac_always_on', 'lights_always_on', 'always_on', 'hvac_off'.
- setpoint_override(heat, cool)   occupied setpoints replaced.
- sat_setpoint_offset(offset)     VAV supply-air setpoint shifted by offset K.
- vav_box_stuck(position)         VAV terminal flow fixed at fraction `position` of maximum.
- boiler_fouling(severity)        boiler efficiency x (1 - severity).
- chiller_fouling(severity)       chiller condenser + 5 x severity K.
"""
import numpy as np
import pandas as pd

FAULT_TYPES = ('oa_damper_stuck', 'oa_damper_leak', 'economizer_disabled', 'economizer_high_limit',
               'cooling_valve_stuck', 'cooling_valve_leak', 'heating_valve_stuck', 'heating_valve_leak',
               'reheat_valve_leak', 'coil_fouling', 'refrigerant_charge', 'condenser_fouling', 'low_airflow',
               'sensor_bias', 'schedule_override', 'setpoint_override', 'sat_setpoint_offset', 'vav_box_stuck',
               'boiler_fouling', 'chiller_fouling')

_PARAM = {'oa_damper_stuck': 'position', 'oa_damper_leak': 'leak', 'economizer_disabled': None,
          'economizer_high_limit': 'high_limit', 'cooling_valve_stuck': 'position', 'cooling_valve_leak': 'leak',
          'heating_valve_stuck': 'position', 'heating_valve_leak': 'leak', 'reheat_valve_leak': 'leak',
          'coil_fouling': 'severity', 'refrigerant_charge': 'charge', 'condenser_fouling': 'severity',
          'low_airflow': 'severity', 'sat_setpoint_offset': 'offset', 'vav_box_stuck': 'position',
          'boiler_fouling': 'severity', 'chiller_fouling': 'severity'}


def compile_faults(faults, times):
    """Returns (dict name -> np.array of parameter (NaN inactive), labels array of str)."""
    n = len(times)
    out = {}
    labels = np.full(n, '', dtype=object)
    for f in faults or []:
        t = f['type']
        if t not in FAULT_TYPES:
            raise ValueError(f'unknown fault {t}; known: {FAULT_TYPES}')
        m = np.ones(n, bool)
        if f.get('start') is not None:
            s = pd.Timestamp(f['start'])
            s = s.tz_localize('UTC') if s.tz is None else s
            m &= times >= s
        if f.get('end') is not None:
            e = pd.Timestamp(f['end'])
            e = e.tz_localize('UTC') if e.tz is None else e
            m &= times < e
        if t == 'sensor_bias':
            key = 'bias_' + f['sensor']
            arr = out.setdefault(key, np.full(n, np.nan))
            arr[m] = float(f['bias'])
            lab = f"sensor_bias:{f['sensor']}:{f['bias']:+g}"
        elif t == 'schedule_override':
            arr = out.setdefault('schedule_override', np.full(n, np.nan))
            code = {'hvac_always_on': 1, 'lights_always_on': 2, 'always_on': 3, 'hvac_off': 4}[f['mode']]
            arr[m] = code
            lab = f"schedule_override:{f['mode']}"
        elif t == 'setpoint_override':
            for kk in ('heat', 'cool'):
                if f.get(kk) is not None:
                    arr = out.setdefault('sp_' + kk, np.full(n, np.nan))
                    arr[m] = float(f[kk])
            lab = f"setpoint_override:{f.get('heat')}/{f.get('cool')}"
        else:
            p = _PARAM[t]
            val = 1.0 if p is None else float(f[p])
            arr = out.setdefault(t, np.full(n, np.nan))
            arr[m] = val
            lab = t if p is None else f'{t}:{val:g}'
        labels[m] = np.where(labels[m] == '', lab, labels[m] + '|' + lab)
    return out, labels


def charge_multipliers(charge):
    """(capacity, EIR) multipliers for a refrigerant charge deviation (negative = undercharge)."""
    if charge != charge or charge == 0:
        return 1.0, 1.0
    if charge < 0:
        u = -charge
        return 1.0 - 0.25 * u - 0.6 * u * u, 1.0 + 0.15 * u + 0.6 * u * u
    o = charge
    return 1.0 - 0.1 * o, 1.0 + 0.3 * o


class FaultView:
    """Fast per-step access: f.get(name, k) -> float or None."""

    def __init__(self, arrays):
        self.a = {k: v.tolist() for k, v in arrays.items()}

    def get(self, name, k):
        a = self.a.get(name)
        if a is None:
            return None
        v = a[k]
        return None if v != v else v

    def has(self, name):
        return name in self.a
