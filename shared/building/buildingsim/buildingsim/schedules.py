"""Occupancy, lighting, plug, hot-water and HVAC-availability schedules (fractions of peak, local civil time).

Night base fractions were raised by +0.05 (lighting) and +0.10 (plug) after the ComStock calibration sample
(BuildingSim night/peak electricity ratio 0.33 vs ComStock 0.43). Commercial schedules are parametric (open and close hour per day type, base fractions), so they can be filled from
ComStock's operating-hours fields (in.weekday_opening_time, in.weekday_operating_hours, ...). Their shapes follow
the DOE commercial prototype schedules (ASHRAE 90.1 User's Manual / PNNL prototype models): occupancy ramps over
about an hour, a lunch dip, lighting and plug loads that never drop below a base fraction. Residential shapes follow
the Building America House Simulation Protocols (Wilson et al. 2014) in spirit: morning and evening peaks, low
weekday midday occupancy. 24/7 types (healthcare, lodging, lab) have a day/night swing. Values are smooth in time
so 15-minute steps see ramps, not hourly steps.
"""
import numpy as np


def _sig(x, w=0.5):
    return 1.0 / (1.0 + np.exp(-x / w))


def _window(h, open_h, close_h, w=0.5):
    """Smooth 0..1 occupied window; close_h may exceed 24 (after midnight)."""
    if open_h is None or close_h is None or close_h <= open_h:
        return np.zeros_like(h)
    hh = np.where(h < open_h - 4, h + 24, h) if close_h > 24 else h
    return _sig(hh - open_h, w) * _sig(close_h - hh, w)


def _gauss(h, mu, s):
    d = (h - mu + 12) % 24 - 12
    return np.exp(-0.5 * (d / s) ** 2)


DEFAULT_SCHEDULES = {
    'office': dict(type='commercial', weekday=(8, 18), saturday=(8, 13), sunday=None, occ_peak=0.95, lunch=0.5,
                   light_base=0.20, plug_base=0.60, dhw_base=0.05, early_start=2.0),
    'education': dict(type='commercial', weekday=(7.5, 16), saturday=None, sunday=None, occ_peak=0.95, lunch=0.8,
                      light_base=0.13, plug_base=0.40, dhw_base=0.05, early_start=2.0, summer_break=(172, 232)),
    'retail': dict(type='commercial', weekday=(9, 21), saturday=(9, 21), sunday=(10, 18), occ_peak=0.7, lunch=1.0,
                   light_base=0.15, plug_base=0.40, dhw_base=0.05, early_start=1.5),
    'food': dict(type='commercial', weekday=(7, 23), saturday=(7, 23), sunday=(8, 22), occ_peak=0.8, lunch=1.2,
                 light_base=0.20, plug_base=0.45, dhw_base=0.10, early_start=1.0),
    'assembly': dict(type='commercial', weekday=(9, 22), saturday=(9, 22), sunday=(9, 18), occ_peak=0.5, lunch=1.0,
                     light_base=0.15, plug_base=0.40, dhw_base=0.05, early_start=1.5),
    'warehouse': dict(type='commercial', weekday=(7, 17), saturday=None, sunday=None, occ_peak=0.9, lunch=0.8,
                      light_base=0.20, plug_base=0.60, dhw_base=0.05, early_start=1.0),
    'other': dict(type='commercial', weekday=(8, 18), saturday=None, sunday=None, occ_peak=0.9, lunch=0.7,
                  light_base=0.20, plug_base=0.60, dhw_base=0.05, early_start=1.5),
    'healthcare': dict(type='24x7', day=(7, 19), night_occ=0.5, night_light=0.5, night_plug=0.7, weekend=0.8),
    'lodging': dict(type='lodging'),
    'lab': dict(type='24x7', day=(7, 19), night_occ=0.15, night_light=0.4, night_plug=0.8, weekend=0.5),
    'residential_single': dict(type='residential', wake=6.5, leave=8.0, back=17.5, sleep=22.5, home_day=0.35),
    'residential_multi': dict(type='residential', wake=6.5, leave=8.0, back=17.5, sleep=23.0, home_day=0.40),
}


# Residential non-HVAC electricity shape (plug + appliances + lighting), mean of 147 occupied ResStock 2025 R1
# dwellings in 2018 (agent B's sample), normalised to mean 1; rows weekday / weekend, hours 0..23 local time.
RES_PLUG_SHAPE = [[0.965, 0.763, 0.637, 0.579, 0.570, 0.600, 0.663, 0.742, 0.836, 0.890, 0.921, 0.932, 0.924,
                   0.911, 0.913, 0.940, 1.007, 1.110, 1.247, 1.404, 1.527, 1.525, 1.396, 1.190],
                  [0.980, 0.770, 0.645, 0.579, 0.559, 0.573, 0.613, 0.689, 0.803, 0.903, 0.983, 1.032, 1.066,
                   1.080, 1.098, 1.137, 1.181, 1.265, 1.390, 1.524, 1.622, 1.605, 1.464, 1.246]]
RES_PLUG_MONTH = [1.073, 1.073, 1.011, 0.988, 0.978, 0.956, 0.948, 0.947, 0.969, 0.976, 0.998, 1.083]
RES_PLUG_PEAK = 1.622          # max of the shape (x monthly factor ~1.08 -> schedule may reach ~1)


def _res_plug(h, dtp, month):
    hh = np.asarray(h, float)
    i0 = np.floor(hh).astype(int) % 24
    i1 = (i0 + 1) % 24
    w = hh - np.floor(hh)
    out = np.empty(len(hh))
    for d in (0, 1):
        m = (dtp > 0) == bool(d)
        sh = np.asarray(RES_PLUG_SHAPE[d])
        out[m] = (1 - w[m]) * sh[i0[m]] + w[m] * sh[i1[m]]
    return out * np.asarray(RES_PLUG_MONTH)[np.asarray(month) - 1] / (RES_PLUG_PEAK * 1.083)


def build(sched, cal):
    """Evaluate a schedule spec on a local calendar (dict of hour, daytype, doy arrays). Returns dict of arrays:
    occ, light, plug, dhw (fractions of peak, 0..1) and hvac (1 = occupied mode incl. early start)."""
    t = sched.get('type', 'commercial')
    h, dtp, doy = cal['hour'], cal['daytype'], cal['doy']
    n = len(h)
    if t == 'commercial':
        occ = np.zeros(n)
        hv = np.zeros(n)
        for d, key in ((0, 'weekday'), (1, 'saturday'), (2, 'sunday')):
            win = sched.get(key)
            m = dtp == d
            if win is None or not m.any():
                continue
            o, c = float(win[0]), float(win[1])
            w = _window(h[m], o, c, 0.35)
            mid = 0.5 * (o + c)
            dip = 1.0 - (1.0 - sched.get('lunch', 1.0)) * _gauss(h[m], 12.5, 0.45) if c - o > 6 else 1.0
            occ[m] = sched.get('occ_peak', 0.9) * w * dip
            es = sched.get('early_start', 1.5)
            hv[m] = ((h[m] >= o - es) & (h[m] < c + 0.5)).astype(float)
            del mid
        sb = sched.get('summer_break')
        if sb is not None:
            br = (doy >= sb[0]) & (doy <= sb[1])
            occ[br] *= 0.15
        act = np.clip(occ / max(sched.get('occ_peak', 0.9), 1e-6), 0, 1)
        lb, pb = sched.get('light_base', 0.1), sched.get('plug_base', 0.4)
        light = lb + (0.9 - lb) * act
        plug = pb + (0.9 - pb) * act
        dhw = sched.get('dhw_base', 0.05) + (1 - sched.get('dhw_base', 0.05)) * act * \
            (0.6 + 0.4 * _gauss(h, 12.0, 1.5))
        return dict(occ=occ, light=light, plug=plug, dhw=dhw, hvac=hv)
    if t == '24x7':
        o, c = sched.get('day', (7, 19))
        wk = np.where(dtp == 0, 1.0, sched.get('weekend', 0.8))
        day = _window(h, o, c, 0.6)
        occ = (sched.get('night_occ', 0.5) + (1 - sched.get('night_occ', 0.5)) * day * wk) * 0.9
        light = sched.get('night_light', 0.5) + (0.9 - sched.get('night_light', 0.5)) * day * wk
        plug = sched.get('night_plug', 0.7) + (0.9 - sched.get('night_plug', 0.7)) * day * wk
        dhw = 0.3 + 0.7 * day * wk
        return dict(occ=occ, light=light, plug=plug, dhw=dhw, hvac=np.ones(n))
    if t == 'lodging':
        # guests out during the day, in at night; lighting peaks in the evening
        occ = 0.85 - 0.55 * _window(h, 9.5, 16.5, 0.7)
        light = 0.15 + 0.5 * _gauss(h, 20.0, 2.0) + 0.25 * _gauss(h, 7.0, 1.0)
        plug = 0.35 + 0.35 * _gauss(h, 20.0, 2.5) + 0.2 * _gauss(h, 7.5, 1.2)
        dhw = 0.08 + 0.85 * _gauss(h, 7.5, 1.3) + 0.45 * _gauss(h, 20.5, 2.0)
        return dict(occ=occ, light=np.clip(light, 0, 1), plug=np.clip(plug, 0, 1), dhw=np.clip(dhw, 0, 1),
                    hvac=np.ones(n))
    if t == 'residential':
        wk = dtp > 0
        wake, leave, back, sleep = (sched.get(k) for k in ('wake', 'leave', 'back', 'sleep'))
        away = _window(h, leave, back, 0.6) * np.where(wk, 0.5, 1.0)
        home = 1.0 - (1.0 - sched.get('home_day', 0.35)) * away
        asleep = _window(h, sleep, wake + 24, 0.5) + _window(h, -1, wake, 0.5)
        awake_home = np.clip(home - asleep, 0, 1)
        occ = np.clip(home, 0, 1)
        light = 0.04 + 0.30 * _gauss(h, wake + 0.75, 0.8) + 0.65 * _gauss(h, 0.5 * (back + sleep) + 0.8, 1.8) \
            + 0.08 * awake_home
        plug = _res_plug(h, dtp, cal['month'])
        dhw = 0.03 + 0.95 * _gauss(h, wake + 0.8 + 1.5 * wk, 1.0) + 0.55 * _gauss(h, 20.0, 1.8) + 0.15 * \
            _gauss(h, 13.0, 2.5) * wk
        # seasonal lighting: longer evenings in winter (+-15 %)
        light = light * (1 + 0.15 * np.cos(2 * np.pi * (doy - 15) / 365.0))
        return dict(occ=occ, light=np.clip(light, 0, 1), plug=np.clip(plug, 0, 1.05), dhw=np.clip(dhw, 0, 1),
                    hvac=np.ones(n), asleep=np.clip(asleep, 0, 1), away=np.clip(away, 0, 1))
    raise ValueError(f'unknown schedule type {t}')
