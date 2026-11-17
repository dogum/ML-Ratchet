"""Simulated fault bank: random BuildingSim buildings with an air handler (packaged RTU, VAV with chiller/boiler or
DX, VAV on district energy) x 17 TMYx cities x any season, each window simulated twice (healthy and with one fault
from day 5 on). Every local day becomes a sample: raw trend channels [13, 96] (float16), class, fault parameter,
energy-waste fraction (faulted vs the same day healthy), and the visible effect size.

Usage: python3 simbank.py <out.npz> <n_windows> <seed> [calib.json]
Resumable: writes shards every 100 windows to <out>.parts/ and concatenates at the end.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, RI, NT, CI, SIM2CLASS, sim_raw, to_day, hvac_energy  # noqa: E402
import buildingsim as bs  # noqa: E402

CITIES = sorted(bs.weather.EPW_CLIMATE_ZONE.keys()) if hasattr(bs, 'weather') else None
AHU_SYSTEMS = ('packaged_rtu', 'vav_chiller_boiler', 'district_steam_chw')
WIN_DAYS = 12          # local days kept per window (fault from day FAULT_DAY on)
FAULT_DAY = 5
_WX = {}


def weather(city):
    if city not in _WX:
        _WX[city] = bs.read_epw(bs.epw_path(city))
    return _WX[city]


def sample_fault(rng, system):
    """One fault dict (type + parameter) drawn for the system. Returns (fault dict, class name, param target)."""
    vav = system != 'packaged_rtu'
    types = ['oa_damper_stuck', 'oa_damper_leak', 'economizer_disabled', 'economizer_high_limit',
             'cooling_valve_stuck', 'cooling_valve_leak', 'heating_valve_stuck', 'heating_valve_leak',
             'coil_fouling', 'condenser_fouling', 'low_airflow', 'sensor_bias', 'sensor_bias', 'sensor_bias',
             'schedule_override', 'setpoint_override', 'boiler_fouling', 'chiller_fouling']
    if vav:
        types += ['reheat_valve_leak', 'sat_setpoint_offset', 'vav_box_stuck']
    else:
        types += ['refrigerant_charge', 'refrigerant_charge']
    t = types[rng.integers(len(types))]
    U = rng.uniform
    f = dict(type=t)
    if t in ('oa_damper_stuck', 'cooling_valve_stuck', 'heating_valve_stuck'):
        r = rng.random()
        p = 0.0 if r < 0.25 else (1.0 if r < 0.5 else round(U(0.05, 0.95), 2))
        f['position'] = p
        par = p
    elif t in ('oa_damper_leak', 'cooling_valve_leak', 'heating_valve_leak', 'reheat_valve_leak'):
        f['leak'] = par = round(U(0.05, 0.6), 2)
    elif t == 'economizer_high_limit':
        f['high_limit'] = round(U(5, 14), 1)
        par = 1.0
    elif t == 'economizer_disabled':
        par = 1.0
    elif t in ('coil_fouling', 'condenser_fouling', 'low_airflow', 'boiler_fouling', 'chiller_fouling'):
        hi = {'coil_fouling': 0.7, 'condenser_fouling': 1.0, 'low_airflow': 0.55, 'boiler_fouling': 0.4,
              'chiller_fouling': 1.0}[t]
        f['severity'] = par = round(U(0.15, hi), 2)
    elif t == 'refrigerant_charge':
        f['charge'] = c = round(U(-0.45, -0.1) if rng.random() < 0.7 else U(0.1, 0.35), 2)
        par = abs(c)
    elif t == 'sensor_bias':
        s = ['T_zone', 'T_sup', 'T_mix', 'T_out', 'T_ret'][rng.integers(5)]
        b = round(U(1.0, 5.0) * (1 if rng.random() < 0.5 else -1), 2)
        f.update(sensor=s, bias=b)
        par = b / 5
        return f, SIM2CLASS['sensor_bias:' + s], par
    elif t == 'schedule_override':
        f['mode'] = ['hvac_always_on', 'always_on', 'hvac_off', 'hvac_always_on'][rng.integers(4)]
        par = 1.0
    elif t == 'setpoint_override':
        d = U(1.5, 4.0) * (1 if rng.random() < 0.5 else -1)
        f['heat'] = None
        f['cool'] = None
        par = d / 4
        f['_shift'] = d
    elif t == 'sat_setpoint_offset':
        f['offset'] = round(U(1.5, 5.0) * (1 if rng.random() < 0.5 else -1), 2)
        par = f['offset'] / 5
    elif t == 'vav_box_stuck':
        f['position'] = par = round(U(0.1, 1.0), 2)
    else:
        par = 1.0
    return f, SIM2CLASS[t], par


def randomize_controls(spec, rng, calib):
    """Control-sequence variety (spec knobs only; physics untouched). calib may widen/shift priors (dev calibration)."""
    h = spec.setdefault('hvac', {})
    v = spec.setdefault('ventilation', {})
    c = calib or {}
    lo, hi = c.get('sat_range', (11.5, 14.5))
    h['sat_cool'] = float(rng.uniform(lo, hi))
    h['sat_reset'] = bool(rng.random() < c.get('p_sat_reset', 0.6))
    h['vav_min_frac'] = float(rng.uniform(*c.get('vav_min_range', (0.2, 0.5))))
    if 'fan_W_scale' in c:
        pass
    h['chw_supply_C'] = float(rng.uniform(5.5, 8.0))
    v['econ_high_limit'] = float(rng.uniform(18.0, 24.0))
    if rng.random() < c.get('p_no_econ', 0.15):
        v['economizer'] = False
    if 'oa_scale' in c:
        sc = float(np.exp(rng.uniform(np.log(c['oa_scale'][0]), np.log(c['oa_scale'][1]))))
        v['L_s_m2'] = (v.get('L_s_m2') or 0.3) * sc
        v['L_s_person'] = (v.get('L_s_person') or 2.5) * sc
    if rng.random() < c.get('p_24x7', 0.0):
        spec['occupancy_schedule'] = {'type': '24x7', 'day': (7, 19), 'night_occ': 0.3, 'night_light': 0.4,
                                      'night_plug': 0.7, 'weekend': 0.6}
    return spec


class Operator:
    """Real-operation irregularity (dev calibration, from the Korean AHUs and the dev days): every local day the
    air handler starts and stops a little early or late (normal, sd in 15-min bins) and sometimes is switched off
    for a block of 1-3 h. A 'schedule_shift' fault moves start or stop by 2-4 h. Applied through BuildingSim's
    controller hook (hvac_on), so the zone physics responds."""

    def __init__(self, seed, tz, sd_start=2.0, sd_stop=3.0, p_block=0.15, shift=None, shift_from=None):
        self.seed, self.tz = seed, tz
        self.sd_start, self.sd_stop, self.p_block = sd_start, sd_stop, p_block
        self.shift, self.shift_from = shift, shift_from
        self.mod = None

    def _build(self, sim):
        on = np.asarray(sim.hvac_on, float).copy()
        t = pd.DatetimeIndex(sim.times).tz_convert(self.tz).tz_localize(None)
        day = (t.normalize() - t.normalize()[0]).days.to_numpy()
        rng = np.random.default_rng(self.seed)
        mod = on.copy()
        for d in np.unique(day):
            ii = np.where(day == d)[0]
            o = on[ii] > 0.5
            if not o.any() or o.all():
                continue
            a, b = ii[np.argmax(o)], ii[len(o) - 1 - np.argmax(o[::-1])]
            s1 = int(round(rng.normal(0, self.sd_start)))
            s2 = int(round(rng.normal(0, self.sd_stop)))
            if self.shift is not None and t[ii[0]] >= self.shift_from:
                if self.shift[0] == 'start':
                    s1 = -self.shift[1]
                else:
                    s2 = self.shift[1]
            a2, b2 = max(ii[0], a + s1), min(ii[-1], b + s2)
            mod[ii] = 0.0
            mod[a2:b2 + 1] = 1.0
            if rng.random() < self.p_block and b2 - a2 > 16:
                c = rng.integers(a2 + 4, b2 - 8)
                mod[c:c + rng.integers(4, 12)] = 0.0
        self.mod = mod.tolist()

    def __call__(self, obs):
        if self.mod is None:
            self._build(obs['sim'])
        return {'hvac_on': self.mod[obs['k']] > 0.5}


def log_commands(xh, xf, f, rng):
    """BMS trend logs record CONTROL SIGNALS, not actuator positions (twin test on dev: FLEXLAB heating valve
    stuck at 100 % logs a 0 % command; a 40 % leak logs 0 %). BuildingSim logs the actual position, so for stuck
    and leaking actuators the logged signal is rebuilt: stuck -> the healthy command, wound up to 0 or 1 when the
    loop cannot reach its target (healthy command beyond the stuck position); leak -> faulty signal minus the
    leak. With probability 0.2 the actual position is kept (units with position feedback)."""
    t = f['type']
    if rng.random() < 0.2:
        return xf
    xf = xf.copy()
    ch = {'oa_damper_stuck': 'damper_oa', 'cooling_valve_stuck': 'valve_cc', 'heating_valve_stuck': 'valve_hc',
          'oa_damper_leak': 'damper_oa', 'cooling_valve_leak': 'valve_cc', 'heating_valve_leak': 'valve_hc'}.get(t)
    if ch is None:
        return xf
    i = RI[ch]
    on = np.nan_to_num(xf[RI['fan_on']]) > 0.5
    if t.endswith('_stuck'):
        p = f['position']
        h = xh[i]
        if ch == 'damper_oa':
            cmd = h.copy()
        else:
            cmd = np.where(h > p + 0.05, 1.0, np.where(h < p - 0.05, 0.0, h))
        xf[i] = np.where(on, cmd, xf[i])
    else:
        xf[i] = np.where(on, np.clip(xf[i] - f['leak'], 0, 1), xf[i])
    return xf


def effect(xh, xf):
    """Visible effect size of a fault on one day: max over channel groups of mean |faulted - healthy|, scaled
    (temperatures / 0.3 K, actuators / 0.03, fan power / 3 % of its p95). >= 1 means visible."""
    T = [RI[c] for c in ('T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_zone')]
    A = [RI[c] for c in ('damper_oa', 'valve_cc', 'valve_hc', 'fan_speed', 'fan_on')]
    dT = np.nanmean(np.abs(xf[T] - xh[T]), axis=1)
    dA = np.nanmean(np.abs(xf[A] - xh[A]), axis=1)
    p95 = np.nanpercentile(xh[RI['fan_power']], 95) if np.isfinite(xh[RI['fan_power']]).any() else 0
    dP = np.nanmean(np.abs(xf[RI['fan_power']] - xh[RI['fan_power']])) / max(p95, 1e-6) if p95 > 0 else 0.0
    return float(np.nanmax([np.nanmax(dT) / 0.3, np.nanmax(dA) / 0.03, dP / 0.03]))


def one_window(rng, specs_pool, calib):
    spec = dict(specs_pool[rng.integers(len(specs_pool))])
    spec = json.loads(json.dumps(spec))
    spec = randomize_controls(spec, rng, calib)
    cities = list(bs.weather.EPW_CLIMATE_ZONE.keys())
    city = cities[rng.integers(len(cities))]
    w = weather(city)
    tz = w[1]['tz']
    doy = int(rng.integers(0, 365 - WIN_DAYS - 3))
    d0 = pd.Timestamp('2019-01-01') + pd.Timedelta(days=doy)          # first local day kept
    start = (d0 - pd.Timedelta(days=1)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    end = (d0 + pd.Timedelta(days=WIN_DAYS + 1)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    f, cls, par = sample_fault(rng, spec['system'])
    fstart = (d0 + pd.Timedelta(days=FAULT_DAY)).tz_localize(tz).tz_convert('UTC').tz_localize(None)
    fd = {k: v for k, v in f.items() if not k.startswith('_')}
    fd['start'] = str(fstart)
    c = calib or {}
    shift = None
    if c.get('schedule_shift') and rng.random() < 0.06:
        f, cls, par = dict(type='schedule_shift'), 'schedule', 1.0
        shift = (['start', 'stop'][rng.integers(2)], int(rng.integers(8, 17)))
    if f['type'] == 'setpoint_override':
        from buildingsim.spec import fill_spec
        fs = fill_spec(json.loads(json.dumps(spec)))
        sp = fs['setpoints']
        fd['heat'] = sp['heat'] + f['_shift']
        fd['cool'] = max(sp['cool'] + f['_shift'], fd['heat'] + 1.0)
    ctl_h = ctl_f = None
    if c.get('operator'):
        op = c['operator']
        oseed = int(rng.integers(1 << 30))
        ctl_h = Operator(oseed, tz, **op)
        ctl_f = Operator(oseed, tz, **op, shift=shift, shift_from=d0 + pd.Timedelta(days=FAULT_DAY))
    h = bs.simulate(spec, w, str(start), str(end), dt='15min', controller=ctl_h)
    if f['type'] == 'schedule_shift':
        fo = bs.simulate(spec, w, str(start), str(end), dt='15min', controller=ctl_f)
    else:
        fo = bs.simulate(spec, w, str(start), str(end), dt='15min', faults=[fd], controller=ctl_f)
    rh, rf = sim_raw(h, tz), sim_raw(fo, tz)
    eh, ef = hvac_energy(h, tz), hvac_energy(fo, tz)
    rows = []
    for k in range(WIN_DAYS):
        day = d0 + pd.Timedelta(days=k)
        xh = to_day(rh, day)
        sel = (eh.index >= day) & (eh.index < day + pd.Timedelta(days=1))
        Eh = float(eh[sel].sum() * 0.25)
        meta = dict(system=spec['system'], use=spec['primary_use'], city=city, month=int(day.month),
                    weekday=int(day.dayofweek), area=float(spec.get('floor_area_m2') or 0))
        if k >= FAULT_DAY:
            xf = to_day(rf, day)
            if c.get('log_commands'):
                xf = log_commands(xh, xf, f, rng)
            Ef = float(ef[sel].sum() * 0.25)
            waste = (Ef - Eh) / max(Eh, 1e-3 * spec.get('floor_area_m2', 1000) * 0.0 + 1.0)
            rows.append(dict(x=xf, cls=CI[cls], par=par, waste=waste, eff=effect(xh, xf), ftype=f['type'],
                             label=fo['trend'].fault.iloc[-1] if f['type'] != 'schedule_shift' else f'schedule_shift:{shift}', E=Ef, **meta))
            if k >= WIN_DAYS - 2:          # keep a few healthy days of the same window as normals too
                rows.append(dict(x=xh, cls=0, par=0.0, waste=0.0, eff=0.0, ftype='none', label='', E=Eh, **meta))
        else:
            if k >= 1:
                rows.append(dict(x=xh, cls=0, par=0.0, waste=0.0, eff=0.0, ftype='none', label='', E=Eh, **meta))
    return rows


def build(out, n_windows, seed, calib=None):
    parts = out + '.parts'
    os.makedirs(parts, exist_ok=True)
    rng_pool = np.random.default_rng(seed)
    specs = [s for s in bs.generate_buildings(max(400, n_windows // 4), seed=seed) if s['system'] in AHU_SYSTEMS]
    print('specs', len(specs), flush=True)
    t0 = time.time()
    for p0 in range(0, n_windows, 100):
        fn = f'{parts}/{p0:06d}.npz'
        if os.path.exists(fn):
            continue
        rows = []
        for i in range(p0, min(p0 + 100, n_windows)):
            rng = np.random.default_rng([seed, i])
            try:
                rows += one_window(rng, specs, calib)
            except Exception as e:  # noqa: BLE001  (a failed window is logged and skipped)
                print('window', i, 'failed:', repr(e)[:200], flush=True)
        save_rows(rows, fn)
        print(f'{p0 + 100} windows, {time.time() - t0:.0f} s', flush=True)
    files = sorted(f for f in os.listdir(parts) if f.endswith('.npz'))
    allp = [dict(np.load(f'{parts}/{f}', allow_pickle=True)) for f in files]
    cat = {k: np.concatenate([a[k] for a in allp]) for k in allp[0]}
    np.savez_compressed(out, **cat)
    print('wrote', out, len(cat['cls']), flush=True)


def save_rows(rows, fn):
    if not rows:
        return
    d = dict(x=np.stack([r['x'] for r in rows]).astype(np.float16),
             cls=np.array([r['cls'] for r in rows], np.int16), par=np.array([r['par'] for r in rows], np.float32),
             waste=np.array([r['waste'] for r in rows], np.float32), eff=np.array([r['eff'] for r in rows], np.float32),
             E=np.array([r['E'] for r in rows], np.float32),
             ftype=np.array([r['ftype'] for r in rows]), label=np.array([r['label'] for r in rows]),
             system=np.array([r['system'] for r in rows]), use=np.array([r['use'] for r in rows]),
             city=np.array([r['city'] for r in rows]), month=np.array([r['month'] for r in rows], np.int8),
             weekday=np.array([r['weekday'] for r in rows], np.int8))
    np.savez_compressed(fn, **d)


if __name__ == '__main__':
    calib = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
    build(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), calib)
