"""OBD log emulator: turns a RoadSim run into what a VED-style logger would have written, then processes it with the
same 1 Hz resampling RoadData used on the real logs.

Real-logger behaviour reproduced (cadences measured on the 16 dev cars' raw rows, data/dev_cadence.json):
  * each signal is polled on its own clock: speed ~1 s, MAF and rpm ~2-2.4 s, absolute load ~5 s, fuel-rate PID ~5 s,
    HV current ~1 s, HV voltage ~5 s; SOC and A/C power change only at their resolution;
  * values are held between updates and quantised (speed 1 km/h, rpm 1, MAF 0.01 g/s, current 0.1 A, voltage 0.5 V, SOC 0.5 %);
  * fuel is mostly ESTIMATED from the MAF (fuel = air x (1+trims)/14.08): it misses enrichment at high load, and
    during deceleration fuel cut-off it still shows the air the motoring engine pumps (not zero);
  * fuel = 0 when rpm < 100 (RoadData rule); dropouts (whole seconds missing) become NaN;
  * resampling to 1 Hz through value-change points, holding plateaus (RoadData stage 2, held_interp, vendored below).
Calibrated parameters (version EMU_VERSION) live in CAL; dev twin tests set them (calib_twin.py)."""
from common import *

EMU_VERSION = 1
CAL = {
    'cut_air_k': (0.4, 1.0),      # MAF-equivalent fuel during fuel cut = k x idle fuel x rpm/idle_rpm  (range drawn per car)
    'enrich_thr': 0.75,           # engine power / max power above which mixture is enriched
    'enrich_factor': 0.90,        # stoichiometric estimate / true fuel when enriched
    'maf_bias_sd': 0.04,          # per-car residual bias of the MAF x trims estimate
    'stft_sd': 0.03,              # fast trim noise
    'p_dropout': 0.25,            # probability a trip has dropouts
    'idle_scale': (1.0, 1.0),     # multiplier on idle fuel (twin rung)
    'idle_rpm_add': (0, 0),       # additive idle rpm (twin rung)
    'hev_eng_on_extra': 0.0,      # extra engine-on probability for HEV at low speed (twin rung)
}


def held_interp(t, v, grid, plateau=2.5):
    """RoadData stage-2 resampler (copied from road/data/scripts/ved_stage2.py)"""
    ok = np.isfinite(v)
    if ok.sum() == 0: return None
    t = t[ok]; v = v[ok]
    if len(v) == 1: return np.full(len(grid), v[0], np.float64)
    ch = np.r_[True, v[1:] != v[:-1]]
    keep = ch.copy()
    idx = np.where(ch)[0]
    nxt = np.r_[idx[1:], len(v)]
    for a, b in zip(idx, nxt):
        if t[b - 1] - t[a] > plateau: keep[b - 1] = True
    keep[-1] = True
    return np.interp(grid, t[keep], v[keep])


def logger_config(arch, rng, fuel_src=None):
    """which channels a car's logger records (fixed per car). Matches VED's mix loosely, plus generic OBD subsets."""
    elec = arch != 'conv'
    plug = arch.startswith('phev') or arch == 'bev'
    c = {}
    if arch == 'bev':
        c['fuel'] = None; c['rpm'] = False
    else:
        c['fuel'] = fuel_src or rng.choice(['maf', 'absload', 'pid', None], p=[0.62, 0.13, 0.12, 0.13])
        c['rpm'] = bool(rng.random() < 0.93) or c['fuel'] in ('maf', 'absload')
    c['batt'] = bool(elec and rng.random() < (0.85 if plug else 0.35))
    c['aux'] = bool(c['batt'] and plug and rng.random() < 0.7)
    c['cad_speed'] = float(rng.uniform(1.0, 1.15))
    c['cad_fast'] = float(rng.choice([2.0, 2.0, 2.1, 2.4]) + rng.uniform(0, 0.1))
    c['cad_slow'] = float(rng.uniform(5.0, 5.3))
    c['cut_k'] = float(rng.uniform(*CAL['cut_air_k']))
    c['maf_bias'] = float(rng.normal(0, CAL['maf_bias_sd']))
    c['idle_scale'] = float(rng.uniform(*CAL['idle_scale']))
    c['p_dropout'] = CAL['p_dropout']
    return c


def _poll(rng, n, cad):
    """update times of a polled signal on [0, n-1]"""
    t0 = rng.uniform(0, cad); k = int((n - 1 - t0) / cad) + 2
    t = t0 + np.arange(k) * cad + rng.normal(0, 0.08, k)
    t = np.sort(t[(t >= 0) & (t <= n - 1)])
    return np.r_[0.0, t, n - 1.0] if len(t) else np.array([0.0, n - 1.0])


def _sample(x, t):
    return np.interp(t, np.arange(len(x)), x)


def _logged(rng, x, cad, q, n):
    t = _poll(rng, n, cad)
    y = np.round(_sample(x, t) / q) * q
    return held_interp(t, y, np.arange(n, dtype=np.float64))


def emulate(out, cfg, rng, spec=None, fuel='gasoline'):
    """out: roadsim output dict (record=True). Returns dict of 1 Hz logged arrays (NaN where not logged) + valid."""
    n = len(out['v'])
    grid = np.arange(n, dtype=np.float64)
    res = {}
    # speed: integer km/h, ~1 s
    res['v'] = (_logged(rng, out['v'] * 3.6, cfg['cad_speed'], 1.0, n) / 3.6).astype(np.float32)
    rpm_true = np.asarray(out['eng_rpm'], np.float64)
    on = np.asarray(out['eng_on'], np.float64) > 0.5
    rpm_true = np.where(on, rpm_true, 0.0)
    rpm = None
    if cfg['rpm'] or cfg['fuel'] in ('maf', 'absload'):
        rpm = _logged(rng, rpm_true, cfg['cad_fast'], 1.0, n)
    fr = np.asarray(out['fuel_rate'], np.float64)
    fobs = None
    if cfg['fuel'] is not None and spec is not None and spec.get('engine'):
        if cfg['fuel'] == 'pid':
            fobs = _logged(rng, fr, cfg['cad_slow'], 0.01 * 0.745 / 3.6, n)   # L/h resolution 0.01
        else:
            # what the MAF sees, as stoichiometric fuel: true fuel, minus enrichment, plus motoring air in fuel cut
            pmax = 1000.0 * float(spec['engine'].get('max_power_kW') or 100.0)
            ep = np.asarray(out['eng_power'], np.float64)
            est = fr * np.where(ep > CAL['enrich_thr'] * pmax, CAL['enrich_factor'], 1.0)
            idle = on & (np.asarray(out['v']) < 0.2) & (fr > 0)
            idle_f = float(np.median(fr[idle])) if idle.sum() >= 5 else 0.10 * float(spec['engine'].get('displacement_L') or 2.0)
            idle_rpm = float(np.median(rpm_true[idle])) if idle.sum() >= 5 else 700.0
            est = np.where(idle, est * cfg['idle_scale'], est)
            cut = on & (fr <= 1e-4) & (rpm_true > 300)
            rr = rpm_true / max(idle_rpm, 400.0) if spec.get('architecture') == 'conv' else 1.0   # hybrids spin the engine slowly in cut
            est = np.where(cut, cfg['cut_k'] * idle_f * rr, est)
            # trims: per-car residual bias + slow AR noise
            from scipy.signal import lfilter
            ar = lfilter([1.0], [1.0, -0.9], rng.normal(0, CAL['stft_sd'] * 0.45, n))
            est = est * (1 + cfg['maf_bias'] + ar)
            cad = cfg['cad_fast'] if cfg['fuel'] == 'maf' else cfg['cad_slow']
            fobs = _logged(rng, np.maximum(est, 0), cad, 0.01 / 14.08, n)
        if rpm is not None:
            fobs = np.where(rpm < 100, 0.0, fobs)
    res['fuel_rate'] = (fobs if fobs is not None else np.full(n, np.nan)).astype(np.float32)
    res['engine_rpm'] = (rpm if (rpm is not None and cfg['rpm']) else np.full(n, np.nan)).astype(np.float32)
    if cfg['batt']:
        res['batt_current'] = _logged(rng, np.asarray(out['batt_current'], np.float64), 1.0, 0.1, n).astype(np.float32)
        res['batt_voltage'] = _logged(rng, np.asarray(out['batt_voltage'], np.float64), cfg['cad_slow'], 0.5, n).astype(np.float32)
        res['soc'] = _logged(rng, np.asarray(out['soc'], np.float64), 1.0, 0.005, n).astype(np.float32)
    else:
        for k in ['batt_current', 'batt_voltage', 'soc']: res[k] = np.full(n, np.nan, np.float32)
    if cfg['aux']:
        res['aux_power'] = _logged(rng, np.asarray(out['hvac_power'], np.float64), 3.0, 10.0, n).astype(np.float32)
    else:
        res['aux_power'] = np.full(n, np.nan, np.float32)
    # dropouts
    valid = np.ones(n, np.uint8)
    if rng.random() < cfg['p_dropout']:
        for _ in range(int(rng.integers(1, 3))):
            ln = int(rng.integers(3, 25)); s = int(rng.integers(0, max(1, n - ln)))
            valid[s:s + ln] = 0
    for k in ['fuel_rate', 'engine_rpm', 'batt_current', 'batt_voltage', 'soc', 'aux_power']:
        res[k][valid == 0] = np.nan
    res['valid'] = valid
    return res
