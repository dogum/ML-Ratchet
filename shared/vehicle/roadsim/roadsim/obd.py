"""Measurement layer: what an OBD logger would have reported for a RoadSim run. Not physics: it is applied to the
outputs of simulate() when, and only when, the comparison is against an OBD-estimated label (VED). Never apply it
to metered fuel (D3 dynamometer tests) or to EPA results.

    from roadsim import obd
    f_label = obd.label_fuel(out, spec, fuel_source)      # per-second fuel [g/s] as VED's label would show it
    cfg = obd.logger_config(arch, rng); log = obd.emulate(out, cfg, rng, spec)   # a full VED-style 1 Hz log

Upstreamed 13 Nov 2026 from the two released models (their own copies stay unchanged so they reproduce):
  * label model (maf_fuel): models/energynet/training/wrappers.py. VED's MAF-based fuel estimate still shows the air a
    motoring engine pumps during deceleration fuel cut-off (~0.2-0.3 g/s, not 0). Dev fit by EnergyNet (7 MAF dev cars,
    10,273 deceleration seconds): f_cut = exp(-1.578) D^0.342 (rpm/1000)^0.226 g/s (residual sd 0.29 in log).
  * logger emulator (emulate): models/tripdetective/training/obdlog.py, EMU_VERSION 1: per-PID polling clocks
    (speed ~1 s, MAF/rpm ~2-2.4 s, load/fuel PID ~5 s), held values, quantisation, MAF fuel estimate (misses
    enrichment above 75 % power, shows motoring air in fuel cut), trims, dropouts, RoadData's 1 Hz resampling.
"""
import numpy as np

MAF_CAL = dict(a=-1.578, b=0.342, c=0.226)
ENRICH = dict(thr=0.75, factor=0.90)          # stoichiometric MAF estimate / true fuel when enriched (obdlog CAL)
MAF_SOURCES = ('maf', 'absload')              # VED meta.fuel_source prefixes estimated from air flow


def maf_fuel(out, spec, cal=None):
    """fuel [g/s] a MAF-based estimate reports: the simulated fuel, but never below the motoring-air equivalent while
    the engine turns (> 300 rpm). EnergyNet's dev-fitted label model, unchanged."""
    c = cal or MAF_CAL
    D = ((spec.get('engine') or {}).get('displacement_L')) or 2.0
    rpm = np.asarray(out['eng_rpm'], float); f = np.asarray(out['fuel_rate'], float)
    fcut = np.exp(c['a']) * D ** c['b'] * np.maximum(rpm / 1000, 0.3) ** c['c']
    return np.where(rpm > 300, np.maximum(f, fcut), f)


def label_fuel(out, spec, fuel_source, enrich=True):
    """per-second fuel [g/s] as a VED label of this fuel_source would show it. MAF / absolute-load estimates: the fuel-
    cut model, and (enrich=True) the missing enrichment at high load; the OBD fuel-rate PID (Volt) and anything else:
    the simulated fuel as is."""
    f = np.nan_to_num(np.asarray(out['fuel_rate'], float))
    if not str(fuel_source or '').startswith(MAF_SOURCES) or not spec.get('engine') or out.get('eng_rpm') is None:
        return f
    if enrich:
        pmax = 1000.0 * float((spec.get('engine') or {}).get('max_power_kW') or 100.0)
        ep = np.asarray(out['eng_power'], float)
        f = f * np.where(ep > ENRICH['thr'] * pmax, ENRICH['factor'], 1.0)
    return maf_fuel(dict(out, fuel_rate=f), spec)


# ------------------------------------------------------------------ OBD logger emulator (TripDetective obdlog v1)
EMU_VERSION = 1
EMU_CAL = {
    'cut_air_k': (0.4, 1.0),      # MAF-equivalent fuel during fuel cut = k x idle fuel x rpm/idle_rpm  (range drawn per car)
    'enrich_thr': 0.75,           # engine power / max power above which mixture is enriched
    'enrich_factor': 0.90,        # stoichiometric estimate / true fuel when enriched
    'maf_bias_sd': 0.04,          # per-car residual bias of the MAF x trims estimate
    'stft_sd': 0.03,              # fast trim noise
    'p_dropout': 0.25,            # probability a trip has dropouts
    'idle_scale': (1.0, 1.0),     # multiplier on idle fuel
    'idle_rpm_add': (0, 0),       # additive idle rpm
    'hev_eng_on_extra': 0.0,      # extra engine-on probability for HEV at low speed
}


def held_interp(t, v, grid, plateau=2.5):
    """RoadData stage-2 resampler (data/scripts/ved_stage2.py)."""
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return None
    t = t[ok]; v = v[ok]
    if len(v) == 1:
        return np.full(len(grid), v[0], np.float64)
    ch = np.r_[True, v[1:] != v[:-1]]
    keep = ch.copy()
    idx = np.where(ch)[0]
    nxt = np.r_[idx[1:], len(v)]
    for a, b in zip(idx, nxt):
        if t[b - 1] - t[a] > plateau:
            keep[b - 1] = True
    keep[-1] = True
    return np.interp(grid, t[keep], v[keep])


def logger_config(arch, rng, fuel_src=None):
    """which channels a car's logger records (fixed per car); VED's mix loosely, plus generic OBD subsets."""
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
    c['cut_k'] = float(rng.uniform(*EMU_CAL['cut_air_k']))
    c['maf_bias'] = float(rng.normal(0, EMU_CAL['maf_bias_sd']))
    c['idle_scale'] = float(rng.uniform(*EMU_CAL['idle_scale']))
    c['p_dropout'] = EMU_CAL['p_dropout']
    return c


def _poll(rng, n, cad):
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
    """out: simulate() output (record=True). Returns 1 Hz logged arrays (NaN where not logged) + 'valid'."""
    from scipy.signal import lfilter
    n = len(out['v'])
    res = {}
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
            fobs = _logged(rng, fr, cfg['cad_slow'], 0.01 * 0.745 / 3.6, n)
        else:
            pmax = 1000.0 * float(spec['engine'].get('max_power_kW') or 100.0)
            ep = np.asarray(out['eng_power'], np.float64)
            est = fr * np.where(ep > EMU_CAL['enrich_thr'] * pmax, EMU_CAL['enrich_factor'], 1.0)
            idle = on & (np.asarray(out['v']) < 0.2) & (fr > 0)
            idle_f = float(np.median(fr[idle])) if idle.sum() >= 5 else 0.10 * float(spec['engine'].get('displacement_L') or 2.0)
            idle_rpm = float(np.median(rpm_true[idle])) if idle.sum() >= 5 else 700.0
            est = np.where(idle, est * cfg['idle_scale'], est)
            cut = on & (fr <= 1e-4) & (rpm_true > 300)
            rr = rpm_true / max(idle_rpm, 400.0) if spec.get('architecture') == 'conv' else 1.0
            est = np.where(cut, cfg['cut_k'] * idle_f * rr, est)
            ar = lfilter([1.0], [1.0, -0.9], rng.normal(0, EMU_CAL['stft_sd'] * 0.45, n))
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
        for k in ['batt_current', 'batt_voltage', 'soc']:
            res[k] = np.full(n, np.nan, np.float32)
    if cfg['aux']:
        res['aux_power'] = _logged(rng, np.asarray(out['hvac_power'], np.float64), 3.0, 10.0, n).astype(np.float32)
    else:
        res['aux_power'] = np.full(n, np.nan, np.float32)
    valid = np.ones(n, np.uint8)
    if rng.random() < cfg['p_dropout']:
        for _ in range(int(rng.integers(1, 3))):
            ln = int(rng.integers(3, 25)); s = int(rng.integers(0, max(1, n - ln)))
            valid[s:s + ln] = 0
    for k in ['fuel_rate', 'engine_rpm', 'batt_current', 'batt_voltage', 'soc', 'aux_power']:
        res[k][valid == 0] = np.nan
    res['valid'] = valid
    return res
