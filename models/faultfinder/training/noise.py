"""Real-noise and quirk model, learned from REAL fault-free days (dev split only) and the unlabelled Korean office
AHUs, then emulated on simulated days.

What it learns (fit) and applies (apply):
1. Residual library: for every real dev fault-free day, each channel's 15-min residual around a 75-min running
   mean while the fan runs (sensor noise + control hunting + small disturbances). apply() adds a random real
   residual day (random circular shift, gain 0.5-1.5) to the same channel of a simulated day.
2. Static sensor offsets and SAT tracking error: per-day constant offsets (sd from dev; heavier tail from the
   Korean AHUs' supply-air tracking error).
3. Fan-off behaviour: real mixed/supply sensors drift toward a blend of return and outdoor air when the fan stops
   (the simulator reports zone air); blend fraction drawn U(0, 0.3).
4. Channel availability: the exam sources log different points (ERS full set; LBNL AHUs no zone temperature or fan
   power; Nesbitt no occupancy, fan current instead of power; ORNL RTU no OA/MA temperatures, dampers or
   setpoint). apply() drops channels in these patterns, plus random gaps and flatlined sensors.
5. Schedule irregularity (Korean AHUs): random start/stop jitter of the occupancy/fan flags by +-1 bin.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, RI, NT, load_real_runs, real_raw, to_day  # noqa: E402

RES_CH = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_zone', 'damper_oa', 'valve_cc', 'valve_hc', 'fan_speed', 'fan_power']

PATTERNS = {   # channels MISSING in each source pattern
    'full': [],
    'lbnl_ahu': ['T_zone', 'fan_power'],
    'nesbitt': ['occ', 'T_zone'],
    'rtu': ['T_oa', 'T_ma', 'damper_oa', 'valve_hc', 'fan_speed', 'T_sa_sp'],
    'korea': ['T_oa', 'T_ma', 'damper_oa', 'valve_cc', 'valve_hc', 'fan_speed', 'fan_power', 'T_zone', 'occ'],
}


def _runmean(x, k=5):
    out = np.full_like(x, np.nan)
    h = k // 2
    for i in range(len(x)):
        w = x[max(0, i - h):i + h + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = w.mean()
    return out


def residuals(x):
    """[13,96] raw day -> residual dict per channel (NaN where fan off / missing)."""
    on = np.nan_to_num(x[RI['fan_on']]) > 0.5
    r = {}
    for c in RES_CH:
        v = x[RI[c]].astype(float)
        if c == 'fan_power':
            p = np.nanpercentile(v, 95) if np.isfinite(v).any() else np.nan
            v = v / p if p and p > 0 else v * np.nan
        res = v - _runmean(v)
        res[~on] = np.nan
        r[c] = res
    return r


def fit(out_path):
    runs = load_real_runs('dev')
    lib = {c: [] for c in RES_CH}
    meta = []
    sa_off = []
    for row, x in runs:
        if row.fault_type != 'none':
            continue
        r = residuals(x)
        for c in RES_CH:
            if np.isfinite(r[c]).sum() >= 16:
                lib[c].append(np.nan_to_num(r[c]).astype(np.float32))
        on = np.nan_to_num(x[RI['fan_on']]) > 0.5
        d = x[RI['T_sa'], on] - x[RI['T_sa_sp'], on]
        if np.isfinite(d).sum() > 8:
            sa_off.append(float(np.nanmedian(d)))
        meta.append(row.run_id)
    # Korean AHUs (unlabelled real background): hourly supply-air tracking error per AHU-day, and fan start/stop hours
    import buildingdata as bd
    k = bd.fault_data('korea_office')
    k['day'] = k.time_local.dt.normalize()
    kor_off, kor_start = [], []
    for (u, dday), g in k.groupby(['run_id', 'day']):
        on = g.fan_sa_status.to_numpy() > 0.5
        sp = g.T_sa_sp.to_numpy()
        ok = on & (sp > 1)
        if ok.sum() >= 4:
            kor_off.append(float(np.median(g.T_sa.to_numpy()[ok] - sp[ok])))
            kor_start.append(float(g.time_local.dt.hour.to_numpy()[on].min()))
    kor_off = np.array(kor_off)
    model = dict(lib_names=np.array(meta))
    for c in RES_CH:
        model['res_' + c] = np.stack(lib[c]) if lib[c] else np.zeros((1, NT), np.float32)
    model['sa_off_dev'] = np.array(sa_off)
    model['sa_off_korea'] = kor_off
    model['korea_start_h'] = np.array(kor_start)
    np.savez(out_path, **model)
    print('residual days per channel:', {c: len(lib[c]) for c in RES_CH})
    print('dev SAT tracking offsets', np.round(sa_off, 2))
    print('korea SAT tracking offsets: n %d, median %.2f, IQR %.2f..%.2f, |off|>2K %.2f' % (
        len(kor_off), np.median(kor_off), *np.percentile(kor_off, [25, 75]), np.mean(np.abs(kor_off) > 2)))
    return model


_P2 = None


def _phase2():
    """buildingsim.phase2 (BuildingSim 1.0.1). The training venv has no pvlib, which the buildingsim package imports;
    phase2.py's FaultFinder functions need only numpy and pandas, so the same file is loaded on its own there."""
    global _P2
    if _P2 is None:
        try:
            from buildingsim import phase2 as p2
        except ImportError:
            import importlib.util
            from _paths import BLDG_CODE
            f = os.path.join(BLDG_CODE, 'buildingsim', 'buildingsim', 'phase2.py')
            spec = importlib.util.spec_from_file_location('buildingsim_phase2', f)
            p2 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(p2)
        _P2 = p2
    return _P2


class Noise:
    def __init__(self, path, level=1.0, offsets=True, residual=True, patterns=True, fanoff=True, damper_char=False,
                 big_offsets=False):
        m = dict(np.load(path))
        self.lib = {c: m['res_' + c] for c in RES_CH}
        self.level = level
        self.offsets, self.residual, self.patterns, self.fanoff = offsets, residual, patterns, fanoff
        self.damper_char, self.big_offsets = damper_char, big_offsets

    @staticmethod
    def damper_command(frac, rng):
        """Installed damper characteristic (dev calibration, v4): BuildingSim logs the outdoor-air FRACTION as the
        damper signal (fraction == damper exactly); real dev units log a COMMAND whose OA fraction is a + (b - a) cmd^g
        (ERS / FLEXLAB at 100 % command mix only 0.81-0.90 outdoor air; Nesbitt at 15 % mixes 0.14-0.28).
        Inverse map with a ~ U(0, 0.15), b ~ U(0.75, 1), g ~ U(0.6, 1.6); 30 % of units keep the linear map."""
        if rng.random() < 0.3:
            return frac
        a, b, g = rng.uniform(0, 0.15), rng.uniform(0.75, 1.0), rng.uniform(0.6, 1.6)
        c = np.clip((frac - a) / (b - a), 0, 1) ** (1 / g)
        return np.where(np.isfinite(frac), c, np.nan).astype(np.float32)

    def apply(self, x, rng, system=None):
        """x: raw [13,96] simulated day (float32, modified copy returned)."""
        x = x.astype(np.float32).copy()
        on = x[RI['fan_on']] > 0.5
        L = self.level
        if self.fanoff and (~on).any():
            b = rng.uniform(0, 0.3)
            Tra, Toa = x[RI['T_ra']], x[RI['T_oa']]
            mix = Tra + b * (Toa - Tra)
            x[RI['T_ma'], ~on] = mix[~on]
            x[RI['T_sa'], ~on] = (mix + rng.uniform(-0.5, 1.0))[~on]
        if self.residual:
            for c in RES_CH:
                lib = self.lib[c]
                r = lib[rng.integers(len(lib))]
                r = np.roll(r, rng.integers(NT)) * rng.uniform(0.5, 1.5) * L
                if c == 'fan_power':
                    p = np.nanpercentile(x[RI[c]], 95)
                    r = r * (p if np.isfinite(p) else 0)
                x[RI[c], on] += r[on]
        if self.damper_char:
            x[RI['damper_oa']] = self.damper_command(x[RI['damper_oa']], rng)
        if self.big_offsets:
            # static sensor disagreement seen on dev normals: mixed-air stratification (FLEXLAB MA below both OA and
            # RA by 0.9 K), zone vs return sensor 1.5 K apart (ORNL RTU), supply tracking error (Korea tail)
            x[RI['T_sa']] += rng.normal(0, 0.5) * L if rng.random() < 0.8 else rng.normal(0, 1.2) * L
            x[RI['T_ma']] += rng.normal(0, 0.8) * L
            x[RI['T_ra']] += rng.normal(0, 0.4) * L
            x[RI['T_oa']] += rng.normal(0, 0.6) * L
            x[RI['T_zone']] += rng.normal(0, 0.9) * L
        elif self.offsets:
            x[RI['T_sa']] += rng.normal(0, 0.45) * L if rng.random() < 0.8 else rng.normal(0, 1.0) * L
            x[RI['T_ma']] += rng.normal(0, 0.4) * L
            x[RI['T_ra']] += rng.normal(0, 0.3) * L
            x[RI['T_oa']] += rng.normal(0, 0.5) * L
            x[RI['T_zone']] += rng.normal(0, 0.3) * L
        else:
            for c in ('T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_zone'):
                x[RI[c]] += rng.normal(0, 0.1, NT)
        for c in ('damper_oa', 'valve_cc', 'valve_hc', 'fan_speed'):
            x[RI[c]] = np.clip(x[RI[c]], 0, 1)
        if self.patterns:
            x = self.drop(x, rng, system)
        return x

    # ---------------------------------------------------------------- noise v5: a unit's habits persist (r7 plan)
    def window_state(self, rng, system=None):
        """Noise v5 (2026-11-17 r7 plan): draws kept for every day of one simulated window (one unit): static sensor
        offsets (buildingsim.phase2.static_offsets, v4 sizes), the damper characteristic (phase2.damper_command, one
        draw per window via a stored seed), which real residual day each channel uses and its gain, the fan-off
        blend and supply offset, the channel pattern and the 6 % extra missing channels. Same marginals as v4."""
        import pandas as pd
        phase2 = _phase2()
        st = dict(fan_b=float(rng.uniform(0, 0.3)), fan_sa=float(rng.uniform(-0.5, 1.0)))
        st['res'] = {c: (int(rng.integers(len(self.lib[c]))), float(rng.uniform(0.5, 1.5))) for c in RES_CH}
        st['damper_seed'] = int(rng.integers(1 << 31))
        z = pd.DataFrame({c: [0.0] for c in ('SAT', 'MAT', 'RAT', 'OAT', 'ZAT')})
        _, off = phase2.static_offsets(z, rng, level=self.level, large=True)
        st['off'] = {'T_sa': off['SAT'], 'T_ma': off['MAT'], 'T_ra': off['RAT'], 'T_oa': off['OAT'],
                     'T_zone': off['ZAT']}
        if system == 'packaged_rtu':
            pat = 'rtu' if rng.random() < 0.5 else ['full', 'lbnl_ahu', 'nesbitt'][rng.integers(3)]
        else:
            pat = ['full', 'full', 'lbnl_ahu', 'nesbitt', 'rtu'][rng.integers(5)] if rng.random() < 0.85 else 'korea'
        miss = list(PATTERNS[pat])
        for c in RAW:
            if c not in ('fan_on',) and rng.random() < 0.06:
                miss.append(c)
        st['pat'], st['miss'] = pat, miss
        st['fan_from_speed'] = bool(rng.random() < 0.1 and pat == 'nesbitt')
        return st

    def apply_window(self, x, rng, st):
        """Noise v5 on one simulated day of a window whose state `st` came from window_state. Per day: the
        residual's circular shift and logger gaps."""
        phase2 = _phase2()
        x = x.astype(np.float32).copy()
        on = x[RI['fan_on']] > 0.5
        L = self.level
        if (~on).any():
            Tra, Toa = x[RI['T_ra']], x[RI['T_oa']]
            mix = Tra + st['fan_b'] * (Toa - Tra)
            x[RI['T_ma'], ~on] = mix[~on]
            x[RI['T_sa'], ~on] = (mix + st['fan_sa'])[~on]
        for c in RES_CH:
            k, g = st['res'][c]
            r = np.roll(self.lib[c][k], rng.integers(NT)) * g * L
            if c == 'fan_power':
                p = np.nanpercentile(x[RI[c]], 95)
                r = r * (p if np.isfinite(p) else 0)
            x[RI[c], on] += r[on]
        x[RI['damper_oa']] = phase2.damper_command(x[RI['damper_oa']], np.random.default_rng(st['damper_seed']))
        for c, v in st['off'].items():
            x[RI[c]] += np.float32(v)
        for c in ('damper_oa', 'valve_cc', 'valve_hc', 'fan_speed'):
            x[RI[c]] = np.clip(x[RI[c]], 0, 1)
        for c in st['miss']:
            x[RI[c]] = np.nan
        if rng.random() < 0.2:      # a gap (logger outage) on a few channels
            a = rng.integers(NT - 8)
            w = rng.integers(4, 40)
            for c in rng.choice(RAW, size=rng.integers(1, 5), replace=False):
                x[RI[c], a:a + w] = np.nan
        if st['fan_from_speed']:
            x[RI['fan_on']] = (np.nan_to_num(x[RI['fan_speed']]) > 0.05).astype(np.float32)
        return x

    def drop(self, x, rng, system):
        if system == 'packaged_rtu':
            pat = 'rtu' if rng.random() < 0.5 else ['full', 'lbnl_ahu', 'nesbitt'][rng.integers(3)]
        else:
            pat = ['full', 'full', 'lbnl_ahu', 'nesbitt', 'rtu'][rng.integers(5)] if rng.random() < 0.85 else 'korea'
        miss = list(PATTERNS[pat])
        for c in RAW:
            if c not in ('fan_on',) and rng.random() < 0.06:
                miss.append(c)
        for c in miss:
            x[RI[c]] = np.nan
        if rng.random() < 0.2:      # a gap (logger outage) on a few channels
            a = rng.integers(NT - 8)
            w = rng.integers(4, 40)
            for c in rng.choice(RAW, size=rng.integers(1, 5), replace=False):
                x[RI[c], a:a + w] = np.nan
        if rng.random() < 0.1 and pat == 'nesbitt':
            x[RI['fan_on']] = (np.nan_to_num(x[RI['fan_speed']]) > 0.05).astype(np.float32)
        return x


if __name__ == '__main__':
    fit(sys.argv[1] if len(sys.argv) > 1 else __import__('common').ROOT + '/data/noise_v1.npz')
