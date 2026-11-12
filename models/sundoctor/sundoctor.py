"""SunDoctor: what's wrong with this solar array?

A neural network trained only on physics simulations of PV systems reads two weeks of a system's power output, with
plane-of-array irradiance and air temperature if you have them (home mode needs neither sensor), for fixed or single-axis
tracking arrays, and returns, for every day, with error bars:

  pf      performance factor: the fraction of nameplate the clean, unshaded, unclipped array delivers
  frac    fraction of the array that is working (dead strings show up here)
  soil    soiling loss
  shade   energy lost to shading
  clip    energy lost to inverter clipping
  outage  energy lost to outages / trips
  snow    energy lost to snow cover
  stuck   energy lost to trackers that are stuck or stowed instead of following the sun
  step_p  probability that part of the array failed (or was repaired) that day

Needs only numpy (pandas for the CSV helper). See MODEL_CARD.md for training, validation and limits.

    import sundoctor
    doc = sundoctor.SunDoctor()                              # loads sundoctor_weights.npz next to this file
    report = doc.diagnose_csv('my_system.csv', lat=38.99, lon=-122.13, tilt=25, azimuth=180,
                              dc_kw=37.2, tz='US/Pacific', n_strings=6)
    print(report['summary'])

    # no irradiance sensor (most home systems): leave poa_col=None or pass home=True
    # single-axis trackers: mount='tracker' (axis_azimuth, max_angle, gcr, backtrack)
"""
import os, json, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T_STEPS = 96; TSOL = (np.arange(T_STEPS) + 0.5) / 4.0; D2R = np.pi / 180

# ------------------------------------------------------------------ physics (validated against pvlib, see MODEL_CARD.md)
def declination(doy):
    g = 2 * np.pi * (doy - 1) / 365.0
    return (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))

def eot_minutes(doy):
    g = 2 * np.pi * (doy - 1) / 365.0
    return 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))

def sun(lat, doy, tsol):
    d = declination(doy); h = (tsol - 12.0) * 15 * D2R; p = lat * D2R
    cz = np.clip(np.sin(p) * np.sin(d) + np.cos(p) * np.cos(d) * np.cos(h), -1, 1); el = 90 - np.arccos(cz) / D2R
    az = np.arctan2(np.sin(h), np.cos(h) * np.sin(p) - np.tan(d) * np.cos(p)) / D2R + 180.0
    return cz, el, az

def cos_aoi(tilt, az_arr, cz, az):
    sz = np.sqrt(np.clip(1 - cz ** 2, 0, 1)); return np.cos(tilt * D2R) * cz + np.sin(tilt * D2R) * sz * np.cos((az - az_arr) * D2R)

def _extra(doy): return 1367.0 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365.0))

def tracker_rotation(cz, az, axis_az=180.0, max_angle=55.0, gcr=0.4, backtrack=True):
    """single-axis tracker rotation (deg, + toward the west for a north-south axis), as pvlib.tracking.singleaxis"""
    sz = np.sqrt(np.clip(1 - cz ** 2, 0, 1)); th = np.arctan2(sz * np.sin((az - axis_az) * D2R), np.maximum(cz, 1e-6))
    if backtrack:
        temp = np.abs(np.cos(th)) / gcr; th = th + np.where(temp < 1, -np.sign(th) * np.arccos(np.clip(temp, -1, 1)), 0.0)
    return np.where(cz > 0, np.clip(th / D2R, -max_angle, max_angle), 0.0)

def clearsky_poa(lat, tilt, azimuth, doy, tl=3.0, mount='fixed', axis_azimuth=180.0, max_angle=55.0, gcr=0.4, backtrack=True):
    cz, el, az = sun(lat, doy, TSOL[None])
    if mount == 'tracker':
        th = tracker_rotation(cz, az, axis_azimuth, max_angle, gcr, backtrack); t = th * D2R
        ca = np.cos(t) * cz + np.sin(t) * np.sqrt(np.clip(1 - cz ** 2, 0, 1)) * np.sin((az - axis_azimuth) * D2R); tilt = np.abs(th)
    else:
        ca = cos_aoi(tilt, azimuth, cz, az)
    z = np.arccos(np.clip(cz, -1, 1)) / D2R
    with np.errstate(all='ignore'):
        am = np.where(cz > 0.01, 1.0 / (cz + 0.50572 * np.maximum(96.07995 - z, 1e-3) ** -1.6364), np.nan)
        I0 = _extra(doy); ghi = 0.868 * I0 * cz * np.exp(-0.0387 * am * tl); dni = 0.827 * I0 * np.exp(-0.09 * am * (tl - 1))
        dni = np.minimum(dni, ghi * (1 - (0.1 - 0.2 * np.exp(-tl)) / (0.1 + 0.882)) / cz)
    ghi = np.nan_to_num(np.where(cz > 0.01, ghi, 0)); dni = np.nan_to_num(np.where(cz > 0.01, dni, 0)); dhi = np.maximum(ghi - dni * np.maximum(cz, 0), 0)
    A = np.clip(dni / I0, 0, 1); rb = np.where(cz > 0.0175, np.maximum(ca, 0) / np.maximum(cz, 0.0175), 0)
    poa = dni * np.maximum(ca, 0) + dhi * (A * rb + (1 - A) * (1 + np.cos(tilt * D2R)) / 2) + ghi * 0.2 * (1 - np.cos(tilt * D2R)) / 2
    return np.maximum(cz, 0), np.maximum(ca, 0), poa / 1000.0

def expected_power(g, ta, ca):
    """clean, healthy, unclipped AC output per kW of DC nameplate (PVWatts + ASHRAE IAM + Faiman), from measured POA (kW/m2)"""
    m = np.clip(1 - 0.05 * (1 / np.clip(ca, 1e-3, 1) - 1), 0, 1)
    geff = g * 1000.0 * (0.25 + 0.75 * m); tc = ta + g * 1000.0 / (25.0 + 6.84 * 2.0)
    pdc = geff / 1000.0 * (1 - 0.0037 * (tc - 25.0)); z = np.clip(pdc, 1e-6, None)
    eta = 0.96 / 0.9637 * (-0.0162 * z - 0.0059 / z + 0.9858)
    return np.where(pdc > 0, np.maximum(eta * pdc, 0), 0.0)

def channels(p, g, ta, cz, ca, gclear, ref, home=False):
    """12 input channels; with home=True (no irradiance sensor) the network leans on clear-sky physics instead"""
    t0 = np.where(np.isfinite(ta), ta, 15.0); pc = expected_power(gclear, t0, ca)
    if home:
        valid = np.isfinite(p); g0 = np.zeros_like(p, dtype=float); pe = np.where(valid, pc, 0.0)
    else:
        valid = np.isfinite(p) & np.isfinite(g); g0 = np.where(valid, g, 0.0); pe = np.where(valid, expected_power(np.maximum(g0, 0), t0, ca), 0.0)
    p0 = np.where(valid, p, 0.0); has = float(ref is not None and np.isfinite(ref) and ref > 0); r = float(ref) if has else 0.0
    return np.stack([p0, g0, np.full_like(p0, 0.0 if home else 1.0), gclear, pe, pc, t0 / 40.0, cz, ca, valid.astype(float),
                     np.full_like(p0, r), np.full_like(p0, has)]).astype(np.float32)

# ------------------------------------------------------------------ network
def _gelu(x):
    a = np.abs(x) * 0.7071067811865476; t = 1 / (1 + 0.3275911 * a)
    erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return 0.5 * x * (1 + np.sign(x) * erf)

def _conv2d(x, w, b, stride, pad):
    kh, kw = w.shape[2:]; xp = np.pad(x, ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])))
    win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::stride[0], ::stride[1]]
    return np.einsum('bchwij,ocij->bohw', win, w, optimize=True) + b[None, :, None, None]

def _conv1d(x, w, b, dil=1):
    k = w.shape[2]; pad = dil * (k // 2); xp = np.pad(x, ((0, 0), (0, 0), (pad, pad))); L = x.shape[2]
    y = sum(np.einsum('bcl,oc->bol', xp[:, :, j * dil: j * dil + L], w[:, :, j]) for j in range(k))
    return y + b[None, :, None]

class SunDoctorNet:
    def __init__(self, path=os.path.join(HERE, 'sundoctor_weights.npz')):
        z = np.load(path, allow_pickle=False); self.meta = json.loads(str(z['meta'])); self.W = {k: z[k] for k in z.files if k != 'meta'}
        self.days = self.meta['days']
    def forward(self, x):
        W, M = self.W, self.meta; s, p = M['stem']
        h = _gelu(_conv2d(x, W['s_w'], W['s_b'], s, p))
        for i, cfg in enumerate(M['conv2d']):
            (s1, p1), (s2, p2), (s3, p3) = cfg
            y = _gelu(_conv2d(h, W[f'b{i}c1_w'], W[f'b{i}c1_b'], s1, p1)); y = _conv2d(y, W[f'b{i}c2_w'], W[f'b{i}c2_b'], s2, p2)
            h = _gelu(y + _conv2d(h, W[f'b{i}sk_w'], W[f'b{i}sk_b'], s3, p3))
        h = np.concatenate([h.mean(-1), h.max(-1)], 1)
        z = _gelu(_conv1d(h, W['p_w'], W['p_b']))
        for i, dil in enumerate(M['tconv']):
            y = _gelu(_conv1d(z, W[f't{i}c1_w'], W[f't{i}c1_b'], dil)); z = _gelu(z + _conv1d(y, W[f't{i}c2_w'], W[f't{i}c2_b']))
        day = _conv1d(z, W['d_w'], W['d_b'])
        g = np.concatenate([z.mean(-1), z.max(-1)], 1); g = _gelu(g @ W['g0_w'].T + W['g0_b']) @ W['g1_w'].T + W['g1_b']
        return day, g
    def predict(self, X, bs=32):
        D, G = [], []
        for i in range(0, len(X), bs):
            d, g = self.forward(X[i:i + bs]); D.append(d); G.append(g)
        D = np.concatenate(D); G = np.concatenate(G); M = self.meta; n = len(M['reg'])
        out = {k: D[:, i] * M['scale'][i] for i, k in enumerate(M['reg'])}
        out.update({k + '_sig': np.exp(np.clip(D[:, n + i], -6, 3)) * M['scale'][i] for i, k in enumerate(M['reg'])})
        out['step_p'] = 1 / (1 + np.exp(-D[:, 2 * n]))
        for i, k in enumerate(M['glob']): out[k] = G[:, i] * M['gscale'][i]; out[k + '_sig'] = np.exp(np.clip(G[:, 2 + i], -6, 3)) * M['gscale'][i]
        return out

# ------------------------------------------------------------------ the doctor
class SunDoctor:
    KEYS = ('pf', 'soil', 'frac', 'shade', 'clip', 'outage', 'snow', 'stuck', 'step_p', 'pf_sig', 'frac_sig', 'soil_sig', 'shade_sig', 'clip_sig', 'outage_sig', 'snow_sig', 'stuck_sig')
    def __init__(self, path=os.path.join(HERE, 'sundoctor_weights.npz')):
        self.net = SunDoctorNet(path)

    @staticmethod
    def solar_grid(t_utc_hours, series, lon, doy):
        """t_utc_hours: absolute UTC hours (float) of interval centers; returns first solar day index and (days, 96) array"""
        tsol = t_utc_hours + lon / 15 + eot_minutes(doy) / 60
        day0 = np.floor(np.nanmin(tsol) / 24); nd = int(np.ceil((np.nanmax(tsol) - day0 * 24) / 24))
        grid = (day0 * 24 + np.arange(nd)[:, None] * 24 + TSOL[None]).ravel()
        ok = np.isfinite(series) & np.isfinite(tsol)
        if ok.sum() < 2: return int(day0), np.full((nd, T_STEPS), np.nan)          # e.g. no irradiance column at all
        ts, xs = tsol[ok], series[ok]; o = np.interp(grid, ts, xs, left=np.nan, right=np.nan)
        i = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1); o[(ts[i] - ts[i - 1]) > 0.51] = np.nan
        return int(day0), o.reshape(nd, T_STEPS)

    def run(self, P, G, Ta, lat, tilt, azimuth, doy, ref=None, home=False, mount='fixed', **trk):
        """P: (days, 96) AC power / DC nameplate; G: POA kW/m2 (ignored in home mode); Ta: deg C; doy: (days,) day of year"""
        D = self.net.days; n = len(P); stride = D // 2
        cz, ca, gc = clearsky_poa(lat, tilt, azimuth, doy[:, None], mount=mount, **trk)
        if n < D:      # pad short records with missing days
            pad = D - n; P, G, Ta = [np.concatenate([a, np.full((pad, T_STEPS), np.nan)]) for a in (P, G, Ta)]
            cz, ca, gc = [np.concatenate([a, np.repeat(a[-1:], pad, 0)]) for a in (cz, ca, gc)]
        N = len(P); starts = list(range(0, N - D + 1, stride))
        if starts[-1] != N - D: starts.append(N - D)
        X = np.stack([channels(P[s:s + D], G[s:s + D], Ta[s:s + D], cz[s:s + D], ca[s:s + D], gc[s:s + D], None if ref is None else ref[min(s + D // 2, len(ref) - 1)], home) for s in starts])
        o = self.net.predict(X); out = {k: np.full(N, np.nan) for k in self.KEYS}; out['ac_limit'] = np.full(N, np.nan); out['eta0'] = np.full(N, np.nan)
        for w, s in enumerate(starts):
            lo = 0 if w == 0 else D // 4; hi = D if w == len(starts) - 1 else D - D // 4
            for k in self.KEYS: out[k][s + lo:s + hi] = o[k][w, lo:hi]
            out['ac_limit'][s + lo:s + hi] = o['ac_limit'][w]; out['eta0'][s + lo:s + hi] = o['eta0'][w]
        return {k: v[:n] for k, v in out.items()}

    def diagnose(self, t_utc_hours, p_kw, poa_wm2, tamb_c, lat, lon, tilt=0.0, azimuth=180.0, dc_kw=1.0, n_strings=None, home=None,
                 mount='fixed', axis_azimuth=180.0, max_angle=55.0, gcr=0.4, backtrack=True):
        """Arrays of equal length: interval-center UTC hours (e.g. unix seconds / 3600), AC power (kW), POA (W/m2, or None for
        home mode), air temperature (C; from a weather service is fine). mount='tracker' for single-axis trackers."""
        home = (poa_wm2 is None) if home is None else home
        if poa_wm2 is None: poa_wm2 = np.full(len(p_kw), np.nan)
        trk = dict(axis_azimuth=axis_azimuth, max_angle=max_angle, gcr=gcr, backtrack=backtrack) if mount == 'tracker' else {}
        t = np.asarray(t_utc_hours, float)
        import datetime as _dt
        doy_t = np.array([(_dt.datetime(1970, 1, 1) + _dt.timedelta(hours=float(h))).timetuple().tm_yday if np.isfinite(h) else 1 for h in t])
        d0, P = self.solar_grid(t, np.asarray(p_kw, float) / dc_kw, lon, doy_t)
        _, G = self.solar_grid(t, np.asarray(poa_wm2, float) / 1000.0, lon, doy_t); _, Ta = self.solar_grid(t, np.asarray(tamb_c, float), lon, doy_t)
        days = np.array([_dt.date(1970, 1, 1) + _dt.timedelta(days=int(d0 + i)) for i in range(len(P))])
        doy = np.array([d.timetuple().tm_yday for d in days])
        valid = (np.isfinite(P) & (True if home else np.isfinite(G))).mean(1) > 0.5
        o1 = self.run(P, G, Ta, lat, tilt, azimuth, doy, None, home, mount, **trk)
        ref = self.reference(o1['pf'], valid & (o1['outage'] < 0.2))
        o = self.run(P, G, Ta, lat, tilt, azimuth, doy, ref, home, mount, **trk)
        for k in ('soil', 'shade', 'clip', 'outage', 'snow', 'stuck'): o[k] = np.clip(o[k], 0, 1)
        if mount != 'tracker': o['stuck'] = np.zeros_like(o['stuck'])          # a fixed array has no trackers to stick
        o['ref'] = ref; o['valid'] = valid; o['date'] = days; o['mode'] = 'home (no irradiance sensor)' if home else 'with irradiance sensor'
        if n_strings: o['dead_strings'] = np.round(n_strings * (1 - np.clip(o['frac'], 0, 1)))
        # energy bookkeeping (kWh per day)
        e = np.nansum(np.where(np.isfinite(P), P, 0), 1) * dc_kw * 0.25
        c = lambda k: np.maximum(1 - np.clip(np.nan_to_num(o[k]), 0, 0.99), 0.01)
        with np.errstate(all='ignore'):
            e_clip = e / c('outage'); e_noclip = e_clip / c('clip'); e_nosnow = e_noclip / c('snow'); e_unsh = e_nosnow / c('shade'); e_track = e_unsh / c('stuck'); e_clean = e_track / c('soil')
        L = {'outage': e_clip - e, 'clipping': e_noclip - e_clip, 'snow': e_nosnow - e_noclip, 'shading': e_unsh - e_nosnow, 'stuck_trackers': e_track - e_unsh,
             'soiling': e_clean - e_track, 'dead_strings': e_clean / np.maximum(np.clip(o['frac'], 0.05, 1), 0.05) * (1 - np.clip(o['frac'], 0, 1))}
        # near-total-loss days (buried in snow, an inverter down): undoing the losses from ~zero output gives ~zero, so size
        # those days from the top, from the healthy array's expected energy
        etop = self.top_energy(o, P, G, Ta, lat, tilt, azimuth, doy, dc_kw, home, mount, trk, e)
        F = c('outage') * c('clip') * c('snow') * c('shade') * c('stuck'); top = (F < 0.3) & np.isfinite(etop) & (etop > e)
        k = lambda n: 1 - np.clip(np.nan_to_num(o[n]), 0, 0.99)
        x1 = etop * np.clip(np.nan_to_num(o['frac'], nan=1), 0, 1); x2 = x1 * k('soil'); x3 = x2 * k('stuck'); x4 = x3 * k('shade'); x5 = x4 * k('snow'); x6 = x5 * k('clip')
        T = {'dead_strings': etop - x1, 'soiling': x1 - x2, 'stuck_trackers': x2 - x3, 'shading': x3 - x4, 'snow': x4 - x5, 'clipping': x5 - x6, 'outage': np.maximum(x6 - e, 0)}
        o['kwh'] = e; o['kwh_lost'] = {n: np.where(top, T[n], v) for n, v in L.items()}
        o['summary'] = self.summary(o, n_strings)
        return o

    @staticmethod
    def top_energy(o, P, G, Ta, lat, tilt, azimuth, doy, dc_kw, home, mount, trk, e):
        """the healthy array's energy each day (kWh): its own healthy level times the physics expectation, from measured
        sunlight with a sensor, or without one from clear-sky sunlight times the month's typical clearness (judged from power)"""
        cz, ca, gc = clearsky_poa(lat, tilt, azimuth, doy[:, None], mount=mount, **trk); t0 = np.where(np.isfinite(Ta), Ta, 15.0)
        ref = np.where(np.isfinite(o['ref']), o['ref'], np.nanmedian(o['pf']))
        if not home:
            pe = np.where(np.isfinite(G), expected_power(np.maximum(np.nan_to_num(G), 0), t0, ca), np.nan)
            return np.where(np.isfinite(pe).mean(1) > 0.5, ref * np.nansum(pe, 1) * dc_kw * 0.25, np.nan)
        ecs = ref * expected_power(gc, t0, ca).sum(1) * dc_kw * 0.25
        F = np.prod([1 - np.clip(np.nan_to_num(o[n]), 0, 0.99) for n in ('outage', 'snow', 'stuck')], 0); r = np.where((F > 0.7) & (ecs > 0) & o['valid'], e / np.maximum(ecs, 1e-9), np.nan)
        kt = np.array([np.nanmedian(r[max(0, i - 15):i + 16]) if np.isfinite(r[max(0, i - 15):i + 16]).sum() >= 3 else 0.6 for i in range(len(r))])
        return ecs * kt

    @staticmethod
    def reference(pf, valid, window=365, q=0.975):
        """the system's own clean, healthy level: 97.5th percentile of daily pf over the previous year (causal)"""
        out = np.full(len(pf), np.nan); x = np.where(valid, pf, np.nan)
        for i in range(len(pf)):
            w = x[max(0, i - window):i]; w = w[np.isfinite(w)]
            if len(w) >= 20: out[i] = np.quantile(w, q)
        return out

    @staticmethod
    def summary(o, n_strings=None):
        v = o['valid']; L = {k: float(np.nansum(np.where(v, x, 0))) for k, x in o['kwh_lost'].items()}; made = float(np.nansum(np.where(v, o['kwh'], 0)))
        s = {'days': int(v.sum()), 'kwh_made': made, 'kwh_lost': L, 'loss_pct_of_possible': {k: 100 * x / max(made + sum(L.values()), 1e-9) for k, x in L.items()}}
        if n_strings is not None:
            ds = o['dead_strings']; ds = ds[v & np.isfinite(ds)]
            s['days_with_dead_strings'] = int((ds > 0).sum()); s['latest_dead_strings'] = int(ds[-1]) if len(ds) else None
        st = np.where(v & (o['step_p'] > 0.5))[0]; s['failure_or_repair_days'] = [str(o['date'][i]) for i in st]
        return s

    def diagnose_csv(self, path, lat, lon, tilt=0.0, azimuth=180.0, dc_kw=1.0, tz='UTC', n_strings=None, time_col=None, power_col=None, poa_col=None,
                     temp_col=None, power_unit='kW', temp_unit='C', label='center', home=None, **mount):
        """CSV with a timestamp column (local time in `tz`), AC power, air temperature and, if you have one, POA irradiance
        (W/m2). Without a POA column (or with home=True) it runs in home mode. label: 'center', 'start' or 'end' of each
        averaging interval. mount: mount='tracker', axis_azimuth, max_angle, gcr, backtrack."""
        import pandas as pd
        df = pd.read_csv(path); c = df.columns
        time_col = time_col or c[0]; pick = lambda key, given: given or next((x for x in c if key in x.lower()), None)
        power_col = pick('power', power_col); poa_col = pick('poa', poa_col); temp_col = pick('temp', temp_col)
        home = (poa_col is None) if home is None else home
        t = pd.to_datetime(df[time_col]); dt = t.diff().median()
        if label == 'start': t = t + dt / 2
        elif label == 'end': t = t - dt / 2
        if t.dt.tz is None: t = t.dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
        t = t.dt.tz_convert('UTC'); ok = t.notna().values
        hrs = (t[ok] - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().values / 3600.0
        p = df[power_col].values[ok].astype(float) * (0.001 if power_unit == 'W' else 1.0)
        ta = df[temp_col].values[ok].astype(float) if temp_col else np.full(ok.sum(), 15.0); ta = (ta - 32) / 1.8 if temp_unit == 'F' else ta
        g = None if home else df[poa_col].values[ok].astype(float)
        return self.diagnose(hrs, p, g, ta, lat, lon, tilt, azimuth, dc_kw, n_strings, home, **mount)
