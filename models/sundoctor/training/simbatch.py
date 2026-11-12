"""Batched SunDoctor simulator: B systems x D days x 96 solar-time steps, fully vectorized (no per-sample python loops
except the obstacle profiles). Produces the network's input channels and per-day labels."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, sys
sys.path.insert(0, _P.SRC)
import pvsim as S

T = S.T_STEPS
REALISM = False
CH = ['p', 'poa', 'poa_clear', 'p_expected', 'temp', 'cos_zenith', 'cos_aoi', 'valid']
DAY_LABELS = ['pf', 'soil', 'frac', 'shade', 'clip', 'outage', 'step', 'dfrac']

def expected_power(g, ta, ca):
    """Physics prior used identically on simulated and real data: what a clean, healthy, unclipped array of 1 kW DC
    would deliver given the measured POA and air temperature (PVWatts, ASHRAE IAM on the beam-ish part, Faiman)."""
    geff = g * 1000.0 * (0.25 + 0.75 * S.iam_ashrae(np.maximum(ca, 1e-3), 0.05))
    tc = S.faiman(g * 1000.0, ta, 2.0)
    return S.pvwatts_inv(S.pvwatts_dc(geff, tc, 1.0, -0.0037), 1.0, 10.0)

def inputs(p, g, ta, cz, ca, gclear):
    """assemble (C, D, T) channels from arrays (D, T) with NaNs for missing data"""
    valid = np.isfinite(p) & np.isfinite(g) & np.isfinite(ta)
    p0 = np.where(valid, p, 0.0); g0 = np.where(valid, g, 0.0); t0 = np.where(np.isfinite(ta), ta, 15.0)
    pe = np.where(valid, expected_power(np.maximum(g0, 0), t0, ca), 0.0)
    return np.stack([p0, g0, gclear, pe, t0 / 40.0, cz, ca, valid.astype(float)]).astype(np.float32)

def simulate_batch(rng, B, D=14, fixed=None):
    f = fixed or {}
    u = lambda a, b, *s: rng.uniform(a, b, (B,) + s)
    lat = f.get('lat', u(20, 52)); tilt = np.where(rng.random(B) < 0.25, u(0, 12), u(10, 40)); tilt = f.get('tilt', tilt)
    azr = f.get('azimuth', np.clip(180 + rng.normal(0, 35, B), 90, 270))
    doy0 = f.get('doy0', rng.integers(1, 366, B)); doy = (doy0[:, None] + np.arange(D)[None] - 1) % 365 + 1          # (B, D)
    dcac = f.get('dcac', u(0.95, 1.5)); gamma = u(-0.0048, -0.0028); eta0 = f.get('eta0', u(0.80, 1.02))
    tl = u(2.0, 4.5, D)[..., None]; alt = u(0, 1800)[:, None, None]; ws = u(0.5, 5.0, D)[..., None]
    lowirr = u(0, 0.025)[:, None, None]; albedo = u(0.12, 0.3)[:, None, None]; b0 = u(0.03, 0.07)[:, None, None]
    L = lat[:, None, None]; Tt = S.TSOL[None, None]; DOY = doy[..., None]
    cz, el, az = S.sun(L, DOY, Tt); ca = S.cos_aoi(tilt[:, None, None], azr[:, None, None], cz, az)
    gc, dnic, dhic = S.ineichen(cz, DOY, tl, alt)
    # clouds: 0 clear, 1 hazy, 2 broken, 3 overcast (vectorized over B, D; Markov chain over T)
    typ = rng.choice(4, size=(B, D), p=[0.42, 0.16, 0.27, 0.15])
    walk = np.cumsum(rng.standard_normal((B, D, T)), -1) / np.sqrt(T)
    k_clear = 1.0 + 0.01 * rng.standard_normal((B, D, T))
    k_hazy = u(0.75, 0.95, D)[..., None] + 0.03 * walk
    k_over = u(0.08, 0.45, D)[..., None] * np.exp(0.25 * walk)
    pc = u(0.2, 0.8, D); lo = u(0.15, 0.6, D); st = rng.random((B, D)) < pc; kb = np.empty((B, D, T))
    for t in range(T):
        sw = rng.random((B, D)) < 0.35; st = np.where(sw, rng.random((B, D)) < pc, st)
        kb[..., t] = np.where(st, lo + 0.1 * rng.standard_normal((B, D)), 1.0 + np.abs(0.06 * rng.standard_normal((B, D))))
    k = np.select([typ[..., None] == 0, typ[..., None] == 1, typ[..., None] == 2], [k_clear, k_hazy, kb], k_over)
    k = np.clip(k, 0.03, 1.35); ghi = gc * k
    dni, dhi = S.erbs(ghi, cz, DOY)
    clr = typ[..., None] <= 1
    dni = np.where(clr, dnic * np.where(typ[..., None] == 1, k, 1.0) ** 1.5, dni); dhi = np.where(clr, np.maximum(ghi - dni * np.maximum(cz, 0), 0), dhi)
    beam, sky, gnd = S.haydavies(tilt[:, None, None], ca, cz, dni, dhi, ghi, DOY, albedo)
    poa = beam + sky + gnd
    season = np.cos(2 * np.pi * (doy - 200) / 365.0)
    tmean = u(4, 22)[:, None] + (14 - 0.15 * np.abs(lat[:, None] - 30)) * season + 2 * rng.standard_normal((B, D))
    amp = np.where(typ == 3, u(2, 5, D), u(5, 11, D))
    ta = tmean[..., None] + amp[..., None] * np.sin(2 * np.pi * (Tt - 9.0) / 24.0)
    # ---- dead strings
    N = f.get('n_strings', rng.choice([1, 2, 3, 4, 6, 8, 12], B))
    off0 = np.where(rng.random(B) < 0.6, 0, np.floor(rng.random(B) * (np.maximum(N // 2, 0) + 1))); off0 = np.where(N > 1, off0, 0)
    stepday = rng.integers(1, D, B); has = (rng.random(B) < 0.45) & (N > 1)
    off1 = np.clip(off0 + rng.choice([-2, -1, 1, 1, 2, 3], B), 0, N - 1)
    off = np.where(has[:, None] & (np.arange(D)[None] >= stepday[:, None]), off1[:, None], off0[:, None])
    frac = 1 - off / N[:, None]
    # ---- soiling with rain / cleaning
    rate = np.select([rng.random(B) < 0.35, rng.random(B) < 0.62], [0.0, rng.uniform(0, 0.0015, B)], rng.uniform(0.0015, 0.006, B))
    cur = np.where(rate > 0, u(0, 0.12) if not REALISM else np.where(rng.random(B) < 0.5, u(0, 0.06), u(0, 0.2)), 0.0); s = np.empty((B, D))
    for d in range(D):
        wash = ((typ[:, d] == 3) & (rng.random(B) < 0.55)) | (rng.random(B) < 0.01)
        cur = np.where(wash, cur * rng.uniform(0, 0.3, B), cur); cur = np.minimum(cur + rate, 0.35); s[:, d] = cur
    # ---- shading
    kind = rng.choice(3, size=B, p=[0.45, 0.25, 0.30]); sf = np.zeros((B, D, T))
    rows = (kind == 1) & (tilt > 8)
    if rows.any():
        gcr = rng.uniform(0.3, 0.6, B); amp_r = rng.uniform(1.0, 3.0, B)
        r = S.row_shade(el, az, tilt[:, None, None], azr[:, None, None], gcr[:, None, None]) * amp_r[:, None, None]
        sf = np.where(rows[:, None, None], r, sf)
    for i in np.where(kind == 2)[0]:
        n_ob = int(rng.integers(1, 4)); pa, ph = [0.0, 359.9], [0.0, 0.0]
        for j in range(n_ob):
            c = rng.choice([rng.uniform(60, 140), rng.uniform(220, 300), rng.uniform(140, 220)], p=[0.4, 0.4, 0.2]); w = rng.uniform(10, 60); hgt = rng.uniform(5, 35)
            pa += [c - w / 2 - 5, c - w / 2, c + w / 2, c + w / 2 + 5]; ph += [0, hgt, hgt, 0]
        pa = np.array(pa) % 360; o = np.argsort(pa)
        sf[i] = S.horizon_shade(el[i], az[i], pa[o], np.array(ph)[o], rng.uniform(2, 8)) * rng.uniform(0.7, 1.5)
    sf = np.clip(sf, 0, 1)
    # ---- physics
    m = S.iam_ashrae(ca, b0)
    soil_t = np.clip(s[..., None] * (1 + 0.3 * np.clip(1 / np.maximum(ca, 0.2) - 1, 0, 2)), 0, 0.6)
    g_clean = beam * m + (sky + gnd) * 0.95
    g_eff = (beam * m * (1 - sf) + (sky + gnd) * 0.95 * (1 - 0.15 * sf)) * (1 - soil_t)
    tc = S.faiman(poa, ta, ws)
    lowf = lambda g: np.where(g > 1, 1 + lowirr * np.log(np.maximum(g, 1) / 1000.0), 0.0)
    E0 = eta0[:, None, None] * frac[..., None]
    pdc = S.pvwatts_dc(g_eff, tc, 1.0, gamma[:, None, None]) * E0 * lowf(g_eff)
    if REALISM:      # model mismatch the physics prior does not know about: spectral (airmass) response and a slow seasonal bias
        am = np.nan_to_num(S.airmass(np.maximum(cz, 0.02)), nan=10.0)
        spec = np.clip(1 + u(-0.015, 0.015)[:, None, None] * (np.minimum(am, 6) - 1.5), 0.92, 1.05)
        seas = 1 + u(-0.02, 0.02)[:, None, None] * np.cos(2 * np.pi * (DOY - u(0, 365)[:, None, None]) / 365.0)
        pdc = pdc * spec * seas
    pac0 = (1 / dcac)[:, None, None]
    pac = S.pvwatts_inv(pdc, 1.0, pac0); pac_nc = S.pvwatts_inv(pdc, 1.0, 10.0)
    g_us = g_clean * (1 - soil_t); pac_us = S.pvwatts_inv(S.pvwatts_dc(g_us, tc, 1.0, gamma[:, None, None]) * E0 * lowf(g_us), 1.0, 10.0)
    # outages and curtailment
    r = rng.random((B, D)); out = np.zeros((B, D, T)); out[r < 0.03] = 1
    part = (r >= 0.03) & (r < 0.08); a = rng.integers(20, 70, (B, D)); ln = rng.integers(2, 30, (B, D))
    tt = np.arange(T)[None, None]; out = np.where(part[..., None] & (tt >= a[..., None]) & (tt < (a + ln)[..., None]), 1.0, out)
    curt = np.where((rng.random(B) < 0.08)[:, None] & (rng.random((B, D)) < 0.3), u(0.3, 0.8)[:, None] / dcac[:, None], 10.0)
    pcl = np.minimum(pac, curt[..., None]); pfin = np.where(out > 0, 0.0, pcl)
    # ---- measurements
    noise = (0.003 + u(0, 0.01))[:, None, None]
    pm = pfin * (1 + noise * rng.standard_normal(pfin.shape)) + 0.001 * rng.standard_normal(pfin.shape) * (pfin > 0)
    sat = rng.random(B) < 0.2; gain = u(0.96, 1.04)[:, None, None]
    gm = poa * gain * (1 + np.where(sat, 0.1, 0.015)[:, None, None] * rng.standard_normal(poa.shape))
    roll = rng.random(B) < 0.15
    for i in np.where(roll)[0]: gm[i] = np.roll(gm[i], rng.choice([-1, 1]), axis=1)
    tm = ta + rng.normal(0, 1.0, B)[:, None, None]
    # missing data: random gaps and the odd missing day
    miss = rng.random((B, D, T)) < 0.01
    gap = rng.random((B, D)) < 0.03; miss |= gap[..., None]
    pm = np.where(miss, np.nan, pm); gm = np.where(miss, np.nan, gm / 1000.0)
    # nominal clear-sky POA (TL = 3, sea level), as available for real systems
    g3, dn3, dh3 = S.ineichen(cz, DOY, 3.0); b3, s3, n3 = S.haydavies(tilt[:, None, None], ca, cz, dn3, dh3, g3, DOY)
    X = np.stack([inputs(pm[i], gm[i], tm[i], np.maximum(cz[i], 0), np.maximum(ca[i], 0), (b3[i] + s3[i] + n3[i]) / 1000.0) for i in range(B)])
    eps = 1e-6
    lab = {'pf': eta0[:, None] * (1 - s) * frac, 'soil': s, 'frac': frac,
           'shade': np.clip(1 - pac_nc.sum(-1) / (pac_us.sum(-1) + eps), 0, 1),
           'clip': np.clip(1 - pcl.sum(-1) / (pac_nc.sum(-1) + eps), 0, 1),
           'outage': np.clip(1 - pfin.sum(-1) / (pcl.sum(-1) + eps), 0, 1),
           'step': np.concatenate([np.zeros((B, 1)), (np.abs(np.diff(frac, axis=1)) > 1e-9)], 1),
           'dfrac': np.concatenate([np.zeros((B, 1)), np.diff(frac, axis=1)], 1)}
    Y = np.stack([lab[k] for k in DAY_LABELS], 1).astype(np.float32)        # (B, 8, D)
    G = np.stack([eta0, 1 / dcac], 1).astype(np.float32)                    # (B, 2): baseline efficiency, AC limit / DC nameplate
    meta = dict(n_strings=N, kind=kind, rate=rate, sat=sat, typ=typ, lat=lat, tilt=tilt, azimuth=azr)
    return X, Y, G, meta
