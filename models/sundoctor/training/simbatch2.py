"""SunDoctor v2 simulator: everything in v1 plus single-axis trackers (with backtracking, stow and stuck rows), snow,
thin-film (CdTe-like) modules, irradiance-sensor misalignment, morning-only string dropouts, model mismatch, and a
home mode in which no irradiance sensor exists at all."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np, sys
sys.path.insert(0, _P.SRC)
import pvsim as S, tracker as TR

T = S.T_STEPS
CH = ['p', 'poa', 'has_poa', 'poa_clear', 'p_expected', 'p_clear', 'temp', 'cos_zenith', 'cos_aoi', 'valid']
DAY_LABELS = ['pf', 'soil', 'frac', 'shade', 'clip', 'outage', 'step', 'dfrac', 'snow', 'stuck']

def expected_power(g, ta, ca):
    geff = g * 1000.0 * (0.25 + 0.75 * S.iam_ashrae(np.maximum(ca, 1e-3), 0.05))
    tc = S.faiman(g * 1000.0, ta, 2.0)
    return S.pvwatts_inv(S.pvwatts_dc(geff, tc, 1.0, -0.0037), 1.0, 10.0)

def inputs(p, g, ta, cz, ca, gclear, has_poa):
    """(C, D, T) channels. g may be None / all-NaN when there is no irradiance sensor (home mode)."""
    pv = np.isfinite(p); t0 = np.where(np.isfinite(ta), ta, 15.0)
    if has_poa:
        valid = pv & np.isfinite(g); g0 = np.where(valid, g, 0.0)
        pe = np.where(valid, expected_power(np.maximum(g0, 0), t0, ca), 0.0)
    else:
        valid = pv; g0 = np.zeros_like(p, dtype=float)
    pc = expected_power(gclear, t0, ca)
    if not has_poa: pe = np.where(valid, pc, 0.0)
    p0 = np.where(valid, p, 0.0)
    return np.stack([p0, g0, np.full_like(p0, float(has_poa)), gclear, pe, pc, t0 / 40.0, cz, ca, valid.astype(float)]).astype(np.float32)

def geometry(cz, az, tilt, azr, trk, axis_az, max_ang, gcr, bt=True):
    """per-step surface tilt / azimuth / cos AOI for fixed (trk False) or ideally tracking systems (bt: backtracking)"""
    th = np.where(bt, TR.rotation(cz, az, axis_az, max_ang, gcr, True), TR.rotation(cz, az, axis_az, max_ang, gcr, False))
    t_tilt, t_az = TR.surface(th, axis_az); t_ca = TR.cos_aoi(th, cz, az, axis_az)
    f_ca = S.cos_aoi(tilt, azr, cz, az)
    return np.where(trk, t_tilt, tilt), np.where(trk, t_az, azr), np.where(trk, t_ca, f_ca), th

FRONTS = False     # rung 8: synoptic weather (day-to-day temperature swings and cold fronts arriving with overcast days)

def simulate_batch(rng, B, D=14, home=None, fixed=None):
    f = fixed or {}
    u = lambda a, b, *s: rng.uniform(a, b, (B,) + s)
    lat = f.get('lat', u(20, 52)); trk = f.get('tracker', rng.random(B) < 0.4)
    tilt = np.where(rng.random(B) < 0.25, u(0, 12), u(10, 40)); azr = np.clip(180 + rng.normal(0, 35, B), 90, 270)
    axis_az = 180 + rng.normal(0, 4, B); max_ang = rng.choice([45.0, 50.0, 52.0, 55.0, 60.0], B); gcr = u(0.28, 0.5)
    bt = f.get('backtrack', rng.random(B) < 0.6)                         # 40% of trackers true-track and shade each other at low sun
    doy0 = f.get('doy0', rng.integers(1, 366, B)); doy = (doy0[:, None] + np.arange(D)[None] - 1) % 365 + 1
    dcac = u(0.95, 1.5); eta0 = u(0.80, 1.02)
    cdte = rng.random(B) < 0.3
    gamma = np.where(cdte, u(-0.0032, -0.0022), u(-0.0048, -0.0028)); lowirr = np.where(cdte, u(-0.01, 0.01), u(0, 0.025))[:, None, None]
    tl = u(2.0, 4.5, D)[..., None]; alt = u(0, 1800)[:, None, None]; ws = u(0.5, 5.0, D)[..., None]
    b0 = u(0.03, 0.07)[:, None, None]
    L = lat[:, None, None]; Tt = S.TSOL[None, None]; DOY = doy[..., None]
    cz, el, az = S.sun(L, DOY, Tt)
    K = lambda a: a[:, None, None]
    s_tilt, s_az, ca, th_ideal = geometry(cz, az, K(tilt), K(azr), K(trk), K(axis_az), K(max_ang), K(gcr), K(bt))
    gc, dnic, dhic = S.ineichen(cz, DOY, tl, alt)
    typ = rng.choice(4, size=(B, D), p=[0.42, 0.16, 0.27, 0.15])
    walk = np.cumsum(rng.standard_normal((B, D, T)), -1) / np.sqrt(T)
    k_clear = 1.0 + 0.01 * rng.standard_normal((B, D, T)); k_hazy = u(0.75, 0.95, D)[..., None] + 0.03 * walk
    k_over = u(0.08, 0.45, D)[..., None] * np.exp(0.25 * walk)
    pc = u(0.2, 0.8, D); lo = u(0.15, 0.6, D); st = rng.random((B, D)) < pc; kb = np.empty((B, D, T))
    for t in range(T):
        sw = rng.random((B, D)) < 0.35; st = np.where(sw, rng.random((B, D)) < pc, st)
        kb[..., t] = np.where(st, lo + 0.1 * rng.standard_normal((B, D)), 1.0 + np.abs(0.06 * rng.standard_normal((B, D))))
    k = np.clip(np.select([typ[..., None] == 0, typ[..., None] == 1, typ[..., None] == 2], [k_clear, k_hazy, kb], k_over), 0.03, 1.35)
    ghi = gc * k; dni, dhi = S.erbs(ghi, cz, DOY); clr = typ[..., None] <= 1
    dni = np.where(clr, dnic * np.where(typ[..., None] == 1, k, 1.0) ** 1.5, dni); dhi = np.where(clr, np.maximum(ghi - dni * np.maximum(cz, 0), 0), dhi)
    season = np.cos(2 * np.pi * (doy - 200) / 365.0)
    tmean = u(4, 22)[:, None] + (14 - 0.15 * np.abs(lat[:, None] - 30)) * season + 2 * rng.standard_normal((B, D))
    cold = rng.random(B) < 0.3; tmean = np.where(cold[:, None], tmean - u(8, 18)[:, None] * np.clip(-season, 0, 1), tmean)   # cold-winter climates
    if FRONTS:   # continental weather: warm spells, then a cold front arrives with the clouds and lingers for a few days
        sig = u(1.0, 5.0); x = rng.standard_normal(B) * sig; a = np.zeros((B, D)); drop = np.where(rng.random(B) < 0.5, u(4, 14), u(0, 4))
        for d in range(D):
            front = (typ[:, d] == 3) & (rng.random(B) < 0.6)
            x = 0.7 * x + 0.71 * sig * rng.standard_normal(B) - np.where(front, drop * rng.uniform(0.5, 1.2, B), 0.0); a[:, d] = x
        tmean = tmean + a + 1.4                                     # +1.4: the fronts' average cooling, so climates keep their means
    amp = np.where(typ == 3, u(2, 5, D), u(5, 11, D)); ta = tmean[..., None] + amp[..., None] * np.sin(2 * np.pi * (Tt - 9.0) / 24.0)
    # ---- snow: falls on cold overcast days, slides and melts off over the following days
    snow = np.zeros((B, D)); c = np.zeros(B)
    for d in range(D):
        fall = (typ[:, d] == 3) & (tmean[:, d] < 1.0) & (lat > 28) & (rng.random(B) < 0.6)
        c = np.where(fall, np.maximum(c, rng.uniform(0.6, 1.0, B)), c)
        sunny = typ[:, d] <= 1; warm = (tmean[:, d] + amp[:, d]) > 2
        shed = (0.12 + 0.35 * warm * (0.5 + sunny) + 0.2 * np.where(trk, 1.0, tilt / 45)) * rng.uniform(0.5, 1.5, B) * np.where(trk, 1.4, 1.0)
        snow[:, d] = c; c = np.where(fall, c, np.clip(c - shed, 0, 1))
    alb = np.where(snow[..., None] > 0.05, 0.65, u(0.12, 0.3)[:, None, None])
    # ---- trackers: some rows stuck at a fixed angle, and the occasional all-row stow (wind or snow)
    stuck_frac = np.where(trk & (rng.random(B) < 0.22), rng.choice([0.1, 0.25, 0.5, 1.0], B), 0.0)
    d0 = rng.integers(0, D, B); d1 = d0 + rng.integers(1, D + 1, B); dd = np.arange(D)[None]
    stuck_on = (dd >= d0[:, None]) & (dd < d1[:, None]) & (stuck_frac[:, None] > 0)
    th_stuck = rng.uniform(-1, 1, B) * max_ang
    stow = trk[:, None] & ((rng.random((B, D)) < 0.04) | (snow > 0.3))
    s0 = rng.integers(28, 60, (B, D)); s1 = s0 + rng.integers(4, 30, (B, D)); tt = np.arange(T)[None, None]
    stow_t = stow[..., None] & (tt >= s0[..., None]) & (tt < s1[..., None])
    stow_ang = np.where(snow[..., None] > 0.3, K(max_ang), 0.0)
    def comps(theta_or_none, tilt_, az_, ca_):
        b_, s_, g_ = S.haydavies(tilt_, ca_, cz, dni, dhi, ghi, DOY, alb); return b_, s_, g_, ca_
    # ideal tracking (or fixed) surface
    bI, sI, gI, caI = comps(None, s_tilt, s_az, ca)
    # actual surface for stuck / stowed rows (trackers only)
    th_act = np.where(stow_t, stow_ang, np.where(stuck_on[..., None], K(th_stuck), th_ideal))
    a_tilt, a_az = TR.surface(th_act, K(axis_az)); a_ca = TR.cos_aoi(th_act, cz, az, K(axis_az))
    bA, sA, gA, _ = comps(None, a_tilt, a_az, a_ca)
    wA = np.where(stow_t, 1.0, np.where(stuck_on[..., None], K(stuck_frac), 0.0)) * K(trk)       # share of the array off-tracking
    # ---- dead strings, soiling, shading (as v1)
    N = rng.choice([1, 2, 3, 4, 6, 8, 12], B)
    off0 = np.where(rng.random(B) < 0.6, 0, np.floor(rng.random(B) * (np.maximum(N // 2, 0) + 1))); off0 = np.where(N > 1, off0, 0)
    stepday = rng.integers(1, D, B); has = (rng.random(B) < 0.45) & (N > 1); off1 = np.clip(off0 + rng.choice([-2, -1, 1, 1, 2, 3], B), 0, N - 1)
    off = np.where(has[:, None] & (dd >= stepday[:, None]), off1[:, None], off0[:, None]); frac = 1 - off / N[:, None]
    rate = np.select([rng.random(B) < 0.35, rng.random(B) < 0.62], [0.0, rng.uniform(0, 0.0015, B)], rng.uniform(0.0015, 0.006, B))
    cur = np.where(rate > 0, np.where(rng.random(B) < 0.5, u(0, 0.06), u(0, 0.2)), 0.0); s = np.empty((B, D))
    for d in range(D):
        wash = ((typ[:, d] == 3) & (rng.random(B) < 0.55)) | (rng.random(B) < 0.01) | (snow[:, d] > 0.3)
        cur = np.where(wash, cur * rng.uniform(0, 0.3, B), cur); cur = np.minimum(cur + rate, 0.35); s[:, d] = cur
    kind = rng.choice(3, size=B, p=[0.5, 0.22, 0.28]); sf = np.zeros((B, D, T))
    rows = (kind == 1) & (tilt > 8) & ~trk
    if rows.any():
        r = S.row_shade(el, az, K(tilt), K(azr), K(rng.uniform(0.3, 0.6, B))) * K(rng.uniform(1.0, 3.0, B)); sf = np.where(rows[:, None, None], r, sf)
    for i in np.where(kind == 2)[0]:
        n_ob = int(rng.integers(1, 4)); pa, ph = [0.0, 359.9], [0.0, 0.0]
        for j in range(n_ob):
            cc = rng.choice([rng.uniform(60, 140), rng.uniform(220, 300), rng.uniform(140, 220)], p=[0.4, 0.4, 0.2]); w = rng.uniform(10, 60); hgt = rng.uniform(5, 35)
            pa += [cc - w / 2 - 5, cc - w / 2, cc + w / 2, cc + w / 2 + 5]; ph += [0, hgt, hgt, 0]
        pa = np.array(pa) % 360; o = np.argsort(pa); sf[i] = S.horizon_shade(el[i], az[i], pa[o], np.array(ph)[o], rng.uniform(2, 8)) * rng.uniform(0.7, 1.5)
    # true-tracking rows shade each other at low sun (thin-film strip cells lose power about linearly; c-Si via bypass diodes faster)
    th_s = TR.true_angle(cz, az, K(axis_az)); rsf = TR.shaded_fraction(th_ideal, th_s, K(gcr)) * K(np.where(cdte, u(0.8, 1.2), u(1.0, 3.0)))
    sf = np.where(K(trk & ~bt) & (cz > 0), np.maximum(sf, rsf), sf)
    sf = np.clip(sf, 0, 1)
    # ---- physics chain
    def g_eff(b_, s_, g_, ca_, shade, soil):
        m = S.iam_ashrae(ca_, b0); so = np.clip(soil[..., None] * (1 + 0.3 * np.clip(1 / np.maximum(ca_, 0.2) - 1, 0, 2)), 0, 0.6)
        return (b_ * m * (1 - shade) + (s_ + g_) * 0.95 * (1 - 0.15 * shade)) * (1 - so)
    gi = g_eff(bI, sI, gI, caI, sf, s); ga = g_eff(bA, sA, gA, a_ca, sf, s)
    g_act = (1 - wA) * gi + wA * ga                                          # what the array really collects (shaded, soiled)
    g_ns = (1 - wA) * g_eff(bI, sI, gI, caI, 0 * sf, s) + wA * g_eff(bA, sA, gA, a_ca, 0 * sf, s)
    g_ideal_ns = g_eff(bI, sI, gI, caI, 0 * sf, s)
    tsn = 1 - np.clip(1.3 * snow, 0, 1)[..., None]
    poa_true = bI + sI + gI
    tc = S.faiman(poa_true, ta, ws)
    lowf = lambda g: np.where(g > 1, 1 + lowirr * np.log(np.maximum(g, 1) / 1000.0), 0.0)
    am = np.nan_to_num(S.airmass(np.maximum(cz, 0.02)), nan=10.0)
    spec = np.clip(1 + np.where(cdte, u(-0.03, 0.01), u(-0.015, 0.015))[:, None, None] * (np.minimum(am, 6) - 1.5), 0.9, 1.05)
    seas = 1 + u(-0.02, 0.02)[:, None, None] * np.cos(2 * np.pi * (DOY - u(0, 365)[:, None, None]) / 365.0)
    E0 = K(eta0) * frac[..., None] * spec * seas
    # morning-only string dropouts (e.g. insulation faults until dew dries): counted as outage energy
    dew = (rng.random(B) < 0.1)[:, None] & (rng.random((B, D)) < 0.5); t_r = rng.uniform(7, 10.5, (B, D))
    mfrac = np.where(dew[..., None] & (Tt < t_r[..., None]), 1 - 1.0 / np.maximum(K(N), 1), 1.0)
    P = lambda g, cap: S.pvwatts_inv(S.pvwatts_dc(g, tc, 1.0, K(gamma)) * E0 * lowf(g), 1.0, cap)
    pac0 = K(1 / dcac)
    p_ideal = P(g_ideal_ns, 10.0); p_track = P(g_ns, 10.0); p_shade = P(g_act, 10.0); p_snow = P(g_act * tsn, 10.0)
    pac = S.pvwatts_inv(S.pvwatts_dc(g_act * tsn, tc, 1.0, K(gamma)) * E0 * lowf(g_act * tsn), 1.0, pac0)
    r = rng.random((B, D)); out = np.zeros((B, D, T)); out[r < 0.03] = 1
    part = (r >= 0.03) & (r < 0.08); a = rng.integers(20, 70, (B, D)); ln = rng.integers(2, 30, (B, D))
    out = np.where(part[..., None] & (tt >= a[..., None]) & (tt < (a + ln)[..., None]), 1.0, out)
    curt = np.where((rng.random(B) < 0.08)[:, None] & (rng.random((B, D)) < 0.3), u(0.3, 0.8)[:, None] / dcac[:, None], 10.0)
    pcl = np.minimum(pac, curt[..., None]); pfin = np.where(out > 0, 0.0, pcl * mfrac)
    # ---- measurements
    noise = (0.003 + u(0, 0.01))[:, None, None]
    pm = pfin * (1 + noise * rng.standard_normal(pfin.shape)) + 0.001 * rng.standard_normal(pfin.shape) * (pfin > 0)
    # irradiance sensor: misaligned (40%), noisy or satellite-like, sometimes snow-covered, or absent (home mode)
    mis = rng.random(B) < 0.4; dtilt = np.where(mis, u(-8, 8), 0); daz = np.where(mis, u(-15, 15), 0)
    sens_tilt = np.clip(s_tilt + K(dtilt), 0, 90); sens_az = s_az + K(daz)
    sens_ca = S.cos_aoi(sens_tilt, sens_az, cz, az); bS, sS, gS = S.haydavies(sens_tilt, sens_ca, cz, dni, dhi, ghi, DOY, alb)
    sat = rng.random(B) < 0.2; gain = u(0.96, 1.04)[:, None, None]
    gm = (bS + sS + gS) * gain * (1 + np.where(sat, 0.1, 0.015)[:, None, None] * rng.standard_normal(poa_true.shape))
    sens_snow = (rng.random(B) < 0.3)[:, None] & (snow > 0.3); gm = np.where(sens_snow[..., None], gm * rng.uniform(0.1, 0.6, (B, D, 1)), gm)
    for i in np.where(rng.random(B) < 0.15)[0]: gm[i] = np.roll(gm[i], rng.choice([-1, 1]), axis=1)
    home = (rng.random(B) < 0.4) if home is None else np.asarray(home, bool)
    tm = ta + rng.normal(0, 1.0, B)[:, None, None] + np.where(home, 1, 0)[:, None, None] * rng.normal(0, 2.0, (B, D, 1))
    miss = rng.random((B, D, T)) < 0.01; miss |= (rng.random((B, D)) < 0.03)[..., None]
    pm = np.where(miss, np.nan, pm); gm = np.where(miss, np.nan, gm / 1000.0)
    g3, dn3, dh3 = S.ineichen(cz, DOY, 3.0); b3, s3, n3 = S.haydavies(s_tilt, ca, cz, dn3, dh3, g3, DOY)
    X = np.stack([inputs(pm[i], None if home[i] else gm[i], tm[i], np.maximum(cz[i], 0), np.maximum(ca[i], 0), (b3[i] + s3[i] + n3[i]) / 1000.0, not home[i]) for i in range(B)])
    eps = 1e-6; E = lambda x: x.sum(-1) + eps
    lab = {'pf': K(eta0)[..., 0] * (1 - s) * frac, 'soil': s, 'frac': frac,
           'stuck': np.clip(1 - E(p_track) / E(p_ideal), 0, 1), 'shade': np.clip(1 - E(p_shade) / E(p_track), 0, 1),
           'snow': np.clip(1 - E(p_snow) / E(p_shade), 0, 1), 'clip': np.clip(1 - E(pcl) / E(p_snow), 0, 1),
           'outage': np.clip(1 - E(pfin) / E(pcl), 0, 1),
           'step': np.concatenate([np.zeros((B, 1)), (np.abs(np.diff(frac, axis=1)) > 1e-9)], 1), 'dfrac': np.concatenate([np.zeros((B, 1)), np.diff(frac, axis=1)], 1)}
    Y = np.stack([lab[k] for k in DAY_LABELS], 1).astype(np.float32)
    G = np.stack([eta0, 1 / dcac], 1).astype(np.float32)
    meta = dict(backtrack=bt & trk, n_strings=N, kind=kind, rate=rate, sat=sat, typ=typ, tracker=trk, home=home, cdte=cdte, snow_any=(snow > 0).any(1), stuck_any=stuck_on.any(1) | stow.any(1))
    return X, Y, G, meta
