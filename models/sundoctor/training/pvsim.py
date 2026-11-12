"""SunDoctor physics: a vectorized PV-system simulator in local solar time (15-min steps, 96 per day).

Sun position (Spencer/NOAA), Ineichen clear sky, a stochastic cloud process, Erbs decomposition, Hay-Davies transposition,
ASHRAE incidence-angle loss, Faiman cell temperature, PVWatts DC and inverter models with AC clipping, plus the things
that go wrong: dead strings, soiling with rain cleaning, obstacle and row-to-row shading, outages and sensor error.
Each building block is checked against pvlib in test_pvsim.py."""
import numpy as np

T_STEPS = 96
TSOL = (np.arange(T_STEPS) + 0.5) / 4.0          # local solar time at step centers (h)
D2R = np.pi / 180

# ----------------------------------------------------------------- sun and sky
def declination(doy):
    g = 2 * np.pi * (doy - 1) / 365.0
    return (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))

def eot_minutes(doy):
    g = 2 * np.pi * (doy - 1) / 365.0
    return 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))

def sun(lat, doy, tsol):
    """lat (deg), doy, solar time (h) -> cos(zenith), elevation (deg), azimuth (deg from north, clockwise)"""
    d = declination(doy); h = (tsol - 12.0) * 15 * D2R; p = lat * D2R
    cz = np.sin(p) * np.sin(d) + np.cos(p) * np.cos(d) * np.cos(h)
    cz = np.clip(cz, -1, 1); el = 90 - np.arccos(cz) / D2R
    az = np.arctan2(np.sin(h), np.cos(h) * np.sin(p) - np.tan(d) * np.cos(p)) / D2R + 180.0
    return cz, el, az

def cos_aoi(tilt, az_arr, cz, az):
    sz = np.sqrt(np.clip(1 - cz ** 2, 0, 1))
    return np.cos(tilt * D2R) * cz + np.sin(tilt * D2R) * sz * np.cos((az - az_arr) * D2R)

def extra(doy):
    return 1367.0 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365.0))

def airmass(cz):
    z = np.arccos(np.clip(cz, -1, 1)) / D2R
    am = 1.0 / (cz + 0.50572 * np.maximum(96.07995 - z, 1e-3) ** -1.6364)
    return np.where(cz > 0.01, am, np.nan)

def ineichen(cz, doy, tl, alt=0.0):
    """Ineichen-Perez clear sky (as pvlib, with absolute airmass at the given altitude)."""
    I0 = extra(doy); am = airmass(cz) * np.exp(-alt / 8434.5)
    fh1, fh2 = np.exp(-alt / 8000.0), np.exp(-alt / 1250.0); cg1, cg2 = 5.09e-5 * alt + 0.868, 3.92e-5 * alt + 0.0387
    ghi = cg1 * I0 * cz * np.exp(-cg2 * am * (fh1 + fh2 * (tl - 1)))
    b = 0.664 + 0.163 / fh1; dni = b * I0 * np.exp(-0.09 * am * (tl - 1))
    dni_lim = ghi * (1 - (0.1 - 0.2 * np.exp(-tl)) / (0.1 + 0.882 / fh1)) / cz
    dni = np.minimum(dni, dni_lim)
    ghi = np.nan_to_num(np.where(cz > 0.01, ghi, 0.0)); dni = np.nan_to_num(np.where(cz > 0.01, dni, 0.0))
    return ghi, dni, np.maximum(ghi - dni * np.maximum(cz, 0), 0)

def erbs(ghi, cz, doy):
    I0 = extra(doy); kt = np.where(cz > 0.065, ghi / (I0 * np.maximum(cz, 0.065)), 0.0); kt = np.clip(kt, 0, 1)
    df = np.where(kt <= 0.22, 1 - 0.09 * kt, np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4, 0.165))
    dhi = df * ghi; dni = np.where(cz > 0.065, (ghi - dhi) / np.maximum(cz, 0.065), 0.0)
    return np.maximum(dni, 0), dhi

def haydavies(tilt, ca, cz, dni, dhi, ghi, doy, albedo=0.2):
    """plane-of-array beam, sky-diffuse and ground-reflected irradiance"""
    I0 = extra(doy); A = np.clip(dni / I0, 0, 1); rb = np.where(cz > 0.0175, np.maximum(ca, 0) / np.maximum(cz, 0.0175), 0)
    beam = dni * np.maximum(ca, 0)
    sky = dhi * (A * rb + (1 - A) * (1 + np.cos(tilt * D2R)) / 2)
    gnd = ghi * albedo * (1 - np.cos(tilt * D2R)) / 2
    return beam, sky, gnd

def iam_ashrae(ca, b0=0.05):
    ca = np.clip(ca, 1e-3, 1); m = 1 - b0 * (1 / ca - 1); return np.clip(m, 0, 1)

def faiman(poa, ta, ws, u0=25.0, u1=6.84):
    return ta + poa / (u0 + u1 * ws)

def pvwatts_dc(g_eff, tc, pdc0, gamma):
    return pdc0 * g_eff / 1000.0 * (1 + gamma * (tc - 25.0))

def pvwatts_inv(pdc, pdc0_inv, pac0, eta_nom=0.96):
    """PVWatts inverter: efficiency curve, start-up threshold, AC clipping at pac0"""
    z = np.clip(pdc / pdc0_inv, 1e-6, None); eta = eta_nom / 0.9637 * (-0.0162 * z - 0.0059 / z + 0.9858)
    pac = np.where(pdc > 0, eta * pdc, 0.0); pac = np.maximum(pac, 0)
    return np.minimum(pac, pac0)

# ----------------------------------------------------------------- shading geometry
def row_shade(el, az, tilt, az_arr, gcr):
    """fraction of a fixed-tilt row's slant height shaded by the row in front (2-D profile-angle model)"""
    tp = np.tan(np.clip(el, 0.1, 89.9) * D2R) / np.maximum(np.cos((az - az_arr) * D2R), 1e-3)
    phi = np.where(np.cos((az - az_arr) * D2R) > 1e-3, np.arctan(tp), np.pi / 2)
    fs = 1 - np.sin(phi) / (gcr * np.sin(phi + tilt * D2R))
    return np.where((el > 0) & (np.cos((az - az_arr) * D2R) > 1e-3), np.clip(fs, 0, 1), np.where(el > 0, 0.0, 1.0))

def horizon_shade(el, az, prof_az, prof_h, width):
    """obstacle profile: elevation prof_h(deg) at azimuths prof_az; returns shaded fraction of the array (soft edge of width deg)"""
    h = np.interp(az, prof_az, prof_h, period=360)
    return np.clip((h - el) / width + 0.5, 0, 1)

# ----------------------------------------------------------------- weather
def cloud_days(rng, n_days):
    """per day: (type, clear-sky index series k(t)). Types: 0 clear, 1 hazy, 2 broken clouds, 3 overcast"""
    p = np.array([0.42, 0.16, 0.27, 0.15]); typ = rng.choice(4, size=n_days, p=p)
    k = np.empty((n_days, T_STEPS))
    for i, ty in enumerate(typ):
        if ty == 0: k[i] = 1.0 + 0.01 * rng.standard_normal(T_STEPS)
        elif ty == 1: k[i] = rng.uniform(0.75, 0.95) + 0.03 * np.cumsum(rng.standard_normal(T_STEPS)) / np.sqrt(T_STEPS)
        elif ty == 3:
            base = rng.uniform(0.08, 0.45); k[i] = base * np.exp(0.25 * np.cumsum(rng.standard_normal(T_STEPS)) / np.sqrt(T_STEPS))
        else:   # broken clouds: two-state Markov chain with cloud enhancement in the clear state
            pc = rng.uniform(0.2, 0.8); st = rng.random() < pc; lo = rng.uniform(0.15, 0.6)
            for t in range(T_STEPS):
                if rng.random() < 0.35: st = rng.random() < pc
                k[i, t] = (lo + 0.1 * rng.standard_normal()) if st else (1.0 + abs(0.06 * rng.standard_normal()))
    return typ, np.clip(k, 0.03, 1.35)

def temperature(rng, n_days, lat, doy, typ):
    season = np.cos(2 * np.pi * (doy - 200) / 365.0)
    tmean = rng.uniform(4, 22) + (14 - 0.15 * abs(lat - 30)) * season + 2 * rng.standard_normal(n_days)
    amp = np.where(typ == 3, rng.uniform(2, 5, n_days), rng.uniform(5, 11, n_days))
    return tmean[:, None] + amp[:, None] * np.sin(2 * np.pi * (TSOL[None] - 9.0) / 24.0)


# ----------------------------------------------------------------- one simulated system, D days
def simulate(rng, D=14, fixed=None):
    """Returns inputs (D, T) for measured AC power / DC nameplate, POA sensor / 1000, ambient temp, plus geometry,
    and per-day labels. `fixed` can pin system parameters (dict)."""
    f = fixed or {}
    lat = f.get('lat', rng.uniform(20, 52)); tilt = f.get('tilt', rng.choice([rng.uniform(0, 12), rng.uniform(10, 40)], p=[0.25, 0.75]))
    az_arr = f.get('azimuth', float(np.clip(180 + rng.normal(0, 35), 90, 270)))
    doy0 = f.get('doy0', rng.integers(1, 366)); doy = (doy0 + np.arange(D) - 1) % 365 + 1
    dcac = f.get('dcac', rng.uniform(0.95, 1.5)); gamma = rng.uniform(-0.0048, -0.0028); eta0 = f.get('eta0', rng.uniform(0.80, 1.02))
    tl = rng.uniform(2.0, 4.5, D)[:, None]; alt = rng.uniform(0, 1800); ws = rng.uniform(0.5, 5.0, D)[:, None]
    lowirr = rng.uniform(0.0, 0.025); albedo = rng.uniform(0.12, 0.3); b0 = rng.uniform(0.03, 0.07)
    cz, el, az = sun(lat, doy[:, None], TSOL[None]); ca = cos_aoi(tilt, az_arr, cz, az)
    gc, dnic, dhic = ineichen(cz, doy[:, None], tl, alt)
    typ, k = cloud_days(rng, D); ghi = gc * k
    dni, dhi = erbs(ghi, cz, doy[:, None])
    clear = typ[:, None] <= 1
    dni = np.where(clear, dnic * np.where(typ[:, None] == 1, k, 1.0) ** 1.5, dni); dhi = np.where(clear, np.maximum(ghi - dni * np.maximum(cz, 0), 0), dhi)
    beam, sky, gnd = haydavies(tilt, ca, cz, dni, dhi, ghi, doy[:, None], albedo)
    poa_true = beam + sky + gnd
    ta = temperature(rng, D, lat, doy, typ)
    # ---- what goes wrong
    lab = {}
    # dead strings: N strings per inverter; a step (failure or repair) may happen inside the window
    N = f.get('n_strings', int(rng.choice([1, 2, 3, 4, 6, 8, 12])))
    off0 = 0 if rng.random() < 0.6 else int(rng.integers(0, max(1, N // 2) + 1)) if N > 1 else 0
    off = np.full(D, off0, float)
    if rng.random() < 0.45 and N > 1:
        d = int(rng.integers(1, D)); off[d:] = int(np.clip(off0 + rng.choice([-2, -1, 1, 1, 2, 3]), 0, N - 1))
    frac = 1 - off / N
    # soiling: linear accumulation, rain / cleaning resets
    rate = rng.choice([0.0, rng.uniform(0, 0.0015), rng.uniform(0.0015, 0.006)], p=[0.35, 0.4, 0.25])
    s = np.empty(D); cur = rng.uniform(0, 0.12) if rate > 0 else 0.0
    for d in range(D):
        if (typ[d] == 3 and rng.random() < 0.55) or rng.random() < 0.01: cur *= rng.uniform(0.0, 0.3)
        cur = min(cur + rate, 0.35); s[d] = cur
    # shading: row-to-row (ground mount) and/or an obstacle on the horizon
    shade_kind = rng.choice(3, p=[0.45, 0.25, 0.30])
    sf = np.zeros_like(el)
    if shade_kind == 1 and tilt > 8:
        sf = row_shade(el, az, tilt, az_arr, rng.uniform(0.3, 0.6)) * rng.uniform(1.0, 3.0)
    elif shade_kind == 2:
        n_ob = int(rng.integers(1, 4)); pa = np.sort(rng.uniform(0, 360, n_ob * 2)); ph = np.zeros(n_ob * 2)
        for j in range(n_ob):
            c = rng.choice([rng.uniform(60, 140), rng.uniform(220, 300), rng.uniform(140, 220)], p=[0.4, 0.4, 0.2]); w = rng.uniform(10, 60); hgt = rng.uniform(5, 35)
            pa = np.concatenate([pa, [c - w / 2 - 5, c - w / 2, c + w / 2, c + w / 2 + 5]]); ph = np.concatenate([ph, [0, hgt, hgt, 0]])
        o = np.argsort(pa); sf = horizon_shade(el, az, pa[o], ph[o], rng.uniform(2, 8)) * rng.uniform(0.7, 1.5)
    sf = np.clip(sf, 0, 1)
    # ---- physics: what the array actually produces
    m = iam_ashrae(ca, b0)
    g_eff_clean = beam * m + (sky + gnd) * 0.95
    soil_t = s[:, None] * (1 + 0.3 * np.clip(1 / np.maximum(ca, 0.2) - 1, 0, 2))
    g_eff = (beam * m * (1 - sf) + (sky + gnd) * 0.95 * (1 - 0.15 * sf)) * (1 - np.clip(soil_t, 0, 0.6))
    tc = faiman(poa_true, ta, ws)
    lowf = np.where(g_eff > 1, 1 + lowirr * np.log(np.maximum(g_eff, 1) / 1000.0), 0.0)
    pdc0 = 1.0
    pdc = pvwatts_dc(g_eff, tc, pdc0, gamma) * eta0 * frac[:, None] * lowf
    pac0 = pdc0 / dcac
    pac = pvwatts_inv(pdc, pdc0, pac0)
    # references for labels: no clipping, no shading, no outage
    pac_noclip = pvwatts_inv(pdc, pdc0, 10.0)
    pdc_unshaded = pvwatts_dc(g_eff_clean * (1 - np.clip(soil_t, 0, 0.6)), tc, pdc0, gamma) * eta0 * frac[:, None] * np.where(g_eff_clean > 1, 1 + lowirr * np.log(np.maximum(g_eff_clean, 1) / 1000.0), 0.0)
    pac_unshaded = pvwatts_inv(pdc_unshaded, pdc0, 10.0)
    # outages: whole day, or part of a day (trip and restart); occasional export curtailment
    out = np.zeros_like(pac)
    for d in range(D):
        r = rng.random()
        if r < 0.03: out[d] = 1
        elif r < 0.08:
            a = int(rng.integers(20, 70)); out[d, a:a + int(rng.integers(2, 30))] = 1
    curt = np.full(D, 10.0)
    if rng.random() < 0.08: curt[rng.random(D) < 0.3] = rng.uniform(0.3, 0.8) * pac0
    pac_final = np.where(out > 0, 0.0, np.minimum(pac, curt[:, None]))
    # ---- measurements
    noise = 0.003 + rng.uniform(0, 0.01)
    p_meas = pac_final * (1 + noise * rng.standard_normal(pac.shape)) + 0.001 * rng.standard_normal(pac.shape) * (pac_final > 0)
    gain = rng.uniform(0.96, 1.04); sat = rng.random() < 0.2        # 20%: irradiance from a noisier source (satellite-like)
    gerr = (0.1 if sat else 0.015) * rng.standard_normal(poa_true.shape)
    poa_meas = poa_true * gain * (1 + gerr)
    if rng.random() < 0.15: poa_meas = np.roll(poa_meas, rng.choice([-1, 1]), axis=1)      # timestamp misalignment
    ta_meas = ta + rng.normal(0, 1.0)
    # ---- labels (per day): energy-weighted
    eps = 1e-6; e_ns = pac_unshaded.sum(1) + eps
    lab['frac'] = frac; lab['soil'] = s; lab['eta0'] = np.full(D, eta0)
    lab['pf'] = eta0 * (1 - s) * frac                                   # performance factor of the unshaded array
    lab['shade'] = np.clip(1 - pac_noclip.sum(1) / e_ns, 0, 1)
    lab['clip'] = np.clip(1 - np.minimum(pac, curt[:, None]).sum(1) / (pac_noclip.sum(1) + eps), 0, 1)
    lab['outage'] = np.clip(1 - pac_final.sum(1) / (np.minimum(pac, curt[:, None]).sum(1) + eps), 0, 1)
    lab['step'] = np.concatenate([[0.0], (np.abs(np.diff(frac)) > 1e-9).astype(float)])
    lab['dfrac'] = np.concatenate([[0.0], np.diff(frac)])
    meta = dict(lat=lat, tilt=tilt, azimuth=az_arr, dcac=dcac, eta0=eta0, n_strings=N, doy0=int(doy0), shade_kind=int(shade_kind), rate=rate, sat=bool(sat))
    geo = dict(cz=np.maximum(cz, 0), ca=np.maximum(ca, 0), el=el, az=az, poa_clear=haydavies(tilt, ca, cz, dnic, dhic, gc, doy[:, None], albedo)[0] + haydavies(tilt, ca, cz, dnic, dhic, gc, doy[:, None], albedo)[1])
    return dict(p=p_meas.astype(np.float32), poa=poa_meas.astype(np.float32), ta=ta_meas.astype(np.float32), geo=geo, lab=lab, meta=meta, typ=typ)
