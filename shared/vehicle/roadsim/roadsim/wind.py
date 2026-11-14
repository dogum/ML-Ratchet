"""Wind at the vehicle (RoadSim physics v2, work package B3; options `wind_profile`, `wind_gusts`, `drafting`).

v1 applies the trip's wind (in real trips: the 10 m wind of the nearest airport station) at road level, steady over
each second. This module adds, each behind an option:

* Road-height mean wind (`wind_profile: 'log'`): the EN 1991-1-4 (Eurocode 1, sec. 4.3) logarithmic profile with
  terrain roughness, carried from the station's terrain (z0_ref, open airport 0.03 m) to the road's (z0_site):
      v_b = U10 / (k_r(z0_ref) ln(10 / z0_ref)),    U(z) = k_r(z0_site) ln(max(z, z_min) / z0_site) v_b,
      k_r(z0) = 0.19 (z0 / 0.05)^0.07,   z_min(z0) from EN 1991-1-4 table 4.1 (1 m at z0 <= 0.01 ... 10 m at 1 m).
  z is the drag-effective height, 0.6 x the vehicle height. Times a sheltering factor (`wind_shelter`, or a per-second
  trip array 'wind_shelter') for cuttings, noise walls, forest.
  Example: 5 m/s at the airport -> 3.2 m/s at a car on a road with hedges (z0 0.1), 2.9 m/s in suburbs (z0 0.3).
* Turbulence (`wind_gusts`): along- and across-wind gusts with the EN 1991-1-4 Annex B (Kaimal-type) spectrum
      f S(f) / sigma^2 = 6.8 f_L / (1 + 10.2 f_L)^(5/3),  f_L = f L / V,  L(z) = 300 (max(z, z_min)/200)^(0.67+0.05 ln z0)
  with sigma_u = k_r(z0_site) v_b (EN 1991-1-4 eq. 4.6, k_I = 1) and sigma_v = 0.75 sigma_u, L_v = 0.35 L_u (ESDU 85020
  ratios near the ground). The field is frozen (Taylor) and laid out along the air-relative path the vehicle travels,
  xi = integral (v_target + U_road) dt, then block-averaged over each integration step, so a 1 Hz run sees 1-second
  means and a 10 Hz run sees the faster gusts. Seeded (`wind_turb_seed`): the same trip gives the same gusts.
* Drafting (`drafting`): CdA x (1 - d0 / (1 + gap/g0)^1.5) behind a lead vehicle at `gap` m (trip array 'gap_m',
  NaN = no lead; or `draft_headway_s` x v + 2 m). d0 = 0.45, g0 = 6 m for light vehicles, 15 m for heavy ones: about
  -10 % CdA at a 10 m gap and -1.6 % at 50 m for cars (Zabat et al. 1995, PATH two-car platoons), -21 % at 10 m for
  trucks (McAuliffe et al. 2018, NRC truck-platoon track tests: 25-35 % drag reduction of the trailing truck at 10-17 m).
"""
import math
import numpy as np

KR_REF_Z0 = 0.05                     # EN 1991-1-4 terrain category II roughness, m
ZMIN_TABLE = ([0.003, 0.01, 0.05, 0.3, 1.0], [1.0, 1.0, 2.0, 5.0, 10.0])     # z0 (m) -> z_min (m), EN 1991-1-4 table 4.1
DRAG_HEIGHT_FRAC = 0.6               # drag-effective height / vehicle height
SIGMA_V_RATIO = 0.75                 # sigma_v / sigma_u near the ground (ESDU 85020)
LV_RATIO = 0.35                      # L_v / L_u
DRAFT = dict(light=dict(d0=0.45, g0=6.0), heavy=dict(d0=0.45, g0=15.0))
DRAFT_STANDSTILL_GAP_M = 2.0


def k_r(z0):
    return 0.19 * (z0 / KR_REF_Z0) ** 0.07


def z_min(z0):
    lz = np.log(ZMIN_TABLE[0])
    return float(np.interp(math.log(max(z0, 1e-4)), lz, ZMIN_TABLE[1]))


def profile_factor(z, z0_ref, z0_site):
    """U(z at the site) / U10 (at the station)."""
    vb = 1.0 / (k_r(z0_ref) * math.log(10.0 / z0_ref))
    return k_r(z0_site) * math.log(max(z, z_min(z0_site)) / z0_site) * vb


def sigma_u_factor(z0_ref, z0_site, log_profile=True, z=1.0):
    """sigma_u / U_input: per station wind (log profile) or per road-level wind (no profile, I = 1/ln(z/z0))."""
    if log_profile:
        return k_r(z0_site) / (k_r(z0_ref) * math.log(10.0 / z0_ref))
    return 1.0 / math.log(max(z, z_min(z0_site)) / z0_site)


def length_scale(z, z0):
    zz = max(z, z_min(z0))
    return 300.0 * (zz / 200.0) ** (0.67 + 0.05 * math.log(z0))


def unit_field(length_m, L, seed, dx=0.5):
    """unit-variance frozen turbulence along a path, EN 1991-1-4 Annex B spectrum. -> (x grid, cumulative integral)."""
    n = int(length_m / dx) + 2
    if n > 2 ** 21:
        dx = length_m / (2 ** 21 - 2); n = 2 ** 21
    nfft = 1 << int(math.ceil(math.log2(max(n, 64))))
    k = np.fft.rfftfreq(nfft, dx)                      # cycles / m
    S = 6.8 * L / (1.0 + 10.2 * k * L) ** (5.0 / 3.0)  # one-sided wavenumber spectrum per unit variance
    S[0] = 0.0
    dk = 1.0 / (nfft * dx)
    rng = np.random.default_rng(seed)
    amp = np.sqrt(S * dk / 2.0) * nfft
    X = amp * (rng.standard_normal(len(k)) + 1j * rng.standard_normal(len(k)))
    g = np.fft.irfft(X, nfft)[:n]
    sd = g.std()
    var_th = float(np.sum(S[1:]) * dk)                 # resolved share of the variance (path >> L: close to 1)
    if sd > 0:
        g = g / sd * math.sqrt(min(var_th, 1.0))
    x = np.arange(n) * dx
    G = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * dx)])
    return x, G


def block_mean(x, G, xi):
    """mean of the field over [xi_i, xi_i+1] for each step."""
    Gi = np.interp(xi, x, G)
    d = np.diff(xi)
    gi = np.interp(0.5 * (xi[1:] + xi[:-1]), x, np.gradient(G, x))
    return np.where(d > 1e-6, np.diff(Gi) / np.where(d > 1e-6, d, 1.0), gi)


def road_wind(trip, trip_inputs, veh, opt, nsub, dt_trip, vt_sub, height_m):
    """-> (headwind list, |crosswind| list) per integration step (length (N-1)*nsub), or None for the v1 path."""
    prof = opt.get('wind_profile', 'none')
    gust = bool(opt.get('wind_gusts', False))
    if prof not in ('none', 'log'):
        raise ValueError(f"wind_profile must be 'none' or 'log' (got {prof!r})")
    N = trip['N']; M = (N - 1) * nsub
    h = dt_trip / nsub
    z = DRAG_HEIGHT_FRAC * height_m
    z0r = float(opt.get('wind_z0_ref_m', 0.03)); z0s = float(opt.get('wind_z0_site_m', 0.1))
    W = trip['wind_speed'].astype(float)
    shelter = float(opt.get('wind_shelter', 1.0))
    sh = trip_inputs.get('wind_shelter')
    if sh is not None:
        sh = np.asarray(sh, float)
        shelter = shelter * (np.nan_to_num(sh, nan=1.0) if sh.ndim else float(sh))
    if prof == 'log':
        Wr = W * profile_factor(z, z0r, z0s) * shelter
    else:
        Wr = W * shelter
    rel = np.radians(trip['wind_dir'] - trip['heading'])
    idx = np.minimum(np.arange(M) // nsub, N - 1)          # trip second of each step
    Wk = Wr[idx]; ck = np.cos(rel)[idx]; sk = np.sin(rel)[idx]
    head = Wk * ck; cross = Wk * sk
    if gust and M > 0:
        su = W * sigma_u_factor(z0r, z0s, prof == 'log', z)
        Lu = length_scale(z, z0s); Lv = LV_RATIO * Lu
        vt = np.asarray(vt_sub, float)
        rate = 0.5 * (vt[1:] + vt[:-1]) + np.maximum(np.abs(Wk), 0.5)
        xi = np.concatenate([[0.0], np.cumsum(rate * h)])
        seed = int(opt.get('wind_turb_seed', 0) or 0)
        xu, Gu = unit_field(xi[-1] + 1.0, Lu, [seed, 1])
        xv, Gv = unit_field(xi[-1] + 1.0, Lv, [seed, 2])
        up = su[idx] * block_mean(xu, Gu, xi)
        vp = SIGMA_V_RATIO * su[idx] * block_mean(xv, Gv, xi)
        head = (Wk + up) * ck + vp * sk
        cross = (Wk + up) * sk - vp * ck
    return head.tolist(), np.abs(cross).tolist()


def draft_factor(trip_inputs, veh, opt, nsub, vt_sub, N):
    """CdA multiplier per integration step, or None when there is no lead vehicle."""
    if not opt.get('drafting', False):
        return None
    M = (N - 1) * nsub
    gap = trip_inputs.get('gap_m')
    vt = np.asarray(vt_sub, float)
    vmid = 0.5 * (vt[1:] + vt[:-1])
    if gap is not None:
        g = np.asarray(gap, float)
        g = np.full(N, float(g)) if g.ndim == 0 else g
        g = g[np.minimum(np.arange(M) // nsub, N - 1)]
    elif opt.get('draft_headway_s') is not None:
        g = DRAFT_STANDSTILL_GAP_M + float(opt['draft_headway_s']) * vmid
    else:
        return None
    p = DRAFT['heavy' if veh.hd else 'light']
    gg = np.where(np.isfinite(g) & (g >= 0), g, np.inf)
    f = 1.0 - p['d0'] / (1.0 + gg / p['g0']) ** 1.5
    return np.where(np.isfinite(gg), f, 1.0).tolist()
