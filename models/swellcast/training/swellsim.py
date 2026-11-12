"""SwellCast simulator: swell from a distant storm arriving at one buoy.

Physics (deep water, linear waves):
  * group speed c_g(f) = g / (4 pi f); a component emitted at time t_s from distance D arrives at t_s + D / c_g(f),
    so at the buoy the swell frequency rises linearly: df/dt = g / (4 pi D)  (Munk et al. 1963; Snodgrass et al. 1966).
  * the storm is a moving, finite patch of sea. Each of 7 sub-areas emits, at each source time, a JONSWAP spectrum whose
    Hs and Tp follow a growth/decay intensity curve; the emission is partly isotropic (rotating cyclone winds) and partly
    beamed along the storm heading (trapped fetch).
  * energy balance along the ray: energy emitted per Hz per radian, P dt_s, spreads over a crest length R_E sin(D/R_E)
    (spherical spreading, Munk 1963) and passes the buoy during the interval it takes those waves to arrive, so
    E_buoy(f,t) dt = E_src(f) L_eff Dir(theta) dt_s / (R_E sin(D / R_E)) x attenuation.
  * swell dissipation by the air-sea boundary layer (Ardhuin et al. 2009, as in WAVEWATCH III ST4):
    dE/dt = -(rho_a / rho_w) (16 f_e sigma^2 u_orb / g) E (turbulent), -(rho_a / rho_w) 2 k sqrt(2 nu_a sigma) E (viscous,
    Reynolds number below 1e5); integrated along the great circle with the swell's own decaying orbital velocity.
  * directional moments: each sub-area's swell arrives from the bearing buoy->sub-area with intrinsic r1 = exp(-s^2/2);
    first moments (r1 cos a1, r1 sin a1) mix with the real background, weighted by energy.
  * an optional local wind-sea burst; chi-square(24 dof) spectral estimation noise and 0.01 m^2/Hz quantisation (NDBC format).
"""
import numpy as np
import geo
G = 9.81; R_E = geo.R_E; RHO_A, RHO_W, NU_A = 1.225, 1025.0, 1.5e-5
FGRID = 0.0325 + 0.005 * np.arange(34); DF = 0.005
NSUB = 4; FFINE = (FGRID[:, None] + DF * ((np.arange(NSUB) + 0.5) / NSUB - 0.5)[None]).ravel()   # 4 sub-bins per bin
DT_H = 3; HALF_H = 120; NT = 2 * HALF_H // DT_H     # 80 three-hour steps over 240 h
F_REF = 0.06

def cg(f): return G / (4 * np.pi * np.asarray(f, float))

def jonswap(f, hs, tp, gam=2.0):
    """JONSWAP shape scaled so that 4 sqrt(m0) = hs. f (n,), hs/tp (m,) -> (m, n) m^2/Hz"""
    fp = 1.0 / np.asarray(tp, float)[:, None]; f = f[None]
    sig = np.where(f <= fp, 0.07, 0.09); r = np.exp(-(f - fp) ** 2 / (2 * sig ** 2 * fp ** 2))
    S = f ** -5.0 * np.exp(-1.25 * (fp / f) ** 4) * gam ** r
    fq = np.linspace(0.02, 1.0, 600); fpq = fp; sq = np.where(fq[None] <= fpq, 0.07, 0.09)
    Sq = fq[None] ** -5.0 * np.exp(-1.25 * (fpq / fq[None]) ** 4) * gam ** np.exp(-(fq[None] - fpq) ** 2 / (2 * sq ** 2 * fpq ** 2))
    m0 = np.trapezoid(Sq, fq, axis=1)[:, None]
    return S * (np.asarray(hs, float)[:, None] ** 2 / 16) / m0

def attenuation(f, D, m0_src, tp, L_eff, dirw, fe, nstep=48):
    """energy attenuation factor exp(-int mu dx) for frequencies f (n,) after distance D, swell starting with m0_src (m^2)
    of peak period tp leaving a storm of width L_eff into a direction with weight dirw (1/rad)."""
    if fe <= 0: return np.ones_like(f)
    x = np.linspace(max(L_eff, 2e5), max(D, max(L_eff, 2e5) + 1), nstep); dx = np.diff(x, prepend=x[0])
    sp = 2 * np.pi / tp; m0 = m0_src; logA = np.zeros_like(f); logAp = 0.0
    sig = 2 * np.pi * f; k = sig ** 2 / G
    for xi, d in zip(x, dx):
        spread = min(1.0, L_eff * dirw / (R_E * np.sin(min(xi, 0.999 * np.pi * R_E) / R_E)))
        m = m0 * spread * np.exp(logAp); uorb = 2 * sp * np.sqrt(max(m, 0)); aorb = uorb / sp
        Re = 4 * uorb * aorb / NU_A
        if Re > 1e5: gam = RHO_A / RHO_W * 16 * fe * sig ** 2 * uorb / G
        else: gam = RHO_A / RHO_W * 2 * k * np.sqrt(2 * NU_A * sig)
        mu = gam / cg(f); logA -= mu * d
        gp = (RHO_A / RHO_W * 16 * fe * sp ** 2 * uorb / G) if Re > 1e5 else RHO_A / RHO_W * 2 * sp ** 2 / G * np.sqrt(2 * NU_A * sp)
        logAp -= gp / cg(1 / tp) * d
    return np.exp(logA)

def sample_storm(rng, D=None, prior='v1'):
    p = dict(hs=rng.uniform(4, 15), tp_c=rng.uniform(3.6, 5.2), sig1=rng.uniform(8, 36), sig2=rng.uniform(8, 48),
             speed=rng.uniform(0, 9), heading=rng.uniform(0, 360), R=10 ** rng.uniform(np.log10(80e3), np.log10(600e3)),
             gam=rng.uniform(1.0, 3.3), w_iso=rng.uniform(0.2, 1.0), s_beam=rng.uniform(1, 6), fe=10 ** rng.uniform(np.log10(0.002), np.log10(0.02)),
             r1_0=np.exp(-0.5 * np.radians(rng.uniform(6, 22)) ** 2), stationary=rng.random() < 0.2)
    if prior >= 'v2':   # calibrated on dev arrivals (calib_dev.py): weaker, longer-lived storms, shorter peak periods, broader directions
        p.update(hs=10 ** rng.uniform(np.log10(2.5), np.log10(14)), tp_c=rng.uniform(2.9, 4.3), sig1=rng.uniform(12, 48), sig2=rng.uniform(12, 72),
                 r1_0=rng.uniform(0.35, 0.8))
    p['tp'] = p['tp_c'] * np.sqrt(p['hs'])
    if p['stationary']: p['speed'] = 0.0
    return p

def simulate(buoy_lat, buoy_lon, lat0, lon0, t0_h, p, t_axis_h, ws=None, rng=None, noise=True):
    """swell energy and first directional moments on the 1-hour grid t_axis_h (hours, same clock as t0_h).
    Returns E (nt, 34), Ec, Es (energy-weighted r1 cos, r1 sin) on the coarse frequency grid."""
    nt = len(t_axis_h); tlo = t_axis_h[0]
    ts = np.arange(-3.5 * p['sig1'], 3.5 * p['sig2'] + 1e-9, 0.5)                       # source times rel. to peak, h
    A = np.where(ts < 0, np.exp(-ts ** 2 / (2 * p['sig1'] ** 2)), np.exp(-ts ** 2 / (2 * p['sig2'] ** 2)))
    keep = A > 0.02; ts, A = ts[keep], A[keep]
    hs, tp = p['hs'] * np.sqrt(A), np.maximum(p['tp'] * A ** 0.25, 4.0)
    Esrc = jonswap(FFINE, hs, tp, p['gam'])                                                 # (ns, nf)
    clat, clon = geo.dest(lat0, lon0, p['heading'], p['speed'] * ts * 3600.0)              # storm centre track
    off_b = np.arange(6) * 60.0; R = p['R']
    subs = [(clat, clon)] + [geo.dest(clat, clon, b, 0.6 * R) for b in off_b]
    L_eff = 2 * R; Eacc = np.zeros((nt, len(FFINE))); Cacc = np.zeros_like(Eacc); Sacc = np.zeros_like(Eacc)
    sbeam = p['s_beam']; norm_beam = 1.0 / np.trapezoid(np.cos(np.linspace(-np.pi, np.pi, 721) / 2) ** (2 * sbeam), np.linspace(-np.pi, np.pi, 721))
    att_cache = {}
    for la, lo in subs:
        D = geo.dist(la, lo, buoy_lat, buoy_lon); D = np.maximum(D, 1e5)
        th = geo.bearing(la, lo, buoy_lat, buoy_lon); beta = np.radians(geo.bearing(buoy_lat, buoy_lon, la, lo))
        rel = (np.radians(th - p['heading']) + np.pi) % (2 * np.pi) - np.pi
        dirw = p['w_iso'] / (2 * np.pi) + (1 - p['w_iso']) * norm_beam * np.cos(rel / 2) ** (2 * sbeam)       # 1/rad
        spread = L_eff / len(subs) * dirw / (R_E * np.abs(np.sin(D / R_E)) + L_eff)                           # (ns,)
        ipk = int(np.argmax(A)); key = round(float(D[ipk]) / 2e5)
        if key not in att_cache:
            att_cache[key] = attenuation(FFINE, float(D[ipk]), p['hs'] ** 2 / 16, p['tp'], L_eff, float(dirw[ipk]), p['fe'])
        att = att_cache[key]
        dep = Esrc * spread[:, None] * att[None] * 0.5                                          # x dt_s (h) / bin width (1 h)
        ta = ts[:, None] + t0_h + D[:, None] / cg(FFINE)[None] / 3600.0                          # arrival time, h
        it = np.floor(ta - tlo).astype(int); ok = (it >= 0) & (it < nt)
        jf = np.broadcast_to(np.arange(len(FFINE))[None], ta.shape)
        np.add.at(Eacc, (it[ok], jf[ok]), dep[ok])
        cb = np.broadcast_to((p['r1_0'] * np.cos(beta))[:, None], ta.shape); sb = np.broadcast_to((p['r1_0'] * np.sin(beta))[:, None], ta.shape)
        np.add.at(Cacc, (it[ok], jf[ok]), (dep * cb)[ok]); np.add.at(Sacc, (it[ok], jf[ok]), (dep * sb)[ok])
    E = Eacc.reshape(nt, 34, NSUB).mean(2); Ec = Cacc.reshape(nt, 34, NSUB).mean(2); Es = Sacc.reshape(nt, 34, NSUB).mean(2)
    if ws is not None:   # local wind sea burst: JONSWAP growing and decaying in place
        tt = t_axis_h - ws['tc']; a = np.exp(-tt ** 2 / (2 * ws['sig'] ** 2)); m = a > 0.01
        if m.any():
            W = np.zeros((nt, 34)); W[m] = jonswap(FGRID, ws['hs'] * np.sqrt(a[m]), np.maximum(ws['tp'] * a[m] ** 0.25, 3.0), 3.3)
            E = E + W; Ec = Ec + W * 0.7 * np.cos(np.radians(ws['dir'])); Es = Es + W * 0.7 * np.sin(np.radians(ws['dir']))
    if noise and rng is not None:
        chi = rng.chisquare(24, E.shape) / 24; E, Ec, Es = E * chi, Ec * chi, Es * chi
    return E, Ec, Es

def windsea(t_axis_h, ws):
    """local wind-sea burst (same as in simulate): JONSWAP growing and decaying in place"""
    nt = len(t_axis_h); W = np.zeros((nt, 34)); tt = t_axis_h - ws['tc']; a = np.exp(-tt ** 2 / (2 * ws['sig'] ** 2)); m = a > 0.01
    if m.any(): W[m] = jonswap(FGRID, ws['hs'] * np.sqrt(a[m]), np.maximum(ws['tp'] * a[m] ** 0.25, 3.0), 3.3)
    return W, W * 0.7 * np.cos(np.radians(ws['dir'])), W * 0.7 * np.sin(np.radians(ws['dir']))

def to_coarse(Eh, Ch, Sh):
    """hourly (240, 34) -> 3-hourly mean (80, 34); C,S passed as energy-weighted moments"""
    n = Eh.shape[0] // DT_H
    return [x[:n * DT_H].reshape(n, DT_H, -1).mean(1) for x in (Eh, Ch, Sh)]

def sample_track_params(rng):
    """v3 storm-physics parameters for a real historical track: sea state from the 1-min wind (Hs = a_hs Vmax), the v2
    period / direction / dissipation priors, and a size from R34 where IBTrACS has it, else log-uniform 100-400 km."""
    return dict(a_hs=rng.uniform(0.12, 0.24), tp_c=rng.uniform(2.9, 4.3), gam=rng.uniform(1.0, 3.3), w_iso=rng.uniform(0.2, 1.0), s_beam=rng.uniform(1, 6),
                fe=10 ** rng.uniform(np.log10(0.002), np.log10(0.02)), r1_0=rng.uniform(0.35, 0.8), R_def=10 ** rng.uniform(np.log10(100e3), np.log10(400e3)),
                R_fac=10 ** rng.uniform(-0.15, 0.15))

def simulate_track(buoy_lat, buoy_lon, s, t0_h, p, t_axis_h, rng=None, noise=True):
    """swell from a REAL storm track s (IBTrACS fixes: t h, lat, lon, wind kt, r34 km; s['k'] = peak point). The track's
    peak time is put at t0_h on the t_axis_h clock. Each half hour the storm emits a JONSWAP sea with Hs = a_hs * V,
    Tp = tp_c sqrt(Hs), from 7 sub-areas, beamed partly along the track heading; same propagation, spreading and
    dissipation as simulate(). Returns E, Ec, Es (nt, 34) and the label dict (hs at peak, mean speed near peak)."""
    t = np.array(s['t']); w = np.array(s['wind'], float); ok = w > 0
    t, w = t[ok], w[ok]; la = np.array(s['lat'])[ok]; lo = np.array(s['lon'])[ok]; r34 = np.array(s['r34'], float)[ok]
    tpk = s['t'][s['k']]
    ts = np.arange(t[0], t[-1] + 1e-9, 0.5); W = np.interp(ts, t, w) * 0.5144
    lat = np.interp(ts, t, la); lon = np.degrees(np.unwrap(np.radians(lo))); lon = (np.interp(ts, t, lon) + 180) % 360 - 180
    R = np.interp(ts, t, np.where(r34 > 0, r34 * 1e3, p['R_def'])) * p['R_fac']
    hs = p['a_hs'] * W; keep = hs > 1.5
    hd = np.r_[geo.bearing(lat[:-1], lon[:-1], lat[1:], lon[1:]), 0.0]; hd[-1] = hd[-2] if len(hd) > 1 else 0
    spd = np.r_[geo.dist(lat[:-1], lon[:-1], lat[1:], lon[1:]) / 1800.0, 0.0]
    near = np.abs(ts - tpk) <= 24; speed = float(np.mean(spd[near])) if near.any() else 0.0
    ts, lat, lon, hs, R, hd = ts[keep], lat[keep], lon[keep], hs[keep], R[keep], hd[keep]
    nt = len(t_axis_h); tlo = t_axis_h[0]; Eacc = np.zeros((nt, len(FFINE))); Cacc = np.zeros_like(Eacc); Sacc = np.zeros_like(Eacc)
    lab = dict(hs=float(p['a_hs'] * s['vmax'] * 0.5144), tp=float(p['tp_c'] * np.sqrt(p['a_hs'] * s['vmax'] * 0.5144)), R=float(np.median(R)) if len(R) else p['R_def'], speed=speed)
    if not len(ts): return np.zeros((nt, 34)), np.zeros((nt, 34)), np.zeros((nt, 34)), lab
    tp = np.maximum(p['tp_c'] * np.sqrt(hs), 4.0); Esrc = jonswap(FFINE, hs, tp, p['gam'])
    sbeam = p['s_beam']; norm_beam = 1.0 / np.trapezoid(np.cos(np.linspace(-np.pi, np.pi, 721) / 2) ** (2 * sbeam), np.linspace(-np.pi, np.pi, 721))
    ipk = int(np.argmax(hs))
    for ob in [None] + list(np.arange(6) * 60.0):
        if ob is None: sla, slo = lat, lon
        else: sla, slo = geo.dest(lat, lon, ob, 0.6 * R)
        D = np.maximum(geo.dist(sla, slo, buoy_lat, buoy_lon), 1e5); th = geo.bearing(sla, slo, buoy_lat, buoy_lon); beta = np.radians(geo.bearing(buoy_lat, buoy_lon, sla, slo))
        rel = (np.radians(th - hd) + np.pi) % (2 * np.pi) - np.pi
        dirw = p['w_iso'] / (2 * np.pi) + (1 - p['w_iso']) * norm_beam * np.cos(rel / 2) ** (2 * sbeam)
        spread = 2 * R / 7 * dirw / (R_E * np.abs(np.sin(D / R_E)) + 2 * R)
        att = attenuation(FFINE, float(D[ipk]), hs[ipk] ** 2 / 16, tp[ipk], 2 * R[ipk], float(dirw[ipk]), p['fe'])
        dep = Esrc * spread[:, None] * att[None] * 0.5
        ta = (ts - tpk + t0_h)[:, None] + D[:, None] / cg(FFINE)[None] / 3600.0
        it = np.floor(ta - tlo).astype(int); okk = (it >= 0) & (it < nt); jf = np.broadcast_to(np.arange(len(FFINE))[None], ta.shape)
        np.add.at(Eacc, (it[okk], jf[okk]), dep[okk])
        cb = np.broadcast_to((p['r1_0'] * np.cos(beta))[:, None], ta.shape); sb = np.broadcast_to((p['r1_0'] * np.sin(beta))[:, None], ta.shape)
        np.add.at(Cacc, (it[okk], jf[okk]), (dep * cb)[okk]); np.add.at(Sacc, (it[okk], jf[okk]), (dep * sb)[okk])
    E = Eacc.reshape(nt, 34, NSUB).mean(2); Ec = Cacc.reshape(nt, 34, NSUB).mean(2); Es = Sacc.reshape(nt, 34, NSUB).mean(2)
    if noise and rng is not None:
        chi = rng.chisquare(24, E.shape) / 24; E, Ec, Es = E * chi, Ec * chi, Es * chi
    return E, Ec, Es, lab
