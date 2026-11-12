"""TransitNet physics: the light of a limb-darkened star partly covered by a dark planet or by a second star.

occult(z, p, u1, u2): fraction of a star's light blocked by a disk of radius p (in stellar radii) whose centre is z stellar
radii from the star's centre, for quadratic limb darkening I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2. Exact geometry: at each
radius r the disk covers an arc of the annulus of half-angle theta(r) (pi if the disk covers the whole annulus); the blocked
light is the integral of I(r) * 2 theta(r) r dr over the covered radii, done by Gauss-Legendre quadrature in a variable
that removes the square-root behaviour at the edges. Works for any p, so the same function eclipses a star by a bigger star.
Orbits: Keplerian, eccentric allowed (Kepler's equation by Newton steps), transit at the time of inferior conjunction T0."""
import numpy as np
GL_X, GL_W = np.polynomial.legendre.leggauss(48)

def _intensity(r, u1, u2):
    mu = np.sqrt(np.clip(1 - r * r, 0, 1)); return 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2

def occult(z, p, u1, u2):
    """z, p, u1, u2 broadcastable arrays -> blocked fraction of the star's flux"""
    z, p, u1, u2 = np.broadcast_arrays(np.abs(np.asarray(z, float)), np.asarray(p, float), np.asarray(u1, float), np.asarray(u2, float))
    out = np.zeros(z.shape); m = z < 1 + p
    if not m.any(): return out
    zz, pp, a1, a2 = z[m], p[m], u1[m], u2[m]
    lo = np.maximum(0.0, zz - pp); hi = np.minimum(1.0, zz + pp)
    # substitution r = lo + (hi - lo) * (1 - cos(phi)) / 2, phi in [0, pi]: smooths the sqrt endpoints of theta(r) and of I(r) at r = 1
    phi = (GL_X[None, :] + 1) * np.pi / 2; w = GL_W[None, :] * np.pi / 2
    r = lo[:, None] + (hi - lo)[:, None] * (1 - np.cos(phi)) / 2; dr = (hi - lo)[:, None] * np.sin(phi) / 2
    c = (r * r + zz[:, None] ** 2 - pp[:, None] ** 2) / (2 * np.maximum(r, 1e-12) * np.maximum(zz[:, None], 1e-12))
    th = np.arccos(np.clip(c, -1, 1)); th = np.where(r <= pp[:, None] - zz[:, None], np.pi, th); th = np.where(zz[:, None] < 1e-9, np.where(r < pp[:, None], np.pi, 0.0), th)
    I = _intensity(r, a1[:, None], a2[:, None]); blocked = (I * 2 * th * r * dr * w).sum(1)
    total = np.pi * (1 - a1 / 3 - a2 / 6)
    out[m] = blocked / total; return out

def ecc_anomaly(M, e, n=12):
    E = M + e * np.sin(M)
    for _ in range(n): E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E

def sky_separation(t, T0, P, a, inc, e=0.0, w=np.pi / 2):
    """projected separation (units of a's length unit) and whether the companion is in front (towards us). w: argument of
    periastron of the companion's orbit; transit (companion in front) at true anomaly f = pi/2 - w."""
    t, T0, P, a, inc, e, w = np.broadcast_arrays(*(np.asarray(x, float) for x in (t, T0, P, a, inc, e, w)))
    f_tr = np.pi / 2 - w; E_tr = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f_tr / 2)); M_tr = E_tr - e * np.sin(E_tr)
    M = 2 * np.pi * (t - T0) / P + M_tr
    if not np.any(e): E = f = M                                                        # circular: no Kepler equation to solve
    else: E = ecc_anomaly(np.mod(M, 2 * np.pi), e); f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * np.cos(E)); X = -r * np.cos(w + f); Y = -r * np.sin(w + f) * np.cos(inc); Zf = r * np.sin(w + f) * np.sin(inc)
    return np.sqrt(X * X + Y * Y), Zf > 0

def planet_flux(t, T0, P, k, aR, inc, u1, u2, e=0.0, w=np.pi / 2, supersample=3, texp=120 / 86400):
    """relative flux of a star with a dark transiting planet (radius ratio k, a/R*), averaged over each exposure"""
    t = np.asarray(t, float); off = (np.arange(supersample) + 0.5) / supersample - 0.5
    tt = t[..., None] + off * texp; z, front = sky_separation(tt, *(np.asarray(x, float)[..., None] for x in (T0, P, aR, inc, e, w)))
    d = np.where(front, occult(z, np.asarray(k, float)[..., None], np.asarray(u1, float)[..., None], np.asarray(u2, float)[..., None]), 0.0)
    return 1 - d.mean(-1)

def eb_flux(t, T0, P, k, aR1, inc, ua, ub, J, e=0.0, w=np.pi / 2, A_ell=0.0, third=0.0, supersample=3, texp=120 / 86400):
    """an eclipsing binary: primary of radius 1, secondary of radius k (both in primary radii), separation aR1 primary radii,
    surface-brightness ratio J; ua, ub: (u1, u2) of each star. Primary eclipse at T0 (secondary star in front). Ellipsoidal
    variation of semi-amplitude A_ell; third light 'third' (in units of the binary's light) dilutes everything."""
    t = np.asarray(t, float); off = (np.arange(supersample) + 0.5) / supersample - 0.5; tt = t[..., None] + off * texp
    ex = lambda x: np.asarray(x, float)[..., None]
    z, front = sky_separation(tt, ex(T0), ex(P), ex(aR1), ex(inc), ex(e), ex(w))
    L1 = 1 - ex(ua[0]) / 3 - ex(ua[1]) / 6; L2 = ex(J) * ex(k) ** 2 * (1 - ex(ub[0]) / 3 - ex(ub[1]) / 6)
    d1 = np.where(front, occult(z, ex(k), ex(ua[0]), ex(ua[1])), 0.0) * L1                     # star 2 in front of star 1
    d2 = np.where(~front, occult(z / ex(k), 1 / ex(k), ex(ub[0]), ex(ub[1])), 0.0) * L2       # star 1 in front of star 2
    ph = 2 * np.pi * (tt - ex(T0)) / ex(P); ell = -ex(A_ell) * np.cos(2 * ph) * (L1 + L2)
    F = (L1 + L2 - d1 - d2 + ell) / (L1 + L2); F = (F + ex(third)) / (1 + ex(third))
    return F.mean(-1)

def a_over_R(P_days, rho_cgs):
    """Kepler's third law: a/R* from period and stellar density (g/cm^3), planet mass neglected"""
    G = 6.674e-8; return (G * rho_cgs * (P_days * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)

def t14(P, aR, k, b, inc, e=0.0, w=np.pi / 2):
    """total transit duration (days), Winn 2010 eq. 14 with the eccentric correction (eq. 16)"""
    arg = np.sqrt(np.clip((1 + k) ** 2 - b ** 2, 0, None)) / (aR * np.sin(inc))
    return P / np.pi * np.arcsin(np.clip(arg, 0, 1)) * np.sqrt(1 - e ** 2) / (1 + e * np.sin(w))
