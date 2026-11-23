"""Okada (1985) surface displacement of a finite rectangular dislocation in an elastic half-space.
Written from Okada, Y. (1985), BSSA 75(4), 1135-1154, eqs. (25)-(30) with Chinnery's notation; vertical component only
(what a tsunami source needs). Poisson solid (lambda = mu, alpha = mu/(lambda+mu) = 0.5).

Geometry follows the NOAA SIFT convention used by info_sz.dat: (lon, lat) is the centre of the fault's LOWER edge,
`depth_top` the depth of its upper edge (km), strike/dip/rake in degrees, length/width in km, slip in m."""
import numpy as np
R_EARTH = 6371.0e3
DEG = np.pi / 180.0

def _uz_ss(xi, eta, q, dip, alpha):
    sd, cd = np.sin(dip), np.cos(dip)
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    dt = eta * sd - q * cd
    I4 = alpha / cd * (np.log(R + dt) - sd * np.log(R + eta))
    return -(dt * q / (R * (R + eta)) + q * sd / (R + eta) + I4 * sd) / (2 * np.pi)

def _uz_ds(xi, eta, q, dip, alpha):
    sd, cd = np.sin(dip), np.cos(dip)
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    dt = eta * sd - q * cd
    X = np.sqrt(xi ** 2 + q ** 2)
    xi_s = np.where(np.abs(xi) < 1e-6, 1e-6, xi)
    I5 = alpha * 2 / cd * np.arctan((eta * (X + q * cd) + X * (R + X) * sd) / (xi_s * (R + X) * cd))
    q_s = np.where(np.abs(q) < 1e-6, 1e-6, q)
    return -(dt * q / (R * (R + xi)) + sd * np.arctan(xi * eta / (q_s * R)) - I5 * sd * cd) / (2 * np.pi)

def okada_uz(lon, lat, src, slip=1.0, alpha=0.5):
    """Vertical seafloor displacement (m, up positive) at points lon, lat (deg, any shape) for one rectangular source.
    src: dict(lon, lat, depth (top, km), strike, dip, rake, length (km), width (km))."""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    dip = src['dip'] * DEG; strike = src['strike'] * DEG; rake = src.get('rake', 90.0) * DEG
    L = src['length'] * 1e3; W = src['width'] * 1e3
    d_bot = src['depth'] * 1e3 + W * np.sin(dip)          # depth of the lower edge
    # local east/north metres from the lower-edge centre (flat-earth, adequate within a few hundred km)
    dlon = (lon - src['lon'] + 180.0) % 360.0 - 180.0
    xe = R_EARTH * np.cos(lat * DEG) * dlon * DEG
    yn = R_EARTH * (lat - src['lat']) * DEG
    xs = xe * np.sin(strike) + yn * np.cos(strike)          # along strike
    yp = -(xe * np.cos(strike) - yn * np.sin(strike))       # perpendicular, positive toward the up-dip side
    x = xs + L / 2                                          # Okada's origin: lower edge, start of the fault
    p = yp * np.cos(dip) + d_bot * np.sin(dip)
    q = yp * np.sin(dip) - d_bot * np.cos(dip)
    def chin(f):
        return f(x, p, q, dip, alpha) - f(x, p - W, q, dip, alpha) - f(x - L, p, q, dip, alpha) + f(x - L, p - W, q, dip, alpha)
    return slip * (np.cos(rake) * chin(_uz_ss) + np.sin(rake) * chin(_uz_ds))
