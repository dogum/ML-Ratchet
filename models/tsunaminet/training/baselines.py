"""Classical and dumb baselines, on exactly the network's input windows.
- SIFT-style inversion: non-negative least squares for slip on the 18 basis unit sources (9 columns x 2 rows around
  the epicentre) fitted to the input DART windows (observed minutes only), as NOAA's SIFT does with its unit sources.
  Mw from the moment (mu = 4e10 Pa), centroid = moment-weighted unit-source centre, forecast = superposition.
- Seismic scaling: Mw from the USGS W-phase, a standard rupture centred on the epicentre's column (length from
  Strasser et al. 2010, both rows, equal split) = the pre-DART SIFT forecast.
- Dumb: the dev-median Mw, the epicentre as centroid, and the dev-median target amplitude everywhere."""
import numpy as np
from scipy.optimize import nnls
import tsim, features as F

def sift_nnls(gf, prep, ridge=0.0, delay_frac=0.0, gain=1.0):
    units = prep['frame']['units']; rows, rhs = [], []
    for k, site in enumerate(prep['inputs']):
        ta = prep['input_ta'][k]; x = prep['X'][0, k]
        m = x[1] > 0
        B = F.basis(gf, site, units, ta)            # mm per m slip, (18, NW)
        obs_mm = x[0] * 10.0 * np.abs(B).max()       # undo features scaling -> mm
        if delay_frac or gain != 1.0:                # dev-calibrated variant: delayed, amplified Green's functions
            tt = np.arange(B.shape[1]); B = gain * np.stack([np.interp(tt - ta * delay_frac, tt, b, left=0.0) for b in B])
        rows.append(B[:, m].T); rhs.append(obs_mm[m])
    A = np.concatenate(rows); b = np.concatenate(rhs)
    ok = units >= 0
    if ridge:
        A2 = np.concatenate([A[:, ok], ridge * np.eye(ok.sum())]); b2 = np.concatenate([b, np.zeros(ok.sum())])
    else:
        A2, b2 = A[:, ok], b
    s, _ = nnls(A2.astype(float), b2.astype(float), maxiter=2000)
    slip = np.zeros(len(units)); slip[ok] = s
    mo = slip.sum() * tsim.MU * tsim.UNIT_AREA
    return units[slip > 0], slip[slip > 0], (tsim.mw_of(mo) if mo > 0 else 6.0), slip

def nnls_theta(slip):
    """Along-strike centroid (100 km) and deep fraction of an 18-unit slip vector."""
    if slip.sum() <= 0: return 0.0, 0.5
    cols = np.tile(np.arange(tsim.NCOL) - tsim.HALF, 2).astype(float)
    return float((cols * slip).sum() / slip.sum()), float(slip[tsim.NCOL:].sum() / slip.sum())

def seismic_theta(mw):
    return np.array([mw, 0.0, 0.5, -2.477 + 0.585 * mw], np.float32)
