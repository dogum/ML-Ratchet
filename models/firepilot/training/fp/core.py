"""FirePilot core geometry (numpy/scipy only; vendored in the release): crew clock, line geometry and build
times, line cost by fuel group, forecast head direction."""
import numpy as np
from scipy import ndimage as ndi

# Scott & Burgan / Anderson fuel-model number ranges by group (as in firesim.simulate.GROUPS)
GROUPS = {'GR': range(101, 110), 'GS': range(121, 125), 'SH': range(141, 150), 'TU': range(161, 166),
          'TL': range(181, 190), 'SB': range(201, 205), 'A_grass': range(1, 4), 'A_shrub': range(4, 8),
          'A_timber': range(8, 11), 'A_slash': range(11, 14)}

HORIZON_H = 72.0
SHIFT_LOCAL = (8.0, 20.0)       # crews work 08:00-20:00 local solar time
MOBILIZE_H = 2.0                # first line work starts 2 h after detection
FL_HOLD_M = 2.4                 # 8 ft: above this the line is overrun (NWCG hauling chart)
DT_BLOCK = 0.25                 # build times are rounded UP to 15 min

# time to build 1 m of line relative to timber litter (approximate ordering of NWCG production rates:
# grass fastest, brush and slash slowest). Non-burnable cells cost nothing (crews tie into roads, rock, water).
COST_MULT = {'GR': 0.6, 'GS': 0.8, 'SH': 1.7, 'TU': 1.3, 'TL': 1.0, 'SB': 2.0,
             'A_grass': 0.6, 'A_shrub': 1.7, 'A_timber': 1.1, 'A_slash': 2.0, 'NB': 0.0}


def cost_raster(fuel):
    out = np.zeros(fuel.shape, np.float32)
    for g, r in GROUPS.items():
        out[np.isin(fuel, list(r))] = COST_MULT[g]
    return out


def burnable(fuel):
    f = np.asarray(fuel)
    return (f > 0) & ((f < 91) | (f > 99)) & (cost_raster(f) > 0)


def work_clock(start, lon, horizon_h=HORIZON_H, mobilize_h=MOBILIZE_H, step=0.05):
    """Cumulative working hours at clock hours t (since start). Returns (t, W) arrays for np.interp."""
    t = np.arange(0.0, horizon_h + step, step)
    loc = (start.hour + start.minute / 60.0 + lon / 15.0 + t) % 24.0
    on = ((loc >= SHIFT_LOCAL[0]) & (loc < SHIFT_LOCAL[1]) & (t >= mobilize_h)).astype(float)
    W = np.concatenate([[0.0], np.cumsum(on[:-1] * step)])
    return t, W


def clock_of_work(w, t, W):
    """Clock time (h) at which cumulative work reaches w (inf if beyond the horizon)."""
    w = np.asarray(w, float)
    out = np.full(w.shape, np.inf)
    ok = w <= W[-1]
    # first time W >= w
    idx = np.searchsorted(W, w[ok], side='left')
    out[ok] = t[np.minimum(idx, len(t) - 1)]
    return out


class Frame:
    """Geometry of a scenario at detection: distance from the burned area, angle about the fire's centroid
    measured from the forecast head direction."""

    def __init__(self, burned0, res, head_az_deg):
        self.burned0 = burned0
        self.res = res
        self.D = ndi.distance_transform_edt(~burned0) * res
        rr, cc = np.nonzero(burned0)
        self.r0, self.c0 = rr.mean(), cc.mean()
        R, C = np.indices(burned0.shape)
        az = np.degrees(np.arctan2(C - self.c0, -(R - self.r0)))      # clockwise from grid north
        self.head_az = head_az_deg
        self.Phi = (az - head_az_deg + 180.0) % 360.0 - 180.0
        self.shape = burned0.shape


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def line_cells(fr, plan, fuel_ok):
    """Plan -> line cells (flat indices), their signed angle from the plan centre (deg).
    plan: dict(mode 0 centre-out / 1 ends-in, phi_c deg, omega deg (half-width), d_h m, d_f m).
    The line is the inner boundary of {D >= d(Phi)}, d(Phi) = d_f + (d_h - d_f) * ((1 + cos(Phi - phi_c)) / 2)^2,
    inside the sector |Phi - phi_c| <= omega; this 8-boundary blocks every 8- and 16-neighbour path."""
    dl = wrap180(fr.Phi - plan['phi_c'])
    w = ((1.0 + np.cos(np.radians(dl))) / 2.0) ** 2
    d = plan['d_f'] + (plan['d_h'] - plan['d_f']) * w
    d = np.maximum(d, fr.res)                         # never inside the burned area
    out = fr.D >= d
    inner = ~out
    # 8-neighbour boundary: FireSim's corner rule only checks permanent non-burnable cells, not time-blocked line
    # cells, so a 4-connected time-blocked line leaks diagonally (upstream candidate U1). The 8-boundary does not.
    nb = ndi.binary_dilation(inner, structure=np.ones((3, 3), bool))
    nb4 = ndi.binary_dilation(inner)
    sec = (np.abs(dl) <= plan['omega']) & ~fr.burned0
    m = out & nb & sec
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    n4 = int((out & nb4 & sec).sum())
    idx = np.flatnonzero(m)
    # line length: 4-boundary cell count x res x pi/4 (staircase correction), shared evenly by the 8-boundary cells
    cell_len = fr.res * (np.pi / 4.0) * n4 / max(len(idx), 1)
    return idx, dl.ravel()[idx], cell_len


def build_times(fr, plan, cost, rate_mph, clock, horizon_h=HORIZON_H):
    """Build completion time (h since detection) of each line cell; two crews, each at rate/2 m per working hour.
    mode 0: both crews start at the plan centre and work outward; mode 1: they start at the two ends (anchors)
    and work toward the centre."""
    idx, dl, cell_len = line_cells(fr, plan, None)
    tb = np.full(idx.shape, np.inf)
    if len(idx) == 0:
        return idx, tb, 0.0
    c = cost.ravel()[idx] * cell_len
    t, W = clock
    for side in (dl >= 0, dl < 0):
        s = np.flatnonzero(side)
        if not len(s):
            continue
        key = np.abs(dl[s]) if plan['mode'] == 0 else -np.abs(dl[s])
        o = s[np.argsort(key, kind='stable')]
        cum = np.cumsum(c[o]) / (rate_mph / 2.0)          # working hours to finish each cell
        tb[o] = clock_of_work(cum, t, W)
    tb[tb > horizon_h] = np.inf
    return idx, tb, float(c[np.isfinite(tb)].sum() / 1000.0)


def head_azimuth(arr_fc, burned0, w=None, horizon=24.0):
    """Forecast head direction: azimuth (deg clockwise from grid north) from the burned centroid to the centroid
    of the forecast's new burned cells in the first `horizon` hours (fallback: speed-weighted mean wind)."""
    rr, cc = np.nonzero(burned0)
    r0, c0 = rr.mean(), cc.mean()
    for h in (horizon, 72.0):
        new = np.isfinite(arr_fc) & (arr_fc <= h) & ~burned0
        if new.sum() >= 3:
            r, c = np.nonzero(new)
            dr, dc = r.mean() - r0, c.mean() - c0
            if np.hypot(dr, dc) > 0.5:
                return float(np.degrees(np.arctan2(dc, -dr)) % 360.0)
    if w is not None:
        to = np.radians((w['wind_dir'].to_numpy() + 180.0) % 360.0)
        s = w['wind_speed'].to_numpy()
        return float(np.degrees(np.arctan2((s * np.sin(to)).sum(), (s * np.cos(to)).sum())) % 360.0)
    return 0.0
