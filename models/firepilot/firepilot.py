"""FirePilot v1.1: a learned containment-line planner (numpy + scipy only).

Generated from the training code (fp/core.py + fp/planner.py). See MODEL_CARD.md.
Main entry points: Context(...) or Context.from_store(npz, rec), ValueNet(path), plan_value(ctx, net, rng),
plan_proxy(ctx, rng), BASELINES, build_times(...).
"""
import warnings
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


# ---- planner ----



BASELINES = {
    'direct_head': dict(mode=0, phi_c=0.0, omega=90.0, d_h=90.0, d_f=90.0),
    'indirect_1km': dict(mode=0, phi_c=0.0, omega=90.0, d_h=1000.0, d_f=1000.0),
    'indirect_2km': dict(mode=0, phi_c=0.0, omega=90.0, d_h=2000.0, d_f=2000.0),
    'indirect_4km': dict(mode=0, phi_c=0.0, omega=90.0, d_h=4000.0, d_f=4000.0),
    'anchor_flank': dict(mode=1, phi_c=0.0, omega=180.0, d_h=90.0, d_f=90.0),
}
D_MIN, D_MAX = 90.0, 8000.0


def sample_plan(rs):
    mode = int(rs.random() < 0.5)
    phi = float(rs.normal(0, 50)) if rs.random() < 0.75 else float(rs.uniform(-180, 180))
    phi = (phi + 180.0) % 360.0 - 180.0
    omega = float(rs.uniform(15, 180))
    d_h = float(np.exp(rs.uniform(np.log(D_MIN), np.log(D_MAX))))
    d_f = float(np.exp(rs.uniform(np.log(D_MIN), np.log(D_MAX)))) if rs.random() < 0.6 else d_h
    return dict(mode=mode, phi_c=phi, omega=omega, d_h=d_h, d_f=d_f)


def to_vec(p):
    return np.array([p['phi_c'], p['omega'], np.log(p['d_h']), np.log(p['d_f'])])


def from_vec(mode, v):
    phi = (v[0] + 180.0) % 360.0 - 180.0
    return dict(mode=int(mode), phi_c=float(phi), omega=float(np.clip(v[1], 5, 180)),
                d_h=float(np.exp(np.clip(v[2], np.log(D_MIN), np.log(D_MAX)))),
                d_f=float(np.exp(np.clip(v[3], np.log(D_MIN), np.log(D_MAX)))))


TRAINED_WORK_H = (34.0, 35.95)   # v1.1 (27 Nov 2026): working hours in 72 h over the training scenarios (crew start
                                 # 06-18 h local solar); v1.0 saw 34.0 only (afternoon starts; firepilot_weights_v1.0.npz)


class Context:
    """What the planner knows. Built from a Scenario (runs one forecast simulation) or from stored arrays."""

    def __init__(self, burned0, arr_f, fl_f, cost, rate, clock, res, w=None, horizon=HORIZON_H):
        self.burned0, self.arr_f, self.fl_f, self.cost = burned0, arr_f, fl_f, cost
        self.rate, self.clock, self.res, self.horizon = rate, clock, res, horizon
        self.head = head_azimuth(arr_f, burned0, w)
        self.fr = Frame(burned0, res, self.head)
        self.fburn = np.isfinite(arr_f) & (arr_f <= horizon)
        self.a0 = burned0.sum()
        self.af = self.fburn.sum()
        self.gf = max(self.af - self.a0, 1)
        self._scn = self.scenario_features()

    @classmethod
    def from_store(cls, z, rec):
        """From a stored case (bank npz + json): no simulation needed."""
        arr = z['arr_f'].astype(np.float64)
        arr[arr >= 9e3] = np.nan
        c = cls.__new__(cls)
        c.burned0, c.arr_f, c.fl_f = z['burned0'], arr, z['fl_f'].astype(np.float64)
        c.cost = z['cost'].astype(np.float32)
        c.rate, c.res, c.horizon = float(rec['rate']), float(rec['res']), HORIZON_H
        c.clock = (z['clock_t'].astype(np.float64), z['clock_w'].astype(np.float64))
        c.head = float(rec['head'])
        c.fr = Frame(c.burned0, c.res, c.head)
        c.fburn = np.isfinite(arr) & (arr <= c.horizon)
        c.a0 = c.burned0.sum()
        c.af = c.fburn.sum()
        c.gf = max(c.af - c.a0, 1)
        c._scn = c.scenario_features()
        return c

    def scenario_features(self):
        a, b0, cell = self.arr_f, self.burned0, self.res ** 2 / 1e4
        new = self.fburn & ~b0
        g = [((np.isfinite(a) & (a <= h)) & ~b0).sum() / self.gf for h in (12, 24, 48)]
        fln = self.fl_f[new]
        edge = np.zeros_like(b0)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        r_eq0 = np.sqrt(self.a0 * self.res ** 2 / np.pi)
        r_eq1 = np.sqrt(self.af * self.res ** 2 / np.pi)
        t, W = self.clock
        if not (TRAINED_WORK_H[0] - 0.5 <= W[-1] <= TRAINED_WORK_H[1] + 0.5):
            # Audit, 25 Nov 2026: v1.0 trained on afternoon starts only (34 working hours in 72 h; input sd 1e-6), and a
            # morning clock raised its predicted savings by .05-.15. v1.1 trained on crew starts 06-18 h (34.0-35.95 h,
            # sd 0.013 in units of 36 h). All 48 exam fires had 34 h.
            warnings.warn(f'FirePilot: the crew clock gives {W[-1]:.1f} working hours in {self.horizon:.0f} h; the value '
                          f'network saw {TRAINED_WORK_H[0]}-{TRAINED_WORK_H[1]} h, so its savings are not reliable here',
                          stacklevel=2)
        return np.array([
            np.log10(self.a0 * cell + 1), np.log10((self.af - self.a0) * cell + 1), g[0], g[1], g[2],
            np.log10(self.rate), W[-1] / 36.0, (r_eq1 - r_eq0) / self.horizon / 100.0,
            float(np.mean(fln > FL_HOLD_M)) if len(fln) else 0.0,
            float(np.log10(np.median(fln) + 0.1)) if len(fln) else -1.0,
            float((self.fburn & edge).sum() > 0), float(np.mean(self.cost[new])) if new.any() else 1.0,
            np.log10(self.rate * W[-1] / max(2 * np.pi * r_eq1, 1.0) + 1e-3),   # buildable line vs final forecast perimeter
        ], float)

    def plan_eval(self, plan):
        """Plan-vs-forecast features and the proxy outcome (connectivity of the forecast burn behind held line)."""
        idx, tb, km = build_times(self.fr, plan, self.cost, self.rate, self.clock, self.horizon)
        n = len(idx)
        if n == 0:
            return dict(s_proxy=0.0, s_proxy3=0.0, f=np.zeros(18), km=0.0), (idx, tb)
        af = self.arr_f.ravel()[idx]
        af = np.where(np.isfinite(af), af, 1e9)
        fl = self.fl_f.ravel()[idx]
        built = np.isfinite(tb)
        won = built & (tb < af)
        won3 = built & (tb < af - 3.0)
        hold = fl <= FL_HOLD_M
        reached = af <= self.horizon
        eff, eff3 = won & hold, won3 & hold
        sp = []
        for e in (eff, eff3):
            m = np.zeros(self.burned0.size, bool)
            m[idx[e]] = True
            m = m.reshape(self.burned0.shape)
            lab, _ = ndi.label(self.fburn & ~m, structure=np.ones((3, 3), bool))
            keep = np.unique(lab[self.burned0])
            keep = keep[keep > 0]
            ap = np.isin(lab, keep).sum() if len(keep) else self.a0
            sp.append(float(np.clip((self.af - ap) / self.gf, 0, 1)))
        # shadow of the line: forecast growth beyond the line contour inside the sector (what the line could protect
        # if the fire could not go around its ends), and how late the forecast fire gets there
        dl = (self.fr.Phi - plan['phi_c'] + 180.0) % 360.0 - 180.0
        wgt = ((1.0 + np.cos(np.radians(dl))) / 2.0) ** 2
        dd = np.maximum(plan['d_f'] + (plan['d_h'] - plan['d_f']) * wgt, self.res)
        shadow = self.fburn & (self.fr.D >= dd) & (np.abs(dl) <= plan['omega'])
        s_sh = shadow.sum() / self.gf
        t_sh = float(np.nanmean(self.arr_f[shadow])) / self.horizon if shadow.any() else 1.0
        e_frac = eff.sum() / max(n, 1)
        tfin = tb[built].max() if built.any() else self.horizon
        f = np.array([
            plan['mode'], np.cos(np.radians(plan['phi_c'])), np.sin(np.radians(plan['phi_c'])), plan['omega'] / 180.0,
            np.log10(plan['d_h']) - 3, np.log10(plan['d_f']) - 3,
            np.log10(km + 0.1), built.mean(), (won & reached).sum() / max(reached.sum(), 1), (reached & ~hold).mean(),
            reached.mean(), tfin / self.horizon, s_sh, t_sh, s_sh * e_frac, e_frac, sp[0], sp[1]], float)
        return dict(s_proxy=sp[0], s_proxy3=sp[1], f=f, km=km), (idx, tb)

    def features(self, plan):
        pe, _ = self.plan_eval(plan)
        return np.concatenate([self._scn, pe['f']]), pe


N_FEAT = 13 + 18


class ValueNet:
    """numpy MLP: x (standardised) -> 2 x tanh hidden -> mean saved fraction (sigmoid) and log-sigma.
    An ensemble file (key n_members) averages the members' means; sigma = sqrt(mean member var + spread var)."""

    def __init__(self, path):
        z = np.load(path)
        n = int(z['n_members']) if 'n_members' in z.files else 0
        sfx = [f'_{k}' for k in range(n)] if n else ['']
        self.members = [([z[f'W{i}{s}'] for i in range(3)], [z[f'b{i}{s}'] for i in range(3)]) for s in sfx]
        self.mu, self.sd = z['mu'], z['sd']
        self.sig_scale = float(z['sig_scale']) if 'sig_scale' in z.files else 1.0

    def __call__(self, X):
        h0 = (np.atleast_2d(X) - self.mu) / self.sd
        ms, ss = [], []
        for W, b in self.members:
            h = np.tanh(h0 @ W[0] + b[0])
            h = np.tanh(h @ W[1] + b[1])
            o = h @ W[2] + b[2]
            ms.append(1.0 / (1.0 + np.exp(-o[:, 0])))
            ss.append(np.exp(np.clip(o[:, 1], -6, 2)))
        ms, ss = np.array(ms), np.array(ss)
        m = ms.mean(0)
        s = np.sqrt((ss ** 2).mean(0) + ms.var(0)) * self.sig_scale
        return m, s


def candidate_set(rs, n=160):
    c = [dict(p) for p in BASELINES.values()]
    for d in (500, 1000, 2000, 3000, 5000):
        for om in (60, 90, 120, 180):
            for mode in (0, 1):
                c.append(dict(mode=mode, phi_c=0.0, omega=float(om), d_h=float(d), d_f=float(d)))
    c += [sample_plan(rs) for _ in range(n)]
    return c


def plan_proxy(ctx, rs, n=160):
    """r0 (no learning): pick the candidate with the best proxy saved fraction (ties -> shorter line)."""
    best, bk = None, None
    for p in candidate_set(rs, n):
        pe, _ = ctx.plan_eval(p)
        k = (pe['s_proxy'], -pe.get('km', 0.0))
        if bk is None or k > bk:
            best, bk = p, k
    return best, dict(pred=bk[0])


def plan_value(ctx, net, rs, n=160, refine=2, n_ref=48, top=6):
    """Learned planner: score candidates with the value network, then refine around the best (CEM on the
    surrogate). Returns the argmax plan and the predicted saved fraction with its sigma."""
    cands = candidate_set(rs, n)
    X = np.array([ctx.features(p)[0] for p in cands])
    m, s = net(X)
    for _ in range(refine):
        order = np.argsort(-m)[:top]
        new = []
        for k in order:
            v = to_vec(cands[k])
            sc = np.array([25.0, 20.0, 0.35, 0.35])
            for _ in range(n_ref // top):
                mode = cands[k]['mode'] if rs.random() < 0.85 else 1 - cands[k]['mode']
                new.append(from_vec(mode, v + sc * rs.standard_normal(4)))
        Xn = np.array([ctx.features(p)[0] for p in new])
        mn, sn = net(Xn)
        cands += new
        X = np.vstack([X, Xn])
        m, s = np.concatenate([m, mn]), np.concatenate([s, sn])
    k = int(np.argmax(m))
    return cands[k], dict(pred=float(m[k]), sigma=float(s[k]))


