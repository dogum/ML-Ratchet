"""Plans, baselines, features, the proxy planner, the learned value-function planner and the search bounds.

A plan is dict(mode, phi_c, omega, d_h, d_f) (see env.line_cells). Everything a planner may use comes from
Context: the detection mask, the landscape (fuel -> line cost), the crew budget and clock, and ONE no-action run
of the FORECAST world (arrival time and flame length). The truth world is only used to score plans (and by the
perfect-foresight bound).
"""
import numpy as np
from scipy import ndimage as ndi

from .core import Frame, head_azimuth, build_times, FL_HOLD_M, HORIZON_H

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
    def from_scenario(cls, S):
        o = S.run('forecast', full=True)
        fl = np.nan_to_num(o['rasters']['flame_length'], nan=0.0)
        return cls(S.burned0, o['arrival_time'], fl, S.cost, S.rate, S.clock, S.res, S.w, S.horizon)

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


def cem_search(S, fr, world='truth', rs=None, iters=4, pop=8, init=(), elite=3):
    """Simulator-in-the-loop CEM over plans in both modes (truth world = perfect-foresight bound; forecast world =
    model-based search with the forecast). Seeds include the baselines and any `init` plans. Returns the best plan
    (by the chosen world) and the number of simulations."""
    rs = rs or np.random.default_rng(0)
    nn = S.none() if world == 'truth' else None
    if world == 'forecast':
        fo = S.run('forecast', full=True)
        fl_fc = np.nan_to_num(fo['rasters']['flame_length'], nan=0.0)
        a_none = S.area_ha(fo['arrival_time'])
    cache = {}

    def score(p):
        key = (p['mode'], round(p['phi_c'], 1), round(p['omega'], 1), round(p['d_h']), round(p['d_f']))
        if key in cache:
            return cache[key]
        if world == 'truth':
            r = S.evaluate(fr, p)
            v = r['area']
        else:
            idx, tb, km, n, nov = S.lines_for(fr, p, fl_world=fl_fc)
            v = S.area_ha(S.run('forecast', (idx, tb)))
        cache[key] = v
        return v

    seeds = [dict(p) for p in BASELINES.values()] + [dict(p) for p in init]
    res = [(score(p), i, p) for i, p in enumerate(seeds)]
    for mode in (0, 1):
        ms = [r for r in res if r[2]['mode'] == mode]
        ms.sort(key=lambda r: r[0])
        mu = to_vec(ms[0][2]) if ms else np.array([0, 90, np.log(1000), np.log(1000)])
        sd = np.array([45.0, 45.0, 0.8, 0.8])
        for it in range(iters):
            pts = [from_vec(mode, mu + sd * rs.standard_normal(4)) for _ in range(pop)]
            sc = [(score(p), 1000 + it * 100 + j, p) for j, p in enumerate(pts)]
            res += sc
            allm = sorted([r for r in res if r[2]['mode'] == mode], key=lambda r: r[0])[:elite]
            V = np.array([to_vec(r[2]) for r in allm])
            mu = V.mean(0)
            sd = np.maximum(V.std(0) * 1.2, np.array([8.0, 8.0, 0.12, 0.12])) * 0.8 + sd * 0.2
    best = min(res, key=lambda r: r[0])
    return best[2], dict(n_sims=len(cache), area=best[0])
