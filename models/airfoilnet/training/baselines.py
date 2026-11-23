"""Baseline methods for the dev gate, twin tests and exam (all share the realeval method interface).

physics_method(set)   : raw simulator (mfoil, ncrit 9) sweeps at the run's Reynolds number, linear in alpha;
                        NaN where no converged point lies within 2.5 deg on each side (the solver gave no answer)
thin_airfoil(p, ...)  : Cl = 2 pi (alpha - alpha_L0), Cm = thin-airfoil Cm_c/4, no drag
mean_polar(...)       : the mean measured dev polar (Cl, ln Cd, Cm vs alpha), ignoring shape and Re
neuralfoil(...)       : NeuralFoil 0.3.3 xxxlarge (trained on XFOIL), ncrit 9, numpy port
"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'release')); sys.path.insert(2, os.path.abspath(ROOT))   # ROOT: airfoilnet.py in the repository layout
import airfoilnet as an  # noqa


def load_sweeps(tag):
    S = {}
    for f in glob.glob(os.path.join(ROOT, 'work', 'bank', f'{tag}_w*.jsonl')):
        for line in open(f):
            r = json.loads(line)
            aid, model, Re = r['id'].split('|')
            q = np.array(r['pts']) if r['pts'] else np.zeros((0, 7))
            if len(q):
                q = q[np.argsort(q[:, 0])]
            S[(aid, model, float(Re))] = q
    return S


def _interp_gap(al, x, y, maxgap=2.5):
    out = np.full(len(al), np.nan)
    if len(x) < 2:
        return out
    for i, a in enumerate(al):
        if a < x[0] - 0.01 or a > x[-1] + 0.01:
            continue
        j = np.searchsorted(x, a)
        j0, j1 = max(j - 1, 0), min(j, len(x) - 1)
        if abs(x[j1] - a) < 1e-6:
            out[i] = y[j1]; continue
        if x[j1] - x[j0] > maxgap or j0 == j1:
            continue
        out[i] = y[j0] + (y[j1] - y[j0]) * (a - x[j0]) / (x[j1] - x[j0])
    return out


class PhysicsMethod:
    """keyed by the CST params identity -> (aid, model); Re rounded to 1e4 as in simbank"""
    def __init__(self, tag, pmap):
        self.S = load_sweeps(tag)
        self.inv = {tuple(np.round(v, 10)): k for k, v in pmap.items()}

    def __call__(self, p, Re, al):
        aid, model = self.inv[tuple(np.round(p, 10))].split('|')
        q = self.S.get((aid, model, float(round(Re / 1e4) * 1e4)))
        al = np.asarray(al, float)
        if q is None or len(q) < 2:
            n = np.full(len(al), np.nan)
            return dict(cl=n, cd=n.copy(), cm=n.copy())
        return dict(cl=_interp_gap(al, q[:, 0], q[:, 1]), cd=np.exp(_interp_gap(al, q[:, 0], np.log(q[:, 2]))),
                    cm=_interp_gap(al, q[:, 0], q[:, 3]))

    def has(self, aid, model, Re):
        q = self.S.get((aid, model, float(round(Re / 1e4) * 1e4)))
        return q is not None


def thin_airfoil(p, Re, al):
    aL0, cm0 = an.thin_airfoil(p)
    al = np.asarray(al, float)
    return dict(cl=2 * np.pi * np.radians(al - aL0), cd=None, cm=np.full(len(al), cm0))


class MeanPolar:
    def __init__(self, units, dev):
        g = np.arange(-10, 25.01, 0.5)
        CL, CD, CM = [], [], []
        for aid in dev:
            for r in units[aid]['runs']:
                d = np.array([[np.nan if v is None else v for v in row] for row in r['data']], float)
                if r['kind'] == 'lift':
                    k = int(np.argmax(d[:, 0])); d = d[:k + 1]
                    o = np.argsort(d[:, 0]); d = d[o]
                    cl = np.where((g >= d[0, 0]) & (g <= d[-1, 0]), np.interp(g, d[:, 0], d[:, 1]), np.nan)
                    CL.append(cl)
                    if np.all(np.isfinite(d[:, 2])):
                        CM.append(np.where((g >= d[0, 0]) & (g <= d[-1, 0]), np.interp(g, d[:, 0], d[:, 2]), np.nan))
                else:
                    o = np.argsort(d[:, 0]); d = d[o]
                    CD.append(np.where((g >= d[0, 0]) & (g <= d[-1, 0]), np.interp(g, d[:, 0], np.log(d[:, 2])), np.nan))
        def avg(L):
            A = np.array(L); m = np.nanmean(A, 0); n = np.sum(np.isfinite(A), 0)
            m[n < 2] = np.nan
            ok = np.isfinite(m)
            return np.interp(g, g[ok], m[ok])        # constant extrapolation beyond the data
        self.g = g; self.cl = avg(CL); self.lncd = avg(CD); self.cm = avg(CM)

    def __call__(self, p, Re, al):
        al = np.asarray(al, float)
        return dict(cl=np.interp(al, self.g, self.cl), cd=np.exp(np.interp(al, self.g, self.lncd)),
                    cm=np.interp(al, self.g, self.cm))


class NeuralFoilMethod:
    def __init__(self, size='xxxlarge'):
        import neuralfoil_np as nfn
        self.nfn = nfn; self.nf = nfn.NeuralFoil(size); self.cache = {}

    def __call__(self, p, Re, al):
        key = tuple(np.round(p, 10))
        if key not in self.cache:
            from shapes import cst_coords
            self.cache[key] = self.nfn.kulfan_fit(cst_coords(np.array(p), 100))
        o = self.nf(self.cache[key], np.asarray(al, float), Re)
        return dict(cl=o['CL'], cd=o['CD'], cm=o['CM'])
