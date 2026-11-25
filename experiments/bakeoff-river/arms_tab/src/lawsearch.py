"""lawsearch: a small symbolic-regression engine (numpy + scipy only).

PySR-style: island populations evolve expression trees by mutation and crossover; constants inside each tree are
fitted by Levenberg-Marquardt (scipy.optimize.least_squares); a hall of fame keeps the best expression found at
every complexity, so the result is a Pareto front of complexity vs error.

Trees are nested tuples:
    ('x', i)            input variable i
    ('c', j)            constant j (values live in a separate vector)
    (op, a)             unary op
    (op, a, b)          binary op
Complexity = number of nodes (PySR's default).
"""
import math, time
import numpy as np
from scipy.optimize import least_squares

BIN = ('+', '-', '*', '/', '^')
UNA = ('exp', 'log', 'sqrt', 'inv', 'sq')


def _ev(t, X, c):
    k = t[0]
    if k == 'x':
        return X[:, t[1]]
    if k == 'c':
        return np.full(X.shape[0], c[t[1]])
    a = _ev(t[1], X, c)
    if k == '+': return a + _ev(t[2], X, c)
    if k == '-': return a - _ev(t[2], X, c)
    if k == '*': return a * _ev(t[2], X, c)
    if k == '/': return a / _ev(t[2], X, c)
    if k == '^':
        b = _ev(t[2], X, c)
        return np.where(a > 0, np.abs(a) ** b, np.nan)          # real powers of positive bases only
    if k == 'exp': return np.exp(np.clip(a, -50, 50))
    if k == 'log': return np.where(a > 0, np.log(np.abs(a) + 1e-300), np.nan)
    if k == 'sqrt': return np.where(a >= 0, np.sqrt(np.abs(a)), np.nan)
    if k == 'inv': return 1.0 / a
    if k == 'sq': return a * a
    raise ValueError(k)


def evaluate(t, X, c):
    with np.errstate(all='ignore'):
        y = _ev(t, X, c)
    return np.broadcast_to(y, (X.shape[0],)).astype(float)


def size(t):
    if t[0] in ('x', 'c'):
        return 1
    return 1 + sum(size(s) for s in t[1:])


def n_const(t):
    if t[0] == 'c': return 1
    if t[0] == 'x': return 0
    return sum(n_const(s) for s in t[1:])


def renumber(t, c):
    """Renumber constants 0..k-1 in tree order; return (tree, consts)."""
    out = []

    def rec(u):
        if u[0] == 'c':
            out.append(c[u[1]]); return ('c', len(out) - 1)
        if u[0] == 'x':
            return u
        return (u[0],) + tuple(rec(s) for s in u[1:])
    t2 = rec(t)
    return t2, np.array(out, float)


def depth(t):
    if t[0] in ('x', 'c'): return 1
    return 1 + max(depth(s) for s in t[1:])


def nodes(t, path=()):
    yield path, t
    if t[0] not in ('x', 'c'):
        for i, s in enumerate(t[1:]):
            yield from nodes(s, path + (i + 1,))


def get(t, path):
    for p in path:
        t = t[p]
    return t


def put(t, path, new):
    if not path:
        return new
    p = path[0]
    lst = list(t)
    lst[p] = put(t[p], path[1:], new)
    return tuple(lst)


def to_str(t, c, names, digits=4):
    k = t[0]
    if k == 'x': return names[t[1]]
    if k == 'c':
        v = c[t[1]]
        return f'{v:.{digits}g}' if v >= 0 else f'({v:.{digits}g})'
    if k in BIN:
        return f'({to_str(t[1], c, names, digits)} {k} {to_str(t[2], c, names, digits)})'
    return f'{k}({to_str(t[1], c, names, digits)})'


def to_js(t, c, names):
    """JavaScript expression (names are JS identifiers)."""
    k = t[0]
    if k == 'x': return names[t[1]]
    if k == 'c': return repr(float(c[t[1]]))
    a = to_js(t[1], c, names)
    if k in ('+', '-', '*', '/'):
        return f'({a} {k} {to_js(t[2], c, names)})'
    if k == '^': return f'Math.pow({a}, {to_js(t[2], c, names)})'
    if k == 'exp': return f'Math.exp({a})'
    if k == 'log': return f'Math.log({a})'
    if k == 'sqrt': return f'Math.sqrt({a})'
    if k == 'inv': return f'(1 / {a})'
    if k == 'sq': return f'({a} * {a})'


def to_json(t):
    return list(t[:1]) + [to_json(s) if isinstance(s, tuple) else s for s in t[1:]]


def from_json(j):
    if j[0] in ('x', 'c'):
        return (j[0], int(j[1]))
    return (j[0],) + tuple(from_json(s) for s in j[1:])


class Search:
    """Symbolic regression by regularised evolution with constant fitting.

    loss(y, yhat, w) is weighted MSE. Inputs X (n, d), target y (n,), optional weights w.
    """

    def __init__(self, names, unary=UNA, binary=BIN, maxsize=25, maxdepth=8, n_islands=4, pop=120,
                 tourn=8, parsimony=0.002, p_opt=0.15, seed=0, n_fit=1200, fixed_seeds=()):
        self.names = list(names); self.d = len(names)
        self.una = tuple(unary); self.bin = tuple(binary)
        self.maxsize = maxsize; self.maxdepth = maxdepth
        self.n_islands = n_islands; self.pop = pop; self.tourn = tourn
        self.parsimony = parsimony; self.p_opt = p_opt
        self.rng = np.random.default_rng(seed); self.n_fit = n_fit
        self.fixed_seeds = list(fixed_seeds)    # list of (tree, consts) to plant in the first population
        self.hof = {}                           # complexity -> (loss, tree, consts)
        self.n_eval = 0

    # ---------- random trees ----------
    def _leaf(self):
        if self.rng.random() < 0.6:
            return ('x', int(self.rng.integers(self.d))), []
        return ('c', 0), [float(self.rng.normal(0, 2))]

    def _rand(self, dep):
        if dep <= 1 or self.rng.random() < 0.3:
            return self._leaf()
        if self.una and self.rng.random() < 0.25:
            op = self.una[self.rng.integers(len(self.una))]
            a, ca = self._rand(dep - 1)
            return (op, a), ca
        op = self.bin[self.rng.integers(len(self.bin))]
        a, ca = self._rand(dep - 1); b, cb = self._rand(dep - 1)
        b = self._shift(b, len(ca))
        return (op, a, b), ca + cb

    @staticmethod
    def _shift(t, k):
        if t[0] == 'c': return ('c', t[1] + k)
        if t[0] == 'x': return t
        return (t[0],) + tuple(Search._shift(s, k) for s in t[1:])

    def random_tree(self, dep=3):
        t, c = self._rand(dep)
        return renumber(t, np.array(c, float))

    # ---------- fitness ----------
    def _loss(self, t, c, X, y, w):
        yh = evaluate(t, X, c)
        if not np.all(np.isfinite(yh)):
            return np.inf
        r = yh - y
        return float(np.sum(w * r * r) / np.sum(w))

    def optimise(self, t, c, X, y, w, nfev=40):
        if len(c) == 0:
            return c
        sw = np.sqrt(w)

        def res(p):
            yh = evaluate(t, X, p)
            r = (yh - y) * sw
            return np.where(np.isfinite(r), r, 1e3)
        try:
            r = least_squares(res, c, method='lm', max_nfev=nfev * (len(c) + 1))
            p = r.x
        except Exception:
            return c
        return p if np.all(np.isfinite(p)) else c

    def score(self, ind):
        loss, t = ind[0], ind[1]
        return loss * (1 + self.parsimony * size(t)) if np.isfinite(loss) else np.inf

    def _record(self, t, c, loss):
        s = size(t)
        if s > self.maxsize or not np.isfinite(loss):
            return
        cur = self.hof.get(s)
        if cur is None or loss < cur[0] - 1e-15:
            self.hof[s] = (loss, t, np.array(c, float))

    # ---------- mutations ----------
    def mutate(self, t, c):
        r = self.rng.random()
        allnodes = list(nodes(t))
        path, sub = allnodes[self.rng.integers(len(allnodes))]
        c = np.array(c, float)
        if r < 0.20 and len(c):                                  # perturb a constant
            j = self.rng.integers(len(c))
            c[j] = c[j] * float(np.exp(self.rng.normal(0, 0.5))) + self.rng.normal(0, 0.1)
            return t, c
        if r < 0.40:                                             # replace a subtree
            new, cn = self._rand(int(self.rng.integers(1, 4)))
            new = self._shift(new, len(c))
            t2 = put(t, path, new)
            return renumber(t2, np.concatenate([c, cn]))
        if r < 0.55:                                             # change an operator / variable
            if sub[0] == 'x':
                new = ('x', int(self.rng.integers(self.d)))
            elif sub[0] == 'c':
                new = ('x', int(self.rng.integers(self.d)))
            elif len(sub) == 2 and self.una:
                new = (self.una[self.rng.integers(len(self.una))], sub[1])
            elif len(sub) == 3:
                new = (self.bin[self.rng.integers(len(self.bin))], sub[1], sub[2])
            else:
                new = sub
            return renumber(put(t, path, new), c)
        if r < 0.72:                                             # insert: wrap a subtree
            if self.una and self.rng.random() < 0.35:
                new = (self.una[self.rng.integers(len(self.una))], sub)
                return renumber(put(t, path, new), c)
            op = self.bin[self.rng.integers(len(self.bin))]
            other, co = self._leaf()
            other = self._shift(other, len(c))
            new = (op, sub, other) if self.rng.random() < 0.5 else (op, other, sub)
            return renumber(put(t, path, new), np.concatenate([c, co]))
        if r < 0.90:                                             # delete: replace a node by one of its children
            if sub[0] in ('x', 'c'):
                return t, c
            child = sub[1 + self.rng.integers(len(sub) - 1)]
            return renumber(put(t, path, child), c)
        new_c = ('c', len(c))                                    # replace a subtree by a constant
        return renumber(put(t, path, new_c), np.concatenate([c, [float(self.rng.normal(0, 1))]]))

    def crossover(self, a, ca, b, cb):
        pa, _ = list(nodes(a))[self.rng.integers(size(a))]
        pb, sb = list(nodes(b))[self.rng.integers(size(b))]
        sb = self._shift(sb, len(ca))
        return renumber(put(a, pa, sb), np.concatenate([ca, cb]))

    # ---------- main loop ----------
    def fit(self, X, y, w=None, seconds=120, max_gen=10**9, verbose=False, log=None):
        X = np.asarray(X, float); y = np.asarray(y, float)
        w = np.ones_like(y) if w is None else np.asarray(w, float)
        n = len(y)
        idx = self.rng.permutation(n)[:min(n, self.n_fit)]
        Xf, yf, wf = X[idx], y[idx], w[idx]
        t0 = time.time()
        isl = []
        for k in range(self.n_islands):
            P = []
            seeds = self.fixed_seeds if k == 0 else []
            for (t, c) in seeds:
                c = self.optimise(t, np.array(c, float), Xf, yf, wf)
                P.append([self._loss(t, c, Xf, yf, wf), t, c])
            while len(P) < self.pop:
                t, c = self.random_tree(int(self.rng.integers(1, 4)))
                c = self.optimise(t, c, Xf, yf, wf, nfev=15)
                P.append([self._loss(t, c, Xf, yf, wf), t, c])
            for ind in P:
                self._record(ind[1], ind[2], ind[0])
            isl.append(P)
        gen = 0
        while time.time() - t0 < seconds and gen < max_gen:
            gen += 1
            for P in isl:
                for _ in range(self.pop // 4):
                    cand = [P[i] for i in self.rng.integers(len(P), size=self.tourn)]
                    cand.sort(key=self.score)
                    a = cand[0]
                    if self.rng.random() < 0.15:
                        b = cand[1]
                        t, c = self.crossover(a[1], a[2], b[1], b[2])
                    else:
                        t, c = self.mutate(a[1], a[2])
                    if size(t) > self.maxsize or depth(t) > self.maxdepth:
                        continue
                    if self.rng.random() < self.p_opt or n_const(t) <= 2:
                        c = self.optimise(t, c, Xf, yf, wf, nfev=20)
                    loss = self._loss(t, c, Xf, yf, wf)
                    self.n_eval += 1
                    self._record(t, c, loss)
                    # regularised evolution: replace the oldest member
                    P.pop(0); P.append([loss, t, c])
            if gen % 20 == 0:                                    # migration: hall-of-fame members into islands
                hof = list(self.hof.values())
                for P in isl:
                    for _ in range(3):
                        h = hof[self.rng.integers(len(hof))]
                        P[self.rng.integers(len(P))] = [h[0], h[1], h[2].copy()]
            if verbose and gen % 25 == 0:
                msg = f'gen {gen} t={time.time()-t0:.0f}s evals={self.n_eval} front=' + ' '.join(
                    f'{s}:{v[0]:.4g}' for s, v in sorted(self.hof.items()))
                print(msg, flush=True)
                if log:
                    log.write(msg + '\n'); log.flush()
        return self

    def front(self, X=None, y=None, w=None, refit=True):
        """Pareto front (complexity ascending, strictly improving loss). Optionally refit constants on (X, y)."""
        out = []
        for s in sorted(self.hof):
            loss, t, c = self.hof[s]
            if refit and X is not None:
                w_ = np.ones(len(y)) if w is None else w
                c2 = self.optimise(t, c, X, y, w_, nfev=100)
                l2 = self._loss(t, c2, X, y, w_)
                if np.isfinite(l2):
                    c, loss = c2, l2
            out.append((s, loss, t, c))
        pf, best = [], np.inf
        for s, loss, t, c in out:
            if loss < best * (1 - 1e-4):
                pf.append((s, loss, t, c)); best = loss
        return pf
