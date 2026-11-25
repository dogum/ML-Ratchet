"""Round 2: numpy inference for ensembles (PREREG section 0's combination rule). numpy only.

TraceEnsemble   k members of one v1.0 trace architecture (weights in tracelib's format). The 5-s block encoding of each
                trace and the physics anchor (bake.e_phys_from_trace, the harness's numpy function) are computed once
                and shared; each member runs its own statics, core and head.
combine(list)   members' 5 log-quantiles -> the ensemble's: mean log-quantile per level; pred = q50.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.dirname(os.path.dirname(HERE))
BAKEOFF = os.path.dirname(R2)
for p in (os.path.join(BAKEOFF, 'arms_trace', 'common'),):
    if p not in sys.path:
        sys.path.insert(0, p)
import tracelib as TL  # noqa: E402

QKEYS = ['q05', 'q25', 'q50', 'q75', 'q95']


def combine(logq_list):
    """list of (n, 5) log-quantile arrays (absolute log MJ) -> (n, 5) ensemble log-quantiles (sorted per row)."""
    L = np.mean(np.stack(logq_list, 0), 0)
    return np.sort(L, 1)


def as_output(logq, comp=None):
    Q = np.exp(logq)
    out = {'pred': Q[:, 2].copy()}
    for j, k in enumerate(QKEYS):
        out[k] = Q[:, j]
    if comp is not None:
        out['components'] = comp
    return out


class TraceEnsemble:
    def __init__(self, paths, group_max=64):
        self.models = [TL.Model(p, group_max) for p in paths]
        self.group_max = group_max
        self._dyn = {}

    def _trace(self, key, tr):
        if key is not None and key in self._dyn:
            return self._dyn[key]
        r = (TL.dyn_features(tr),) + TL.trip_dist_dur(tr)
        if key is not None:
            if len(self._dyn) > 4096:
                self._dyn.clear()
            self._dyn[key] = r
        return r

    def member_logq(self, batch):
        """-> list of (n, 5) absolute log-quantiles (one per member), list of (n, 5) log components, enc dist."""
        n = int(batch['n'])
        keys = batch.get('trace_key')
        enc = [self._trace(None if keys is None else str(keys[i]), batch['traces'][i]) for i in range(n)]
        dist = np.array([e[1] for e in enc]); dur = np.array([e[2] for e in enc])
        m0 = self.models[0]
        A0, Eph, anc = m0.anchors(batch, dist)
        items = []
        for i, e in enumerate(enc):
            for a in range(0, len(e[0]), TL.CHUNK):
                items.append((i, e[0][a:a + TL.CHUNK]))
        bylen = {}
        for j, (i, dy) in enumerate(items):
            bylen.setdefault(len(dy), []).append(j)
        LQ, LC = [], []
        for m in self.models:
            A0m = TL.anchor_dumb(m.W, batch['arch'], batch['cls'], dist)
            st = TL.statics(m.W, m.cfg, batch['spec'], batch['rl'], dist, dur, A0m, Eph)
            d = m.W['lnf.weight'].shape[0]
            pooled = np.zeros((n, d)); wsum = np.zeros(n)
            for T, js in bylen.items():
                for a in range(0, len(js), self.group_max):
                    g = js[a:a + self.group_max]
                    rows = np.array([items[j][0] for j in g])
                    dyn = np.stack([items[j][1] for j in g])
                    P = TL.encode(m.W, m.cfg, dyn, st[rows])
                    np.add.at(pooled, rows, P * T)
                    np.add.at(wsum, rows, T)
            pooled /= wsum[:, None]
            q, c = TL.head(m.W, pooled, st)
            LQ.append(np.log(anc)[:, None] + q)
            LC.append(np.log(anc)[:, None] + c)
        return LQ, LC

    def predict(self, batch):
        LQ, LC = self.member_logq(batch)
        return as_output(combine(LQ), np.exp(np.mean(np.stack(LC, 0), 0)))

    def predict_many(self, batches):
        return [self.predict(b) for b in batches]
