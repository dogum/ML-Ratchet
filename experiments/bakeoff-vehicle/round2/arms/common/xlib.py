"""Round 2: the cross-family ensemble (PREREG section 3) and the table-from-trace input rebuild. numpy only.

table_from_trace(batch)   a trace-track batch -> the table track's (X (n, 151), E_phys (n,)) with EnergyNet's own
                          feature code (features.prep values from the trace, route_vec, phys_ints, e_phys, phys_vec; the
                          vehicle vector from spec with the two label entries made relative to E_phys again). The same
                          functions the harness uses (bake.e_phys_from_trace, tests/check_trace.py).
CrossFamily(members)      members = [(name, predict.py path, 'trace' | 'table'), ...]; equal weights; every member's
                          5 log-quantiles averaged per level (enslib.combine); pred = exp(mean log q50).
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enslib  # noqa: E402

QKEYS = enslib.QKEYS


def table_from_trace(batch):
    import bake                       # the harness (numpy): prep_trace, and features via its sys.path
    import features as F
    n = int(batch['n'])
    X = np.zeros((n, F.NR + F.NF + F.NV), np.float32); E = np.zeros(n)
    for i in range(n):
        tr = batch['traces'][i]; sp = np.asarray(batch['spec'][i], np.float64); rl = batch['rl'][i]
        P = bake.prep_trace(tr)
        arch = F.ARCHS[int(batch['arch'][i])]; cls = F.CLASSES[int(batch['cls'][i])]
        hd = cls in F.HD
        m = float(rl[0])
        ph = F.phys_ints(P, m, float(rl[1]), float(rl[2]), float(rl[3]), float(rl[4]), hd)
        soc = float(sp[41]) if sp[42] > 0.5 else None
        cap = float(np.exp(sp[33]) * 10.0) if sp[34] > 0.5 else None
        Ei, Eh, Eb, f = F.e_phys(ph, arch, m, float(P['N']), soc, cap, hd)
        dist = float(np.sum(P['v']))
        Fv = F.phys_vec(ph, Ei, Eh, Eb, f, m, dist)
        V = sp.copy()
        adj = np.log(max(Ei, 1e-9) / max(dist / 1000.0, 1e-6))
        for j in (35, 38):
            if V[j + 2] > 0.5:
                V[j] = V[j] - adj
        X[i] = np.concatenate([F.route_vec(P), Fv, V]); E[i] = Ei
    return X, E


def _load_module(path, name):
    d = os.path.dirname(path)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class CrossFamily:
    def __init__(self, members):
        self.members = [(nm, _load_module(p, 'r2x_member_' + nm), kind) for nm, p, kind in members]

    def member_outputs(self, batch):
        tb = None
        outs = {}
        for nm, mod, kind in self.members:
            if kind == 'trace':
                outs[nm] = mod.predict(batch)
            else:
                if tb is None:
                    X, E = table_from_trace(batch)
                    tb = {k: batch[k] for k in ('uid', 'trace_key', 'source', 'period', 'n', 'arch', 'cls') if k in batch}
                    tb.update(track='table', X=X, E_phys=E)
                outs[nm] = mod.predict(tb)
        return outs

    def predict(self, batch):
        outs = self.member_outputs(batch)
        LQ = [np.log(np.maximum(np.stack([o[k] for k in QKEYS], 1), 1e-12)) for o in outs.values()]
        return enslib.as_output(enslib.combine(LQ))

    def predict_many(self, batches):
        return [self.predict(b) for b in batches]
