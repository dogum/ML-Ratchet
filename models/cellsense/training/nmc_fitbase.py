"""The classical baseline for CellSense NMC811: voltage fitting (Dubarry / Birkl style) with the same electrode model the
network was trained on. For each reading:
  1. the cell when new: with a reference, fit the cell's own first test (nmc_calibrate.fit_fresh); without one, take the
     dev cells' mean fresh parameters
  2. fit lithium lost, negative lost, positive lost and resistance growth to whatever part of the C/20 curve was recorded
     (the partial discharge), by a batched Levenberg-Marquardt over all readings at once
  3. simulate that aged cell down to 2.5 V: its capacity is the baseline's answer
No network, no training data: a strong, standard method.
"""
import _paths as _P
import os, sys, json, numpy as np
sys.path.insert(0, _P.SRC); import nmcsim as NS, nmc_data as D, nmc_calibrate as CA
LO, HI = np.array([-0.05, -0.05, -0.05, -0.6]), np.array([0.4, 0.4, 0.4, 5.0])


def sim_curves(P0, M, corr, qgs):
    """aged cells' voltage at the readings' capacity points (list of arrays) and their capacity to 2.5 V"""
    P = CA.aged(P0, M, corr=corr); o = NS.simulate(P, dt=60.0, corr=corr); V = []
    for b, qg in enumerate(qgs):
        m = np.isfinite(o['V'][b]); q, v = np.append(o['Q'][b][m], o['Q_total'][b]), np.append(o['V'][b][m], 2.5)
        V.append(np.interp(qg, q, v, right=2.5) if len(q) > 3 else np.full(len(qg), 9.0))
    return V, o['Q_total']


def fit(P0, targets, corr, iters=30):
    """P0: fresh parameters per reading (dict of (N,) arrays); targets: list of (qg, v) measured on the recorded part.
    Batched Levenberg-Marquardt on the four modes; returns modes (N, 4) and capacity to 2.5 V (N,)"""
    N = len(targets); qgs = [t[0] for t in targets]; vs = [t[1] for t in targets]; m = np.zeros((N, 4)); lam = np.full(N, 1e-2)
    h = np.array([2e-3, 2e-3, 2e-3, 0.04]); rep = lambda P, k: {a: np.repeat(b, k) for a, b in P.items()}
    res = lambda V: [(a - b) * 1000 for a, b in zip(V, vs)]
    V, _ = sim_curves(P0, m, corr, qgs); r = res(V); cost = np.array([np.sum(x ** 2) for x in r])
    for it in range(iters):
        Mj = np.repeat(m, 5, 0) + np.tile(np.vstack([np.zeros(4), np.diag(h)]), (N, 1))
        Vj, _ = sim_curves(rep(P0, 5), Mj, corr, [q for q in qgs for _ in range(5)])
        new = m.copy()
        for n in range(N):
            r0 = (Vj[5 * n] - vs[n]) * 1000; J = np.stack([(Vj[5 * n + k + 1] - Vj[5 * n]) * 1000 / h[k] for k in range(4)], 1)
            A = J.T @ J; g = J.T @ r0; step = -np.linalg.solve(A + lam[n] * np.diag(np.diag(A) + 1e-6), g); new[n] = np.clip(m[n] + step, LO, HI)
        Vn, _ = sim_curves(P0, new, corr, qgs); cn = np.array([np.sum(x ** 2) for x in res(Vn)]); better = cn < cost
        m[better] = new[better]; cost[better] = cn[better]; lam = np.where(better, lam / 3, lam * 4)
    _, qtot = sim_curves(P0, m, corr, qgs); rms = np.sqrt(cost / np.array([len(q) for q in qgs]))
    return m, qtot, rms


def targets_of(row, depth, n=100):
    """measured V(Q) on the part of the test that was recorded (from 0.03 Ah to the cut)"""
    end = row['q_total'] if depth is None else min(depth, row['q_total']); qg = np.linspace(0.03, end - 0.02, n)
    return qg, np.interp(qg, row['Q'], row['V'])
