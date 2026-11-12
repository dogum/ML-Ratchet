"""Whole-life curve check for the v2 aging model on the six calibration cells: with each cell's fitted trajectory and its
early slow-graphite share, how well do simulated 4C curves match the real ones at 50%, 80% and 100% of life, for a grid of
the late-life knobs (transport-polarization growth k_e and its onset power p_e, its coupling to lost lithium and graphite
k_e2, the slow-graphite ceiling F_cap and late growth k_f2)? Scored on dQ_n-10(V) smoothed over 24 mV, and on the charge
left for the 2.0 V hold."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, itertools, numpy as np
sys.path.insert(0, _P.SRC); import realdata as RD, calibrate as CA, agesim as AS, simbank as SB, fit_life as FL, dq_early as DQ, calib_early as CE
FR = (0.5, 0.8, 1.0)

def real_targets(cells):
    T = {}
    for k in CA.CAL:
        c = cells[k]; L = c['cycle_life_pub']; idx = c['cyc_idx']; qd = c['summary']['QDischarge']; rows = []
        for f in FR:
            n = int(round(f * L / 10) * 10); r = int(np.argmin(np.abs(idx + 1 - n))); d = DQ.real_dq(c, 10, int(idx[r] + 1))
            j = idx[r]; ok = 0.5 < qd[j] < 1.2
            rows.append(dict(n=n, dq=None if d is None else CE.sm(d), hold=float(qd[j] - np.nanmax(c['Qdlin'][r])) if ok else np.nan))
        T[k] = rows
    return T

def sim_cell(k, knobs):
    """knobs: list of dicts -> per knob set, per FR: (smoothed dQ, hold)"""
    F = json.load(open(_P.WORK + '/aging_calibration.json'))[k]; E = json.load(open(_P.WORK + '/early_calibration.json'))[k]; B = len(knobs)
    P0 = CA.params(np.array(FL.CA_FITS[k]['theta']), 30.0); P0['Q_Li'] = P0['Q_Li'] * SB.Q_LI_TRIM; P0 = {a: np.repeat(v, B) for a, v in P0.items()}
    R = FL.rates_from(np.repeat(np.array(F['z'])[None], B, 0), F['c']); R.update(F=np.full(B, E['F_adj']), gam=np.full(B, 0.8), R_s=np.full(B, np.exp(E['log_R_s'])))
    for key in knobs[0]:
        R[key] = np.array([kn[key] for kn in knobs]) if not isinstance(knobs[0][key], str) else knobs[0][key]
    L = F['life']; tr = AS.trajectory(P0, R, n_max=int(L * 1.1) + 20, kappa=np.full(B, F['kappa']))
    Pa = AS.state_at(P0, R, tr, np.full(B, 1)); Qa = CE.fill(SB.curves(Pa)[0]); out = []
    for f in FR:
        n = int(round(f * L / 10) * 10); Q, T, cc, tot, ir = SB.curves(AS.state_at(P0, R, tr, np.full(B, n // 10)))
        out.append((CE.sm(CE.fill(Q) - Qa), tot - cc))
    return out

def main():
    cells, V = RD.load(); T = real_targets(cells)
    grid = [dict(k_e=ke, p_e=2.0, k_e2=k2, F_cap=fc, k_f2=kf, k_f=0.0) for ke, k2, fc, kf in itertools.product((0.0, 0.1, 0.3), (0.0, 0.1, 0.3), (0.1, 0.13), (0.0, 0.05))]
    grid.append(dict(k_e=1.5, p_e=1.0, k_e2=1.0, F_cap=1e-6, k_f2=0.2, k_f=0.02))      # v1's central late-life knobs, no early slow graphite
    S = np.zeros((len(grid), len(CA.CAL), len(FR))); H = np.zeros_like(S)
    for ci, k in enumerate(CA.CAL):
        res = sim_cell(k, grid)
        for fi, (dq, hold) in enumerate(res):
            t = T[k][fi]
            S[:, ci, fi] = np.sqrt(np.mean((dq - t['dq']) ** 2, 1)) * 1000 if t['dq'] is not None else np.nan; H[:, ci, fi] = (hold - t['hold']) * 1000
        print(k, 'done', flush=True)
    score = np.nanmean(S, (1, 2)) + 0.5 * np.nanmean(np.abs(H), (1, 2)); o = np.argsort(score)
    print('real hold mAh', {k: [round(1000 * t['hold'], 1) for t in T[k]] for k in CA.CAL})
    for j in list(o[:8]) + [len(grid) - 1]:
        print(round(score[j], 1), grid[j], 'dQ rms by stage', np.nanmean(S[j], 0).round(1), 'hold err by stage', np.nanmean(H[j], 0).round(1))
    v1 = dict(k_e=1.5, p_e=1.0, k_e2=1.0, F_cap=0.0001, k_f2=0.2, ge_drv='lli')
    json.dump(dict(grid=grid, score=score.tolist(), S=S.tolist(), H=H.tolist()), open(_P.WORK + '/late_fit.json', 'w'))

if __name__ == '__main__':
    main()
