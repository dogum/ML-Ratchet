"""Twin test for CellSense NMC811, on the dev cells only. Each dev test gets a simulated twin: the cell's own fresh fit plus
the degradation modes the voltage fit found for that test (nmc_calibrate.py, same physics version). A rung reads twin and
real alike as partial discharges (1.5, 2.5, 3.5 Ah) and full. If it reads twins well and reals badly, the gap is in the
simulator; the twin's curve minus the real one shows where.

    python nmc_twin.py <ckpt> <physics: cal|ocp>
"""
import _paths as _P
import os, sys, json, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, os.path.dirname(os.path.dirname(_P.MODEL)))
import nmc_data as D, nmc_ratchet as RT, nmc_calibrate as CA, nmcsim as NS
OUT = os.path.join(_P.WORK, 'nmc')


def twins(physics):
    tag = '_ocp' if physics == 'ocp' else ''; corr = json.load(open(os.path.join(OUT, 'ocp_corr.json'))) if physics == 'ocp' else None
    F = json.load(open(os.path.join(OUT, f'cal_fresh{tag}.json'))); M = json.load(open(os.path.join(OUT, f'cal_modes{tag}.json')))
    rows = D.readings(D.DEV); P0 = {k: np.array([F[r['cell']]['params'][k] for r in rows]) for k in F[rows[0]['cell']]['params']}
    m = np.array([[M[f"{r['cell']}_{r['diag']}"]['modes'][k] for k in CA.MODES] for r in rows])
    P = CA.aged(P0, m, corr=corr); o = NS.simulate(P, dt=30.0, corr=corr); Qt = NS.q_of_v(o, D.VG)
    return rows, Qt, o['Q_total']


if __name__ == '__main__':
    ck, physics = sys.argv[1], sys.argv[2]; net = RT.load_net(os.path.join(OUT, 'ckpt', ck)); rows, Qt, qt_twin = twins(physics)
    ref_twin = {r['cell']: Qt[i] for i, r in enumerate(rows) if r['diag'] == 1}
    out = {}
    for d in (1.5, 2.5, 3.5, None):
        for kind in ('real', 'twin'):
            Q1 = np.stack([r['q'] for r in rows]) if kind == 'real' else Qt.copy(); Q0 = np.stack([r['q_ref'] if kind == 'real' else ref_twin[r['cell']] for r in rows])
            truth = np.array([r['q_total'] for r in rows]) if kind == 'real' else qt_twin
            if d is not None: Q1 = np.where(Q1 > d, np.nan, Q1)
            for ref in (True, False):
                P = RT.predict(net, Q1, Q0 if ref else None); e = 1000 * np.abs(P['q_total'] - truth)
                out[f"{kind}_{d}_{'ref' if ref else 'noref'}"] = float(e.mean())
    gap = 1000 * np.nanmean(np.abs(Qt - np.stack([r['q'] for r in rows])), 0)
    out['curve_gap_mAh_by_V'] = dict(zip([f'{v:.2f}' for v in D.VG[::25]], gap[::25].round(1).tolist())); out['twin_capacity_err_mAh'] = float(1000 * np.mean(np.abs(qt_twin - np.array([r['q_total'] for r in rows]))))
    json.dump(out, open(os.path.join(OUT, f"twin_{ck.replace('.pt', '')}_{physics}.json"), 'w'), indent=1)
    for k, v in out.items(): print(k, v if isinstance(v, dict) else round(v, 1))
