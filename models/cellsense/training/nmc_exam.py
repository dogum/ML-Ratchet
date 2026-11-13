"""CellSense NMC811 final exam: the seven exam cells (W5, W7, W9, W10, G1, V4, V5; 75 capacity tests), never used to choose
anything. Each test is read in full and as a partial discharge stopped after 1.5, 2.5 and 3.5 Ah, with the cell's first
test as the reference (tests 2 onward) and without it (all tests). Answer key: the capacity the cycler counted to 2.5 V.

Beside the network:
  dumb       guess the dev cells' median capacity (no reference); assume no fade, i.e. the reference's capacity (with one)
  classical  shift at the cut: the reference's capacity plus the gap between the two curves where the partial one stops
             (with a reference); voltage fitting (nmc_fitbase.py): fit lithium / negative / positive lost and resistance
             to the recorded part with the same electrode model, simulate to 2.5 V
Error bars: one widen-only factor per output (capacity, fade), fitted on the dev cells' readings (mlratchet.calibrate),
then the exam coverage of the 68% bars. Writes work/nmc/exam_<rung>.json.

    python nmc_exam.py <ckpt> <physics> [--no-fit]
    python nmc_exam.py --fit-only <physics>         (the classical voltage fit alone, cached)
"""
import _paths as _P
import os, sys, json, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, os.path.dirname(os.path.dirname(_P.MODEL)))
import nmc_data as D, nmc_ratchet as RT, nmc_fitbase as FB, nmc_calibrate as CA, nmcsim as NS
from mlratchet.calibrate import widen_factor, coverage, ONE_SIGMA
OUT = os.path.join(_P.WORK, 'nmc')


def fresh_fits(cells, corr, tag):
    """the classical method's new-cell fit of each cell's own first test (cached)"""
    path = os.path.join(OUT, f'fitbase_fresh{tag}.json'); F = json.load(open(path)) if os.path.exists(path) else {}
    CA.CORR = corr; rng = np.random.default_rng(1)
    for r in D.readings(cells):
        if r['diag'] != 1 or r['cell'] in F: continue
        z, info = CA.fit_fresh(r, rng); F[r['cell']] = dict(z=z.tolist(), params={k: float(v[0]) for k, v in CA.params(z).items()}, v_rms_mV=info['v_rms_mV'])
        json.dump(F, open(path, 'w'), indent=1); print('fresh fit', r['cell'], round(info['v_rms_mV'], 1), flush=True)
    return F


def classical_fit(rows, physics):
    corr = json.load(open(os.path.join(OUT, 'ocp_corr.json'))) if physics == 'ocp' else None; tag = '_ocp' if physics == 'ocp' else ''
    F = fresh_fits(sorted({r['cell'] for r in rows}), corr, tag); dev = json.load(open(os.path.join(OUT, f'cal_fresh{tag}.json')))
    mean_dev = {k: float(np.mean([dev[c]['params'][k] for c in dev])) for k in dev[next(iter(dev))]['params']}
    raw = {(r['cell'], r['diag']): r for r in D.readings(sorted({r['cell'] for r in rows}))}
    P0 = {k: np.array([(F[r['cell']]['params'] if r['ref'] else mean_dev)[k] for r in rows]) for k in mean_dev}
    T = [FB.targets_of(raw[(r['cell'], r['diag'])], r['depth']) for r in rows]; out_m, out_q, out_rms = [], [], []
    for i in range(0, len(rows), 400):
        sl = slice(i, i + 400); m, q, rms = FB.fit({k: v[sl] for k, v in P0.items()}, T[sl], corr); out_m.append(m); out_q.append(q); out_rms.append(rms)
        print('voltage fit', i + len(T[sl]), '/', len(rows), flush=True)
    return np.concatenate(out_m), np.concatenate(out_q), np.concatenate(out_rms)


def table(rows, P, extra, dev_med):
    qt = np.array([r['q_total'] for r in rows]); out = []
    for ref in (True, False):
        for d in (None,) + RT.DEPTHS:
            m = np.array([r['ref'] == ref and r['depth'] == d for r in rows]); e = lambda p: float(1000 * np.mean(np.abs(p[m] - qt[m])))
            row = dict(ref=ref, depth=d, n=int(m.sum()), model=e(P['q_total']), median_guess=e(np.full(len(rows), dev_med)))
            for k, v in extra.items():
                if np.isfinite(v[m]).all(): row[k] = e(v)
            out.append(row)
    return out


if __name__ == '__main__':
    if sys.argv[1] == '--fit-only':          # the classical baseline alone (no network), cached in work/nmc/fitbase_<physics>.json
        physics = sys.argv[2]; rows = RT.real_readings(D.EXAM); key = lambda r: f"{r['cell']}|{r['diag']}|{r['depth']}|{int(r['ref'])}"
        Mf, Qf, rms = classical_fit(rows, physics)
        json.dump({key(r): dict(q=float(q), rms_mV=float(e), modes=m.tolist()) for r, q, e, m in zip(rows, Qf, rms, Mf)}, open(os.path.join(OUT, f'fitbase_{physics}.json'), 'w'), indent=1)
        qt = np.array([r['q_total'] for r in rows]); print('voltage fit capacity error (mAh)', round(float(1000 * np.mean(np.abs(Qf - qt))), 1)); sys.exit()
    ck, physics = sys.argv[1], sys.argv[2]; net = RT.load_net(os.path.join(OUT, 'ckpt', ck))
    dev_rows = RT.real_readings(D.DEV); Pd = RT.run_real(net, dev_rows); qd = np.array([r['q_total'] for r in dev_rows])
    fac = {'q_total': widen_factor(Pd['q_total'] - qd, Pd['q_total_sig'])}
    fd = np.array([1 - r['q_total'] / r['q_ref'] for r in dev_rows]); mref = np.array([r['ref'] for r in dev_rows]); fac['fade'] = widen_factor((Pd['fade'] - fd)[mref], Pd['fade_sig'][mref])
    dev_med = float(np.median([r['q_total'] for r in D.readings(D.DEV)]))
    rows = RT.real_readings(D.EXAM); P = RT.run_real(net, rows); qt = np.array([r['q_total'] for r in rows]); ref = np.array([r['ref'] for r in rows])
    q_ref = np.array([r['q_ref'] if r['ref'] else np.nan for r in rows])
    shift = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        if r['ref']:
            j = np.where(np.isfinite(r['Q1']))[0].max(); shift[i] = r['q_ref'] + r['Q1'][j] - r['Q0'][j]
    extra = {'no_fade': q_ref, 'shift_at_cut': shift}
    if '--no-fit' not in sys.argv:
        cache = os.path.join(OUT, f'fitbase_{physics}.json'); key = lambda r: f"{r['cell']}|{r['diag']}|{r['depth']}|{int(r['ref'])}"
        C = json.load(open(cache)) if os.path.exists(cache) else {}
        if not all(key(r) in C for r in rows):
            Mf, Qf, rms = classical_fit(rows, physics)
            C = {key(r): dict(q=float(q), rms_mV=float(e), modes=m.tolist()) for r, q, e, m in zip(rows, Qf, rms, Mf)}; json.dump(C, open(cache, 'w'), indent=1)
        extra['voltage_fit'] = np.array([C[key(r)]['q'] for r in rows])
    T = table(rows, P, extra, dev_med)
    res = qt - P['q_total']; cov = {'q_total_raw': coverage(res, P['q_total_sig']), 'q_total': coverage(res, P['q_total_sig'], fac['q_total'])}
    for d in (None,) + RT.DEPTHS:
        m = np.array([r['depth'] == d for r in rows]); cov[f'q_total_{d}'] = coverage(res[m], P['q_total_sig'][m], fac['q_total'])
    fe = np.array([1 - r['q_total'] / r['q_ref'] for r in rows]); cov['fade'] = coverage((fe - P['fade'])[ref], P['fade_sig'][ref], fac['fade'])
    cells = {}
    for c in D.EXAM:
        m = np.array([r['cell'] == c and r['depth'] is not None for r in rows]); cells[c] = float(1000 * np.mean(np.abs(res[m])))
    out = dict(ckpt=ck, physics=physics, factors=fac, dev_median_q=dev_med, table=T, coverage=cov, per_cell_partial_mAh=cells,
               fade_corr=float(np.corrcoef(P['fade'][ref], fe[ref])[0, 1]),
               rows=[dict(cell=r['cell'], diag=r['diag'], depth=r['depth'], ref=r['ref'], q_true=r['q_total'], q_model=float(P['q_total'][i]), q_sig=float(P['q_total_sig'][i] * fac['q_total']),
                          lli=float(P['lli'][i]), lam_ne=float(P['lam_ne'][i]), lam_pe=float(P['lam_pe'][i]), **{k: (None if not np.isfinite(v[i]) else float(v[i])) for k, v in extra.items()}) for i, r in enumerate(rows)])
    json.dump(out, open(os.path.join(OUT, f"exam_{ck.replace('.pt', '')}.json"), 'w'), indent=1)
    print('widen factors', {k: round(v, 2) for k, v in fac.items()}, 'dev median %.3f Ah' % dev_med)
    for t in T: print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()})
    print('coverage', {k: round(v, 2) for k, v in cov.items()}); print('per cell partial mAh', {k: round(v, 1) for k, v in cells.items()}); print('fade corr %.3f' % out['fade_corr'])
