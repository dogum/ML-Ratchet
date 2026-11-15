"""CellSense NMC811, round 2 (15 Nov 2026): does the network fall below the dev median when the curve is short, and does
an uncertainty-weighted pull toward a dev prior fix it? Dev cells only (W3, W4, W8); the exam is never read here.

The idea, written down before any number was seen:
  * A C/20 curve stopped after 1.5 Ah carries little about the end of discharge; the network's guess there is an
    extrapolation from simulated cells, and real curves are out of its distribution.
  * The honest fallback when the curve says little is what a population of similar cells says: the dev cells' capacity
    (no reference) or the reference minus the dev cells' typical fade (with a reference).
  * Combine the two by inverse variance: q = (q_net / s_net^2 + q_prior / s_prior^2) / (1 / s_net^2 + 1 / s_prior^2), with
    s_net the network's own bar times a widen-only factor fitted on dev, and s_prior the spread of the dev prior.
    Full curves keep the end-of-curve rule (coulomb counting) and are not touched.
  * Judged on dev in-sample (the gate as written) AND leave-one-dev-cell-out (prior and widen factor from the other two
    dev cells), so the prior is never scored on the cell it came from.

    python nmc_blend.py probe        dev table for n8 (released) and the n9-n11 networks, raw / ensemble / blended
    python nmc_blend.py sim          the same variants on the frozen simulated set val_ocpd (objective as nmc_ratchet.py)
"""
import _paths as _P
import os, sys, json, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, os.path.dirname(os.path.dirname(_P.MODEL)))
import nmc_ratchet as RT, nmc_data as D, nmc_model as NM
from mlratchet.calibrate import widen_factor
CK = os.path.join(_P.WORK, 'nmc', 'ckpt')
NETS = {'n8': 'n8_ocpj.pt+endrule', 'n9': 'n9_ocpj2.pt+endrule', 'n10': 'n10_ocpd.pt+endrule', 'n11': 'n11_ocpd_mix.pt+endrule'}


def ensemble(Ps):
    """mean of the members; bar = sqrt(mean member variance + spread of the members)"""
    out = {}
    for k in Ps[0]:
        if k.endswith('_sig'): continue
        M = np.stack([p[k] for p in Ps]); S = np.stack([p[k + '_sig'] for p in Ps])
        out[k] = M.mean(0); out[k + '_sig'] = np.sqrt((S ** 2).mean(0) + M.var(0))
    return out


def prior_stats(cells):
    """dev prior: capacity median and spread (no reference); fade median and spread (with reference). Spread = 1.4826 MAD."""
    R = D.readings(cells); q = np.array([r['q_total'] for r in R]); f = np.array([1 - r['q_total'] / r['q_total_ref'] for r in R if r['diag'] > 1])
    mad = lambda x: 1.4826 * float(np.median(np.abs(x - np.median(x))))
    return dict(q_med=float(np.median(q)), q_sd=mad(q), f_med=float(np.median(f)), f_sd=mad(f))


def blend(P, rows, pri, fac, mode='ivw'):
    """pull partial readings' capacity toward the dev prior by inverse variance; full curves untouched"""
    q = P['q_total'].copy(); s = P['q_total_sig'] * fac; out = q.copy()
    for i, r in enumerate(rows):
        if r['depth'] is None and np.isfinite(r['Q1'][-1]): continue
        if r['ref']: m0 = r['q_ref'] * (1 - pri['f_med']); s0 = r['q_ref'] * pri['f_sd']
        else: m0, s0 = pri['q_med'], pri['q_sd']
        w = s0 ** 2 / (s0 ** 2 + s[i] ** 2); out[i] = w * q[i] + (1 - w) * m0
    return out


def table(rows, q):
    qt = np.array([r['q_total'] for r in rows]); e = 1000 * np.abs(q - qt); out = {}
    part = np.array([r['depth'] is not None for r in rows]); out['partial'] = float(e[part].mean()); out['full'] = float(e[~part].mean())
    for d in RT.DEPTHS:
        for ref in (True, False):
            m = np.array([r['depth'] == d and r['ref'] == ref for r in rows]); out[f'{d}{"r" if ref else "n"}'] = float(e[m].mean())
    return out


def partial_fac(P, rows):
    """widen-only factor for the capacity bar, fitted on the partial readings of these rows"""
    m = np.array([r['depth'] is not None for r in rows]); qt = np.array([r['q_total'] for r in rows])
    return widen_factor((P['q_total'] - qt)[m], P['q_total_sig'][m])


def probe():
    rows = RT.real_readings(D.DEV); cell = np.array([r['cell'] for r in rows]); P = {k: RT.run_real(RT.load_net(os.path.join(CK, v)), rows) for k, v in NETS.items()}
    P['ens4'] = ensemble([P[k] for k in NETS]); P['ens_n8_n10'] = ensemble([P['n8'], P['n10']])
    qt = np.array([r['q_total'] for r in rows]); res = {}
    med = prior_stats(D.DEV); res['dev_median_guess'] = table(rows, np.full(len(rows), med['q_med']))
    res['prior_alone'] = table(rows, blend(dict(q_total=np.zeros(len(rows)), q_total_sig=np.full(len(rows), 1e6)), rows, med, 1.0))
    for k, p in P.items():
        res[k] = table(rows, p['q_total'])
        fac = partial_fac(p, rows); res[k + '+blend(in-sample)'] = table(rows, blend(p, rows, med, fac)) | {'fac': fac}
        q = np.full(len(rows), np.nan)                    # leave one dev cell out: prior and factor from the other two
        for c in D.DEV:
            o = [x for x in D.DEV if x != c]; mo = cell != c; pri = prior_stats(o)
            f = partial_fac({kk: vv[mo] for kk, vv in p.items()}, [r for r, z in zip(rows, mo) if z])
            idx = np.nonzero(cell == c)[0]; q[idx] = blend({kk: vv[idx] for kk, vv in p.items()}, [rows[i] for i in idx], pri, f)
        res[k + '+blend(LOCO)'] = table(rows, q)
    res['prior'] = med
    for k, v in res.items(): print(f'{k:24s}', {kk: round(vv, 1) if kk != 'fac' else round(vv, 2) for kk, vv in v.items()} if k != 'prior' else v)
    json.dump(res, open(os.path.join(_P.WORK, 'nmc', 'blend_probe.json'), 'w'), indent=1)


def sim_objective(P, Y, sets):
    out = {}
    for (name, Q1, Q0), p in zip(sets, P):
        s = {}
        for k in RT.SK + (['fade'] if Q0 is not None else []):
            e = np.abs(p[k] - Y[k]); s[k] = float(1 - e.mean() / np.abs(Y[k] - np.median(Y[k])).mean())
        out[name] = float(np.mean([max(0.0, v) for v in s.values()]))
    out['objective'] = float(np.mean(list(out.values()))); return out


if __name__ == '__main__':
    probe() if sys.argv[1] == 'probe' else None


def blend_arrays(q, s, fac, qref, full, pri):
    """array form of blend() for the simulated set: qref NaN where there is no reference; full curves untouched"""
    has = np.isfinite(qref); m0 = np.where(has, qref * (1 - pri['f_med']), pri['q_med']); s0 = np.where(has, qref * pri['f_sd'], pri['q_sd'])
    w = s0 ** 2 / (s0 ** 2 + (s * fac) ** 2); return np.where(full, q, w * q + (1 - w) * m0)


def sim(variants):
    sets, Y = RT.val_sets('ocpd'); Pn = {}
    for k, v in NETS.items():
        net = RT.load_net(os.path.join(CK, v)); Pn[k] = [RT.predict(net, Q1, Q0) for _, Q1, Q0 in sets]; print('predicted', k, flush=True)
    Pn['ens4'] = [ensemble([Pn[k][i] for k in NETS]) for i in range(len(sets))]
    dev = prior_stats(D.DEV); simpri = dict(q_med=float(np.median(Y['q_total'])), q_sd=1.4826 * float(np.median(np.abs(Y['q_total'] - np.median(Y['q_total'])))))
    f = 1 - Y['q_total'] / sets[0][2][:, -1]; simpri.update(f_med=float(np.median(f)), f_sd=1.4826 * float(np.median(np.abs(f - np.median(f)))))
    rows = RT.real_readings(D.DEV); res = {}
    for k in ('n8', 'ens4'):
        res[k] = sim_objective(Pn[k], Y, sets)
        Pr = RT.run_real(RT.load_net(os.path.join(CK, NETS['n8'])), rows) if k == 'n8' else ensemble([RT.run_real(RT.load_net(os.path.join(CK, v)), rows) for v in NETS.values()])
        fac = partial_fac(Pr, rows)
        for pname, pri in (('devprior', dev), ('simprior', simpri)):
            P2 = []
            for (name, Q1, Q0), p in zip(sets, Pn[k]):
                p = dict(p); qref = Q0[:, -1] if Q0 is not None else np.full(len(Q1), np.nan); full = np.isfinite(Q1[:, -1])
                p['q_total'] = blend_arrays(p['q_total'], p['q_total_sig'], fac, qref, full, pri)
                if Q0 is not None: p['fade'] = np.where(full, p['fade'], 1 - p['q_total'] / qref)
                P2.append(p)
            res[f'{k}+blend({pname})'] = sim_objective(P2, Y, sets)
    res['sim_prior'] = simpri
    for k, v in res.items(): print(f'{k:24s}', {kk: round(vv, 4) for kk, vv in v.items()})
    json.dump(res, open(os.path.join(_P.WORK, 'nmc', 'blend_sim.json'), 'w'), indent=1)


if __name__ == '__main__' and sys.argv[1] == 'sim':
    sim(None)
