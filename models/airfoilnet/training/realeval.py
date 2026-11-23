"""Metrics on real UIUC polars, shared by the dev gate and the exam (the exam script imports these).

A method is a function  f(p, Re, alphas) -> dict(cl, cd, cm [, cl_sd, lncd_sd, cm_sd])  arrays over alphas,
NaN where it has no answer. Stall quantities come from the method's polar on a 0.25 deg grid.
Aggregation: per run, then per airfoil (mean over its runs), then mean over airfoils.
"""
import json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
GRID = np.arange(-8, 22.01, 0.25)
CL_DESIGN = 0.5


def upsweep(rows):
    a = np.array([r[0] for r in rows]); k = int(np.argmax(a))
    return np.array(rows[:k + 1], dtype=float)


def measured_stall(L):
    """L: up-sweep lift rows (alpha, cl, cm). Returns (alpha_s, clmax, has_stall)."""
    k = int(np.argmax(L[:, 1]))
    has = (k < len(L) - 1) and (L[k, 1] - L[k + 1:, 1].min() > 0.03)
    return float(L[k, 0]), float(L[k, 1]), bool(has)


def _first_rise_cross(al, cl, target):
    """alpha where cl first crosses target going up, on the pre-stall branch"""
    k = int(np.nanargmax(cl)) if np.any(np.isfinite(cl)) else -1
    for i in range(min(k, len(al) - 1)):
        if np.isfinite(cl[i]) and np.isfinite(cl[i + 1]) and cl[i] <= target < cl[i + 1]:
            return al[i] + (target - cl[i]) / (cl[i + 1] - cl[i]) * (al[i + 1] - al[i]), i
    return None, None


def eval_run(f, p, run):
    """returns dict of per-run errors and coverage counts"""
    out = {}
    if run['kind'] == 'lift':
        L = upsweep(run['data'])
        if len(L) < 4:
            return None
        a_s, clmax, has = measured_stall(L)
        pr = f(p, run['Re'], L[:, 0])
        e = pr['cl'] - L[:, 1]
        pre = L[:, 0] <= a_s; post = (L[:, 0] > a_s) & (L[:, 0] <= a_s + 10)
        out['cl_pre'] = np.nanmean(np.abs(e[pre])) if np.any(np.isfinite(e[pre])) else np.nan
        out['cl_post'] = np.nanmean(np.abs(e[post])) if (has and np.any(post) and np.any(np.isfinite(e[post]))) else np.nan
        out['answered'] = float(np.mean(np.isfinite(e)))
        cmv = np.array([np.nan if v is None else v for v in L[:, 2]], float)
        if np.any(np.isfinite(cmv[pre])) and 'cm' in pr and pr['cm'] is not None:
            ec = pr['cm'] - cmv
            out['cm_pre'] = np.nanmean(np.abs(ec[pre])) if np.any(np.isfinite(ec[pre])) else np.nan
            if 'cm_sd' in pr:
                ok = np.isfinite(ec) & pre
                out['cov_cm'] = (np.sum(np.abs(ec[ok]) <= pr['cm_sd'][ok]), int(ok.sum()))
        if 'cl_sd' in pr:
            ok = np.isfinite(e) & pre
            out['cov_cl'] = (np.sum(np.abs(e[ok]) <= pr['cl_sd'][ok]), int(ok.sum()))
        if has:
            g = GRID[(GRID >= L[:, 0].min() - 1) & (GRID <= L[:, 0].max() + 3)]
            pg = f(p, run['Re'], g)
            if np.any(np.isfinite(pg['cl'])):
                k = int(np.nanargmax(pg['cl']))
                out['clmax'] = abs(pg['cl'][k] - clmax)
                out['a_stall'] = abs(g[k] - a_s)
                out['clmax_signed'] = pg['cl'][k] - clmax
                out['a_stall_signed'] = g[k] - a_s
    elif run['kind'] == 'drag':
        D = np.array(run['data'], float)
        D = D[np.isfinite(D[:, 2]) & (D[:, 2] > 0)]
        if len(D) < 3:
            return None
        pr = f(p, run['Re'], D[:, 0])
        if pr.get('cd') is None:
            return None
        le = np.log(pr['cd']) - np.log(D[:, 2])
        out['lncd'] = np.nanmean(np.abs(le)) if np.any(np.isfinite(le)) else np.nan
        out['lncd_signed'] = np.nanmean(le) if np.any(np.isfinite(le)) else np.nan
        out['answered_d'] = float(np.mean(np.isfinite(le)))
        if 'lncd_sd' in pr:
            ok = np.isfinite(le)
            out['cov_lncd'] = (np.sum(np.abs(le[ok]) <= pr['lncd_sd'][ok]), int(ok.sum()))
        # Cd at the design lift coefficient
        o = np.argsort(D[:, 0]); Ds = D[o]
        a_m, i_m = _first_rise_cross(Ds[:, 0], Ds[:, 1], CL_DESIGN)
        if a_m is not None:
            cd_m = np.exp(np.interp(a_m, Ds[i_m:i_m + 2, 0], np.log(Ds[i_m:i_m + 2, 2])))
            g = GRID[(GRID >= Ds[0, 0] - 2) & (GRID <= Ds[-1, 0] + 2)]
            pg = f(p, run['Re'], g)
            a_p, i_p = _first_rise_cross(g, pg['cl'], CL_DESIGN)
            if a_p is not None and np.all(np.isfinite(pg['cd'][i_p:i_p + 2])):
                cd_p = np.exp(np.interp(a_p, g[i_p:i_p + 2], np.log(pg['cd'][i_p:i_p + 2])))
                out['cd_design'] = abs(np.log(cd_p / cd_m))
    return out


KEYS = ['cl_pre', 'cl_post', 'clmax', 'a_stall', 'lncd', 'cd_design', 'cm_pre']


def evaluate(f, units, airfoils, pmap, run_filter=None):
    """pmap: key 'aid|model' -> CST params. Returns summary dict and per-airfoil table."""
    per = {}
    cov = {'cl': [0, 0], 'lncd': [0, 0], 'cm': [0, 0]}
    for aid in airfoils:
        rs = []
        for run in units[aid]['runs']:
            if run_filter is not None and not run_filter(aid, run):
                continue
            r = eval_run(f, pmap[f"{aid}|{run['model']}"], run)
            if r is None:
                continue
            rs.append(r)
            for c in cov:
                if f'cov_{c}' in r:
                    cov[c][0] += r[f'cov_{c}'][0]; cov[c][1] += r[f'cov_{c}'][1]
        if rs:
            per[aid] = {k: (float(np.nanmean([r[k] for r in rs if k in r]))
                            if any(k in r and np.isfinite(r[k]) for r in rs) else np.nan)
                        for k in KEYS + ['clmax_signed', 'a_stall_signed', 'lncd_signed']}
            per[aid]['n_runs'] = len(rs)
    summ = {k: float(np.nanmean([v[k] for v in per.values()])) if any(np.isfinite(v[k]) for v in per.values())
            else np.nan for k in KEYS + ['clmax_signed', 'a_stall_signed', 'lncd_signed']}
    summ['n_airfoils'] = len(per)
    summ['coverage'] = {c: (v[0] / v[1] if v[1] else np.nan) for c, v in cov.items()}
    return summ, per
