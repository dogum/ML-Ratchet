"""Score strategies on a fixed case set.
  baselines: every RoadSim strategy for the architecture + the DP curve (cached per set and physics version)
  policies : HybridPilot variants, graded against the cached DP curve at matched battery energy
usage: evaluate.py base SET           (build the baseline cache; HP_WORKERS processes, default 5)
"""
import sys, os, pickle, json, time, copy
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from hp_env import *
from hp_score import run_case, metrics, interp_dp, LHV, add_matched, grade
from roadsim.sim import get_vehicle
from hp_policy import HybridPilot, MLP
from roadsim import simulate
import sets

WORKERS = int(os.environ.get('HP_WORKERS', '5'))
RULE = {'hev_powersplit': 'powersplit_rule', 'hev_parallel_p2': 'parallel_rule', 'hev_series': 'series_thermostat',
        'phev_powersplit': 'phev_cdcs', 'phev_series': 'phev_cdcs'}


def stratum(meta):
    a = meta['arch']
    if meta.get('mild'): return 'mild P1 (CR-Z type)'
    if meta['cls'] == 'bus': return 'bus (' + ('series' if 'series' in a else 'P2') + ')'
    if meta['cls'] in ('truck_md', 'truck_hd'): return 'truck (' + ('PHEV series' if a.startswith('phev') else 'P2') + ')'
    return {'hev_powersplit': 'power-split HEV', 'hev_parallel_p2': 'P2 HEV', 'hev_series': 'series HEV',
            'phev_powersplit': 'power-split PHEV', 'phev_series': 'series PHEV (EREV)'}[a]


def base_path(setname):
    return f'{ROOT}/data/base_{setname}_{sets.PHYS}.pkl'


def _base_job(c):
    try:
        return sets.case_key(c), run_case(c)
    except Exception as ex:
        import traceback
        return sets.case_key(c), {'_fail': repr(ex) + traceback.format_exc()[-600:]}


def unique_keys(cases):
    """(15 Nov 2026 correction) results are dicts filled by imap_unordered: two cases with one key would overwrite each
    other (last finisher wins) and a policy could be graded against the other case's DP. Refuse that."""
    keys = [sets.case_key(c) for c in cases]
    if len(set(keys)) != len(keys):
        raise ValueError(f'duplicate case keys in the set ({len(keys) - len(set(keys))}); use sets.assign_keys')


def score_baselines(setname, limit=None, workers=WORKERS):
    cases = sets.get(setname)
    unique_keys(cases)
    fn = base_path(setname)
    res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
    todo = [c for c in cases[:limit] if sets.case_key(c) not in res]
    t0 = time.time()
    with Pool(workers) as pool:
        for i, (cid, r) in enumerate(pool.imap_unordered(_base_job, todo)):
            res[cid] = r
            if '_fail' in r:
                print('FAIL', cid, r['_fail'], flush=True)
            if i % 10 == 0:
                pickle.dump(res, open(fn, 'wb'))
                print(f'{i}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    pickle.dump(res, open(fn, 'wb'))
    return res


def regrade(base):
    """recompute pct_DP of cached results with the current definition (no re-simulation)."""
    from hp_score import grade
    for cid, b in base.items():
        if '_fail' in b: continue
        pts = b['_dp_points']
        for p in pts:
            p.setdefault('matched', abs(p['target'] - b['_meta']['soc0']) > 1e-6 and p['target'] not in
                         [round(b['_meta']['soc_lo'] + 0.05, 4)] and False)
        for nm, m in b.items():
            if not nm.startswith('_'):
                grade(m, pts, 42.9, b['_meta']['soc0'])
    return base


_NETS = {}


def _net(ckpt):
    if ckpt not in _NETS:
        _NETS[ckpt] = MLP.load(ckpt)
    return _NETS[ckpt]


def policy_params(ckpt, preview_s=0, nav=False, **kw):
    p = dict(net=_net(ckpt) if isinstance(ckpt, str) else ckpt, preview_s=preview_s, nav=nav)
    p.update(kw)
    return p


def case_params(c, ckpt, params):
    """HybridPilot params for one case: params may hold preview_s, nav ('case' = the case's own flag, or 0 / 1) and knobs."""
    q = dict(params); prev = q.pop('preview_s', 0); navmode = q.pop('nav', 'case')
    nav = c['nav'] if navmode == 'case' else bool(int(navmode))
    return policy_params(ckpt, preview_s=prev, nav=nav, **q)


def _policy_job(job):
    c, b, ckpt, params, match = job
    st = HybridPilot(**case_params(c, ckpt, params))
    try:
        o = simulate(c['spec'], c['trip'], strategy=st, init_soc=c['soc0'], init_temps=c['temps'], record=True, options=sets.PHYS)
    except Exception as ex:
        print('policy FAIL', sets.case_key(c), repr(ex), flush=True); return sets.case_key(c), None
    m = metrics(o, b['_meta']['arch'], c['soc0'])
    pts = [dict(p) for p in b['_dp_points']]
    if '_stage' in b and (match is True or (match == 'phev' and b['_meta']['phev'])):
        st_ = b['_stage']
        stage = (st_[0], st_[1].astype(np.float64), st_[2].astype(np.float64), st_[3], st_[4], st_[5],
                 None if st_[6] is None else st_[6].astype(np.float64))
        add_matched(c, get_vehicle(c['spec']), stage, pts, m['soc_end'])
    grade(m, pts, LHV.get(get_vehicle(c['spec']).fuel, 42.9e3) / 1e3, c['soc0'])
    return sets.case_key(c), m


def score_policy(setname, ckpt, params, base=None, cases=None, match=True, workers=WORKERS):
    """HybridPilot (checkpoint path + params, see case_params) on a set. Returns {case key: metrics incl. pct_dp}
    (key = case_id, or case_id#position for a case_id drawn twice; sets.assign_keys)."""
    cases = cases or sets.get(setname)
    unique_keys(cases)
    base = base or pickle.load(open(base_path(setname), 'rb'))
    jobs = [(c, base[sets.case_key(c)], ckpt, params, match) for c in cases
            if base.get(sets.case_key(c)) is not None and '_fail' not in base[sets.case_key(c)]]
    out = {}
    with Pool(workers) as pool:
        for cid, m in pool.imap_unordered(_policy_job, jobs, chunksize=2):
            if m is not None:
                out[cid] = m
    return out


def cs_ok(m, meta):
    """charge sustaining held. HEV: never leaves the SOC window by more than 0.02 and ends no farther from the window
    middle than it started, plus 15 % of the window. PHEV: never drops more than 0.03 below its CS level
    (soc_lo + 0.05, or the start SOC if it started lower)."""
    lo, hi = meta['soc_lo'], meta['soc_hi']
    if meta['phev']:
        return m['soc_min'] >= min(lo + 0.05, meta['soc0']) - 0.03
    mid = 0.5 * (lo + hi); w = hi - lo
    end_ok = abs(m['soc_end'] - mid) <= abs(meta['soc0'] - mid) + 0.15 * w
    return bool(end_ok and m['soc_min'] >= lo - 0.02 and m['soc_max'] <= hi + 0.02)


def table(base, pols=None, names=None):
    """aggregate per stratum and overall: mean pct_dp, CS-holding rate, starts/km, chatter/km."""
    rows = {}
    for cid, b in base.items():
        if '_fail' in b: continue
        meta = b['_meta']; sk = stratum(meta)
        strat = {k: v for k, v in b.items() if not k.startswith('_')}
        strat['rule'] = strat.get(RULE[meta['arch']])
        if pols:
            for pn, pr in pols.items():
                if cid in pr: strat[pn] = pr[cid]
        for nm, m in strat.items():
            if m is None or (names and nm not in names): continue
            for key in (sk, 'ALL'):
                d = rows.setdefault(key, {}).setdefault(nm, dict(pct=[], cs=[], starts=[], chat=[], ex=[]))
                d['pct'].append(m['pct_dp']); d['cs'].append(cs_ok(m, meta)); d['starts'].append(m['starts_per_km'])
                d['chat'].append(m['chatter_per_km']); d['ex'].append(m['extrap'])
    out = {}
    for key, D in rows.items():
        out[key] = {nm: dict(n=len(d['pct']), pct_dp_mean=float(np.mean(d['pct'])), pct_dp_median=float(np.median(d['pct'])),
                             pct_dp_p10=float(np.percentile(d['pct'], 10)), cs_rate=float(np.mean(d['cs'])),
                             starts_per_km=float(np.mean(d['starts'])), chatter_per_km=float(np.mean(d['chat'])),
                             extrap_frac=float(np.mean(d['ex']))) for nm, d in D.items()}
    return out


def skill(base, pol, ref='rule'):
    """fraction of the rule-based strategies' fuel gap to DP (matched battery energy) that `pol` closes:
    gap = 1 - min(pct_DP, 1) (no credit for beating the coarse DP); per-stratum mean gaps, strata weigh the same;
    skill = 1 - sum_strata gap_pol / sum_strata gap_rule. 0 = as good as the rules, 1 = DP."""
    per = {}
    for cid, b in base.items():
        if '_fail' in b or cid not in pol: continue
        meta = b['_meta']; sk = stratum(meta)
        r = b[ref] if ref in b else b[RULE[meta['arch']]]
        d = per.setdefault(sk, [0.0, 0.0, 0, 0])
        d[0] += 1 - min(pol[cid]['pct_dp'], 1.0); d[1] += 1 - min(r['pct_dp'], 1.0); d[2] += 1; d[3] += cs_ok(pol[cid], meta)
    gp = sum(v[0] / v[2] for v in per.values()); gr = sum(v[1] / v[2] for v in per.values())
    sks = {k: dict(gap_pol=v[0] / v[2], gap_rule=v[1] / v[2], cs=v[3] / v[2], n=v[2]) for k, v in per.items()}
    return float(1 - gp / max(gr, 1e-9)), sks, float(sum(v[3] for v in per.values()) / max(sum(v[2] for v in per.values()), 1))


if __name__ == '__main__':
    mode, setname = sys.argv[1], sys.argv[2]
    if mode == 'base':
        res = score_baselines(setname)
        print(json.dumps(table(res)['ALL'], indent=1))
