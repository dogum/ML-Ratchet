"""Score strategies on a fixed case set.
  baselines: every RoadSim strategy for the architecture + the DP curve (cached per set and physics version)
  policies : HybridPilot variants, graded against the cached DP curve at matched battery energy
usage: evaluate.py base SET           (build the baseline cache)
       evaluate.py pol SET NAME CKPT [preview_s] [nav: auto|0|1] [extra json params]
"""
import sys, os, pickle, json, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hp_env import *
from hp_score import run_case, metrics, interp_dp, LHV, add_matched, grade
from roadsim.sim import get_vehicle
from hp_policy import HybridPilot, MLP
from roadsim import simulate
import sets

RULE = {'hev_powersplit': 'powersplit_rule', 'hev_parallel_p2': 'parallel_rule', 'hev_series': 'series_thermostat',
        'phev_powersplit': 'phev_cdcs', 'phev_series': 'phev_cdcs'}


def stratum(meta):
    a = meta['arch']
    if meta['cls'] == 'bus': return 'bus (' + ('series' if 'series' in a else 'P2') + ')'
    if meta['cls'] in ('truck_md', 'truck_hd'): return 'truck (' + ('PHEV series' if a.startswith('phev') else 'P2') + ')'
    return {'hev_powersplit': 'power-split HEV', 'hev_parallel_p2': 'P2 HEV', 'hev_series': 'series HEV',
            'phev_powersplit': 'power-split PHEV', 'phev_series': 'series PHEV (EREV)'}[a]


def base_path(setname):
    return f'{ROOT}/data/base_{setname}_{sets.PHYS}.pkl'


def score_baselines(setname, limit=None):
    cases = sets.get(setname)
    fn = base_path(setname)
    res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
    t0 = time.time()
    for i, c in enumerate(cases[:limit]):
        if c['case_id'] in res:
            continue
        try:
            r = run_case(c)
        except Exception as ex:
            print('FAIL', c['case_id'], repr(ex), flush=True); r = {'_fail': repr(ex)}
        res[c['case_id']] = r
        if i % 10 == 0:
            pickle.dump(res, open(fn, 'wb'))
            print(f'{i}/{len(cases)} {time.time() - t0:.0f}s', flush=True)
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


def policy_params(ckpt, preview_s=0, nav=False, **kw):
    p = dict(net=MLP.load(ckpt) if isinstance(ckpt, str) else ckpt, preview_s=preview_s, nav=nav)
    p.update(kw)
    return p


def score_policy(setname, params_fn, base=None, cases=None, match=True):
    """params_fn(case) -> HybridPilot params. Returns {case_id: metrics incl. pct_dp}."""
    cases = cases or sets.get(setname)
    base = base or pickle.load(open(base_path(setname), 'rb'))
    out = {}
    for c in cases:
        b = base.get(c['case_id'])
        if b is None or '_fail' in b:
            continue
        st = HybridPilot(**params_fn(c))
        try:
            o = simulate(c['spec'], c['trip'], strategy=st, init_soc=c['soc0'], init_temps=c['temps'], record=True)
        except Exception as ex:
            print('policy FAIL', c['case_id'], repr(ex), flush=True); continue
        m = metrics(o, b['_meta']['arch'], c['soc0'])
        pts = [dict(p) for p in b['_dp_points']]
        if '_stage' in b and (match is True or (match == 'phev' and b['_meta']['phev'])):
            st_ = b['_stage']
            stage = (st_[0], st_[1].astype(np.float64), st_[2].astype(np.float64), st_[3], st_[4], st_[5])
            add_matched(c, get_vehicle(c['spec']), stage, pts, m['soc_end'])
        grade(m, pts, LHV.get(get_vehicle(c['spec']).fuel, 42.9e3) / 1e3, c['soc0'])
        out[c['case_id']] = m
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
