"""Ratchet: score a rung on the frozen set and the dev gate, compare with the incumbent, log to ratchet.json.

Frozen skill = fraction of the rule strategy's fuel gap to DP (matched battery energy) that the policy closes,
averaged over the 18 vehicle strata (0 = rule, 1 = DP). Kept if
  frozen skill > incumbent + 0.01  AND  dev gate: dev mean %DP >= incumbent - 0.2 pp and dev CS-holding rate >= incumbent - 3 pp.
usage: ratchet.py NAME CKPT_OR_- 'json params' 'note'
"""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import evaluate as E
import sets
from hp_env import ROOT
from hp_policy import MLP

RJ = f'{ROOT}/ratchet.json'
MARGIN = 0.01


def load():
    return json.load(open(RJ)) if os.path.exists(RJ) else []


def summarise(base, pol, setname):
    sk, per, cs = E.skill(base, pol)
    pct = [m['pct_dp'] for m in pol.values()]
    starts = [m['starts_per_km'] for m in pol.values()]
    chat = [m['chatter_per_km'] for m in pol.values()]
    return dict(set=setname, n=len(pol), skill=sk, skill_strata=per, pct_dp_mean=float(np.mean(pct)), pct_dp_median=float(np.median(pct)),
                cs_rate=cs, starts_per_km=float(np.mean(starts)), chatter_per_km=float(np.mean(chat)))


def baseline_as_pol(base, name):
    out = {}
    for cid, b in base.items():
        if '_fail' in b: continue
        nm = name if name in b else E.RULE[b['_meta']['arch']]
        if name == 'rule': nm = E.RULE[b['_meta']['arch']]
        out[cid] = b[nm]
    return out


def score(name, ckpt, params):
    res = {}
    net = MLP.load(ckpt) if ckpt and ckpt != '-' else None
    for setname in ('frozen', 'dev'):
        base = pickle.load(open(E.base_path(setname), 'rb'))
        if net is None:
            pol = baseline_as_pol(base, params['baseline'])
        else:
            def pf(c, p=params):
                q = dict(p); prev = q.pop('preview_s', 0); navmode = q.pop('nav', 'case')
                nav = c['nav'] if navmode == 'case' else bool(navmode)
                return E.policy_params(net, preview_s=prev, nav=nav, **q)
            pol = E.score_policy(setname, pf, base=base)
            pickle.dump(pol, open(f'{ROOT}/data/pol_{name}_{setname}.pkl', 'wb'))
        res[setname] = summarise(base, pol, setname)
    return res


def decide(entry, rungs):
    inc = None
    for r in rungs:
        if r.get('kept'):
            inc = r
    if inc is None:
        return True, 'first rung (incumbent by definition)'
    f, fi = entry['frozen']['skill'], inc['frozen']['skill']
    d, di = entry['dev'], inc['dev']
    ok_f = f > fi + MARGIN
    ok_d = d['pct_dp_mean'] >= di['pct_dp_mean'] - 0.002 and d['cs_rate'] >= di['cs_rate'] - 0.03
    why = f"frozen skill {f:.4f} vs incumbent {fi:.4f} ({'+' if f >= fi else ''}{f - fi:.4f}, margin {MARGIN}); " \
          f"dev %DP {100 * d['pct_dp_mean']:.2f} vs {100 * di['pct_dp_mean']:.2f}, dev CS {d['cs_rate']:.3f} vs {di['cs_rate']:.3f}"
    return bool(ok_f and ok_d), ('KEPT: ' if ok_f and ok_d else 'REJECTED: ') + why


if __name__ == '__main__':
    name, ckpt, params, note = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), sys.argv[4]
    rungs = load()
    t0 = time.time()
    entry = dict(rung=name, ckpt=ckpt, params=params, note=note, physics=sets.PHYS, **score(name, ckpt, params))
    kept, why = decide(entry, rungs)
    entry['kept'] = kept; entry['decision'] = why; entry['wall_s'] = time.time() - t0
    entry['time'] = time.strftime('%Y-%m-%d %H:%M')
    rungs = [r for r in rungs if r['rung'] != name] + [entry]
    json.dump(rungs, open(RJ, 'w'), indent=1)
    print(why)
    print(json.dumps({k: entry[k] for k in ('frozen', 'dev')}, indent=1))
