"""The in-band pre-cool rule with its two settings tuned on dev (integrity pass, 17 Nov 2026).

  precool_tuned.py tune               grid lead 1-4 h x depth 1-3 K on the 30 dev episodes -> results/dev/precool_grid.json
  precool_tuned.py run SET            the dev-chosen setting on SET -> results/SET/precool_tuned.json

The rule is Episode.precool with the lead (hours before the weekday peak window) and the depth (K below the occupied
band's top) as parameters; lead 3 h, depth 2 K is the v1.0 rule exactly. The setting is chosen by dev skill (sum of
relative J savings vs fixed over the sum for perfect foresight), never by an exam set.
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK as ROOT
import numpy as np
from evaluate import load_cases, episode, clean
from ratchet import load_base

LEADS = (1, 2, 3, 4)
DEPTHS = (1.0, 2.0, 3.0)


def precool_rule(ep, lead, depth):
    c, P = ep.ctx, ep.peak_hours
    lo0, hi0 = ep.pref['lo'], ep.pref['hi']

    def f(k, obs, info):
        ph = P.get(c['month'][k], set())
        hs, cs = float(c['lo'][k]), float(c['hi'][k])
        if ph and c['wd'][k]:
            h = c['hour'][k]; p0 = min(ph)
            if h in [(p0 - j) % 24 for j in range(1, lead + 1)]:
                cs = max(lo0 + 0.5, hi0 - depth)
                hs = min(hs, cs - 0.5)
        return dict(heat_sp=hs, cool_sp=cs)
    return f


def skill(base, res, ids):
    Jf = np.array([base[i]['fixed']['J'] for i in ids]); Jo = np.array([base[i]['oracle']['J'] for i in ids])
    Jc = np.array([res[i]['J'] for i in ids])
    return float(np.sum((Jf - Jc) / Jf) / max(np.sum((Jf - Jo) / Jf), 1e-9))


def main():
    cmd = sys.argv[1]
    if cmd == 'tune':
        cs = load_cases('dev'); base = load_base('dev')
        out = {}
        t0 = time.time()
        for lead in LEADS:
            for depth in DEPTHS:
                res = {}
                for c in cs:
                    ep = episode(c)
                    m, _ = ep.run(precool_rule(ep, lead, depth))
                    res[c['id']] = clean(m)
                ids = [i for i in base if i in res]
                if lead == 3 and depth == 2.0:   # must equal the v1.0 rule
                    assert all(abs(res[i]['J'] - base[i]['precool']['J']) < 1e-9 for i in ids), 'not the v1.0 rule'
                out[f'{lead}h_{depth:g}K'] = dict(lead=lead, depth=depth, skill=skill(base, res, ids),
                                                  DH=float(np.mean([res[i]['DH'] for i in ids])), n=len(ids))
                print(lead, depth, json.dumps(out[f'{lead}h_{depth:g}K']), '%.0fs' % (time.time() - t0), flush=True)
        best = max(out, key=lambda k: out[k]['skill'])
        R = dict(grid=out, chosen=best, rule='max dev skill', v10_rule='3h_2K')
        json.dump(R, open(os.path.join(ROOT, 'results', 'dev', 'precool_grid.json'), 'w'), indent=1)
        print('chosen', best, out[best])
    elif cmd == 'run':
        name = sys.argv[2]
        G = json.load(open(os.path.join(ROOT, 'results', 'dev', 'precool_grid.json')))
        ch = G['grid'][G['chosen']]
        res = {}
        for c in load_cases(name):
            try:
                ep = episode(c)
                m, _ = ep.run(precool_rule(ep, ch['lead'], ch['depth']))
                res[c['id']] = clean(m)
            except Exception as e:
                res[c['id']] = dict(error=repr(e)[:300])
        json.dump(dict(setting=G['chosen'], results=res),
                  open(os.path.join(ROOT, 'results', name, 'precool_tuned.json'), 'w'))
        print(name, G['chosen'], len(res))


if __name__ == '__main__':
    main()
