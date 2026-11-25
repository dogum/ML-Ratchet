"""Score every Pareto-front law of a search on the frozen sim val set and the dev gate, per powertrain group.

usage: python3 ladder.py <tag>        (reads work/search/<tag>_{ice,hev,bev}.json -> work/ladder_<tag>.json)
Per group and front member: sim val (the group's scored val trips; plug-ins excluded) MSLE, median |log error|,
R^2 of log e, MdAPE; dev = mean MdAPE over the group's D16 sets (ice: VED ICE, EPA conv x5, Mersin x2, ICCT;
hev: VED HEV, EPA hev FTP / HWY; bev: VED EV, EPA bev FTP / HWY). Also the knee (LawFinder's rule: on the front of
complexity vs sim val MSLE, among members with MSLE <= 1.5 x the lowest, the largest drop in log MSLE per added node)
and the road-load reference (E_phys) per group.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fscore as FS  # noqa: E402
import lawvars as LV  # noqa: E402
import lawmodel as LM  # noqa: E402
import score as S  # noqa: E402

ROOT = os.path.dirname(HERE)
GROUP_ARCH = {'ice': [0], 'hev': [1, 2, 3], 'bev': [6], 'phev': [4, 5]}
GROUP_SETS = {'ice': ['VED_ICE', 'EPA_conv_FTP', 'EPA_conv_HWY', 'EPA_conv_US06', 'EPA_conv_FTPcold', 'EPA_conv_SC03',
                      'Mersin_diesel', 'Mersin_cng', 'ICCT'],
              'hev': ['VED_HEV', 'EPA_hev_FTP', 'EPA_hev_HWY'], 'bev': ['VED_EV', 'EPA_bev_FTP', 'EPA_bev_HWY'],
              'phev': ['VED_PHEV']}


def views():
    X, y, meta, dumb = FS.simval()
    Xd, Fr = FS.dev()
    return LV.from_table(X), y, LV.from_table(Xd)


def group_scores(g, law, Vs, y, Vd, E_s=None, E_d=None):
    laws = {k: law for k in ('ice', 'hev', 'bev', 'phev')}
    m = np.isin(Vs['arch'], GROUP_ARCH[g])
    E = LM.energy(laws, Vs) if E_s is None else E_s
    le = np.log(E[m] / y[m])
    ly = np.log(y[m] / Vs['dist_km'][m])
    out = {'sim_msle': float(np.mean(le ** 2)), 'sim_medlog': float(np.median(np.abs(le))),
           'sim_r2_log': float(1 - np.sum(le ** 2) / np.sum((ly - ly.mean()) ** 2)),
           'sim_MdAPE': float(np.median(np.abs(np.exp(le) - 1))), 'sim_n': int(m.sum())}
    Ed = LM.energy(laws, Vd) if E_d is None else E_d
    d16, sd = FS.score_dev(Ed)
    out['dev_sets'] = {k: round(sd[k]['MdAPE'], 4) for k in GROUP_SETS[g] if k in sd and sd[k].get('n')}
    out['dev'] = float(np.mean(list(out['dev_sets'].values())))
    return out


def knee(members, ratio=1.5):
    rs = sorted(members, key=lambda r: (r['complexity'], r['sim_msle']))
    front, best = [], np.inf
    for r in rs:
        if r['sim_msle'] < best:
            front.append(r); best = r['sim_msle']
    lmin = min(r['sim_msle'] for r in front)
    cand = []
    for a, b in zip(front[:-1], front[1:]):
        if b['sim_msle'] <= ratio * lmin:
            cand.append((-np.log(b['sim_msle'] / a['sim_msle']) / (b['complexity'] - a['complexity']), b))
    return (max(cand, key=lambda t: t[0])[1] if cand else front[-1])['complexity']


def main(tag):
    Vs, y, Vd = views()
    out = {'tag': tag, 'groups': {}}
    for g in ('ice', 'hev', 'bev', 'phev'):
        p = os.path.join(ROOT, 'work', 'search', f'{tag}_{g}.json')
        if not os.path.exists(p):
            print('missing', p); continue
        cp = os.path.join(ROOT, 'work', f'ladder_{tag}_{g}.json')
        if os.path.exists(cp):
            out['groups'][g] = json.load(open(cp)); print('cached', cp); continue
        S_ = json.load(open(p))
        members = []
        for L in S_['front']:
            if L['complexity'] < 3:
                continue
            sc = group_scores(g, L, Vs, y, Vd)
            members.append({**L, **sc})
        ref = group_scores(g, None, Vs, y, Vd, E_s=Vs['E_phys'], E_d=Vd['E_phys'])
        k = knee(members)
        out['groups'][g] = {'members': members, 'knee': k, 'road_load': ref, 'search': {x: S_[x] for x in
                            ('n_train', 'seconds', 'cpu_s', 'n_eval', 'vars', 'maxsize', 'seed')}}
        json.dump(out['groups'][g], open(cp, 'w'), indent=1, default=float)
        print(f'== {g}: road-load sim medlog {ref["sim_medlog"]:.4f} msle {ref["sim_msle"]:.4f} dev {ref["dev"]:.4f}')
        for r in members:
            print(f"  c={r['complexity']:2d} sim msle {r['sim_msle']:.4f} medlog {r['sim_medlog']:.4f} "
                  f"MdAPE {r['sim_MdAPE']:.4f} R2log {r['sim_r2_log']:.3f} | dev {r['dev']:.4f} "
                  f"{'KNEE ' if r['complexity'] == k else ''}{r['expr'][:80]}")
    json.dump(out, open(os.path.join(ROOT, 'work', f'ladder_{tag}.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1])
