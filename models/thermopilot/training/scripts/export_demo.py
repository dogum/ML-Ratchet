"""Pack a few exam episodes for release/demo.py: the inputs ThermoPilot saw each hour (context arrays, building
card, forecasts, measured indoor temperature / HVAC heat / electricity from the closed-loop BuildingSim run) and the
setpoints it chose, plus the exam metrics of every controller for those episodes.

usage: export_demo.py WEIGHTS.npz OUT.npz
"""
import os, sys, json, pickle
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # training/: tp, _paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK as ROOT   # data/ and results/ live in WORK (was the sandbox's thermopilot/ folder)
import numpy as np
import tp
from tp import features as F, policy as P
from evaluate import load_cases, episode

PICK = [('exam_recs', 'heat_pump_air', 1), ('exam_recs', 'split_ac_furnace', 7), ('exam_bdg2', None, 7)]


def main():
    W = P.load(sys.argv[1]); out = {}
    meta = []
    for i, (name, sysname, mo) in enumerate(PICK):
        cs = load_cases(name)
        r = np.random.default_rng(7 + i)
        cand = [c for c in cs if c['month'] == mo and (sysname is None or c['spec_nom']['system'] == sysname)]
        c = cand[int(r.integers(len(cand)))]
        ep = episode(c)
        if 'tau_a' in W:   # R8: residual on the card MPC (tp/card_mpc.py); the card carries the house map
            card = F.full_card(ep.model, ep.ctx, c['pref']['hi'])
            pol = P.residual(ep, W, c['pref']['hi'])
        else:
            card = F.card_from_model(ep.model, c['pref']['hi'])
            pol = P.ThermoPilot(W, card, ep.ctx, ep.lam, ep.model.base_W, ep.fc_noisy)
        rec = dict(T=[], q=[], e=[], fT=[], fG=[], h=[], c=[])

        def st(k, obs, info):
            a = pol.act(k, obs, info)
            fT, fG = ep.fc_noisy.get(k)
            rec['T'].append(obs['T_zone']); rec['q'].append(info['q_hvac_kW'] * 1000 if info else 0.0)
            rec['e'].append(info['electricity_kW'] if info else 0.0); rec['fT'].append(fT); rec['fG'].append(fG)
            rec['h'].append(a['heat_sp']); rec['c'].append(a['cool_sp'])
            return a
        m, _ = ep.run(st)
        n = len(rec['T'])
        p = f'ep{i}_'
        for k in ('price', 'lo', 'hi', 'state', 'dperiod', 'drate', 'frate', 'hour', 'wd', 'ym', 'doy', 'drem', 'frem'):
            out[p + 'ctx_' + k] = np.asarray(ep.ctx[k])[:n + 24]
        for k in ('Tg', 'Qmax', 'Qh', 'Ph', 'eff', 'Qs', 'Pc'):
            out[p + 'card_' + k] = card[k]
        for k, v in rec.items():
            out[p + 'rec_' + k] = np.asarray(v)
        out[p + 'base_W'] = np.asarray(ep.model.base_W)[:n + 24]
        base = json.load(open(os.path.join(ROOT, 'results', name, 'base', f'{c["id"]}.json')))
        meta.append(dict(id=c['id'], group=name, system=c['spec_nom']['system'], month=mo, wsrc=c['wsrc'],
                         tariff=c['tariff'], lam=ep.lam, pref_lo=c['pref']['lo'], pref_hi=c['pref']['hi'],
                         card={k: card[k] for k in ('UA', 'C', 'area', 'kind', 'gas_heat', 'fan_cont', 'res', 'P_fan', 'backup',
                                                    'map', 'Qheat', 'eff0', 'gx', 'hp_speed', 'cool_speed') if k in card},
                         gas_price=ep.gas_price,
                         thermopilot=dict(bill=m['bill'], DH=m['DH'], J=m['J']),
                         baselines={k: dict(bill=v['bill'], DH=v['DH'], J=v['J']) for k, v in base.items()
                                    if isinstance(v, dict) and 'bill' in v}))
        fa = os.path.join(ROOT, 'results', name, 'pol_card4.json')
        if os.path.exists(fa) and c['id'] in json.load(open(fa)):
            v = json.load(open(fa))[c['id']]
            meta[-1]['baselines']['card_mpc'] = dict(bill=v['bill'], DH=v['DH'], J=v['J'])
        print(c['id'], 'TP bill %.1f DH %.1f' % (m['bill'], m['DH']))
    out['meta'] = np.array(json.dumps(meta))
    np.savez_compressed(sys.argv[2], **out)


if __name__ == '__main__':
    main()
