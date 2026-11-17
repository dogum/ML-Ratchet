"""Export the page lab data: three exam-twin homes x five real 2017 climates x two weeks x five real tariffs, every
controller run in BuildingSim (traces + bills), plus the linear house maps for the in-browser live lab and a set of
feature vectors / logits to check the JS port of the network.

usage: lab_export.py WEIGHTS.npz OUT.json
"""
import os, sys, json, pickle, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # training/: tp, _paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK as ROOT   # data/ and results/ live in WORK (was the sandbox's thermopilot/ folder)
import numpy as np
import tp
from tp import features as F, policy as P, env, mpc as M
from tp.episode import Episode

CLIMATES = [('Hog', 'Minneapolis MN (6A)'), ('Bobcat', 'Ogden UT (5B)'), ('Bear', 'Berkeley CA (3C)'),
            ('Bull', 'Austin TX (2A)'), ('Fox', 'Tempe AZ (2B)')]
TARIFFS = [('pge_e_tou_c', 'PG&E E-TOU-C'), ('aps_r3', 'APS R-3 (demand charge)'), ('xcel_mn_a04_tod', 'Xcel MN A04 time-of-day'),
           ('austin_res_tou', 'Austin Energy TOU'), ('rtp_ercot', 'ERCOT real-time + 10 c')]
WEEKS = [('winter', 1, 9), ('summer', 7, 10)]
CTRL = ['fixed', 'setback', 'precool', 'mpc_c', 'oracle', 'thermopilot']


def pick_homes():
    cs = pickle.load(open(os.path.join(ROOT, 'data', 'cases', 'exam_recs.pkl'), 'rb'))
    want = [('heat_pump_air', 'Heat-pump house'), ('split_ac_furnace', 'Gas furnace + central AC house'),
            ('baseboard_resistance', 'Electric baseboard home')]
    out = []
    for sysname, label in want:
        for c in cs:
            if c['spec_nom']['system'] == sysname and c['month'] == 1:
                if sysname == 'split_ac_furnace' and not (c['spec_nom'].get('hvac') or {}).get('cooling'):
                    continue
                out.append((label, c)); break
    return out


def r1(x, d=1):
    return [round(float(v), d) for v in x]


def main():
    W = P.load(sys.argv[1]); out_path = sys.argv[2]
    homes = pick_homes()
    data = dict(climates=CLIMATES, tariffs=TARIFFS, weeks=[w[0] for w in WEEKS], controllers=CTRL, homes=[], runs={},
                live={}, check=[])
    for hi_, (label, c) in enumerate(homes):
        s = c['spec_nom']
        data['homes'].append(dict(label=label, id=c['unit'], system=s['system'], area=round(s['floor_area_m2']),
                                  year=s.get('year_built'), pref=c['pref'], state=c.get('state')))
        for ci, (site, _) in enumerate(CLIMATES):
            for wi, (wname, mo, day) in enumerate(WEEKS):
                for ti, (tar, _) in enumerate(TARIFFS):
                    key = f'{hi_}_{ci}_{wi}_{ti}'
                    t0 = time.time()
                    ep = Episode(c['spec_true'], c['spec_nom'], f'site:{site}', 2017, mo, tar, c['pref'], seed=c['seed'],
                                 day=day, days=7)
                    n = ep.n
                    run = dict(T_out=r1(ep.model.To[:n]), price=r1(ep.ctx['price'][:n], 3), lo=r1(ep.ctx['lo'][:n]),
                               hi=r1(ep.ctx['hi'][:n]), dperiod=[int(v) for v in ep.ctx['dperiod'][:n]],
                               drate=r1(ep.ctx['drate'][:n], 3), frate=r1(ep.ctx['frate'][:n], 3),
                               drem=r1(ep.ctx['drem'][:n], 0), frem=r1(ep.ctx['frem'][:n], 0), ctrl={})
                    for nm in CTRL:
                        if nm == 'mpc_c':
                            st = ep.mpc(False).act
                        elif nm == 'oracle':
                            st = ep.mpc(True).act
                        elif nm == 'thermopilot':
                            card = F.card_from_model(ep.model, c['pref']['hi'])
                            pol = P.ThermoPilot(W, card, ep.ctx, ep.lam, ep.model.base_W, ep.fc_noisy)
                            st = pol.act
                        else:
                            st = getattr(ep, nm)()
                        m, o = ep.run(st)
                        ts = o['ts']
                        hv = (ts['heating_electricity_kW'] + ts['cooling_electricity_kW'] + ts['fans_electricity_kW']).values
                        run['ctrl'][nm] = dict(T=r1(ts['T_zone'].values, 2), hvac_kW=r1(hv, 2),
                                               gas_kW=r1(ts['heating_gas_kW'].values, 2),
                                               bill=round(m['bill'], 2), energy=round(m['energy_cost'], 2),
                                               demand=round(m['demand_cost'], 2), gas=round(m['gas_cost'], 2),
                                               kWh=round(m['kWh'], 1), DH=round(m['DH'], 2), peak=round(m['peak_kW'], 2),
                                               ppd=round(m['ppd_home'], 1), J=round(m['J'], 2))
                    data['runs'][key] = run
                    print(key, ' '.join('%s %.1f/%.1f' % (k, v['bill'], v['DH']) for k, v in run['ctrl'].items()),
                          '%.0fs' % (time.time() - t0), flush=True)
                # live-lab linear model (nominal house) for this home/climate/week, built once (tariff-independent)
                ep = Episode(c['spec_true'], c['spec_nom'], f'site:{site}', 2017, mo, TARIFFS[0][0], c['pref'],
                             seed=c['seed'], day=day, days=7)
                Mo = ep.model
                n = ep.n + 24
                idx = np.arange(n)
                co = Mo.coeffs(idx, Mo.To[idx], Mo.ghi[idx], 0.5 * (ep.ctx['lo'][np.minimum(idx, ep.n - 1)] + ep.ctx['hi'][np.minimum(idx, ep.n - 1)]))
                card = F.card_from_model(Mo, c['pref']['hi'])
                cal = env.local_calendar(Mo.sim.times[:n], ep.tz)
                data['live'][f'{hi_}_{ci}_{wi}'] = dict(
                    A=r1(co['A'], 4), At=r1(co['At'], 5), Aq=round(float(np.mean(co['Aq'])), 8), C=r1(co['C'], 4),
                    Ct=r1(co['Ct'], 6), Cq=round(float(np.mean(co['Cq'])), 9), To=r1(Mo.To[idx], 2), G=r1(Mo.ghi[idx], 0),
                    base=r1(Mo.base_W[idx], 0), state=[int(v) for v in ep.ctx['state'][np.minimum(idx, ep.n - 1)]],
                    hour=[int(v) for v in cal['hour']], wd=[int(v) for v in cal['wd']], month=[int(v) for v in cal['month']],
                    Tm0=round(float(np.mean(ep.ctx['lo'][:3]) + 1.0), 2),
                    card=dict(UA=card['UA'], C=card['C'], area=card['area'], kind=card['kind'], gas_heat=card['gas_heat'],
                              fan_cont=card['fan_cont'], res=card['res'], P_fan=card['P_fan'], backup=card['backup'],
                              Tg=r1(card['Tg'], 0), Qmax=r1(card['Qmax'], 0), Qh=r1(card['Qh'], 0), Ph=r1(card['Ph'], 0),
                              eff=r1(card['eff'], 3), Qs=r1(card['Qs'], 0), Pc=r1(card['Pc'], 0)),
                    lam=ep.lam, gas_price=ep.gas_price)
    # JS-port check: features and logits on a few stored steps of the first runs
    for key in list(data['runs'])[:3]:
        hi_, ci, wi, ti = map(int, key.split('_'))
        label, c = homes[hi_]
        site = CLIMATES[ci][0]; wname, mo, day = WEEKS[wi]
        ep = Episode(c['spec_true'], c['spec_nom'], f'site:{site}', 2017, mo, TARIFFS[ti][0], c['pref'], seed=c['seed'],
                     day=day, days=7)
        card = F.card_from_model(ep.model, c['pref']['hi'])
        pol = P.ThermoPilot(W, card, ep.ctx, ep.lam, ep.model.base_W, ep.fc_noisy)
        rec = []

        def st(k, obs, info):
            a = pol.act(k, obs, info)
            if k % 20 == 5:
                rec.append(dict(k=k, x=[float(v) for v in pol.x], logits=[float(v) for v in P.mlp_forward(W, pol.x.astype(np.float64))]))
            return a
        ep.run(st)
        data['check'] += rec[:4]
    json.dump(data, open(out_path, 'w'), separators=(',', ':'))
    print('wrote', out_path, os.path.getsize(out_path) / 1e6, 'MB')


if __name__ == '__main__':
    main()
