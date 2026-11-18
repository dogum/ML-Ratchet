"""Dev twin test of what the network adds to the cheap MPC (integrity pass, 17 Nov 2026). Dev episodes only.

usage: twin_test.py WEIGHTS.npz [I0 I1]     -> results/dev/twin/<id>.npz (per hour), then
       twin_test.py report [clip4K]         -> results/dev/twin_report[_clip4K].json

The classical MPC with forecast (nominal twin, noisy forecast, observer: the "cheap MPC" a residual would sit on)
drives the house. Every hour, on the same state, the teacher (true house, true weather, true state; demand peak
target as in bank.py) and ThermoPilot are also asked. Setpoints are compared at the current band edge: a heating
setpoint below the band's bottom (or off) counts as the bottom, a cooling setpoint above the top (or off) as the top,
so only pre-heating / pre-cooling inside the band shows ('report clip4K': clip to 4 K outside the occupied band
instead, the plan's first definition; there the shield's 'off -> band edge' dominates). Reported: correlation of ThermoPilot - MPC with teacher - MPC (heating, cooling),
the teacher's regret ($, from its own action values) of the MPC's action and of ThermoPilot's (pre-shield) action,
and where the teacher disagrees with the MPC.
"""
import os, sys, json, glob, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK as ROOT
import numpy as np
from evaluate import load_cases, episode
from tp import features as F, policy as P

# TWIN_ANCHOR=card (rung R8): the card MPC is the anchor that drives the house and the residual pilot is asked
# instead of v1.0; per-hour files in results/dev/twin_<tag>/, report in results/dev/twin_report_<tag>.json
CARD = os.environ.get('TWIN_ANCHOR') == 'card'
TAG = os.environ.get('TWIN_TAG', 'r8')
OD = os.path.join(ROOT, 'results', 'dev', f'twin_{TAG}' if CARD else 'twin')


def run(c, W):
    ep = episode(c)
    floor = None
    if ep.has_demand():
        floor, _ = ep.oracle_floor(fracs=(0.85, 1.0))
    oracle = ep.mpc(oracle=True, peak_floor=floor)
    if CARD:
        mpc = P.card_mpc(ep, c['pref']['hi'])
        pil = P.residual(ep, W, c['pref']['hi'])
    else:
        mpc = ep.mpc(False)
        card = F.card_from_model(ep.model, c['pref']['hi'])
        pil = P.ThermoPilot(W, card, ep.ctx, ep.lam, ep.model.base_W, ep.fc_noisy)
    L = {k: [] for k in ('k', 'a_m', 'a_p', 'a_t', 'q', 'sp_m', 'sp_p', 'sp_t', 'Tz', 'To')}

    def strat(k, obs, info):
        if info is not None and k > 0:
            oracle.update_peak(k - 1, info['electricity_kW'])
        oracle.observe(k, obs, info)
        a_t, q = oracle.plan(k, obs['W_zone'], return_values=True)
        sp_m = mpc.act(k, obs, info)
        sp_p = pil.act(k, obs, info)
        h, cc = P.action_sp(ep.ctx, k, a_t)
        for key, v in (('k', k), ('a_m', mpc.last), ('a_p', pil.last), ('a_t', a_t), ('q', q - q.min()),
                       ('sp_m', (sp_m['heat_sp'], sp_m['cool_sp'])), ('sp_p', (sp_p['heat_sp'], sp_p['cool_sp'])),
                       ('sp_t', (h, cc)), ('Tz', obs['T_zone']), ('To', ep.model.To[k])):
            L[key].append(v)
        return sp_m

    m, _ = ep.run(strat)
    cx = ep.ctx
    n = len(L['k'])
    price = cx['price'][:n]
    tou = bool(np.ptp(price) > 1e-6)
    out = {k: np.array(v) for k, v in L.items()}
    out.update(lo=cx['lo'][:n], hi=cx['hi'][:n], state=cx['state'][:n], hour=cx['hour'][:n], price=price,
               drate=cx['drate'][:n], lo0=cx['pref_lo'], hi0=cx['pref_hi'])
    meta = dict(id=c['id'], month=c['month'], kind=ep.model.kind, res=c['res'], tou=tou, demand=ep.has_demand(),
                tariff=c['tariff'] if isinstance(c['tariff'], str) else 'synth', J_mpc=m['J'])
    return out, meta


def corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def report():
    rows = []
    for f in sorted(glob.glob(os.path.join(OD, '*.npz'))):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z['meta']))
        lo0, hi0 = float(z['lo0']), float(z['hi0'])
        lo, hi = z['lo'], z['hi']
        if BAND:   # off and the band edge count as the same (both keep the zone at the current band edge)
            clip = lambda sp: np.stack([np.maximum(sp[:, 0], lo), np.minimum(sp[:, 1], hi)], 1)
        else:
            clip = lambda sp: np.stack([np.clip(sp[:, 0], lo0 - 4, hi0 + 4), np.clip(sp[:, 1], lo0 - 4, hi0 + 4)], 1)
        sm, sp, st = clip(z['sp_m']), clip(z['sp_p']), clip(z['sp_t'])
        q = z['q']; n = len(q); ii = np.arange(n)
        st_ = z['state']
        rows.append(dict(meta=meta, dP=sp - sm, dT=st - sm, reg_m=q[ii, z['a_m']], reg_p=q[ii, z['a_p']],
                         dis=(z['a_t'] != z['a_m']), tpdiff=(z['a_p'] != z['a_m']), jm=np.full(n, meta['J_mpc']), hour=z['hour'], state=st_, peak=z['price'] >= z['price'].max() - 1e-9))
    cat = lambda key: np.concatenate([r[key] for r in rows])
    dP, dT = cat('dP'), cat('dT')
    rm, rp, dis, tpd, jm = cat('reg_m'), cat('reg_p'), cat('dis'), cat('tpdiff'), cat('jm')
    hour, state = cat('hour'), cat('state')
    ep_of = np.concatenate([[j] * len(r['dis']) for j, r in enumerate(rows)])
    meta = [r['meta'] for r in rows]
    month = np.array([meta[j]['month'] for j in ep_of]); kind = np.array([meta[j]['kind'] for j in ep_of])
    tou = np.array([meta[j]['tou'] for j in ep_of]); dem = np.array([meta[j]['demand'] for j in ep_of])
    res = np.array([meta[j]['res'] for j in ep_of])

    def block(sel):
        s = sel
        return dict(hours=int(s.sum()), teacher_disagrees=float(dis[s].mean()), tp_differs_from_mpc=float(tpd[s].mean()),
                    regret_mpc_rel=float(np.sum(rm[s] / jm[s])), regret_tp_rel=float(np.sum(rp[s] / jm[s])),
                    corr_heat=corr(dP[s, 0], dT[s, 0]), corr_cool=corr(dP[s, 1], dT[s, 1]),
                    corr_heat_where_teacher_differs=corr(dP[s & dis, 0], dT[s & dis, 0]),
                    corr_cool_where_teacher_differs=corr(dP[s & dis, 1], dT[s & dis, 1]),
                    mean_teacher_minus_mpc=[float(dT[s, 0].mean()), float(dT[s, 1].mean())],
                    mean_tp_minus_mpc=[float(dP[s, 0].mean()), float(dP[s, 1].mean())],
                    regret_mpc_per_ep=float(rm[s].sum() / len(rows)), regret_tp_per_ep=float(rp[s].sum() / len(rows)),
                    tp_moves_same_way=float(np.mean(np.sign(dP[s & dis]).ravel() == np.sign(dT[s & dis]).ravel()))
                    if (s & dis).any() else float('nan'))
    one = np.ones(len(dis), bool)
    R = dict(n_episodes=len(rows), all=block(one),
             by_month={str(m_): block(month == m_) for m_ in np.unique(month)},
             by_kind={str(k_): block(kind == k_) for k_ in np.unique(kind)},
             by_tariff={'tou_no_demand': block(tou & ~dem), 'demand': block(dem), 'flat': block(~tou & ~dem)},
             by_building={'home': block(res), 'commercial': block(~res)},
             by_hour={f'{h0:02d}-{h0 + 5:02d}': block((hour >= h0) & (hour < h0 + 6)) for h0 in (0, 6, 12, 18)},
             by_state={str(s_): block(state == s_) for s_ in np.unique(state)},
             by_episode={meta[j]['id']: dict(regret_mpc=float(rm[ep_of == j].sum()), regret_tp=float(rp[ep_of == j].sum()),
                                             disagree=float(dis[ep_of == j].mean())) for j in range(len(rows))})
    rep = ('twin_report_%s.json' % TAG) if CARD else 'twin_report%s.json' % ('' if BAND else '_clip4K')
    json.dump(R, open(os.path.join(ROOT, 'results', 'dev', rep), 'w'), indent=1)
    print(json.dumps(R['all'], indent=1))
    for k in ('by_month', 'by_kind', 'by_tariff', 'by_building', 'by_hour', 'by_state'):
        for g, b in R[k].items():
            print(k, g, json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in b.items()}))


BAND = True


def main():
    global BAND
    if sys.argv[1] == 'report':
        BAND = not (len(sys.argv) > 2 and sys.argv[2] == 'clip4K')
        return report()
    W = P.load(sys.argv[1])
    cs = load_cases('dev')
    i0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    i1 = int(sys.argv[3]) if len(sys.argv) > 3 else len(cs)
    os.makedirs(OD, exist_ok=True)
    for c in cs[i0:i1]:
        f = os.path.join(OD, f'{c["id"]}.npz')
        if os.path.exists(f):
            continue
        t0 = time.time()
        out, meta = run(c, W)
        np.savez_compressed(f + '.tmp.npz', meta=json.dumps(meta), **out)
        os.replace(f + '.tmp.npz', f)
        print(c['id'], meta['kind'], 'disagree %.2f' % np.mean(out['a_t'] != out['a_m']), '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
