"""Teacher bank for the residual rung R8: the residual student on the card MPC acts, the perfect-foresight DP
teacher labels every hour, and the anchor's (card MPC's) 36 action values are saved with the teacher's.

usage: bank_r8.py OUT_DIR SEED_LO SEED_HI SHIELD                 round 0: the untrained student (= the card MPC) acts
       bank_r8.py OUT_DIR SEED_LO SEED_HI SHIELD POLICY.npz BETA DAgger: the student acts, the teacher's action is
                                                                 executed instead with probability BETA
Each episode -> OUT_DIR/ep_<seed>.npz with X (298 features), Q (teacher values - min), QA (anchor values - min),
S (24 h price scale), A (executed action), A0 (anchor action), AT (teacher action). Resumable.
"""
import os, sys, time, json, traceback
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import tp
from tp import twins, features as F, policy as P
from tp.episode import Episode

core = P.core


def untrained(shield, nf=F.NF + 36):
    """Zero residual: logits = -QA / (S tau), i.e. exactly the card MPC (behind the shield)."""
    return dict(mu=np.zeros(nf), sd=np.ones(nf), W0=np.zeros((nf, 1)), b0=np.zeros(1), W1=np.zeros((1, 36)),
                b1=np.zeros(36), tau_a=np.array(0.1), shield=np.array(float(shield)))


def run_case(case, W, beta, rng):
    ep = Episode(case['spec_true'], case['spec_nom'], case['wsrc'], case['year'], case['month'], case['tariff'],
                 case['pref'], seed=case['seed'])
    floor = None
    if ep.has_demand():
        floor, _ = ep.oracle_floor(fracs=(0.85, 1.0))
    oracle = ep.mpc(oracle=True, peak_floor=floor)
    card = F.full_card(ep.model, ep.ctx, case['pref']['hi'])
    pil = core.ResidualPilot(W, card, ep.ctx, ep.lam, ep.model.base_W, ep.gas_price)
    L = {k: [] for k in ('X', 'Q', 'QA', 'S', 'A', 'A0', 'AT')}

    def strat(k, obs, info):
        if info is not None and k > 0:
            oracle.update_peak(k - 1, info['electricity_kW'])
        oracle.observe(k, obs, info)
        a_t, q = oracle.plan(k, obs['W_zone'], return_values=True)
        qv = info['q_hvac_kW'] * 1000.0 if info is not None else 0.0
        ev = info['electricity_kW'] if info is not None else 0.0
        To, G = ep.fc_noisy.get(k)
        sp = pil.step(k, obs['T_zone'], qv, ev, To, G)
        S = core.price_scale(ep.ctx, k, pil.feat.P)
        a = pil.last
        h, c = sp['heat_sp'], sp['cool_sp']
        if beta > 0 and rng.random() < beta:
            a = a_t
            h, c = P.action_sp(ep.ctx, k, a_t)
        L['X'].append(pil.x); L['Q'].append(q - q.min()); L['QA'].append(pil.qa * S); L['S'].append(S)
        L['A'].append(a); L['A0'].append(pil.last_anchor); L['AT'].append(a_t)
        return dict(heat_sp=h, cool_sp=c)

    m, out = ep.run(strat)
    m.pop('actions', None)
    d = dict(X=np.array(L['X'], np.float16), Q=np.array(L['Q'], np.float32), QA=np.array(L['QA'], np.float32),
             S=np.array(L['S'], np.float32), A=np.array(L['A'], np.int16), A0=np.array(L['A0'], np.int16),
             AT=np.array(L['AT'], np.int16))
    return d, m, ep


def main():
    out_dir, lo, hi, shield = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    if len(sys.argv) > 6:
        W, beta = P.load(sys.argv[5]), float(sys.argv[6])
        W['shield'] = np.array(float(shield))
    else:
        W, beta = untrained(shield), 0.0
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(lo * 7 + 3)
    for seed in range(lo, hi):
        f = os.path.join(out_dir, f'ep_{seed}.npz')
        if os.path.exists(f) or os.path.exists(f + '.skip'):
            continue
        t0 = time.time()
        try:
            case = twins.train_case(seed)
            if case is None:
                raise ValueError('no case')
            d, m, ep = run_case(case, W, beta, rng)
        except Exception as e:
            open(f + '.skip', 'w').write(traceback.format_exc())
            print(seed, 'skip', repr(e)[:200], flush=True)
            continue
        meta = dict(seed=seed, wsrc=case['wsrc'], month=case['month'],
                    tariff=case['tariff'] if isinstance(case['tariff'], str) else 'synth',
                    system=case['spec_nom']['system'], use=case['spec_nom']['primary_use'],
                    area=case['spec_nom']['floor_area_m2'], kind=ep.model.kind, beta=beta,
                    agree_anchor_teacher=float(np.mean(d['A0'] == d['AT'])),
                    **{k: v for k, v in m.items() if isinstance(v, (int, float, bool))})
        np.savez_compressed(f + '.tmp.npz', meta=json.dumps(meta), **d)
        os.replace(f + '.tmp.npz', f)
        print(seed, meta['kind'], meta['wsrc'], meta['month'], meta['tariff'], 'J %.1f bill %.1f DH %.1f agree %.2f %.1fs' % (
            m['J'], m['bill'], m['DH'], meta['agree_anchor_teacher'], time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
