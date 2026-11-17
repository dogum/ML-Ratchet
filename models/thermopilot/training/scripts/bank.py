"""Teacher bank: run training episodes, label every hour with the perfect-foresight DP teacher's action values.

usage: bank.py OUT_DIR SEED_LO SEED_HI [POLICY.npz BETA]
  Without a policy the teacher's own action is executed (behaviour cloning data). With a policy (DAgger) the
  student's action is executed with probability 1 - BETA, and the teacher labels the states the student visits.
Each episode -> OUT_DIR/ep_<seed>.npz (resumable: existing files are skipped).
"""
import os, sys, time, json, traceback
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import tp
from tp import twins, features as F, policy as P
from tp.episode import Episode


def run_case(case, Wp=None, beta=1.0, rng=None):
    ep = Episode(case['spec_true'], case['spec_nom'], case['wsrc'], case['year'], case['month'], case['tariff'],
                 case['pref'], seed=case['seed'])
    floor = None
    if ep.has_demand():
        floor, _ = ep.oracle_floor(fracs=(0.85, 1.0))
    oracle = ep.mpc(oracle=True, peak_floor=floor)
    card = F.card_from_model(ep.model, case['pref']['hi'])
    feat = F.Featurizer(card, ep.ctx, ep.lam, ep.model.base_W)
    X, Q, A, S = [], [], [], []
    pil = P.ThermoPilot(Wp, card, ep.ctx, ep.lam, ep.model.base_W, ep.fc_noisy) if Wp is not None else None

    def strat(k, obs, info):
        if info is not None and k > 0:
            oracle.update_peak(k - 1, info['electricity_kW'])
        oracle.observe(k, obs, info)
        a_or, q = oracle.plan(k, obs['W_zone'], return_values=True)
        qv = info['q_hvac_kW'] * 1000.0 if info is not None else 0.0
        ev = info['electricity_kW'] if info is not None else 0.0
        To, G = ep.fc_noisy.get(k)
        x = feat.step(k, obs['T_zone'], qv, ev, To, G)
        a = a_or
        h, c = P.action_sp(ep.ctx, k, a)
        if pil is not None:
            sp = pil.act(k, obs, info, x=x)        # keeps the student's own history in step
            if rng.random() > beta:
                a = pil.last; h, c = sp['heat_sp'], sp['cool_sp']
        X.append(x); Q.append(q - q.min()); A.append(a)
        idx = np.minimum(np.arange(k, k + 24), len(ep.ctx['price']) - 1)
        S.append(max(float(np.mean(np.abs(ep.ctx['price'][idx]))), 0.02) * feat.P)
        return dict(heat_sp=h, cool_sp=c)

    m, out = ep.run(strat)
    m.pop('actions', None)
    return dict(X=np.array(X, np.float16), Q=np.array(Q, np.float32), A=np.array(A, np.int16),
                S=np.array(S, np.float32)), m, ep


def main():
    out_dir, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    Wp, beta = None, 1.0
    if len(sys.argv) > 5:
        Wp = P.load(sys.argv[4]); beta = float(sys.argv[5])
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
            d, m, ep = run_case(case, Wp, beta, rng)
        except Exception as e:
            open(f + '.skip', 'w').write(traceback.format_exc())
            print(seed, 'skip', repr(e)[:200], flush=True)
            continue
        meta = dict(seed=seed, wsrc=case['wsrc'], month=case['month'], tariff=case['tariff'] if isinstance(case['tariff'], str) else 'synth',
                    system=case['spec_nom']['system'], use=case['spec_nom']['primary_use'], area=case['spec_nom']['floor_area_m2'],
                    kind=ep.model.kind, **{k: v for k, v in m.items() if isinstance(v, (int, float, bool))})
        np.savez_compressed(f + '.tmp.npz', meta=json.dumps(meta), **d)
        os.replace(f + '.tmp.npz', f)
        print(seed, meta['kind'], meta['wsrc'], meta['month'], meta['tariff'], 'J %.1f bill %.1f DH %.1f  %.1fs' % (
            m['J'], m['bill'], m['DH'], time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
