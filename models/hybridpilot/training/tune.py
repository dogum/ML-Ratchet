"""Policy-search fine-tuning ("RL fine-tune", black-box): tune the policy head's knobs on the TRAINING-distribution tune
set (never the frozen set) to maximise mean %DP at matched battery energy minus a drivability penalty.
Knobs: th_on, th_off (hysteresis on P(on)), s_scale, s_bias, guard_k (SOC guard), min_on_s.
Method: (1+lambda) evolution strategy with step-size adaptation, fixed seed.
usage: tune.py CKPT OUT.json [preview_s] [n_iter]"""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import evaluate as E
import sets
from hp_policy import MLP

KNOBS = dict(s_scale=(1.0, 0.85, 1.2), s_bias=(0.0, -0.3, 0.3), Kp=(6.0, 0.0, 20.0), guard_k=(0.0, 0.0, 4.0),
             min_on_s=(4.0, 0.0, 12.0), min_off_s=(2.0, 0.0, 8.0), pon_w=(0.0, 0.0, 3.0))
FIXED = dict(onoff='ecms')
LAM_CHAT = 0.002       # objective penalty per (chatter event / km)


def objective(net, x, cases, base, preview):
    prm = {k: float(v) for k, v in zip(KNOBS, x)}
    pol = E.score_policy('tune', lambda c: E.policy_params(net, preview_s=preview, nav=c['nav'], **prm, **FIXED), base=base, cases=cases, match='phev')
    pct = np.mean([min(m['pct_dp'], 1.0) for m in pol.values()])
    chat = np.mean([m['chatter_per_km'] for m in pol.values()])
    cs = np.mean([E.cs_ok(m, base[cid]['_meta']) for cid, m in pol.items()])
    return pct - LAM_CHAT * chat - 0.05 * max(0.0, 0.95 - cs), dict(pct=pct, chat=chat, cs=cs, **prm)


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    preview = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    n_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    net = MLP.load(ckpt)
    cases = sets.get('tune')
    base = pickle.load(open(E.base_path('tune'), 'rb'))
    lo = np.array([v[1] for v in KNOBS.values()]); hi = np.array([v[2] for v in KNOBS.values()])
    x = np.array([v[0] for v in KNOBS.values()])
    rng = np.random.default_rng(5)
    f, info = objective(net, x, cases, base, preview)
    hist = [dict(it=0, f=f, **info)]
    print('start', f, info, flush=True)
    sig = 0.25
    for it in range(1, n_iter + 1):
        cands = [np.clip(x + sig * (hi - lo) * rng.normal(size=len(x)), lo, hi) for _ in range(3)]
        res = [objective(net, c, cases, base, preview) for c in cands]
        j = int(np.argmax([r[0] for r in res]))
        if res[j][0] > f:
            x, f, info = cands[j], res[j][0], res[j][1]; sig = min(sig * 1.3, 0.5)
        else:
            sig *= 0.7
        hist.append(dict(it=it, f=f, sig=sig, **info))
        print(it, round(f, 5), {k: round(v, 3) for k, v in info.items()}, 'sig', round(sig, 3), flush=True)
    best = {k: float(v) for k, v in zip(KNOBS, x)}
    json.dump(dict(best=best, f=f, info=info, hist=hist, preview=preview, ckpt=ckpt), open(out, 'w'), indent=1)
    print('best', best, f)


if __name__ == '__main__':
    main()
