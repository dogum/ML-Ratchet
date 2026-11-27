"""Build scenario banks (resumable; one npz + one json per case).

  python3 bank.py train 0 600        # training bank: synthetic seeds 0..599, 10 truth-scored plans each
  python3 bank.py val 0 160          # frozen simulated validation set (seeds 900000+i): baselines + r0
  python3 bank.py gate               # real dev gate: 12 seeded dev fires x 2 budgets (dev only)
  python3 bank.py onpol r2 onpol1 600 800   # on-policy round with model r2 -> data/onpol1_p1
  python3 bank.py pf val 0 40        # perfect-foresight CEM bound on val cases 0..39 (and 'pf gate')
  python3 bank.py trainvs 0 300      # 27 Nov 2026: as train, but the crew starts at a seeded local hour 06-18 h
  python3 bank.py valvs 0 160        #   (fp.scenarios.varied_start_hour); same seeds, landscapes and weather draws

Each case stores what the planner may see (detection mask, forecast arrival and flame length, line cost, budget,
clock) plus the truth no-action flame length (holding rule) and areas, so features can be recomputed and rungs
scored without re-running the forecast.
"""
import os
import sys
import json
import time
import traceback

import _paths  # noqa: E402  (repository paths: shared/fire on sys.path; outputs under WORK, see _paths.py)
import numpy as np

from fp.scenarios import synth_scenario, real_scenario, gate_ids, BUDGETS_EVAL, hidden_draw, varied_start_hour
from fp.planner import Context, BASELINES, sample_plan, candidate_set, plan_proxy, cem_search

ROOT = _paths.WORK + '/data'
from fp.env import PHYSICS as VERSION  # noqa: E402  physics/env version of the frozen sets (FIREPILOT_PHYSICS, default p1)


def case_dir(kind):
    d = os.path.join(ROOT, f'{kind}_{VERSION}')
    os.makedirs(d, exist_ok=True)
    return d


def save_case(d, name, S, ctx, rows, extra=None):
    nn = S.none()
    np.savez_compressed(os.path.join(d, name + '.npz'), burned0=S.burned0,
                        arr_f=np.nan_to_num(ctx.arr_f, nan=1e4).astype(np.float16),
                        fl_f=ctx.fl_f.astype(np.float16), fl_t=nn['fl'].astype(np.float16), cost=S.cost.astype(np.float16),
                        clock_t=S.clock[0].astype(np.float32), clock_w=S.clock[1].astype(np.float32))
    rec = dict(name=name, rate=S.rate, res=S.res, a0=S.a0(), area_none=nn['area'], hidden=S.hidden,
               meta={k: v for k, v in S.meta.items() if k not in ('obs',)}, start=str(S.start), lon=S.lon,
               head=ctx.head, scn_f=ctx._scn.tolist(), rows=rows, **(extra or {}))
    with open(os.path.join(d, name + '.json.tmp'), 'w') as f:
        json.dump(rec, f, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(os.path.join(d, name + '.json.tmp'), os.path.join(d, name + '.json'))


def score_rows(S, ctx, plans, tags):
    rows = []
    for p, tag in zip(plans, tags):
        f, pe = ctx.features(p)
        r = S.evaluate(ctx.fr, p)
        rows.append(dict(tag=tag, plan=p, f=f.tolist(), s_proxy=pe['s_proxy'], **r))
    return rows


def _synth(kind, seed):
    """train / val: the afternoon start of v1.0; trainvs / valvs: a varied crew start (27 Nov 2026)."""
    return synth_scenario(seed, start_hour=varied_start_hour(seed) if kind.endswith('vs') else 13.0)


def do_train(seed, kind='train'):
    d = case_dir(kind)
    name = f's{seed:06d}'
    if os.path.exists(os.path.join(d, name + '.json')):
        return
    S = _synth(kind, seed)
    if S.none()['area'] - S.a0() < 50:          # nothing to plan: the fire barely grows
        json.dump(dict(name=name, skip='no growth'), open(os.path.join(d, name + '.json'), 'w'))
        return
    ctx = Context.from_scenario(S)
    rs = np.random.default_rng(seed + 7)
    plans, tags = [], []
    for k in rs.choice(list(BASELINES), 2, replace=False):
        plans.append(dict(BASELINES[k])); tags.append('base:' + k)
    for _ in range(4):
        plans.append(sample_plan(rs)); tags.append('prior')
    cand = [sample_plan(rs) for _ in range(60)]
    sp = np.array([ctx.plan_eval(p)[0]['s_proxy'] for p in cand])
    for k in np.argsort(-sp)[:4]:
        plans.append(cand[k]); tags.append('proxytop')
    save_case(d, name, S, ctx, score_rows(S, ctx, plans, tags))


def do_onpol(seed, model, tag):
    """On-policy round (DAgger-like): the current value-net planner picks a plan on a NEW training scenario; that
    plan, 4 perturbations of it and 2 prior plans are scored in the truth world and added to the bank."""
    from fp.planner import ValueNet, plan_value, to_vec, from_vec
    d = case_dir(tag)
    name = f's{seed:06d}'
    if os.path.exists(os.path.join(d, name + '.json')):
        return
    S = synth_scenario(seed)
    if S.none()['area'] - S.a0() < 50:
        json.dump(dict(name=name, skip='no growth'), open(os.path.join(d, name + '.json'), 'w'))
        return
    ctx = Context.from_scenario(S)
    rs = np.random.default_rng(seed + 11)
    net = ValueNet(os.path.join(_paths.WORK, 'models', model + '.npz'))
    p, info = plan_value(ctx, net, rs)
    plans, tags = [p], ['policy']
    for _ in range(4):
        v = to_vec(p) + np.array([30.0, 25.0, 0.4, 0.4]) * rs.standard_normal(4)
        plans.append(from_vec(p['mode'] if rs.random() < 0.8 else 1 - p['mode'], v)); tags.append('policy_jitter')
    for _ in range(2):
        plans.append(sample_plan(rs)); tags.append('prior')
    save_case(d, name, S, ctx, score_rows(S, ctx, plans, tags), extra=dict(policy_pred=info['pred']))


def do_val(i, kind='val'):
    d = case_dir(kind)
    name = f'v{i:04d}'
    if os.path.exists(os.path.join(d, name + '.json')):
        return
    S = _synth(kind, 900000 + i)
    if S.none()['area'] - S.a0() < 50:
        json.dump(dict(name=name, skip='no growth'), open(os.path.join(d, name + '.json'), 'w'))
        return
    ctx = Context.from_scenario(S)
    plans = [dict(p) for p in BASELINES.values()]
    tags = ['base:' + k for k in BASELINES]
    pp, _ = plan_proxy(ctx, np.random.default_rng(i))
    plans.append(pp); tags.append('r0')
    save_case(d, name, S, ctx, score_rows(S, ctx, plans, tags))


def gate_cases():
    out = []
    rs = np.random.default_rng(20261120)
    for fid in gate_ids():
        for b in BUDGETS_EVAL:
            out.append((fid, b, hidden_draw(rs)))
    return out


def do_gate(k):
    d = case_dir('gate')
    fid, b, hid = gate_cases()[k]
    name = f'g{k:02d}_{fid}_{int(b)}'
    if os.path.exists(os.path.join(d, name + '.json')):
        return
    try:
        S = real_scenario(fid, b, hid)
    except ValueError as e:
        json.dump(dict(name=name, skip=str(e)), open(os.path.join(d, name + '.json'), 'w'))
        return
    ctx = Context.from_scenario(S)
    plans = [dict(p) for p in BASELINES.values()]
    tags = ['base:' + k for k in BASELINES]
    pp, _ = plan_proxy(ctx, np.random.default_rng(k))
    plans.append(pp); tags.append('r0')
    save_case(d, name, S, ctx, score_rows(S, ctx, plans, tags), extra=dict(fire_id=fid))


def rebuild(kind, rec):
    """Rebuild the Scenario of a stored case (deterministic from its seed / fire id)."""
    if kind in ('train', 'trainvs'):
        return _synth(kind, int(rec['name'][1:]))
    if kind in ('val', 'valvs'):
        return _synth(kind, 900000 + int(rec['name'][1:]))
    if kind == 'gate':
        k = int(rec['name'][1:3])
        fid, b, hid = gate_cases()[k]
        return real_scenario(fid, b, hid)
    raise ValueError(kind)


def do_pf(kind, name):
    d = case_dir(kind)
    fn = os.path.join(d, name + '.pf.json')
    if os.path.exists(fn):
        return
    rec = json.load(open(os.path.join(d, name + '.json')))
    if 'skip' in rec:
        return
    S = rebuild(kind, rec)
    ctx = Context.from_scenario(S)
    init = [r['plan'] for r in rec['rows']]
    t0 = time.time()
    pf, info = cem_search(S, ctx.fr, 'truth', np.random.default_rng(1), init=init)
    r = S.evaluate(ctx.fr, pf)
    json.dump(dict(plan=pf, n_sims=info['n_sims'], wall_s=time.time() - t0, **r), open(fn, 'w'))


if __name__ == '__main__':
    kind = sys.argv[1]
    if kind == 'gate':
        ks = range(len(gate_cases()))
        fn = do_gate
    elif kind == 'pf':
        sub = sys.argv[2]
        d = case_dir(sub)
        names = sorted(f[:-5] for f in os.listdir(d) if f.endswith('.json') and not f.endswith('.pf.json'))
        if len(sys.argv) > 4:
            names = names[int(sys.argv[3]):int(sys.argv[4])]
        ks = names
        fn = lambda n: do_pf(sub, n)  # noqa: E731
    elif kind == 'onpol':
        model, tag = sys.argv[2], sys.argv[3]
        ks = range(int(sys.argv[4]), int(sys.argv[5]))
        fn = lambda k: do_onpol(k, model, tag)  # noqa: E731
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        ks = range(a, b)
        fn = (lambda k: do_train(k, kind)) if kind.startswith('train') else (lambda k: do_val(k, kind))  # noqa: E731
    t0 = time.time()
    for k in ks:
        t = time.time()
        try:
            fn(k)
        except Exception:
            print('FAIL', k, traceback.format_exc()[-800:], flush=True)
        print(kind, k, '%.1fs' % (time.time() - t), 'total %.0fs' % (time.time() - t0), flush=True)
