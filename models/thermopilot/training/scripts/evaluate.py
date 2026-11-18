"""Evaluate controllers on a frozen case set.

  evaluate.py SET freeze                     build the case list once -> data/cases/SET.pkl
  evaluate.py SET baselines [I0 I1]          fixed, setback, setback_std, precool, mpc_c, oracle -> results/SET/base/<id>.json
  evaluate.py SET policy WEIGHTS.npz NAME    ThermoPilot -> results/SET/pol_NAME.json
"""
import os, sys, json, time, pickle, traceback
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # training/: tp, _paths
from _paths import WORK as ROOT   # data/ and results/ live in WORK (was the sandbox's thermopilot/ folder)
import numpy as np
import tp
from tp import cases as C, features as F, policy as P
from tp.episode import Episode

BASES = ['fixed', 'setback', 'setback_std', 'precool', 'precool_p1', 'mpc_c', 'oracle']


def load_cases(name):
    f = os.path.join(ROOT, 'data', 'cases', f'{name}.pkl')
    cs = pickle.load(open(f, 'rb'))
    if name == 'exam_recs':
        # compute budget (decided before any exam result): the first 4 of the 6 drawn respondents per tariff slot
        # (cases come in blocks of 6 units x 2 months per slot)
        cs = [c for i, c in enumerate(cs) if i % 12 < 8]
    return cs


def episode(c):
    return Episode(c['spec_true'], c['spec_nom'], c['wsrc'], c['year'], c['month'], c['tariff'], c['pref'], seed=c['seed'])


def clean(m):
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer, bool)) else v) for k, v in m.items()
            if k != 'actions'}


def run_baselines(ep, which=BASES):
    res = {}
    for nm in which:
        if nm == 'mpc_c':
            st = ep.mpc(False).act
        elif nm == 'oracle':
            # perfect-foresight best: the oracle DP (with the two-pass monthly peak target when the tariff has demand
            # charges), or any rule if it did better over the month (foresight + exact re-simulation allow choosing)
            ep.oracle_floor()
            res['oracle_dp'] = clean(ep._oracle_best)
            best = min([res['oracle_dp']] + [res[k] for k in ('fixed', 'setback', 'precool') if k in res], key=lambda d: d['J'])
            res['oracle'] = dict(best)
            continue
        else:
            st = getattr(ep, nm)()
        m, _ = ep.run(st)
        res[nm] = clean(m)
    return res


def thermopilot(ep, W, c):
    if 'tau_a' in W:      # R8: residual on the card MPC (tp/card_mpc.py)
        return P.residual(ep, W, c['pref']['hi'])
    card = F.card_from_model(ep.model, c['pref']['hi'])
    return P.ThermoPilot(W, card, ep.ctx, ep.lam, ep.model.base_W, ep.fc_noisy)


def main():
    name, cmd = sys.argv[1], sys.argv[2]
    if cmd == 'freeze':
        os.makedirs(os.path.join(ROOT, 'data', 'cases'), exist_ok=True)
        f = os.path.join(ROOT, 'data', 'cases', f'{name}.pkl')
        if os.path.exists(f):
            print('exists', f); return
        cs = C.SETS[name]()
        pickle.dump(cs, open(f, 'wb'))
        print(name, len(cs), 'cases')
        return
    cs = load_cases(name)
    od = os.path.join(ROOT, 'results', name)
    os.makedirs(os.path.join(od, 'base'), exist_ok=True)
    if cmd == 'baselines':
        i0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        i1 = int(sys.argv[4]) if len(sys.argv) > 4 else len(cs)
        for c in cs[i0:i1]:
            f = os.path.join(od, 'base', f'{c["id"]}.json')
            if os.path.exists(f):
                continue
            t0 = time.time()
            try:
                ep = episode(c)
                res = run_baselines(ep)
                res['meta'] = dict(id=c['id'], group=c['group'], wsrc=c['wsrc'], month=c['month'],
                                   tariff=c['tariff'] if isinstance(c['tariff'], str) else 'synth',
                                   system=c['spec_nom']['system'], kind=ep.model.kind, res=c['res'],
                                   area=c['spec_nom']['floor_area_m2'], unit=c.get('unit', c['id']), UA=ep.model.UA,
                                   tau_h=ep.model.tau_h, pref=c['pref'])
            except Exception as e:
                res = dict(error=traceback.format_exc()[-800:])
                print(c['id'], 'ERROR', repr(e)[:200], flush=True)
            json.dump(res, open(f + '.tmp', 'w')); os.replace(f + '.tmp', f)
            if 'error' not in res:
                print(c['id'], ' '.join('%s %.1f' % (k, res[k]['J']) for k in BASES), '%.0fs' % (time.time() - t0), flush=True)
    elif cmd == 'reoracle':
        # upgrade baseline files made before the two-pass demand oracle (no change for cases without demand charges
        # except that 'oracle' becomes the best of the DP and the rules)
        for c in cs:
            f = os.path.join(od, 'base', f'{c["id"]}.json')
            if not os.path.exists(f):
                continue
            res = json.load(open(f))
            if 'error' in res or 'oracle_dp' in res:
                continue
            if res['fixed']['demand_cost'] > 0:
                ep = episode(c)
                fl, _ = ep.oracle_floor()
                m, _ = ep.run(ep.mpc(True, peak_floor=fl).act)
                res['oracle_dp'] = clean(m)
            else:
                res['oracle_dp'] = res['oracle']
            res['oracle'] = dict(min([res['oracle_dp']] + [res[k] for k in ('fixed', 'setback', 'precool')], key=lambda d: d['J']))
            json.dump(res, open(f + '.tmp', 'w')); os.replace(f + '.tmp', f)
            print(c['id'], 'oracle %.1f (dp %.1f)' % (res['oracle']['J'], res['oracle_dp']['J']), flush=True)
    elif cmd == 'anchor':
        # the card MPC alone (R8 anchor): evaluate.py SET anchor TAG [SHIELD] [I0 I1] -> results/SET/pol_TAG.json
        tag = sys.argv[3]
        sh = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        i0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        i1 = int(sys.argv[6]) if len(sys.argv) > 6 else len(cs)
        out = {}
        t0 = time.time()
        for c in cs[i0:i1]:
            try:
                ep = episode(c)
                m, _ = ep.run(P.card_mpc(ep, c['pref']['hi'], sh).act)
                out[c['id']] = clean(m)
            except Exception as e:
                out[c['id']] = dict(error=traceback.format_exc()[-600:])
                print(c['id'], 'ERROR', repr(e)[:200], flush=True)
        part = '' if (i0, i1) == (0, len(cs)) else f'.part{i0}'
        json.dump(out, open(os.path.join(od, f'pol_{tag}.json{part}'), 'w'))
        print(name, tag, len(out), 'cases %.0fs' % (time.time() - t0))
    elif cmd == 'policy':
        W = P.load(sys.argv[3]); tag = sys.argv[4]
        if len(sys.argv) > 5 and sys.argv[5] != '-':
            W['shield'] = np.array(float(sys.argv[5]))
        i0 = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        i1 = int(sys.argv[7]) if len(sys.argv) > 7 else len(cs)
        out = {}
        t0 = time.time()
        for c in cs[i0:i1]:
            try:
                ep = episode(c)
                pol = thermopilot(ep, W, c)
                m, _ = ep.run(pol.act)
                out[c['id']] = clean(m)
            except Exception as e:
                out[c['id']] = dict(error=repr(e)[:300])
        part = '' if (i0, i1) == (0, len(cs)) else f'.part{i0}'
        json.dump(out, open(os.path.join(od, f'pol_{tag}.json{part}'), 'w'))
        print(name, tag, len(out), 'cases %.0fs' % (time.time() - t0))
    elif cmd == 'merge':
        # evaluate.py SET merge TAG: join pol_TAG.json.part* -> pol_TAG.json
        import glob
        tag = sys.argv[3]
        out = {}
        for f in sorted(glob.glob(os.path.join(od, f'pol_{tag}.json.part*'))):
            out.update(json.load(open(f)))
        json.dump(out, open(os.path.join(od, f'pol_{tag}.json'), 'w'))
        print(name, tag, len(out), 'merged')


if __name__ == '__main__':
    main()
