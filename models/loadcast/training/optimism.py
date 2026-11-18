"""Dev-gate optimism (17 Nov 2026). Two retrains of the R3 recipe on the same rebuilt bank and seeds:
  arm A (run oA): noise_bank.npz as built (dev residuals of 2016 and 2017)
  arm B (run oB): noise_bank_y16.npz (each dev series' 2017 residuals and gaps replaced by its 2016 ones, shifted 364 days)
Scores both, and the released R3 weights, on the dev gate (84 dev building-meters, 2016 -> 2017), on the frozen set
v1_val (rebuilt) and on the SEEN exam (cold and warm, a year of history). Nothing is chosen from it.

    python optimism.py oA oB          -> WORK/exam/optimism.json
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, dataset as D, evaluate as E, lcnet as N



def nets_for(run):
    if run == 'r3_released':
        import rescore_seen as R           # patches N.predict / N.load to the numpy release
        return R.np_nets(), R.np_predict
    return {m: N.load(f'{C.ROOT}/ckpt/{run}_{m}.pt')[0] for m in ('cold', 'warm')}, None


def main(*runs):
    import torch
    torch_predict = N.predict
    t0 = time.time()
    dev = D.real_units('dev')
    for u in dev: D.prepare(u, ('full',))
    ex = D.real_units('exam')
    for u in ex: D.prepare(u, ('full',))
    sets = [('dev', dev), ('exam', ex)]
    if os.environ.get('LC_FROZEN'):                       # LC_FROZEN=1: also the rebuilt frozen set v1_val
        fz = D.load_bank('v1_val', 'real', seed=12345)
        for u in fz: D.prepare(u)
        ref = E.ref_cvs(fz); sets.append(('frozen', fz))
    print('units ready', round(time.time() - t0), 's', flush=True)
    out = {}
    for run in list(runs) + ['r3_released']:
        nets, pred = nets_for(run)
        N.predict = pred or torch_predict
        r = {}
        for name, U in sets:
            df = E.eval_units(U, nets)
            if name == 'frozen':
                sk = E.skill(df, ref); r[name] = dict(objective=sk['objective'], cold_skill=sk['cold_skill'], warm_skill=sk['warm_skill'])
                continue
            r[name] = dict(score=E.dev_score(df), by_meter={m: dict(n=len(g), cold=float(g.cold_cv.median()), warm=float(g.warm_cv.median()))
                                                             for m, g in df.groupby('meter')})
            df.to_csv(f'{C.ROOT}/exam/optimism_{name}_{run}.csv', index=False)
        out[run] = r
        print(run, json.dumps(r), round(time.time() - t0), 's', flush=True)
    C.jdump(out, f'{C.ROOT}/exam/optimism_' + '_'.join(runs) + '.json')


if __name__ == '__main__':
    main(*sys.argv[1:])
