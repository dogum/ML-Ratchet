"""Ratchet bookkeeping: re-score every trained rung on every frozen simulated set version (physics|theta) and re-take
the keep/reject decisions in order on the LATEST frozen version.
Rule: kept if frozen-set RPSS > incumbent + 0.01 AND dev gate (gate half of dev) escape AUC >= incumbent - 0.01 AND
size MAE log10 <= 1.05 x incumbent.
usage: ratchet.py <latest frozen version e.g. v1|v1>   (rungs and order from rungs.json)"""
import sys, json, os
from common import *
import train

MARGIN = 0.01


def main():
    latest = sys.argv[1]
    R = json.load(open(latest('rungs.json')))   # shipped training/rungs.json unless a re-run wrote WORK/rungs.json
    versions = sorted({r['frozen'] for r in R if 'frozen' in r} | {latest})
    sets = {}
    for v in versions:
        ph, th = v.split('|')
        try:
            sets[v] = train.sim_set(ph, th)
        except Exception as e:
            print('skip version', v, e)
    mat = {}
    for r in R:
        mp = f'{train.MOD}/{r["rung"]}.npz'
        if not os.path.exists(mp):
            continue
        W = dict(np.load(mp))
        mat[r['rung']] = {v: train.score_sim(W, S) for v, S in sets.items()}
        js = json.load(open(f'{train.MOD}/{r["rung"]}.json'))
        r['dev_gate'] = dict(esc_auc=js['dev_gate']['escape']['auc'], esc_ap=js['dev_gate']['escape']['ap'],
                             brier=js['dev_gate']['escape']['brier'], mae=js['dev_gate']['size']['mae_log10'],
                             rho=js['dev_gate']['size']['rho'], mean_p=js['dev_gate']['mean_p_escape'])
        r['sim_latest'] = mat[r['rung']][latest]
        r['sim_own'] = mat[r['rung']].get(r.get('frozen', latest))
    inc = None
    for r in R:
        if r['rung'] not in mat:
            continue
        if inc is None:
            r['kept'] = True; r['reason'] = 'base'; inc = r
            continue
        ds = r['sim_latest']['rpss'] - inc['sim_latest']['rpss']
        g_auc = r['dev_gate']['esc_auc'] >= inc['dev_gate']['esc_auc'] - 0.01
        g_mae = r['dev_gate']['mae'] <= 1.05 * inc['dev_gate']['mae']
        ok = ds > MARGIN and g_auc and g_mae
        r['kept'] = bool(ok)
        r['reason'] = (f'sim RPSS {ds:+.3f} vs {inc["rung"]}' + ('' if g_auc else '; gate AUC fails') +
                       ('' if g_mae else '; gate MAE fails') + ('' if ds > MARGIN else '; below margin'))
        if ok:
            inc = r
    json.dump(dict(latest_frozen=latest, margin=MARGIN, rungs=R, matrix=mat, incumbent=inc['rung']),
              open(f'{ROOT}/ratchet.json', 'w'), indent=1, default=float)
    for r in R:
        if r['rung'] in mat:
            print(f"{r['rung']:6s} {r.get('change','')[:50]:50s} rpss(latest) {r['sim_latest']['rpss']:.3f} "
                  f"escAUCsim {r['sim_latest']['esc_auc']:.3f} | gate AUC {r['dev_gate']['esc_auc']:.3f} "
                  f"AP {r['dev_gate']['esc_ap']:.3f} MAE {r['dev_gate']['mae']:.3f} rho {r['dev_gate']['rho']:.3f} "
                  f"p {r['dev_gate']['mean_p']:.3f} | {'KEPT' if r['kept'] else 'no'} {r['reason']}")
    print('incumbent', inc['rung'])


if __name__ == '__main__':
    main()
