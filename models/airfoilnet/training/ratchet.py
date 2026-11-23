"""The ratchet: re-take every rung decision in order from work/runs/<rung>/result.json -> work/ratchet.json

Rule (fixed before the first rung was scored):
  keep a rung if  sim_skill(frozen set) > incumbent + 0.01   AND   dev error D <= 1.05 x incumbent's D
  sim_skill = mean over Cl, ln Cd, Cm of (1 - MAE / MAE of a constant guess) on the frozen simulated set
  D = mean( Cl_pre/0.10, lnCd/0.20, Clmax/0.20, alpha_stall/3 deg )  on the real dev airfoils
usage: python ratchet.py r1 r2 r3 ...  (order = the order the rungs were tried)
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
from _paths import WORK   # models/airfoilnet/work or $ML_RATCHET_WORK/airfoilnet (training/_paths.py)
MARGIN, GATE = 0.01, 1.05
NORM = dict(cl_pre=0.10, lncd=0.20, clmax=0.20, a_stall=3.0)


def dev_D(dev):
    return sum(dev[k] / v for k, v in NORM.items()) / len(NORM)


def main(order, notes=None, key='sim_skill', frozen='val1 (physics p1)', out='ratchet.json'):
    notes = notes or {}
    log = []; inc = None
    for rid in order:
        f = os.path.join(WORK, 'runs', rid, 'result.json')
        if not os.path.exists(f):
            continue
        r = json.load(open(f)); D = dev_D(r['dev'])
        row = dict(id=rid, change=notes.get(rid, ''), sim=round(r[key], 4), dev_D=round(D, 4),
                   frozen=frozen, dev={k: round(r['dev'][k], 4) for k in NORM},
                   dev_cov=r['dev']['coverage'], n_train_sweeps=r['n_train_sweeps'])
        if inc is None:
            row.update(decision='kept', why='First rung.'); inc = row
        else:
            gain = row['sim'] - inc['sim']; ok_gate = D <= GATE * inc['dev_D']
            if gain > MARGIN and ok_gate:
                row.update(decision='kept', why=f'sim +{gain:.3f} > {MARGIN}; dev D {D:.3f} <= {GATE} x {inc["dev_D"]:.3f}')
                inc = row
            else:
                why = []
                if gain <= MARGIN:
                    why.append(f'sim {gain:+.3f} not > +{MARGIN}')
                if not ok_gate:
                    why.append(f'dev D {D:.3f} > {GATE} x {inc["dev_D"]:.3f}')
                row.update(decision='rejected', why='; '.join(why))
        log.append(row)
        print(row['id'], row['decision'], row['sim'], row['dev_D'], row['why'])
    json.dump(dict(rule=__doc__, log=log, incumbent=inc['id'] if inc else None),
              open(os.path.join(WORK, out), 'w'), indent=1)
    return log, inc


if __name__ == '__main__':
    main(sys.argv[1:])
