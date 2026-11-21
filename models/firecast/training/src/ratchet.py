"""Ratchet bookkeeping. Rule: a candidate rung replaces the incumbent if
   (1) sim: mean AP (F mode) on the CURRENT frozen set > incumbent's + 0.01, and
   (2) dev gate: mean AP (F mode) on the 535 dev growth fire-days >= 0.95 x incumbent's.
Both are re-scored on every frozen set version. Usage:
   python3 ratchet.py decide <candidate> <frozen_ver> "<change>"     (appends to ratchet.json)
   python3 ratchet.py show"""
from common import *  # noqa

FN = f'{ROOT}/ratchet.json'
MARGIN = 0.01
DEV_TOL = 0.95


def load():   # a re-run's ratchet.json under WORK, else the shipped log (training/ratchet.json)
    fn = FN if os.path.exists(FN) else f'{TRAIN}/ratchet.json'
    return json.load(open(fn)) if os.path.exists(fn) else dict(rule=__doc__, incumbent=None, rungs=[])


def ev(r):
    return json.load(open(shipped(f'eval_{r}.json')))


def decide(cand, ver, change):
    R = load()
    E = ev(cand)
    key = f'sim_{ver}_F'
    row = dict(rung=cand, frozen=ver, change=change, sim_ap_F=E[key]['ap_mean'], sim_ap_L=E.get(f'sim_{ver}_L', {}).get('ap_mean'),
               sim_f1_F=E[key]['f1_mean'], dev_ap_F=E['dev_F']['ap_mean'], dev_ap_L=E['dev_L']['ap_mean'],
               dev_f1_F=E['dev_F']['f1_mean'], dev_f1_L=E['dev_L']['f1_mean'],
               dev_logA_F=E['dev_F']['log10_area_ratio_median'], dev_dice_total_F=E['dev_F']['dice_total_median'])
    inc = R['incumbent']
    if inc is None:
        row.update(kept=True, reason='first rung (base)')
    else:
        I = ev(inc)
        if key not in I:
            raise SystemExit(f'incumbent {inc} not scored on frozen set {ver}: run evaluate.py {inc} sim first')
        ds = row['sim_ap_F'] - I[key]['ap_mean']
        dr = row['dev_ap_F'] / I['dev_F']['ap_mean']
        row.update(incumbent=inc, d_sim=ds, dev_ratio=dr, incumbent_sim=I[key]['ap_mean'], incumbent_dev=I['dev_F']['ap_mean'])
        ok = ds > MARGIN and dr >= DEV_TOL
        row['kept'] = bool(ok)
        row['reason'] = (f'sim {row["sim_ap_F"]:.3f} vs {I[key]["ap_mean"]:.3f} ({ds:+.3f}, need > +{MARGIN}); '
                         f'dev {row["dev_ap_F"]:.3f} vs {I["dev_F"]["ap_mean"]:.3f} (x{dr:.3f}, need >= {DEV_TOL})')
    if row['kept']:
        R['incumbent'] = cand
    R['rungs'].append(row)
    jdump(R, FN)
    print(json.dumps(row, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'decide':
        decide(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(json.dumps(load(), indent=1))
