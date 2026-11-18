"""Exam rows with 95 % bootstrap intervals on ThermoPilot minus the best baseline (integrity pass, 17 Nov 2026).

usage: intervals.py TAG [ANCHOR_TAG]
        reads results/<set>/base/*.json, pol_TAG.json and precool_tuned.json (if present) -> results/intervals_TAG.json
        R8: with ANCHOR_TAG, pol_ANCHOR_TAG.json (the card MPC alone) is one more baseline, 'card_mpc'

Units (homes / buildings) are resampled with their episodes, 2,000 resamples, seed 20261117. Rows per set:
  bill   saving vs fixed thermostat (1 - sum bill / sum fixed bill), higher is better
  skill  sum of relative J savings vs fixed / the same for perfect foresight, higher is better
  DH     kelvin-hours outside the band per episode, lower is better
Best baseline = the strongest baseline on the row by the point value (perfect foresight is a bound, not a baseline).
On bill rows a baseline counts only if its K h per episode <= fixed + 1 (the ratchet's comfort test); the others are
listed. Verdict: win if the interval is on the better side of zero, loss if on the worse side, tie if it spans zero.
Also ThermoPilot minus setback (occupant's schedule) on bill.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORK as ROOT
import numpy as np
from ratchet import load_base

SETS = ['exam_recs', 'exam_bdg2', 'exam_hue', 'exam_wpuq', 'dev']
BASES = ['fixed', 'setback', 'setback_std', 'precool', 'precool_p1', 'precool_tuned', 'mpc_c', 'card_mpc']
NB, SEED = 2000, 20261117


def stats(get, base, ids):
    """Per controller: bill saving, skill, DH for the episode list ids (with repeats)."""
    Bf = sum(base[i]['fixed']['bill'] for i in ids)
    Jf = np.array([base[i]['fixed']['J'] for i in ids]); Jo = np.array([base[i]['oracle']['J'] for i in ids])
    gap = max(float(np.sum((Jf - Jo) / Jf)), 1e-12)
    out = {}
    for c in BASES + ['thermopilot', 'oracle']:
        try:
            rows = [get(i, c) for i in ids]
        except KeyError:
            continue
        Jc = np.array([r['J'] for r in rows])
        out[c] = dict(bill=1 - sum(r['bill'] for r in rows) / Bf, skill=float(np.sum((Jf - Jc) / Jf) / gap),
                      DH=float(np.mean([r['DH'] for r in rows])))
    return out


def main():
    tag = sys.argv[1]
    atag = sys.argv[2] if len(sys.argv) > 2 else None
    R = {}
    for s in SETS:
        base = load_base(s)
        f = os.path.join(ROOT, 'results', s, f'pol_{tag}.json')
        if not base or not os.path.exists(f):
            continue
        pol = json.load(open(f))
        ft = os.path.join(ROOT, 'results', s, 'precool_tuned.json')
        pt = json.load(open(ft))['results'] if os.path.exists(ft) else {}
        fa = os.path.join(ROOT, 'results', s, f'pol_{atag}.json') if atag else None
        cm = json.load(open(fa)) if fa and os.path.exists(fa) else {}
        ids = sorted(i for i in base if i in pol and 'error' not in pol[i] and (not pt or i in pt)
                     and (not cm or (i in cm and 'error' not in cm[i])))

        def get(i, c):
            if c == 'thermopilot':
                return pol[i]
            if c == 'precool_tuned':
                return pt[i]
            if c == 'card_mpc':
                return cm[i]
            return base[i][c]

        units = sorted(set(base[i]['meta']['unit'] for i in ids))
        by_unit = {u: [i for i in ids if base[i]['meta']['unit'] == u] for u in units}
        pt0 = stats(get, base, ids)
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(NB):
            pick = rng.choice(len(units), len(units))
            boots.append(stats(get, base, [i for p in pick for i in by_unit[units[p]]]))
        dh_ok = pt0['fixed']['DH'] + 1.0
        rows = {}
        for m, better in (('bill', 'higher'), ('skill', 'higher'), ('DH', 'lower')):
            cand = [c for c in BASES if c in pt0 and (m != 'bill' or pt0[c]['DH'] <= dh_ok)]
            sgn = 1 if better == 'higher' else -1
            best = max(cand, key=lambda c: sgn * pt0[c][m])
            d = np.array([b['thermopilot'][m] - b[best][m] for b in boots])
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff = pt0['thermopilot'][m] - pt0[best][m]
            good = (lo > 0) if better == 'higher' else (hi < 0)
            bad = (hi < 0) if better == 'higher' else (lo > 0)
            rows[m] = dict(better=better, best=best, model=pt0['thermopilot'][m], best_value=pt0[best][m], diff=diff,
                           ci95=[float(lo), float(hi)], verdict='win' if good else 'loss' if bad else 'tie',
                           excluded_for_comfort=[c for c in BASES if c in pt0 and c not in cand])
        d = np.array([b['thermopilot']['bill'] - b['setback']['bill'] for b in boots])
        rows['bill_vs_setback'] = dict(diff=pt0['thermopilot']['bill'] - pt0['setback']['bill'],
                                       ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        if 'card_mpc' in pt0:
            for m in ('bill', 'skill', 'DH'):
                d = np.array([b['thermopilot'][m] - b['card_mpc'][m] for b in boots])
                rows[f'{m}_vs_card_mpc'] = dict(diff=pt0['thermopilot'][m] - pt0['card_mpc'][m],
                                                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
        ci = {c: {m: [float(np.percentile([b[c][m] for b in boots], q)) for q in (2.5, 97.5)] for m in ('bill', 'skill', 'DH')}
              for c in pt0}
        R[s] = dict(n_episodes=len(ids), n_units=len(units), controllers=pt0, controllers_ci95=ci, rows=rows,
                    precool_tuned_setting=json.load(open(ft))['setting'] if os.path.exists(ft) else None)
        print(s, len(ids), 'episodes', len(units), 'units')
        for m, r in rows.items():
            print('  %-16s %s' % (m, json.dumps(r)))
    json.dump(R, open(os.path.join(ROOT, 'results', f'intervals_{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
