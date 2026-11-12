"""Exam negatives: 120 more Kepler dwarfs (same DR25 cut as the noise pool: logg > 4.25, no KOI/TCE, >= 16 quarters),
never in any other split and NOT screened (rotation, eclipses and systematics left in, so false alarms are counted
honestly), magnitude-matched to the exam giants. Fixed seed 2027. Adds 'exam_neg' to split.json."""
import _paths as _P  # portable paths (see _paths.py)
import json, csv, numpy as np
D = _P.WORK + '/data/'
sp = json.load(open(D + 'split.json')); Y = json.load(open(D + 'yu2018.json'))
used = set(k for v in sp.values() for k in v)
W = [r for r in csv.DictReader(open(D + 'dwarfs_dr25.csv')) if r['st_quarters'].count('1') >= 16 and int(r['kepid']) not in used and r['kepid'] not in Y]
wk = np.array([float(r['kepmag']) for r in W]); rng = np.random.default_rng(2027)
kp = np.array([Y[str(k)]['kp'] for k in sp['exam']]); kp = kp[np.isfinite(kp)]; taken = set(); out = []
for m in rng.choice(kp, 120, replace=False):
    cand = [j for j in np.argsort(np.abs(wk - m))[:40] if j not in taken]; j = int(rng.choice(cand[:10])); taken.add(j); out.append(int(W[j]['kepid']))
sp['exam_neg'] = out; json.dump(sp, open(D + 'split.json', 'w')); print(len(out), 'exam negatives; Kp median %.2f vs giants %.2f' % (np.median(wk[list(taken)]), np.median(kp)))
