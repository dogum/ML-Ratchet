"""Dev set: 20 confirmed planets and 20 false positives (clear signals, published radius ratio, impact parameter and a/R*),
drawn at random once. Their stars (every planet and TOI on them) are left out of the exam."""
import _paths as _P  # portable paths (see _paths.py)
import json, numpy as np
D = _P.WORK + '/data/'
M = json.load(open(D + 'real_meta.json')); rng = np.random.default_rng(2026)
pl = [i for i, m in enumerate(M) if m['kind'] == 'planet' and m['info']['snr'] >= 10 and np.isfinite([m['k'], m['b'], m['aR']]).all()]
fp = [i for i, m in enumerate(M) if m['kind'] == 'FP' and m['info']['snr'] >= 10]
dev = sorted(rng.choice(pl, 20, replace=False).tolist() + rng.choice(fp, 20, replace=False).tolist()); dev_tics = {M[i]['tic'] for i in dev}
exam = [i for i, m in enumerate(M) if m['tic'] not in dev_tics]
json.dump(dict(dev=dev, exam=exam, dev_tics=sorted(dev_tics)), open(D + 'split.json', 'w'))
print('dev', len(dev), 'exam readings', len(exam), {k: sum(M[i]['kind'] == k for i in exam) for k in ('planet', 'FP', 'FA')})
print('dev planets', [M[i]['name'] for i in dev if M[i]['kind'] == 'planet'])
