"""Compare twin prior-predictive rows of two physics versions on the same dev fire-days (partial runs allowed).
Usage: python3 twin_compare.py p0 p1 -> out/twin_compare_p0_p1.json"""
import json, sys
import numpy as np
from common import OUT, jdump


def load(v):
    R = {}
    for l in open(f'{OUT}/twin_{v}_rows.jsonl'):
        r = json.loads(l)
        if 'skip' not in r:
            R[r['i']] = r
    return R


def stats(rows, idx):
    lr, pit, byk = [], [], {}
    for i in idx:
        r = rows[i]
        v = np.array([m['new_feds'] for m in r['members'] if 'new_feds' in m])
        if not len(v):
            continue
        x = np.log10((v + 1) / (r['real_new'] + 1))
        lr += list(x)
        kb = '0' if r['k'] == 0 else '1-2' if r['k'] <= 2 else '3-5' if r['k'] <= 5 else '6-10' if r['k'] <= 10 else '>10'
        byk.setdefault(kb, []).extend(x)
        pit.append((np.sum(v < r['real_new']) + 0.5 * np.sum(v == r['real_new'])) / len(v))
    pit = np.array(pit)
    return dict(n_days=len(pit), log10_ratio_median=float(np.median(lr)), log10_ratio_q25=float(np.quantile(lr, .25)),
                log10_ratio_q75=float(np.quantile(lr, .75)), frac_real_below_all=float(np.mean(pit == 0)),
                frac_real_above_all=float(np.mean(pit == 1)),
                by_fire_day={k: dict(n=len(v), median=float(np.median(v))) for k, v in sorted(byk.items())})


if __name__ == '__main__':
    a, b = sys.argv[1], sys.argv[2]
    A, B = load(a), load(b)
    idx = sorted(set(A) & set(B))
    out = dict(note=f'same {len(idx)} dev fire-days (twin {b} partial run)', **{a: stats(A, idx), b: stats(B, idx)})
    jdump(out, f'{OUT}/twin_compare_{a}_{b}.json')
    print(json.dumps(out, indent=1))
