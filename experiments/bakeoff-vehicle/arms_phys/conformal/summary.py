"""Print every wrapper's dev coverage (raw -> in-sample -> grouped 10-fold cross-fit) from work/fit_report.json."""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.normpath(os.path.join(HERE, '..', '..', 'harness', 'registry'))
print(f"{'wrapper':24s} {'track':6s} {'D16-mean cov50/cov90 raw -> in -> cross-fit':48s} {'all units cov90 raw -> cv':26s} crps raw -> cv   hash")
for f in sorted(glob.glob(os.path.join(HERE, '*_conformal', 'work', 'fit_report.json'))):
    w = os.path.basename(os.path.dirname(os.path.dirname(f)))
    R = json.load(open(f))
    h = json.load(open(os.path.join(REG, w + '.json')))['hash'][:16] if os.path.exists(os.path.join(REG, w + '.json')) else 'NOT REGISTERED'
    for t, r in R['tracks'].items():
        m = [r[x]['D16_mean'] for x in ('raw', 'in_sample', 'cross_fit')]
        a = [r[x]['all_units'] for x in ('raw', 'cross_fit')] if 'all_units' in r['raw'] else None
        print(f"{w:24s} {t:6s} {m[0]['cov50']:.2f}/{m[0]['cov90']:.2f} -> {m[1]['cov50']:.2f}/{m[1]['cov90']:.2f} -> "
              f"{m[2]['cov50']:.2f}/{m[2]['cov90']:.2f}{'':20s} " + (f"{a[0]['cov90']:.3f} -> {a[1]['cov90']:.3f}{'':12s}" if a else ' ' * 26)
              + f" {m[0]['crps_log']:.4f} -> {m[2]['crps_log']:.4f}  {h}")
