"""Assemble web/firedetective.html from the template, the released weights, the exam results and the ratchet log."""
from common import *
import pickle
import feats
from exam import exam_list, EXD

T = open(f'{SRC}/template.html', encoding='utf-8').read()   # import: template and JS port kept in training/
JS = open(f'{SRC}/fdnet.js', encoding='utf-8').read().replace("if (typeof module !== 'undefined') module.exports = FDNet;", '')
REL = f'{ROOT}/release' if os.path.exists(f'{ROOT}/release/weights/calib.json') else MODEL   # import: a re-run of export.py, else the released weights
Wz = dict(np.load(f'{REL}/weights/firedetective_v1.npz'))
cal = json.load(open(f'{REL}/weights/calib.json'))
ids = sorted({k.split('.')[0] for k in Wz if '.' in k})


def r4(a):
    a = np.asarray(a, np.float64)
    return [float(f'{v:.5g}') for v in a.ravel()]


weights = []
for m in ids:
    d = {k[len(m) + 1:]: dict(s=list(v.shape), d=r4(v)) for k, v in Wz.items() if k.startswith(m + '.')}
    for k in ('seq_mu', 'seq_sd', 'st_mu', 'st_sd'):
        d[k] = dict(s=list(Wz[k].shape), d=r4(Wz[k]))
    weights.append(d)
EX = json.load(open(latest('out/exam.json')))
rows = {r['fid']: r for r in EX['fires']}
fires = []
for fid in exam_list():
    fn = f'{EXD}/{fid}.pkl'
    if not os.path.exists(fn) or fid not in rows:
        continue
    rec = pickle.load(open(fn, 'rb'))
    r = dict(rows[fid])
    n = int((rec['seq'][:, 0] > 0).sum())
    r['seq'] = [[float(f'{v:.4g}') for v in row] for row in rec['seq'][:n]]
    r['st'] = [float(f'{v:.4g}') for v in rec['static']]
    fires.append(r)
fires.sort(key=lambda r: (-(r['hs'] is not None and r['forest'] >= 0.5), r['name']))
R = json.load(open(latest('ratchet.json')))
rungs = [dict(rung=e['rung'], desc=e['desc'], frozen=e['frozen'], sim=e['sim_primary'],
              inc=e.get('incumbent', {}).get('sim_primary') if e.get('incumbent') else None,
              dev_hs=e['dev']['hs_rho_all'], dev_wind=e['dev']['wind_rho'], decision=e['decision']) for e in R['rungs']]
tw = json.load(open(latest('out/twin_v1.json')))['agg']
names = dict(final_logA='log10 burned area at the last step (km²)', frac_growth='share of steps with growth',
             logN='log10(1 + VIIRS pixels per step)', frp_day='log10(1 + mean FRP per pixel), day',
             frp_night='log10(1 + mean FRP per pixel), night', elong='shape elongation, log(√λ1/λ2)',
             mean_dlogA='mean growth per step, Δlog10 area')
twin = [[names[k], f"{tw[k]['sim_median']:.2f}", f"{tw[k]['real_median']:.2f}", f"{tw[k]['corr']:.2f}"] for k in names if k in tw]
extra = json.load(open(f'{ROOT}/out/page_text.json'))
DATA = dict(weights=weights, calib=cal, fires=fires, wind_cols=feats.WIND_COLS, hidden_col=feats.SEQ_NAMES.index('wind_hidden'),
            seq_names=feats.SEQ_NAMES, stat_names=feats.STAT_NAMES, rungs=rungs, twin=twin, **extra)
html = T.replace('__DATA__', json.dumps(DATA, separators=(',', ':'))).replace('__FDNET__', JS)
os.makedirs(f'{ROOT}/web', exist_ok=True)
open(f'{ROOT}/web/firedetective.html', 'w', encoding='utf-8').write(html)
print('page bytes', len(html), 'fires', len(fires))
