"""Dated supplementary re-score, 27 Nov 2026 (audit follow-up, owner-approved): the v1.0 exam buildings whose BDG2 meter
list has no electricity meter, re-scored with the released weights and `assume_electricity=True` (electricity added to
meters_present, as LoadCast.forecast does). Nothing is trained or chosen here; the exam is SEEN (v1.0 took it once,
17 Nov re-scored it). Plan in NOTEBOOK.md (27 Nov 2026).

    python rescore_noelec.py score     # re-run the affected exam building-meters -> WORK/exam/noelec/units_noelec.csv
    python rescore_noelec.py boot      # every row of rescore_seen.py again with those units replaced -> rows, verdicts

Inputs (from the 17 Nov integrity pass, unchanged): WORK/exam/exam_units_r3.csv, WORK/exam/baselines_r3_seen.csv,
the released widening factors (training/widen_r3.json). Outputs under WORK/exam/noelec/ (git-ignored); the summary is
copied to training/rescore_noelec_r3.json.
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, dataset as D, evaluate as E
import rescore_seen as R          # patches N.predict / N.load to the released numpy weights
import exam as X

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{C.ROOT}/exam/noelec'


def affected_units():
    ex = D.real_units('exam')
    return [u for u in ex if 'electricity' not in u['meters']], len(ex), len({u['bid'] for u in ex})


def score():
    os.makedirs(OUT, exist_ok=True)
    nets = R.np_nets(); t0 = time.time()
    W = json.load(open(f'{HERE}/widen_r3.json')); W = W.get('widen', W)
    aff, n_all, nb_all = affected_units()
    meta = C.bd.bdg2_meta(split='exam')
    nb_noel = int((~meta['meters'].map(lambda s: 'electricity' in C.meters_of(s))).sum())
    print(f'exam buildings {len(meta)}, without electricity {nb_noel}; scored building-meters {n_all} '
          f'({nb_all} buildings); affected {len(aff)} ({len({u["bid"] for u in aff})} buildings)', flush=True)
    for u in aff:
        u['meters'] = list(u['meters']) + ['electricity']      # assume_electricity=True; hvac class does not change
        assert C.hvac_class_from_meters(u['meters']) == u['hvac']
        D.prepare(u, ('full', 'w8', 'w2'))
    df = E.eval_units(aff, nets, modes=X.MODES, widen=W)
    df['building_id'] = [u['bid'] for u in aff]
    df.to_csv(f'{OUT}/units_noelec.csv', index=False)
    old = pd.read_csv(f'{C.ROOT}/exam/exam_units_r3.csv').set_index('id')
    # check the scorer: the same units without the change must reproduce the logged values
    chk = []
    for u in aff[:6]:
        u2 = dict(u, meters=[m for m in u['meters'] if m != 'electricity']); D.prepare(u2, ('full', 'w8', 'w2'))
        r = E.eval_units([u2], nets, modes=X.MODES, widen=W).iloc[0]
        chk.append(max(abs(r[f'{k}_cv'] - old.loc[u['id'], f'{k}_cv']) for k in ('cold', 'warm', 'warm_w8', 'warm_w2')))
    print('repro check on 6 units, max |CV diff| vs logged:', max(chk), flush=True)
    print(round(time.time() - t0), 's', flush=True)


KEYS = ('cold', 'warm', 'warm_w8', 'warm_w2')


def boot():
    new = pd.read_csv(f'{OUT}/units_noelec.csv').set_index('id')
    old = pd.read_csv(f'{C.ROOT}/exam/exam_units_r3.csv')
    ids = set(new.index)
    cols = [c for c in new.columns if c.split('_')[0] in ('cold', 'warm') and c in old.columns]
    pat = old.copy()
    m = pat.id.isin(ids)
    for c in cols:
        pat.loc[m, c] = new.loc[pat.loc[m, 'id'], c].values
    pat.to_csv(f'{OUT}/exam_units_r3_noelec.csv', index=False)
    # affected units only, old vs new
    a_old = old[m].set_index('id'); a_new = pat[m].set_index('id')
    aff = dict(n_units=int(m.sum()), n_buildings=int(old[m].building_id.nunique()), by_meter={})
    for mt, g in a_old.groupby('meter'):
        gn = a_new.loc[g.index]
        aff['by_meter'][mt] = {'n': len(g), **{k: dict(cv_old=float(g[f'{k}_cv'].median()), cv_new=float(gn[f'{k}_cv'].median()),
                                                       nmbe_old=float(g[f'{k}_nmbe'].median()), nmbe_new=float(gn[f'{k}_nmbe'].median()),
                                                       cov_old=float(g[f'{k}_cov'].mean()), cov_new=float(gn[f'{k}_cov'].mean()))
                                                for k in KEYS}}
    aff['all'] = {k: dict(cv_old=float(a_old[f'{k}_cv'].median()), cv_new=float(a_new[f'{k}_cv'].median())) for k in KEYS}
    # full exam: re-run the 17 Nov rows with the patched units (same seed, same baselines)
    res = {}
    for tag, src in (('old', f'{C.ROOT}/exam/exam_units_r3.csv'), ('new', f'{OUT}/exam_units_r3_noelec.csv')):
        res[tag] = R.boot(src, write=False, quiet=True)
    rows = []
    for ro, rn in zip(res['old']['rows'], res['new']['rows']):
        assert (ro['meter'], ro['row']) == (rn['meter'], rn['row'])
        rows.append(dict(meter=ro['meter'], row=ro['row'], n=rn['n'], model_old=ro['model'], model_new=rn['model'],
                         best_old=ro['best'], best_new=rn['best'], best_value=rn['baselines'][rn['best']],
                         diff_old=ro['diff'], diff_new=rn['diff'], ci_old=[ro['lo'], ro['hi']], ci_new=[rn['lo'], rn['hi']],
                         verdict_old=ro['verdict'], verdict_new=rn['verdict'], nmbe_old=ro['model_nmbe'],
                         nmbe_new=rn['model_nmbe'], g14_old=ro['g14_model'], g14_new=rn['g14_model'],
                         cov90_old=ro.get('cov90'), cov90_new=rn.get('cov90'),
                         paired_new=[rn['model_paired'], rn['best_paired']], n_all=rn['n_all']))
    pooled = {k: dict(old=float(old[f'{k}_cv'].median()), new=float(pat[f'{k}_cv'].median()), n=int(pat[f'{k}_cv'].notna().sum()))
              for k in KEYS}
    tally = {t: {v: sum(r[f'verdict_{t}'] == v for r in rows) for v in ('win', 'tie', 'loss')} for t in ('old', 'new')}
    out = dict(date='2026-11-27', note='Supplementary re-score of the SEEN v1.0 exam: exam building-meters whose BDG2 '
               'meter list has no electricity meter, re-run with assume_electricity=True; every other unit unchanged. '
               'Rows and intervals as rescore_seen.py (seed 20261117, 2,000 paired resamples, BuildingData baselines at p = 0).',
               affected=aff, pooled=pooled, tally=tally, rows=rows)
    C.jdump(out, f'{OUT}/rescore_noelec_r3.json')
    print(json.dumps(aff, indent=1)[:4000]); print(json.dumps(pooled)); print(tally)
    for r in rows:
        if r['model_old'] != r['model_new'] or r['verdict_old'] != r['verdict_new']:
            print(f"{r['meter']:12s} {r['row']:13s} {r['model_old']:6.1f} -> {r['model_new']:6.1f}  best {r['best_new']} "
                  f"{r['best_value']:.1f}  {r['verdict_old']} -> {r['verdict_new']}")


if __name__ == '__main__':
    {'score': score, 'boot': boot}[sys.argv[1]]()
