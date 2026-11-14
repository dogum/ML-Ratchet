"""Check: EPA test-car hybrids and plug-ins on their own test cycles (FTP-75, HWY, US06, SC03, 20 F cold FTP where
EPA tested them), specs by rule (checks/hybrid_specs.py), procedures as check_cycles (checks/epa_procedures.py).
Error = simulated / EPA - 1 on fuel economy (+ = the sim burns less). Plug-ins run their charge-sustaining tests: the
twin starts and is held at its charge-sustaining SOC.

Sets:
  dev    every hybrid / plug-in record of RoadData's DEV model families (frozen family split; carry-over copies
         dropped) that RoadSim models, plus the four check_cycles dev hybrids (Sonata Hybrid, RAV4 Hybrid, Escape HEV,
         Accord Hybrid, with their own specs). The calibration may fit on this set.
  devx   dev plus exam-3's dev_extra (dev-family hybrids found by the exam-3 rules: Sienna 2021+, Audi Q5 TFSIe /
         e-tron; v3 pass, 14 Nov 2026). The calibration may fit on this set.
  exam2  the frozen exam-2 list (checks/exam2_hybrids.json, sha256 checked). Final report only.
  exam3  the frozen exam-3 list (checks/exam3_hybrids.json, sha256 checked; the clean judge of the v3 pass). Final
         report only.

usage: python check_hybrids_epa.py --set dev|devx|exam2|exam3 [--options roadsim-physics-v2] [--workers 14] [--out FILE]
Output: hybrids_epa__<set>__<preset>.json
"""
import os, sys, json, time, argparse, copy
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(VEH, 'roadsim')); sys.path.insert(0, VEH)
import _opts
from epa_procedures import ftp, hot_cycle, sc03, test_spec, T_COLD

CHECK_DEV_GROUP = {'hev_parallel_p2': 'p2_tmed', 'hev_powersplit': None, 'hev_series': 'honda_series'}


def dev_records():
    import realdata as rd
    import hybrid_specs as hs
    e = rd.epa_testcars()
    h = e[e.architecture.isin(['hev', 'phev']) & (e.split == 'dev')].copy()
    h = h[h.FTP_mpg.notna() | h.HWY_mpg.notna() | h.US06_mpg.notna()]
    h['_m'] = h.model.astype(str).str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
    h = h.sort_values(['family', '_m', 'model_year']).drop_duplicates(
        ['make', '_m', 'test_vehicle_id', 'FTP_mpg', 'HWY_mpg', 'US06_mpg', 'SC03_mpg', 'FTP_cold_mpg'])
    S = json.load(open(os.path.join(VEH, 'data', 'proc', 'd3', 'specs.json')))
    d3 = json.load(open(os.path.join(VEH, 'data', 'proc', 'd3', 'splits.json')))['vehicles']
    d3_exam_rows = {(s.get('epa_match') or {}).get('epa_vehicle_id') for k, s in S.items() if d3.get(k, {}).get('split') == 'exam'}
    out, skipped = [], {}
    for i, r in h.iterrows():
        spec, kind, grp = hs.spec_of(r)
        if spec is None:
            skipped[grp] = skipped.get(grp, 0) + 1; continue
        out.append(dict(id=i, name=spec['name'], make=str(r.make), family=r.family, group=grp, kind=kind, spec=spec,
                        epa=hs.epa_results(r), source='EPA dev family', d3_exam_match=i in d3_exam_rows))
    V = json.load(open(os.path.join(HERE, 'epa_vehicles.json')))
    for v in V:
        a = v['spec']['architecture']
        if v['split'] != 'dev' or not a.startswith('hev'):
            continue
        g = CHECK_DEV_GROUP.get(a) or ('ford_ps' if 'Ford' in v['name'] else 'toyota_ps')
        epa = {k: v['epa'][k] for k in ('FTP', 'HWY', 'US06', 'SC03', 'COLD') if k in v['epa']}
        out.append(dict(id=v['spec']['id'], name=v['name'], make=v['name'].split()[1].upper(), family='check_cycles dev',
                        group=g, kind=a, spec=v['spec'], epa=epa, source='check_cycles dev vehicle', d3_exam_match=False))
    return out, skipped


def exam2_records():
    import realdata as rd
    import hybrid_specs as hs
    import exam2_hybrids as x2
    d = x2.load()
    e = rd.epa_testcars()
    out = []
    for r in d['epa']:
        spec, kind, grp = hs.spec_of(e.loc[r['epa_vehicle_id']])
        out.append(dict(id=r['epa_vehicle_id'], name=spec['name'], make=str(r['make']), family=r['family'], group=r['group'],
                        kind=kind, spec=spec, epa=r['epa'], source='exam2', family_seen_v2=r['family_seen_v2']))
    return out, {}, d['sha256']


def exam3_records(part='epa'):
    """exam-3 records (part 'epa') or its dev_extra (part 'dev_extra'); specs from the stored test-car fields with the
    supplementary hardware rules (hybrid_specs.spec_of(..., ext=True))."""
    import hybrid_specs as hs
    import exam3_hybrids as x3
    d = x3.load()
    out = []
    for r in d[part]:
        spec, kind, grp = hs.spec_of(x3.row_of(r), ext=True)
        rec = dict(id=r['epa_vehicle_id'], name=spec['name'], make=str(r['make']), family=r['family'], group=r['group'],
                   kind=kind, spec=spec, epa=r['epa'], source='exam3' if part == 'epa' else 'exam3 dev_extra',
                   family_seen_v2=r['family_seen_exam2'])
        rec['model_year'] = r['model_year']
        out.append(rec)
    return out, d['sha256']


def cs_setup(spec):
    """plug-ins: charge-sustaining operation at the CS level (soc_cs = window low + 0.05, the PHEV default)."""
    a = spec['architecture']
    s = copy.deepcopy(spec)
    b = s['battery']; lo = b['soc_window'][0]
    if a.startswith('phev'):
        soc = lo + 0.05
        s['control'] = {'strategy': 'phev_cdcs', 'params': {'soc_cs': soc}}
        return s, soc
    p = ((s.get('control') or {}).get('params') or {})
    if 'soc_target' in p:
        return s, float(p['soc_target'])
    return s, None


def one(job):
    rec, opt = job['rec'], job['opt']
    _opts.set(opt)
    t0 = time.time()
    s0, soc0 = cs_setup(rec['spec'])
    s = test_spec(s0)
    e = rec['epa']
    row = dict(id=rec['id'], name=rec['name'], make=rec['make'], family=rec['family'], group=rec['group'], kind=rec['kind'],
               architecture=s['architecture'], source=rec['source'], sim={}, epa={}, err_pct={}, diag={})
    for k in ('d3_exam_match', 'family_seen_v2', 'model_year'):
        if k in rec:
            row[k] = rec[k]
    try:
        if 'FTP' in e:
            r = ftp(s, soc0=soc0); row['sim']['FTP'] = r['FE']
        for cyc, key in (('hwfet', 'HWY'), ('us06', 'US06')):
            if key in e:
                r = hot_cycle(s, cyc, soc0=soc0); row['sim'][key] = r['FE']
                o = r['o']; row['diag'][key + '_eng_on'] = float(np.mean(o['eng_on']))
                row['diag'][key + '_starts'] = int(np.sum(np.diff((np.asarray(o['eng_on']) > 0.5).astype(int)) == 1))
        if 'SC03' in e:
            r = sc03(s, soc0=soc0); row['sim']['SC03'] = r['FE']; row['diag']['SC03_hvac_W'] = r['hvac_W']
        if 'COLD' in e:
            r = ftp(s, T=T_COLD, soc0=soc0); row['sim']['COLD'] = r['FE']
    except Exception as ex:
        row['error'] = f'{type(ex).__name__}: {ex}'[:300]
    for k, v in row['sim'].items():
        row['epa'][k] = e[k]
        row['err_pct'][k] = round(100 * (v / e[k] - 1), 2)
        row['sim'][k] = round(v, 2)
    row['wall_s'] = round(time.time() - t0, 1)
    return row


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    return dict(n=int(len(x)), median=round(float(np.median(x)), 2), p25=round(float(np.percentile(x, 25)), 2),
                p75=round(float(np.percentile(x, 75)), 2), mean_abs=round(float(np.mean(np.abs(x))), 2))


CYC = ('FTP', 'HWY', 'US06', 'SC03', 'COLD')


def summarize(rows):
    out = {'by_group': {}, 'by_group_make': {}}
    d = pd.DataFrame(rows)
    for g, gg in d.groupby('group'):
        o = dict(n=len(gg))
        for c in CYC:
            st = stats([r.get(c) for r in gg.err_pct])
            if st: o[c] = st
        st = stats([v for r in gg.err_pct for k, v in r.items() if k in ('FTP', 'HWY', 'US06')])
        if st: o['main'] = st
        out['by_group'][g] = o
        for m, mm in gg.groupby('make'):
            o2 = dict(n=len(mm))
            for c in CYC:
                st = stats([r.get(c) for r in mm.err_pct])
                if st: o2[c] = st['median']
            out['by_group_make'][f'{g}|{m}'] = o2
    return out


def table(summ):
    lines = [f"{'group':16s} {'n':>4s} " + ' '.join(f'{c:>14s}' for c in CYC)]
    for g, o in sorted(summ['by_group'].items()):
        lines.append(f"{g:16s} {o['n']:4d} " + ' '.join(
            f"{o[c]['median']:+6.1f} (n{o[c]['n']:3d})" if c in o else f"{'':>14s}" for c in CYC))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='dev', choices=['dev', 'devx', 'exam2', 'exam3'])
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--out', default=None)
    ap.add_argument('--groups', default=None)
    _opts.init()
    a = ap.parse_args()
    t0 = time.time()
    sha = None
    if a.set in ('dev', 'devx'):
        recs, skipped = dev_records()
        if a.set == 'devx':
            recs += exam3_records('dev_extra')[0]
    elif a.set == 'exam3':
        (recs, sha), skipped = exam3_records('epa'), {}
    else:
        recs, skipped, sha = exam2_records()
    if a.groups:
        recs = [r for r in recs if r['group'] in a.groups.split(',')]
    jobs = [dict(rec=r, opt=_opts.get()) for r in recs]
    jobs.sort(key=lambda j: -len(j['rec']['epa']))
    with Pool(min(a.workers, len(jobs))) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    errors = [r for r in rows if 'error' in r]
    summ = summarize([r for r in rows if r['err_pct']])
    import roadsim
    out = dict(check=f'EPA hybrids and plug-ins, set {a.set}', method=__doc__.strip(), set=a.set, exam_sha256=sha, exam2_sha256=sha if a.set == 'exam2' else None,
               options=_opts.get(), options_name=_opts.name(), physics=roadsim.PHYSICS_VERSION, skipped=skipped,
               n=len(rows), n_errors=len(errors), errors=errors[:20], summary=summ, vehicles=rows,
               wall_s=round(time.time() - t0, 1))
    fn = a.out or os.path.join(HERE, f'hybrids_epa__{a.set}__{_opts.name()}.json')
    json.dump(out, open(fn, 'w', encoding='utf-8', newline='\n'), indent=1, default=float)
    print(table(summ))
    print(f'{len(rows)} vehicles, {len(errors)} errors, {out["wall_s"]} s -> {fn}')


if __name__ == '__main__':
    main()
