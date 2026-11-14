"""Exam-3 for hybrids and plug-ins: EPA test-car records no RoadSim check had simulated before 14 Nov 2026, frozen
before any v3 fitting (the RoadSim v3 pass). Exam-2, the EPA held-out vehicles, the D3 exam and the VED exam have all
been seen; this is the clean judge of v3.

Writes exam3_hybrids.json: the list AND each record's test-car fields (the MY2000-2007 rows are not in any tracked
table), with a sha256 over its canonical content (sorted keys, no whitespace, UTF-8, without the sha256 field).
Written once; `check` recomputes the list and the hash and compares.

Sources (gasoline-group hybrids and plug-ins of held-out model families, family split of data/proc/splits.json):
(a) MY2000-2007 test cars (checks/epa_pre2008.py; EPA files RoadData's table does not hold: no check ever ran on
    them). Hybrid / plug-in by RoadData's model-name rule.
(b) MY2008-2026 test cars that RoadData labels conventional ('conv': no hybrid word in the model name) but that are
    hybrids or plug-ins: a name token (FHEV, REX, E-TRON, TFSIE), a hybrid-only model line (Toyota Venza 2021+,
    Crown 2023+, Sequoia 2023+, Sienna 2021+, Karma) or an FTP at least 0.9 x HWY (a city-strong signature no
    conventional car in the table has: their FTP is 0.55-0.80 x HWY).
Dropped (with the reason, listed in the file):
  - families of the dev split (they are dev: listed under 'dev_extra' for the calibration, not part of the exam);
  - the families of the four check_cycles dev hybrids (HYUNDAI|SONATA, TOYOTA|RAV4, FORD|ESCAPE, HONDA|ACCORD), as
    in exam-2: v2 was fitted on those cars;
  - records any check simulated (check_cycles, exam-2, D3 twin rows) and carry-over copies: same family and the same
    FTP and HWY (0.1 mpg) as any MY2008-2026 hybrid / plug-in record or check_cycles vehicle; repeated copies inside
    the new set (same make, model, test vehicle id and results) keep the first model year;
  - designs RoadSim does not model or has no hardware rule for (checks/hybrid_specs.py lookup_ext; the supplementary
    rules SUPP_HW were written from model names before any of these records was simulated).
Each record carries its group (hybrid_specs.group_of), its model year, and whether its family had exam-2 records
('family_seen_exam2': results of siblings were seen at the v2.1 report).

usage: python exam3_hybrids.py write | check
"""
import os, sys, json, hashlib, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VEH, 'roadsim')); sys.path.insert(0, VEH); sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'exam3_hybrids.json')
FIT_FAMILIES = ('HYUNDAI|SONATA', 'TOYOTA|RAV4', 'FORD|ESCAPE', 'HONDA|ACCORD')
TOKENS = re.compile(r'FHEV|\bREX\b|E-TRON|TFSIE', re.I)
FIELDS = ['model_year', 'make', 'model', 'test_vehicle_id', 'displacement_L', 'rated_hp', 'veh_type', 'trans_type', 'gears',
          'drive', 'etw_lb', 'axle_ratio', 'nv_ratio', 'A_lbf', 'B_lbf_mph', 'C_lbf_mph2', 'FTP_mpg', 'HWY_mpg', 'US06_mpg',
          'SC03_mpg', 'FTP_cold_mpg']


def canon(d):
    return json.dumps({k: v for k, v in d.items() if k != 'sha256'}, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def sha(d):
    return hashlib.sha256(canon(d).encode('utf-8')).hexdigest()


def _hidden(r):
    md = str(r.model).upper(); y = int(r.model_year); mk = str(r.make).upper()
    if TOKENS.search(md):
        return 'name token'
    if (('VENZA' in md and y >= 2021) or ('CROWN' in md and y >= 2023) or ('SEQUOIA' in md and y >= 2023)
            or ('SIENNA' in md and y >= 2021) or mk == 'KARMA'):
        return 'hybrid-only model line'
    if pd.notna(r.FTP_mpg) and pd.notna(r.HWY_mpg) and r.FTP_mpg >= 0.9 * r.HWY_mpg:
        return 'FTP >= 0.9 HWY'
    return None


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v if isinstance(v, (int, str, bool)) else str(v)


def build():
    import realdata as rd
    import hybrid_specs as hs
    import epa_pre2008
    import exam2_hybrids as x2
    e = rd.epa_testcars()
    # ---- what was seen
    V = json.load(open(os.path.join(HERE, 'epa_vehicles.json')))
    cand = pd.read_csv(os.path.join(VEH, 'roadsim', 'refdata', 'epa_candidates.csv'))
    seen_ids, seen_fe = set(), set()
    for v in V:
        c = cand.loc[v['row']]
        m = e[(e.model_year == c['Model Year']) & (e.test_vehicle_id == c['Test Vehicle ID'])]
        seen_ids |= set(m.index)
    X2 = x2.load()
    seen_ids |= {r['epa_vehicle_id'] for r in X2['epa']}
    for ids in X2['epa_excluded'].values():
        seen_ids |= set(ids)
    S = json.load(open(os.path.join(VEH, 'data', 'proc', 'd3', 'specs.json')))
    seen_ids |= {(s.get('epa_match') or {}).get('epa_vehicle_id') for s in S.values()} - {None}
    fe_key = lambda r: (r.family, round(float(r.FTP_mpg), 1) if pd.notna(r.FTP_mpg) else None,
                        round(float(r.HWY_mpg), 1) if pd.notna(r.HWY_mpg) else None)
    for i, r in e[e.architecture.isin(['hev', 'phev']) | e.index.isin(seen_ids)].iterrows():
        seen_fe.add(fe_key(r))
    x2_fams = {r['family'] for r in X2['epa']}
    # ---- candidates
    p = epa_pre2008.vehicles()
    a = p[p.architecture.isin(['hev', 'phev']) & (p.fuel == 'gasoline')].copy(); a['source'] = 'EPA MY2000-2007 (new files)'
    a['why_hybrid'] = 'model-name rule'
    b = e[(e.architecture == 'conv') & (e.fuel == 'gasoline') & (e.FTP_mpg.notna() | e.HWY_mpg.notna())].copy()
    b['why_hybrid'] = [_hidden(r) for _, r in b.iterrows()]
    b = b[b.why_hybrid.notna()].copy(); b['source'] = 'EPA MY2008-2026, labelled conventional'
    h = pd.concat([a, b], sort=False)
    h['_m'] = h.model.astype(str).str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
    h = h.sort_values(['family', '_m', 'model_year']).drop_duplicates(
        ['make', '_m', 'test_vehicle_id', 'FTP_mpg', 'HWY_mpg', 'US06_mpg', 'SC03_mpg', 'FTP_cold_mpg'])
    keep, dev_extra, excl = [], [], {}
    for i, r in h.iterrows():
        why = None
        if i in seen_ids:
            why = 'simulated by a check (check_cycles, exam-2, D3 twin row)'
        elif fe_key(r) in seen_fe:
            why = 'carry-over copy of a MY2008-2026 hybrid record or a check vehicle'
        elif r.family in FIT_FAMILIES:
            why = 'family of a check_cycles dev hybrid (fitted on in v2)'
        spec = kind = grp = None
        if why is None:
            spec, kind, grp = hs.spec_of(r, ext=True)
            if spec is None:
                why = f'not modelled: {grp}' if kind == 'exclude' else f'no spec: {grp}'
        if why:
            excl.setdefault(why, []).append(i); continue
        rec = dict(epa_vehicle_id=i, family=r.family, make=str(r.make), model=str(r.model), model_year=int(r.model_year),
                   kind=kind, group=grp, source=r.source, why_hybrid=r.why_hybrid, family_seen_exam2=r.family in x2_fams,
                   epa=hs.epa_results(r), row={k: _clean(r.get(k)) for k in FIELDS})
        (dev_extra if r.split == 'dev' else keep).append(rec)
    out = dict(about=__doc__.strip(), frozen='14 Nov 2026, RoadSim v3 pass, before any v3 fitting',
               raw_files={os.path.basename(f): hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in epa_pre2008.raw_files()},
               fit_families=list(FIT_FAMILIES), epa=keep, dev_extra=dev_extra,
               excluded={k: sorted(v) for k, v in sorted(excl.items())},
               counts=dict(epa=len(keep), epa_by_group=pd.Series([k['group'] for k in keep]).value_counts().to_dict(),
                           epa_by_source=pd.Series([k['source'] for k in keep]).value_counts().to_dict(),
                           family_unseen=int(sum(not k['family_seen_exam2'] for k in keep)),
                           dev_extra=len(dev_extra), excluded={k: len(v) for k, v in excl.items()}))
    out['sha256'] = sha(out)
    return out


def load():
    d = json.load(open(OUT, encoding='utf-8'))
    assert sha(d) == d['sha256'], 'exam3_hybrids.json: sha256 mismatch'
    return d


def row_of(rec):
    """the record's test-car fields as a pandas row (what hybrid_specs.spec_of reads), named by its epa_vehicle_id."""
    return pd.Series(rec['row'], name=rec['epa_vehicle_id'])


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'
    d = build()
    if cmd == 'write':
        if os.path.exists(OUT):
            sys.exit('exam3_hybrids.json exists: it is frozen (use check)')
        json.dump(d, open(OUT, 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)
        print('wrote', OUT, d['sha256'])
    elif cmd == 'show':
        for r in d['epa'] + [dict(x, group='DEV ' + x['group']) for x in d['dev_extra']]:
            print(f"{r['group']:16s} {r['model_year']} {r['make'][:12]:12s} {r['model'][:34]:34s} {r['why_hybrid'][:22]:22s} seen2={r['family_seen_exam2']!s:5s} {r['epa']}")
    else:
        old = load()
        print('rebuilt', d['sha256'], 'frozen', old['sha256'], 'MATCH' if d['sha256'] == old['sha256'] else 'DIFFERENT')
    print(json.dumps(d['counts'], indent=1))
