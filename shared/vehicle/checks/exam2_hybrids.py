"""Exam-2 for hybrids and plug-ins: data no RoadSim check had looked at before 14 Nov 2026, frozen before the v2.1 fit.

Writes exam2_hybrids.json (lists only, no results) with a sha256 over its canonical content (sorted keys, no
whitespace, UTF-8, without the sha256 field). Written once; `check` recomputes the list and the hash and compares.

(a) VED exam HEV and PHEV cars (data/proc/splits.json, split 'exam'): run as label-matched twins on their own
    q_label_ok trips, exactly as checks/check_ved_twin.py runs the dev cars (check_ved_twin.py --split exam). No VED
    exam check existed before. VED exam is the models' exam too: used for the v2.1 report only, nothing is chosen by
    it, and no per-trip prediction is written under models/.
(b) EPA test-car records (data/proc/epa/testcars_vehicles.parquet) of hybrids and plug-ins (RoadData architecture hev
    or phev) in held-out model families (family split 'exam'), one per test vehicle and result set (carry-over copies
    of the same test in later model years dropped), with an FTP, HWY or US06 result, that no check had simulated:
      - not a check_cycles vehicle (checks/epa_vehicles.json; same model year and test vehicle id), nor a carry-over
        copy of one (same family, FTP and HWY);
      - not the EPA row a D3 twin was matched to (data/proc/d3/specs.json, epa_match);
      - not in the family of a check_cycles DEV hybrid (HYUNDAI|SONATA, TOYOTA|RAV4, FORD|ESCAPE, HONDA|ACCORD): v2
        was fitted on those cars (regen share, Toyota and e:HEV rules), so their siblings are not a fresh judge;
      - an architecture RoadSim models (checks/hybrid_specs.py; excluded designs are listed with the reason).
    Each record carries its reporting group (hybrid_specs.GROUP) and whether its family held a check_cycles held-out
    vehicle (seen once, at the v2 report: 'family_seen_v2').

usage: python exam2_hybrids.py write | check
"""
import os, sys, json, hashlib, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VEH, 'roadsim')); sys.path.insert(0, VEH); sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'exam2_hybrids.json')
FIT_FAMILIES = ('HYUNDAI|SONATA', 'TOYOTA|RAV4', 'FORD|ESCAPE', 'HONDA|ACCORD')


def canon(d):
    return json.dumps({k: v for k, v in d.items() if k != 'sha256'}, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def sha(d):
    return hashlib.sha256(canon(d).encode('utf-8')).hexdigest()


def build():
    import realdata as rd
    import hybrid_specs as hs
    e = rd.epa_testcars()
    V = json.load(open(os.path.join(HERE, 'epa_vehicles.json')))
    cand = pd.read_csv(os.path.join(VEH, 'roadsim', 'refdata', 'epa_candidates.csv'))
    sim_ids, check_fe, seen_fam = set(), set(), set()
    for v in V:
        c = cand.loc[v['row']]
        m = e[(e.model_year == c['Model Year']) & (e.test_vehicle_id == c['Test Vehicle ID'])]
        sim_ids |= set(m.index)
        for i, r in m.iterrows():
            check_fe.add((r.family, round(float(r.FTP_mpg), 1) if pd.notna(r.FTP_mpg) else None,
                          round(float(r.HWY_mpg), 1) if pd.notna(r.HWY_mpg) else None))
            if v['split'] == 'heldout':
                seen_fam.add(r.family)
    S = json.load(open(os.path.join(VEH, 'data', 'proc', 'd3', 'specs.json')))
    d3_ids = {(s.get('epa_match') or {}).get('epa_vehicle_id') for s in S.values()} - {None}
    h = e[e.architecture.isin(['hev', 'phev']) & (e.split == 'exam')].copy()
    h = h[h.FTP_mpg.notna() | h.HWY_mpg.notna() | h.US06_mpg.notna()]
    h['_m'] = h.model.astype(str).str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
    h = h.sort_values(['family', '_m', 'model_year']).drop_duplicates(
        ['make', '_m', 'test_vehicle_id', 'FTP_mpg', 'HWY_mpg', 'US06_mpg', 'SC03_mpg', 'FTP_cold_mpg'])
    keep, excl = [], {}
    for i, r in h.iterrows():
        why = None
        fe = (r.family, round(float(r.FTP_mpg), 1) if pd.notna(r.FTP_mpg) else None,
              round(float(r.HWY_mpg), 1) if pd.notna(r.HWY_mpg) else None)
        if i in sim_ids or fe in check_fe:
            why = 'simulated by check_cycles (or a carry-over copy)'
        elif i in d3_ids:
            why = 'EPA row of a D3 twin'
        elif r.family in FIT_FAMILIES:
            why = 'family of a check_cycles dev hybrid (fitted on in v2)'
        else:
            spec, kind, grp = hs.spec_of(r)
            if spec is None:
                why = f'not modelled: {grp}' if kind == 'exclude' else f'no spec: {grp}'
        if why:
            excl.setdefault(why, []).append(i); continue
        keep.append(dict(epa_vehicle_id=i, family=r.family, make=r.make, model=str(r.model), model_year=int(r.model_year),
                         architecture=r.architecture, kind=kind, group=grp, family_seen_v2=r.family in seen_fam,
                         epa=hs.epa_results(r)))
    vv = rd.ved_vehicles()
    x = vv[(vv.split == 'exam') & vv.architecture.isin(['hev_powersplit', 'hev_parallel_p2', 'phev_powersplit', 'phev_series'])]
    idx = rd.ved_index()
    ntr = idx[(idx.split == 'exam') & idx.q_label_ok].groupby('vehicle_id').size()
    ved = [dict(vehicle_id=vid, architecture=r.architecture, example_models=str(r.example_models)[:120],
                match_confidence=r.match_confidence, n_label_trips=int(ntr.get(vid, 0))) for vid, r in x.sort_index().iterrows()]
    out = dict(about=__doc__.strip(), frozen='14 Nov 2026, calibration agent 3, before any v2.1 fitting',
               fit_families=list(FIT_FAMILIES), epa=keep, epa_excluded={k: sorted(v) for k, v in sorted(excl.items())},
               ved=ved, counts=dict(epa=len(keep), epa_by_group=pd.Series([k['group'] for k in keep]).value_counts().to_dict(),
                                    epa_excluded={k: len(v) for k, v in excl.items()},
                                    ved_cars=len(ved), ved_by_arch=pd.Series([v['architecture'] for v in ved]).value_counts().to_dict(),
                                    ved_trips=int(sum(v['n_label_trips'] for v in ved))))
    out['sha256'] = sha(out)
    return out


def load():
    d = json.load(open(OUT, encoding='utf-8'))
    assert sha(d) == d['sha256'], 'exam2_hybrids.json: sha256 mismatch'
    return d


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'
    d = build()
    if cmd == 'write':
        if os.path.exists(OUT):
            sys.exit('exam2_hybrids.json exists: it is frozen (use check)')
        json.dump(d, open(OUT, 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)
        print('wrote', OUT, d['sha256'])
    else:
        old = load()
        print('rebuilt', d['sha256'], 'frozen', old['sha256'], 'MATCH' if d['sha256'] == old['sha256'] else 'DIFFERENT')
    print(json.dumps(d['counts'], indent=1))
