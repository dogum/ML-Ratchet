"""Freeze the BuildingData splits (seed 20261116) -> proc/splits.json (+ sha256 of the frozen content).

Decided once; must not change after Phase 2 starts. Design in DATA.md section 8.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import json, hashlib, os
import numpy as np, pandas as pd

D = f'{BLDG_ROOT}/data'; P = f'{D}/proc'
SEED = 20261116


def draw(df, key, strata, k_per, min_stratum, rng):
    picks = []
    for st, g in df.groupby(strata):
        if len(g) < min_stratum: continue
        picks += list(g[key].iloc[rng.choice(len(g), min(k_per, len(g)), replace=False)])
    return sorted(picks)


def main():
    rng = np.random.default_rng(SEED)
    out = {'seed': SEED, 'version': 'bldg-data-1.0', 'frozen': '2026-11-16'}
    # ---------------- BDG2 buildings
    b = pd.read_parquet(f'{P}/bdg2/buildings.parquet')
    okcols = [c for c in b.columns if c.startswith('q_ok_') and not c.startswith('q_ok_daily')]
    b['any_ok'] = b[okcols].fillna(False).any(axis=1)
    b['stratum'] = b.primary_use + '|' + b.climate_zone
    elig = b[b.any_ok]
    dev = draw(elig, 'id', 'stratum', 2, 8, rng)        # 2 buildings per (use, climate) stratum with >= 8 eligible buildings
    exam = sorted(set(b.id) - set(dev))
    out['bdg2'] = {
        'unit': 'building', 'stratified_by': ['primary_use', 'climate_zone'],
        'dev': dev, 'exam': exam,
        'rule': 'dev = 2 buildings drawn per (primary_use, climate_zone) stratum having >= 8 buildings with at least one q_ok meter; exam = all others',
        'time': {'history': ['2016-01-01', '2017-01-01'], 'forecast': ['2017-01-01', '2018-01-01'],
                 'note': 'calendar years in each building\'s LOCAL time; models needing history fit on 2016 and are graded on 2017'},
        'labels_ok': 'grade a building-meter only where meter_quality.q_ok (hourly) or q_ok_daily (daily/monthly tasks) is True'}
    # ---------------- RECS / CBECS respondents
    for name, f, strata in [('recs2020', 'eia/recs2020.parquet', ['division', 'housing_type']),
                            ('cbecs2018', 'eia/cbecs2018.parquet', ['division', 'pba'])]:
        e = pd.read_parquet(f'{P}/{f}')
        lab = pd.Series('exam', index=e.index)
        for _, g in e.groupby(strata):
            idx = g.index.to_numpy().copy(); rng.shuffle(idx)
            n = len(idx); nd = max(1, int(round(0.10 * n))); npri = int(round(0.50 * n))
            lab.loc[idx[:nd]] = 'dev'; lab.loc[idx[nd:nd + npri]] = 'prior'
        out[name] = {'unit': 'respondent', 'stratified_by': strata, 'fractions': {'prior': 0.5, 'dev': 0.1, 'exam': 0.4},
                     'prior': sorted(e.id[lab == 'prior']), 'dev': sorted(e.id[lab == 'dev']), 'exam': sorted(e.id[lab == 'exam']),
                     'note': 'prior = may feed generator priors / simulated training; dev = calibration and gates; exam = final report only. '
                             'Survey weights refer to the whole sample: for population estimates within a split multiply weights by 1/fraction.'}
    # ---------------- faults (group = facility + date; ERS AHU-A day and its AHU-B baseline stay together)
    r = pd.read_parquet(f'{P}/faults/runs.parquet')
    r['group'] = r.facility + '|' + r.date.fillna('')
    g = r[r.gradable].copy()
    g['is_fault'] = g.fault_type != 'none'
    grp = g.groupby('group').agg(dataset=('dataset', 'first'), fault=('is_fault', 'max'), season=('season', 'first')).reset_index()
    dev_groups = []
    plan = {'ers_rp1312': [('fault', 'summer', 1), ('fault', 'winter', 1), ('fault', 'transition', 1)],
            'lbnl_mzvav2_ers_exp': [('normal', None, 1)], 'lbnl_szcav_flexlab': [('fault', None, 1)],
            'lbnl_szvav_flexlab': [('fault', None, 1), ('normal', None, 1)], 'lbnl_rtu_ornl': [('fault', None, 1), ('normal', None, 1)],
            'nesbitt': [('fault', None, 2), ('normal', None, 2)]}
    for ds, items in plan.items():
        for kind, season, k in items:
            c = grp[(grp.dataset == ds) & (grp.fault == (kind == 'fault'))]
            if season: c = c[c.season == season]
            c = c[~c.group.isin(dev_groups)]
            dev_groups += list(c.group.iloc[rng.choice(len(c), min(k, len(c)), replace=False)])
    lab = np.where(~r.gradable, np.where(r.kind.isin(['simulated', 'hil']), 'sim_reference',
                                         np.where(r.dataset == 'korea_office', 'noise_unlabelled', 'excluded')),
                   np.where(r.group.isin(dev_groups), 'dev', 'exam'))
    r['split'] = lab
    out['faults'] = {'unit': 'run (unit-day); grouped by facility+date so a day never spans splits',
                     'dev': sorted(r.run_id[r.split == 'dev']), 'exam': sorted(r.run_id[r.split == 'exam']),
                     'sim_reference': sorted(r.run_id[r.split == 'sim_reference']),
                     'noise_unlabelled': sorted(r.run_id[r.split == 'noise_unlabelled']), 'excluded': sorted(r.run_id[r.split == 'excluded']),
                     'rule': 'dev = seeded draw: ERS 1 fault day per season (+ its AHU-B baseline), 1 ERS normal day, FLEXLAB SZCAV 1 fault day, '
                             'SZVAV 1 fault + 1 normal day, ORNL RTU 1 fault + 1 normal day, Nesbitt 2 fault + 2 baseline days (both AHUs); '
                             'exam = every other gradable (experimental or real-building) run. Simulated / HIL sets are never graded.'}
    r.to_parquet(f'{P}/faults/runs.parquet', index=False)
    # ---------------- NREL stock samples (modelled): reference only
    out['stock'] = {'resstock_sample': 'reference (modelled; generator priors and simulator-vs-simulator check; never an exam)',
                    'comstock_sample': 'reference (modelled; same)',
                    'priors_rule': 'whole-stock metadata may be used freely for priors (it is modelled, not a real exam)'}
    body = json.dumps(out, sort_keys=True)
    out['sha256'] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(out, open(f'{P}/splits.json', 'w'), indent=0)
    print('bdg2 dev', len(dev), 'exam', len(exam))
    print(b[b.id.isin(dev)].groupby(['primary_use', 'climate_zone']).size().to_string())
    for n in ['recs2020', 'cbecs2018']: print(n, {k: len(out[n][k]) for k in ['prior', 'dev', 'exam']})
    print(r.groupby(['split', 'dataset']).size().to_string())


if __name__ == '__main__':
    if os.path.exists(f'{P}/splits.json') and os.environ.get('REFREEZE') != '1':
        raise SystemExit('splits.json exists and is frozen; set REFREEZE=1 only before Phase 2 starts')
    main()
