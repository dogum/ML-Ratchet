"""Freeze the WaterData splits (proc/splits.json).  Sections are written once and never changed afterwards;
a later run only ADDS sections that do not exist yet (other regions, SNOTEL, reservoirs) and refuses to rewrite.

Seed 20261118 everywhere (numpy default_rng). Selection uses only attributes and data availability, never skill.
"""
import json, os, sys, hashlib, time, glob
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
OUT = f'{ROOT}/proc/splits.json'
SEED = 20261118
CAL = ('1999-10-01', '2008-09-30')
VAL = ('1989-10-01', '1999-09-30')
RECENT = ('2015-10-01', '2023-09-30')


def coverage(gid, region='camels', periods=(CAL, VAL)):
    q = pd.read_parquet(f'{ROOT}/proc/caravan/{region}/{gid}.parquet', columns=['Q_mm']).Q_mm
    return [float(q.loc[a:b].notna().mean()) for a, b in periods]


def camels_us():
    a = pd.read_parquet(f'{ROOT}/proc/attributes/camels.parquet').set_index('gauge_id')
    cov = pd.DataFrame({g: coverage(g) for g in a.index}, index=['cov_cal', 'cov_val']).T
    a = a.join(cov)
    a['eligible'] = (a.cov_cal >= 0.9) & (a.cov_val >= 0.9) & (a.area_discrepancy.abs() < 0.1)
    rng = np.random.default_rng(SEED)
    a['snow_cls'] = pd.cut(a.frac_snow, [-1, 0.1, 0.3, 2], labels=[0, 1, 2]).astype(int)
    dev = []
    alloc = {}
    for r, grp in a.groupby('huc2'):
        n = max(1, int(round(50 * len(grp) / len(a))))
        alloc[r] = n
        el = grp[grp.eligible].copy()
        el['key'] = el.snow_cls * 10 + el.aridity_FAO_PM.rank(pct=True)   # climate gradient within region
        el = el.sort_values('key')
        step = len(el) / n
        start = rng.uniform(0, step)
        pick = [el.index[min(len(el) - 1, int(start + i * step))] for i in range(n)]
        dev += pick
    # make sure regulation is represented: 3 extra regulated basins from distinct HUC2s
    regd = [g for g in dev if a.loc[g, 'reg_class'] in ('mid', 'high')]
    pool = a[a.eligible & a.reg_class.isin(['mid', 'high']) & ~a.index.isin(dev)]
    extra = []
    for g in rng.permutation(pool.index.values):
        if len(extra) >= 3:
            break
        if a.loc[g, 'huc2'] not in [a.loc[x, 'huc2'] for x in extra]:
            extra.append(g)
    dev = sorted(dev + extra)
    exam = sorted(set(a.index) - set(dev))
    folds = {g: int(f) for g, f in zip(a.index, rng.permutation(np.arange(len(a)) % 12))}
    checks = {}
    for c in ['aridity_FAO_PM', 'frac_snow', 'p_mean', 'area', 'ele_mt_sav', 'for_pc_sse', 'dor_frac', 'gauge_lat', 'gauge_lon']:
        x, y = a.loc[dev, c].dropna(), a.loc[exam, c].dropna()
        checks[c] = {'dev_median': round(float(x.median()), 4), 'exam_median': round(float(y.median()), 4),
                     'ks_p': round(float(ks_2samp(x, y).pvalue), 4)}
    return {
        'dev': dev, 'exam': exam,
        'exam_531': sorted(set(exam) & set(a.index[a.in_531])),
        'dev_regulated_extra': extra,
        'n_dev': len(dev), 'n_exam': len(exam),
        'allocation_per_huc2': alloc,
        'eligibility': 'daily Q coverage >= 90% in both CAMELS periods (Caravan record) and |area(Caravan) - area(NWIS)| < 10%',
        'stratification': ('per HUC2 region n = max(1, round(50 * N_region / 671)); within the region the eligible basins are '
                           'sorted by snow class (frac_snow <0.1, 0.1-0.3, >=0.3) then aridity rank and picked by systematic '
                           'sampling with a seeded random start; then 3 extra regulated basins (reg_class mid/high, '
                           'distinct HUC2) are added at random.'),
        'balance_check_dev_vs_exam': checks,
        'ungauged_folds_k12': folds,
        'ineligible_count': int((~a.eligible).sum()),
    }


def other_regions(done):
    """3 dev basins per region, one per aridity tercile; first candidate in a seeded order with >= 80% Q coverage
    1990-2009. Each region is frozen separately once all its timeseries are on disk (regions already frozen: skipped).
    The rng is seeded per region (SEED + 1 + region index) so the order in which regions arrive does not matter."""
    out = {}
    for k, r in enumerate(['camelsgb', 'camelsaus', 'camelscl', 'camelsbr']):
        if r in done:
            continue
        a = pd.read_parquet(f'{ROOT}/proc/attributes/{r}.parquet').set_index('gauge_id')
        have = set(os.path.basename(p)[:-8] for p in glob.glob(f'{ROOT}/proc/caravan/{r}/*.parquet'))
        if len(have) < len(a):
            print(f'{r}: only {len(have)}/{len(a)} timeseries present; not frozen yet')
            continue
        rng = np.random.default_rng(SEED + 1 + k)
        terc = pd.qcut(a.aridity_FAO_PM.rank(method='first'), 3, labels=['humid', 'mid', 'arid'])
        dev = []
        for t in ['humid', 'mid', 'arid']:
            cand = rng.permutation(a.index[terc == t].values)
            for g in cand:
                if coverage(g, r, periods=(('1990-01-01', '2009-12-31'),))[0] >= 0.8:
                    dev.append(g); break
        out[r] = {'dev': sorted(dev), 'exam': sorted(set(a.index) - set(dev)), 'n': len(a),
                  'rule': 'one basin per aridity tercile (FAO-PM aridity), first in a seeded permutation (seed+1+region index) '
                          'with >= 80% daily Q coverage 1990-2009',
                  'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    return out


def reservoirs():
    r = pd.read_parquet(f'{ROOT}/proc/reservoirs/reservoirs.parquet')
    r = r[r.cap_hm3.notna()]
    core = r[(r.n_all3 >= 3650)].copy()
    core['use_grp'] = core.main_use.where(core.main_use.isin(['fcon', 'irri', 'elec', 'supp']), 'other')
    core['size'] = np.where(core.cap_hm3 >= core.cap_hm3.median(), 'large', 'small')
    rng = np.random.default_rng(SEED + 3)
    dev = []
    for (u, sz), g in core.groupby(['use_grp', 'size']):
        n = max(1, int(round(14 * len(g) / len(core))))
        dev += list(rng.choice(g.res_id.values, size=min(n, len(g)), replace=False))
    dev = sorted(dev)
    allr = pd.read_parquet(f'{ROOT}/proc/reservoirs/reservoirs.parquet')
    return {'dev': dev, 'exam': sorted(set(core.res_id) - set(dev)),
            'aux': sorted(set(allr.res_id) - set(core.res_id)),
            'rule': ('core = ResOpsUS dams with >= 10 years of simultaneous storage, inflow and release and a GRanD/ResOpsUS '
                     'capacity; dev = per (main use in fcon/irri/elec/supp/other) x (capacity above/below median) cell, '
                     'n = max(1, round(14 * N_cell / N_core)) at random (seed+3); exam = rest of core; aux = everything else '
                     '(incomplete variables; free for simulator calibration of single processes, never scored).'),
            'temporal': 'each exam reservoir: first half of its simultaneous record (>= 1980) = calibration of rule curve/demand, '
                        'second half = scored. Dev reservoirs: free.'}


def snotel():
    m = pd.read_parquet(f'{ROOT}/proc/snotel/stations.parquet')
    a = pd.read_parquet(f'{ROOT}/proc/attributes/camels.parquet')
    cu = json.load(open(OUT))['camels_us']
    exam_h8 = set(a[a.gauge_id.isin(cu['exam'])].huc8.dropna())
    m['h8'] = m.huc12.astype(str).str[:8]
    elig = m[m.n_wy_complete >= 10].copy()
    pool = elig[~elig.h8.isin(exam_h8)]
    rng = np.random.default_rng(SEED + 2)
    dev = []
    for st, g in elig.groupby('state'):
        n = max(1, int(round(40 * len(g) / len(elig))))
        gp = pool[pool.state == st].sort_values('elev_m')
        if len(gp) == 0:
            continue
        step = len(gp) / min(n, len(gp)); start = rng.uniform(0, step)
        dev += [gp.station_id.iloc[min(len(gp) - 1, int(start + i * step))] for i in range(min(n, len(gp)))]
    dev = sorted(set(dev))
    return {'dev': dev, 'exam': sorted(set(elig.station_id) - set(dev)),
            'ineligible': sorted(set(m.station_id) - set(elig.station_id)),
            'rule': ('eligible = >= 10 complete water years; dev pool = eligible stations whose HUC8 contains no EXAM CAMELS-US '
                     'gauge; per state n = max(1, round(40 * N_state / N_eligible)), systematic sample by elevation with a '
                     'seeded start (seed+2); exam = all other eligible stations.'),
            'temporal': 'classical per-station calibration WY2001-2010, scoring WY2011-2025; networks may use dev stations fully'}


def main():
    s = json.load(open(OUT)) if os.path.exists(OUT) else {}
    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    what = sys.argv[1:] or ['core']
    if 'core' in what and 'camels_us' not in s:
        s['seed'] = SEED
        s['version'] = 'water-splits-v1'
        s['rules'] = [
            'Only dev basins / dev stations / dev reservoirs may be used to calibrate, gate or choose anything in Phase 2.',
            'Exam units are for final reporting only. Their streamflow (any period) is never a training target.',
            'Gauged exam protocol: an exam basin may use its own observed Q from the calibration period as model INPUT or '
            'for per-basin classical calibration; it is scored in the validation (and recent) periods only.',
            'Ungauged exam protocol: no streamflow from the scored basin in any period. Donor information may come only '
            'from dev basins (protocol U-dev, same information an ML model gets) or, for the literature-style '
            'regionalisation baseline, from other folds of ungauged_folds_k12 (protocol U-k12).',
            'Simulated training may use real forcing of ANY basin from non-exam periods (before 1989-10-01, 1999-10-01..2015-09-30), '
            'and real attributes only from dev basins (and from regions/datasets not used in any exam).',
            'Sections of this file are frozen when written; later runs only add new sections.',
        ]
        s['temporal'] = {
            'calibration': {'start': CAL[0], 'end': CAL[1], 'spinup_start': '1998-10-01',
                            'note': 'CAMELS standard protocol (Newman et al. 2017; Kratzert et al. 2019): train/calibrate'},
            'validation': {'start': VAL[0], 'end': VAL[1], 'spinup_start': '1988-10-01',
                           'note': 'CAMELS standard test period: the main temporal exam'},
            'recent_exam': {'start': RECENT[0], 'end': RECENT[1], 'spinup_start': '2014-10-01',
                            'note': 'water years 2016-2023; Q from USGS NWIS (beyond Caravan end 2014); forcing ERA5-Land from '
                                    'Caravan ends 2023-12-30; NWM retrospective v3.0 ends 2023-01-31 (NWM scores use WY2016-2022)'},
            'free_forcing_periods': [['1951-01-01', '1988-09-30'], ['1999-10-01', '2015-09-30']],
            'water_year': 'Oct 1 - Sep 30, named by the ending year',
        }
        s['camels_us'] = camels_us()
        s['camels_us']['frozen_at'] = stamp
    if 'other' in what:
        s.setdefault('other_regions', {})
        s['other_regions'].update(other_regions(set(s['other_regions'])))
        print('other regions frozen:', {r: v['dev'] for r, v in s['other_regions'].items()})
    if 'reservoirs' in what and 'reservoirs' not in s:
        s['reservoirs'] = reservoirs()
        s['reservoirs']['frozen_at'] = stamp
        print('reservoirs dev', len(s['reservoirs']['dev']), 'exam', len(s['reservoirs']['exam']), 'aux', len(s['reservoirs']['aux']))
    if 'snotel' in what and 'snotel' not in s:
        s['snotel'] = snotel()
        s['snotel']['frozen_at'] = stamp
        print('snotel dev', len(s['snotel']['dev']), 'exam', len(s['snotel']['exam']))
    body = json.dumps({k: v for k, v in s.items() if k != 'sha256'}, sort_keys=True)
    s['sha256'] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(s, open(OUT + '.tmp', 'w'), indent=1); os.replace(OUT + '.tmp', OUT)
    cu = s['camels_us']
    print('dev', cu['n_dev'], 'exam', cu['n_exam'], 'exam531', len(cu['exam_531']), 'inelig', cu['ineligible_count'])
    print(json.dumps(cu['balance_check_dev_vs_exam'], indent=0))


if __name__ == '__main__':
    main()
