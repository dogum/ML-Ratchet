"""BuildingDetective exam 2 (frozen 17 Nov 2026, before any round-2 rung is judged): HEAPO, Zurich heat-pump homes.

    python exam2_heapo.py fetch     # download heapo_data.zip (Zenodo 15056919 v1, CC BY 4.0, 458 MB) into work/exam2/heapo/raw
    python exam2_heapo.py build     # QC, split by home, hash; writes work/exam2/heapo/proc and training/split_exam2_heapo.json

Scores nothing. The plan is the NOTEBOOK.md entry of 2026-11-17 "Fresh exam frozen before round-2 rungs: HEAPO".
split_exam2_heapo.json is frozen: build refuses to overwrite it (use --check to rebuild and compare hashes).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths as _P  # noqa: E402

SEED = 20261117
OUT = os.path.join(_P.WORK, 'exam2', 'heapo')
RAW, PROC = os.path.join(OUT, 'raw'), os.path.join(OUT, 'proc')
SPLIT = os.path.join(HERE, 'split_exam2_heapo.json')
UA = 'Mozilla/5.0 (research data client)'
FILES = [dict(name='heapo_data.zip', url='https://zenodo.org/api/records/15056919/files/heapo_data.zip/content',
              md5='2b92b0d52487e1a733c35ee59860450d', size=458317017)]
SOURCE = dict(dataset='HEAPO', version='v1 (24 Mar 2025)', doi='10.5281/zenodo.15056919',
              landing='https://zenodo.org/records/15056919', licence='CC BY 4.0',
              citation='Brudermueller, T. et al. (2025) HEAPO - An Open Dataset for Heat Pump Optimization with Smart '
                       'Electricity Meter Data and On-Site Inspection Protocols, Zenodo, doi:10.5281/zenodo.15056919')


def file_hash(path, algo='sha256'):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fetch():
    os.makedirs(RAW, exist_ok=True)
    rec = []
    for f in FILES:
        p = os.path.join(RAW, f['name'])
        if not (os.path.exists(p) and os.path.getsize(p) == f['size']):
            subprocess.run(['curl', '-L', '-sS', '--fail', '--retry', '5', '-A', UA, '-C', '-', '-o', p, f['url']],
                           check=True)
        md5 = file_hash(p, 'md5')
        if md5 != f['md5']:
            raise SystemExit(f"{f['name']}: md5 {md5} != published {f['md5']}")
        rec.append(dict(name=f['name'], url=f['url'], size=os.path.getsize(p), md5=md5, sha256=file_hash(p),
                        downloaded=date.today().isoformat()))
    json.dump(dict(SOURCE, files=rec), open(os.path.join(RAW, 'SOURCES.json'), 'w'), indent=1)
    print(json.dumps(rec, indent=1))


# ----------------------------------------------------------------------------- BDG2 electricity flags (copied rules)
# Same rules as shared/building/data/scripts/build_bdg2.py flags_for (electricity), on any hourly UTC index.
F = dict(MISSING=1, ZERO=2, ZERO_RUN=4, NEGATIVE=8, SPIKE=16, FLATLINE=32, INTERP=64, LEVEL_SHIFT=128)
BAD = F['MISSING'] | F['NEGATIVE'] | F['SPIKE'] | F['FLATLINE'] | F['INTERP'] | F['ZERO'] | F['ZERO_RUN'] | F['LEVEL_SHIFT']
# HEAPO change (17 Nov 2026, after reading QC counts, before any scoring): the BDG2 level-shift rule (14-day median of
# daily medians > 3x or < 1/3 of the record median) flags the heating season itself in heat-pump homes (median 145
# flagged days per home; 72 of 214 homes lost to it). It is not applied: LEVEL_SHIFT is computed and reported only.
BAD_HEAPO = BAD & ~F['LEVEL_SHIFT']


def run_len(mask):
    n = len(mask)
    if not mask.any():
        return np.zeros(n, np.int32)
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    out = np.zeros(n, np.int32)
    for a, b in zip(st, en):
        out[a:b] = b - a
    return out


def flags_electricity(x, index):
    """x: hourly kWh on `index` (hourly UTC). Returns (uint16 flags, set of level-shift days)."""
    n = len(x)
    f = np.zeros(n, np.uint16)
    miss = ~np.isfinite(x)
    f[miss] |= F['MISSING']
    v = np.where(miss, np.nan, x)
    zero = v == 0
    f[zero] |= F['ZERO']
    f[run_len(zero) >= 24] |= F['ZERO_RUN']
    f[v < 0] |= F['NEGATIVE']
    pos = v[np.isfinite(v) & (v > 0)]
    if len(pos) < 500:
        return f, set()
    med_all = float(np.median(pos))
    same = np.r_[False, (np.diff(v) == 0)] & (v != 0) & np.isfinite(v)
    f[run_len(same) >= 23] |= F['FLATLINE']
    d1 = np.r_[np.nan, np.diff(v)]
    lin = np.r_[False, np.abs(np.diff(d1)) <= 1e-4 * max(med_all, 1e-3)] & (np.abs(d1) > 1e-6 * max(med_all, 1e-3))
    f[run_len(lin & np.isfinite(d1)) >= 11] |= F['INTERP']
    s = pd.Series(v, index=index)
    good = s.where(s > 0)
    med24 = good.rolling(25, center=True, min_periods=6).median()
    r = good - med24
    scale = r.abs().rolling(24 * 7 + 1, center=True, min_periods=48).median() * 1.4826 + 0.02 * med_all
    spike = (r > 8 * scale) & (good > 2.0 * med24.clip(lower=0.05 * med_all))
    p999 = float(np.nanpercentile(pos, 99.9))
    if p999 > 0:
        spike |= good > 10 * p999
    f[spike.fillna(False).to_numpy()] |= F['SPIKE']
    dm = good.where((f & (F['SPIKE'] | F['FLATLINE'] | F['INTERP'])) == 0).resample('D').median()
    d14 = dm.rolling(14, center=True, min_periods=7).median()
    G = float(dm.median())
    days = set()
    if G > 0:
        shift = ((d14 > 3 * G) | (d14 < G / 3)) & dm.notna()
        sd = shift[shift].index
        if len(sd):
            hrs = index.floor('D').isin(sd)
            f[hrs & ~zero] |= F['LEVEL_SHIFT']
            days = set(sd)
    return f, days


def hash_series(index, cols):
    """sha256 over 'UTC ISO timestamp,v1,v2,...' lines (6 decimals, 'nan' for missing)."""
    h = hashlib.sha256()
    ts = index.strftime('%Y-%m-%dT%H:%M:%SZ')
    X = np.column_stack([np.asarray(c, float) for c in cols])
    for t, row in zip(ts, X):
        h.update((t + ',' + ','.join('nan' if not np.isfinite(v) else f'{v:.6f}' for v in row) + '\n').encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------- build
LAT, LON, TZ = 47.38, 8.54, 'Europe/Zurich'    # station coordinates are not published (anonymised IDs): Zurich
MIN_DAYS = 300
AREA_OK, YEAR_OK = (30.0, 2000.0), (1800, 2024)
TYPE = {'single family house': 'residential_single', 'multi family house': 'residential_multi'}


def _zip_csv(zf, name, **kw):
    with zf.open(name) as fh:
        return pd.read_csv(fh, sep=';', **kw)


def truth_table(zf):
    p = _zip_csv(zf, 'heapo_data/reports/protocols.csv')
    p = p[p.Household_ID.notna()].copy()
    p['Household_ID'] = p.Household_ID.astype(int)
    p['_d'] = pd.to_datetime(p.Visit_Date, errors='coerce')
    first = p.sort_values(['Household_ID', 'Visit_Year', '_d', 'Report_ID']).groupby('Household_ID').first()
    pv_any = p.groupby('Household_ID').Building_PVSystem_Available.apply(lambda s: bool((s == True).any()))  # noqa: E712
    h = _zip_csv(zf, 'heapo_data/meta_data/households.csv').set_index('Household_ID')
    t = pd.DataFrame(index=first.index)
    t['report_id'] = first.Report_ID.astype(int)
    t['n_reports'] = p.groupby('Household_ID').size()
    t['area_m2'] = first.Building_FloorAreaHeated_Total.astype(float)
    t['year_built'] = first.Building_ConstructionYear
    t['building_type'] = first.Building_Type
    t['use'] = first.Building_Type.map(TYPE)
    t['hp_type'] = first.HeatPump_Installation_Type
    t['floor_heating'] = first.HeatDistribution_System_FloorHeating
    t['radiators'] = first.HeatDistribution_System_Radiators
    t['pv'] = pv_any | (h.reindex(t.index).Installation_HasPVSystem == True)  # noqa: E712
    t['weather_id'] = h.reindex(t.index).Weather_ID
    return t


def home_series(zf, hid):
    d = _zip_csv(zf, f'heapo_data/smart_meter_data/15min/{hid}.csv', usecols=['Timestamp', 'kWh_received_Total',
                                                                             'kWh_received_HeatPump'])
    if d.kWh_received_Total.notna().sum() == 0:
        return None, 'only a heat-pump meter (no whole-home meter)'
    t = pd.to_datetime(d.Timestamp, utc=True)
    q = pd.Series(d.kWh_received_Total.to_numpy(float), index=t)
    q = q[~q.index.duplicated()].sort_index()
    idx = pd.date_range(q.index[0].floor('h'), q.index[-1].floor('h'), freq='h', tz='UTC')
    hq = q.groupby(q.index.floor('h'))
    s, c = hq.sum(min_count=1), hq.count()
    e = s.where(c == 4).reindex(idx)          # an hour counts only with all four quarter-hours
    return e, None


def best_window(good, index):
    best = None
    for t0 in pd.date_range(index[0].normalize() + pd.offsets.MonthBegin(0), index[-1] - pd.Timedelta('365D'),
                            freq='MS', tz='UTC'):
        sel = (index >= t0) & (index < t0 + pd.Timedelta('365D'))
        n = int(good[sel].sum())
        if best is None or n > best[0]:
            best = (n, t0)
    return best


def build(check=False):
    import pvlib
    os.makedirs(PROC, exist_ok=True)
    zf = zipfile.ZipFile(os.path.join(RAW, 'heapo_data.zip'))
    T = truth_table(zf)
    wx = {}
    for w in sorted(T.weather_id.dropna().unique()):
        x = _zip_csv(zf, f'heapo_data/weather_data/hourly/{w}.csv', usecols=['Timestamp', 'Temperature_avg_hourly'])
        s = pd.Series(x.Temperature_avg_hourly.to_numpy(float), index=pd.to_datetime(x.Timestamp, utc=True))
        wx[w] = s[~s.index.duplicated()].sort_index()
    kept, dropped = {}, {}
    for hid, r in T.iterrows():
        if r.pv:
            dropped[int(hid)] = 'PV (meter reads net import)'
            continue
        e, why = home_series(zf, hid)
        if e is None:
            dropped[int(hid)] = why
            continue
        f, shift_days = flags_electricity(e.to_numpy(), e.index)
        good = (f & BAD_HEAPO) == 0
        bw = best_window(good, e.index)
        if bw is None or bw[0] < MIN_DAYS * 24:
            dropped[int(hid)] = f'< {MIN_DAYS} days of good hours in any 365-day window ({0 if bw is None else bw[0] // 24})'
            continue
        t0 = bw[1]
        idx = pd.date_range(t0, t0 + pd.Timedelta('365D'), freq='h', inclusive='left', tz='UTC')
        nshift = sum(1 for d in shift_days if t0 <= d < t0 + pd.Timedelta('365D'))   # reported, not a filter
        elec = pd.Series(np.where(good, e.to_numpy(), np.nan), index=e.index).reindex(idx).to_numpy(float)
        w = wx[r.weather_id].reindex(idx)
        n_wgap = int(w.isna().sum())
        Tout = w.interpolate(limit=12).bfill().ffill().to_numpy(float)
        ghi = pvlib.location.Location(LAT, LON).get_clearsky(idx + pd.Timedelta('30min'))['ghi'].to_numpy() * 0.6
        pd.DataFrame({'time_utc': idx, 'elec_kW': elec, 'T_out': Tout, 'GHI': ghi}).to_parquet(
            os.path.join(PROC, f'{hid}.parquet'), index=False)
        kept[int(hid)] = dict(window_start=str(t0), good_hours=int(np.isfinite(elec).sum()), level_shift_days=nshift,
                              weather_id=r.weather_id, weather_gap_hours_filled=n_wgap,
                              sha256=hash_series(idx, [elec, Tout, ghi]))
    # truth validity per row
    K = T.loc[list(kept)].copy()
    K['area_ok'] = K.area_m2.between(*AREA_OK)
    K['year_ok'] = pd.to_numeric(K.year_built, errors='coerce').between(*YEAR_OK)
    # split by home: 20 % dev per (building type x heat-pump type) stratum with >= 3 homes, seed 20261117
    rng = np.random.default_rng(SEED)
    K['stratum'] = K.building_type.fillna('unknown') + ' | ' + K.hp_type.fillna('unknown')
    role = {}
    for st in sorted(K.stratum.unique()):
        ids = sorted(K.index[K.stratum == st])
        nd = max(1, int(round(0.2 * len(ids)))) if len(ids) >= 3 else 0
        perm = rng.permutation(ids)
        for i, hid in enumerate(perm):
            role[int(hid)] = 'dev' if i < nd else 'exam'
    units = {}
    for hid in sorted(kept):
        r = K.loc[hid]
        units[str(hid)] = dict(role=role[hid], stratum=r.stratum, **kept[hid],
                               # truth values stay in the raw protocols (re-read at scoring); only what defines the
                               # rows is frozen here: no readings, no floor areas or years
                               truth_source=dict(report_id=int(r.report_id), n_reports=int(r.n_reports),
                                                 **{k: (r[k] if isinstance(r[k], str) else None)
                                                    for k in ('building_type', 'use', 'hp_type')}),
                               area_row=bool(r.area_ok), year_row=bool(r.year_ok))
    ex = K[[role[h] == 'exam' for h in K.index]]
    dv = K[[role[h] == 'dev' for h in K.index]]
    minority = int(min((ex.use == 'residential_single').sum(), (ex.use == 'residential_multi').sum()))
    src = json.load(open(os.path.join(RAW, 'SOURCES.json')))
    doc = dict(
        frozen='2026-11-17',
        purpose='BuildingDetective exam 2 (HEAPO): Zurich heat-pump homes no version has seen; scored once for v1.0 '
                'and once for any round-2 release, after the round-2 release decision',
        status='exam', source=dict(SOURCE, files=src['files']), seed=SEED,
        truth='first protocol per home (Visit_Year, Visit_Date, Report_ID order): Building_FloorAreaHeated_Total (area '
              'row if 30-2,000 m2), Building_ConstructionYear (year row if 1800-2024), Building_Type, '
              'HeatPump_Installation_Type; values are re-read from reports/protocols.csv at scoring, not stored here',
        split='by home; 20 % dev per (building type x heat-pump type) stratum with >= 3 homes, rest exam; dev only '
              'fits the baselines',
        n=dict(protocol_homes=int(len(T)), kept=len(kept), dropped=len(dropped), dev=int(len(dv)), exam=int(len(ex)),
               exam_area_row=int(ex.area_ok.sum()), exam_year_row=int(ex.year_ok.sum()),
               exam_single=int((ex.use == 'residential_single').sum()),
               exam_multi=int((ex.use == 'residential_multi').sum()),
               exam_hp_type=ex.hp_type.value_counts().to_dict()),
        dropped_units={str(k): v for k, v in sorted(dropped.items())},
        dropped_by_reason=pd.Series([v.split(' (')[0] if v.startswith('PV') or v.startswith('only') else
                                     ('short window' if v.startswith('<') else 'level shift')
                                     for v in dropped.values()]).value_counts().to_dict(),
        input=dict(meters="{'elec': hourly kW}: kWh_received_Total summed over the hour (all four quarter-hours "
                          "present), bad hours (BDG2 electricity flags) NaN; HEAPO has no fuel / district meters",
                   T='hourly mean air temperature of the home\'s HEAPO station (Weather_ID); gaps <= 12 h '
                     'interpolated, then filled from the nearest value (as realdata.hue_cases)',
                   GHI='clear-sky GHI x 0.6 (pvlib, Ineichen, 47.38 N 8.54 E, at the half hour): the stations log '
                       'sunshine duration, not irradiance',
                   loc='window index converted to Europe/Zurich', lat=LAT,
                   window='most complete 365-day window starting on a month boundary (UTC), >= 300 days of good '
                          'hours',
                   flags='BDG2 electricity flags (missing, zero, zero runs >= 24 h, negative, spikes, flatlines >= '
                         '24 h, linear-fill runs >= 12 h) make an hour bad; the BDG2 level-shift rule is NOT applied '
                         '(it flags the heating season of heat-pump homes); its day count is reported per home',
                   monthly='Detective.read_monthly_from_hourly(meters, T, loc, lat)',
                   annual="read_annual({'elec': nanmean(elec_kW) * 8760}, HDD, CDD), HDD/CDD = sum over local days "
                          "of max(18.33 - Tday, 0) / max(Tday - 18.33, 0) (as realfeats.bdg2)"),
        output_gap='The Detective has no air- vs ground-source output and no emitter output (floor heating / '
                   'radiators): recorded, not scored. Every home has a heat pump, so the heating class has no '
                   'negative case.',
        rows=[
            dict(id='area_mdape_<mode>', modes=['hourly', 'monthly', 'annual'],
                 metric='heated floor area, median abs. % error', better='lower', unit_set='exam homes with area_row'),
            dict(id='year_mae_<mode>', modes=['hourly', 'monthly', 'annual'],
                 metric='construction year, mean abs. error (yr)', better='lower', unit_set='exam homes with year_row'),
            dict(id='type_acc_<mode>', modes=['hourly', 'monthly', 'annual'],
                 metric='single vs multi-family: accuracy of argmax(P(residential_single), P(residential_multi))',
                 better='higher', unit_set='exam homes with use single or multi',
                 note=('info only, no verdict: the exam has %d multi-family homes (< 10)' % minority) if minority < 10
                 else 'verdict row'),
            dict(id='heatpump_info_<mode>', modes=['hourly', 'monthly', 'annual'],
                 metric='share of exam homes with heat_pump as the top heating class; median P(heat_pump)',
                 better='higher', unit_set='all exam homes', note='info only (every home has a heat pump)'),
        ],
        model_input="released weights/ through Detective.read_hourly(meters, T, GHI, loc, lat), "
                    "read_monthly_from_hourly and read_annual, from work/exam2/heapo/proc/<home>.parquet; point "
                    "estimates are the decoded value (floor_area_m2[0], year_built[0]). A round-2 release is "
                    "scored the same way with its own weights. No refit on these data.",
        baselines=[
            dict(name='annual elec / dev median kWh per m2 of the TRUE building type', kind='classical',
                 rows=['area'], note='given the true type, which the Detective is not'),
            dict(name='annual elec / dev median kWh per m2 (all dev homes)', kind='classical', rows=['area']),
            dict(name='dev-fitted line log10 A = a + b log10 annual kWh (least squares on dev)', kind='classical',
                 rows=['area']),
            dict(name='dev median heated floor area', kind='dumb', rows=['area']),
            dict(name='dev median construction year', kind='dumb', rows=['year']),
            dict(name='dev-fitted line year = a + b log10 annual kWh', kind='classical', rows=['year']),
            dict(name='dev majority building type', kind='dumb', rows=['type']),
        ],
        baseline_inputs='annual kWh = nanmean(elec_kW) * 8760 over the same window the Detective reads; dev rows '
                        'with area_row / year_row only',
        interval=dict(method='95 % percentile bootstrap over exam homes, paired', resamples=2000, seed=SEED,
                      statistic="model metric minus the best baseline's metric on the same resample (best = better "
                                'point value on the full exam row)',
                      verdict='win if entirely on the good side of zero, loss if entirely on the bad side, tie if '
                              'it spans zero; n homes stated'),
        scored_once_for=['BuildingDetective v1.0', 'any round-2 release'],
        units=units,
    )
    from hashlib import sha256
    doc['split_sha256'] = sha256(json.dumps(doc['units'], sort_keys=True).encode('utf-8')).hexdigest()
    if os.path.exists(SPLIT):
        old = json.load(open(SPLIT, encoding='utf-8'))
        if check:
            print('rebuild matches frozen split' if old['split_sha256'] == doc['split_sha256'] else 'REBUILD DIFFERS')
            return
        raise SystemExit(f'{SPLIT} exists and is frozen; not rewriting it (use --check).')
    with open(SPLIT, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(doc, fh, indent=1, default=str)
        fh.write('\n')
    print(json.dumps(doc['n'], default=str), json.dumps(doc['dropped_by_reason']))
    print(f'wrote {SPLIT} sha256 {file_hash(SPLIT)}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if cmd == 'fetch':
        fetch()
    else:
        build(check='--check' in sys.argv)
