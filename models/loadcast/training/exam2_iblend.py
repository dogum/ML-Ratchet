"""LoadCast exam 2 (frozen 17 Nov 2026, before any round-2 rung is judged): I-BLEND, IIIT Delhi campus buildings.

    python exam2_iblend.py fetch    # I-BLEND energy + calendar (figshare, CC0), ISD-Lite Delhi IGI, NASA POWER hourly
    python exam2_iblend.py build    # hourly series, QC, year pair per building, hash; writes training/split_exam2_iblend.json

Scores nothing. The plan is the NOTEBOOK.md entry of 2026-11-17 "Fresh exam frozen before round-2 rungs: I-BLEND".
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths as _P  # noqa: E402

SEED = 20261117
OUT = os.path.join(_P.WORK, 'exam2', 'iblend')
RAW, PROC = os.path.join(OUT, 'raw'), os.path.join(OUT, 'proc')
SPLIT = os.path.join(HERE, 'split_exam2_iblend.json')
UA = 'Mozilla/5.0 (research data client)'
LAT, LON = 28.5459, 77.2732          # NASA POWER request point (same 0.5 x 0.625 deg cell as the campus)
MODEL_LAT = 28.5463                  # campus latitude in the I-BLEND paper (28.5463 N, 77.2732 E, 226 m)
TZ = 'Asia/Kolkata'
FILES = [
    dict(name='energy_dataset.zip', url='https://ndownloader.figshare.com/files/10797959',
         md5='754d5c8549fd4907a7566b3b82c849ca', size=589226996, doi='10.6084/m9.figshare.6007637.v1', licence='CC0 1.0'),
    dict(name='iiitd_calender_schedule.zip', url='https://ndownloader.figshare.com/files/10797956',
         md5='efdcc0a64ea1e412447c89a251e42a9a', size=8206, doi='10.6084/m9.figshare.6007634.v1', licence='CC0 1.0'),
    dict(name='weather_comparison.zip', url='https://ndownloader.figshare.com/files/12470675',
         md5='8be6cdb6ed18413336fd56d91b358a10', size=47862, doi='10.6084/m9.figshare.6007649.v1', licence='CC0 1.0'),
]
# NCEI's /pub/data/noaa/isd-lite/ now redirects to NOAA's open-data bucket (an HTML explorer page, not the file)
ISD = 'https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/{y}/421810-99999-{y}.gz'
POWER = ('https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,T2MDEW,WS10M,ALLSKY_SFC_SW_DWN'
         '&community=RE&longitude={lon}&latitude={lat}&start={y}0101&end={y}1231&format=CSV&time-standard=UTC')
YEARS = range(2013, 2019)
SOURCE = dict(dataset='I-BLEND', version='figshare collection 3893581 v1 (15 Feb 2019)',
              doi='10.6084/m9.figshare.c.3893581.v1', paper='Rashid, H. et al. (2019) Sci. Data 6:190015, '
              'doi:10.1038/sdata.2019.15', licence='CC0 1.0',
              weather='NOAA ISD-Lite 421810-99999 (Delhi IGI; NOAA data, no restrictions) and NASA POWER hourly '
                      '(public, no registration)')


def file_hash(path, algo='sha256'):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _curl(url, path):
    subprocess.run(['curl', '-L', '-sS', '--fail', '--retry', '5', '-A', UA, '-C', '-', '-o', path, url], check=True)


def fetch():
    os.makedirs(RAW, exist_ok=True)
    rec = []
    for f in FILES:
        p = os.path.join(RAW, f['name'])
        if not (os.path.exists(p) and os.path.getsize(p) == f['size']):
            _curl(f['url'], p)
        md5 = file_hash(p, 'md5')
        if md5 != f['md5']:
            raise SystemExit(f"{f['name']}: md5 {md5} != published {f['md5']}")
        rec.append(dict(name=f['name'], url=f['url'], doi=f['doi'], licence=f['licence'], size=os.path.getsize(p),
                        md5=md5, sha256=file_hash(p), downloaded=date.today().isoformat()))
    for y in YEARS:
        for name, url in ((f'isd_421810_{y}.gz', ISD.format(y=y)),
                          (f'power_{y}.csv', POWER.format(y=y, lat=LAT, lon=LON))):
            p = os.path.join(RAW, name)
            if os.path.exists(p) and open(p, 'rb').read(15).lstrip().lower().startswith(b'<!doctype'):
                os.remove(p)                       # an HTML page saved under a data name
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                try:
                    subprocess.run(['curl', '-L', '-sS', '--fail', '--retry', '5', '-A', UA, '-o', p, url], check=True)
                except subprocess.CalledProcessError as e:
                    rec.append(dict(name=name, url=url, error=str(e)))
                    continue
            rec.append(dict(name=name, url=url, size=os.path.getsize(p), sha256=file_hash(p),
                            downloaded=date.today().isoformat()))
    json.dump(dict(SOURCE, files=rec), open(os.path.join(RAW, 'SOURCES.json'), 'w'), indent=1)
    print(json.dumps(rec, indent=1))


# ----------------------------------------------------------------------------- buildings (I-BLEND paper, Table 2)
FT2 = 0.09290304
BUILDINGS = {   # id: (columns in all_buildings_power.csv summed per minute, paper name, LoadCast use, total area ft2)
    'academic': (['Academic'], 'Academics', 'education', 61308),
    'lecture': (['Lecture'], 'Lecture', 'education', 17548),
    'library': (['Library'], 'Library', 'education', 26401),
    'dining': (['Mess'], 'Dining', 'food', 50191),
    'facilities': (['Facilities'], 'Facilities', 'other', 9769),
    'boys_hostel': (['Boys_main', 'Boys_backup'], 'Boys Dormitory', 'lodging', 72745),
    'girls_hostel': (['Girls_main', 'Girls_backup'], 'Girls Dormitory', 'lodging', 38126),
}

# ----------------------------------------------------------------------------- BDG2 electricity flags (copied rules)
# Same rules as shared/building/data/scripts/build_bdg2.py flags_for (electricity), on any hourly UTC index.
F = dict(MISSING=1, ZERO=2, ZERO_RUN=4, NEGATIVE=8, SPIKE=16, FLATLINE=32, INTERP=64, LEVEL_SHIFT=128)
BAD = F['MISSING'] | F['NEGATIVE'] | F['SPIKE'] | F['FLATLINE'] | F['INTERP'] | F['ZERO'] | F['ZERO_RUN'] | F['LEVEL_SHIFT']


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


# ----------------------------------------------------------------------------- weather
def read_isd(path):
    import gzip
    rows = []
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            p = line.split()
            if len(p) < 9:
                continue
            rows.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[8])))
    d = pd.DataFrame(rows, columns=['y', 'm', 'd', 'h', 'T', 'dew', 'ws'])
    t = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d, hour=d.h), utc=True)
    out = pd.DataFrame({'T_out': d['T'].to_numpy() / 10.0, 'dewpoint': d.dew.to_numpy() / 10.0,
                        'wind_speed': d.ws.to_numpy() / 10.0}, index=pd.DatetimeIndex(t))
    return out.mask(out <= -999.0)


def read_power(path):
    txt = open(path, encoding='utf-8').read()
    import io
    x = pd.read_csv(io.StringIO(txt[txt.index('YEAR,MO'):])).replace(-999.0, np.nan)
    t = pd.to_datetime(dict(year=x.YEAR, month=x.MO, day=x.DY, hour=x.HR), utc=True)
    return pd.DataFrame({'T2M': x.T2M.to_numpy(), 'T2MDEW': x.T2MDEW.to_numpy(), 'WS10M': x.WS10M.to_numpy(),
                         'GHI': x.ALLSKY_SFC_SW_DWN.to_numpy()}, index=pd.DatetimeIndex(t))


def weather(idx):
    isd = pd.concat([read_isd(os.path.join(RAW, f'isd_421810_{y}.gz')) for y in YEARS
                     if os.path.exists(os.path.join(RAW, f'isd_421810_{y}.gz'))])
    isd = isd[~isd.index.duplicated()].reindex(idx)
    pw = pd.concat([read_power(os.path.join(RAW, f'power_{y}.csv')) for y in YEARS])
    pw = pw[~pw.index.duplicated()].reindex(idx)
    w = pd.DataFrame(index=idx)
    src = {}
    for c, pc in (('T_out', 'T2M'), ('dewpoint', 'T2MDEW'), ('wind_speed', 'WS10M')):
        x = isd[c].interpolate(limit=6, limit_area='inside')
        src[c] = dict(isd=float(isd[c].notna().mean()), isd_or_interp=float(x.notna().mean()))
        w[c] = x.fillna(pw[pc])
    w['GHI'] = pw.GHI
    both = isd.T_out.notna() & pw.T2M.notna()
    src['nasa_minus_isd_T_bias_C'] = round(float((pw.T2M - isd.T_out)[both].mean()), 2)
    src['nasa_vs_isd_T_mae_C'] = round(float((pw.T2M - isd.T_out)[both].abs().mean()), 2)
    return w, src


# ----------------------------------------------------------------------------- build
MIN_MIN_PER_HOUR, GOOD_FRAC, MAX_SHIFT_DAYS, EUI_OK, RATIO_OK = 45, 0.85, 30, (10.0, 2000.0), (0.4, 2.5)


def hourly_buildings():
    import zipfile
    zf = zipfile.ZipFile(os.path.join(RAW, 'energy_dataset.zip'))
    with zf.open('energy_dataset/all_buildings_power.csv') as fh:
        x = pd.read_csv(fh)
    t = pd.to_datetime(x.timestamp, unit='s', utc=True)
    dup = int(t.duplicated().sum())
    x.index = t
    x = x[~x.index.duplicated()].drop(columns='timestamp').sort_index()
    out, info = {}, {}
    for b, (cols, *_rest) in BUILDINGS.items():
        p = x[cols].sum(axis=1, min_count=len(cols)).where(x[cols].notna().all(axis=1))   # W, every part present
        g = p.groupby(p.index.floor('h'))
        kw = (g.mean() / 1000.0).where(g.count() >= MIN_MIN_PER_HOUR)
        out[b] = kw
        info[b] = dict(first_minute=str(p.first_valid_index()), last_minute=str(p.last_valid_index()),
                       minutes_present=int(p.notna().sum()))
    idx = pd.date_range(min(s.index.min() for s in out.values()), max(s.index.max() for s in out.values()),
                        freq='h', tz='UTC')
    return {b: s.reindex(idx) for b, s in out.items()}, idx, info, dup


def build(check=False):
    os.makedirs(PROC, exist_ok=True)
    H, idx, minfo, dup = hourly_buildings()
    W, wsrc = weather(idx)
    local_year = idx.tz_convert(TZ).year
    local_doy = idx.tz_convert(TZ).dayofyear
    units, dropped, yearly = {}, {}, {}
    for b, (cols, pname, use, ft2) in BUILDINGS.items():
        area = ft2 * FT2
        x = H[b].to_numpy(float)
        f, shift_days = flags_electricity(x, idx)
        good = (f & BAD) == 0
        sdays = pd.DatetimeIndex(sorted(shift_days))
        sd_year = pd.Series(1, index=sdays.tz_convert(TZ).year if len(sdays) else []).groupby(level=0).sum() \
            if len(sdays) else pd.Series(dtype=int)
        Y = {}
        for y in sorted(set(local_year)):
            sel = local_year == y
            n_hours = int(pd.Timestamp(y, 12, 31).dayofyear) * 24
            gf = float(good[sel].sum() / n_hours)
            eui = float(np.nanmean(np.where(good[sel], x[sel], np.nan)) * 8760 / area) if good[sel].any() else np.nan
            ls = int(sd_year.get(y, 0))
            eui_ok = bool(np.isfinite(eui) and EUI_OK[0] <= eui <= EUI_OK[1])      # value not stored (a reading)
            Y[int(y)] = dict(good_frac=round(gf, 4), level_shift_days=ls, eui_ok=eui_ok,
                             ok=bool(gf >= GOOD_FRAC and ls < MAX_SHIFT_DAYS and eui_ok))
        yearly[b] = Y
        pair = None
        for y in sorted(Y, reverse=True):
            if y - 1 in Y and Y[y]['ok'] and Y[y - 1]['ok']:
                a = pd.Series(np.where(good & (local_year == y - 1), x, np.nan)).groupby(local_doy).mean()
                c = pd.Series(np.where(good & (local_year == y), x, np.nan)).groupby(local_doy).mean()
                j = pd.concat([a, c], axis=1).dropna()
                ratio = float(j.iloc[:, 1].mean() / j.iloc[:, 0].mean()) if len(j) else np.nan
                Y[y]['ratio_to_previous_ok'] = bool(RATIO_OK[0] <= ratio <= RATIO_OK[1])
                if RATIO_OK[0] <= ratio <= RATIO_OK[1]:
                    pair = (y - 1, y)
                    break
        if pair is None:
            bad = {y: v for y, v in Y.items() if not v['ok']}
            dropped[b] = 'no consecutive pair of ok local years: ' + '; '.join(
                f"{y}: good {v['good_frac']:.2f}, level-shift days {v['level_shift_days']}, EUI ok {v['eui_ok']}"
                for y, v in bad.items())
            continue
        hy, fy = pair
        span = (local_year == hy) | (local_year == fy)
        wk = np.flatnonzero(span)
        w0, w1 = max(wk[0] - 24 * 8, 0), wk[-1] + 1        # 8 days of weather before history (rolling features)
        sl = slice(w0, w1)
        y_out = np.where(good, x, np.nan)
        frame = pd.DataFrame({'time_utc': idx[sl], 'kWh': y_out[sl], 'role': np.where(
            local_year[sl] == hy, 'history', np.where(local_year[sl] == fy, 'forecast', 'weather_only')),
            'T_out': W.T_out.to_numpy()[sl], 'dewpoint': W.dewpoint.to_numpy()[sl], 'GHI': W.GHI.to_numpy()[sl],
            'wind_speed': W.wind_speed.to_numpy()[sl]})
        frame.to_parquet(os.path.join(PROC, f'{b}.parquet'), index=False)
        hsel, fsel = frame.role == 'history', frame.role == 'forecast'
        units[b] = dict(role='exam', paper_name=pname, use=use, area_m2=round(area, 1), meters_present=['electricity'],
                        meter='electricity', history_year=int(hy), forecast_year=int(fy),
                        history_good_hours=int(frame.kWh[hsel].notna().sum()),
                        forecast_good_hours=int(frame.kWh[fsel].notna().sum()), years=Y,
                        sha256=hash_series(pd.DatetimeIndex(frame.time_utc), [frame.kWh, frame.T_out, frame.dewpoint,
                                                                              frame.GHI, frame.wind_speed]))
    src = json.load(open(os.path.join(RAW, 'SOURCES.json')))
    doc = dict(
        frozen='2026-11-17',
        purpose='LoadCast exam 2 (I-BLEND): Delhi campus buildings no version has seen; scored once for v1.0 and once '
                'for any round-2 release, after the round-2 release decision',
        status='exam', source=dict(SOURCE, files=[f for f in src['files'] if 'error' not in f]), seed=SEED,
        split='by building; every building that passes QC is exam (7 at most, too few to split). Nothing is fitted '
              'on them: dev guesses and the twin come from BDG2 dev; history baselines from each building\'s own '
              'history year.',
        buildings_source='I-BLEND paper Table 2 (total area, ft2 -> m2); hostels = mains + UPS backup meters',
        series=dict(meter='1-min active power (W, all_buildings_power.csv; UNIX time) -> UTC hour-beginning mean kW '
                          f'(= kWh); hours with < {MIN_MIN_PER_HOUR} minutes missing; a hostel minute counts only if '
                          'both its meters are present',
                    duplicate_minutes_dropped=dup, minutes=minfo),
        qc=dict(flags='BDG2 electricity flags (missing, zero, zero runs >= 24 h, negative, spikes, flatlines >= 24 h, '
                      'linear-fill runs >= 12 h, level-shift days) on the whole record; bad hours are NaN and not '
                      'graded',
                year_ok=f'>= {GOOD_FRAC:.0%} good hours in the local year, < {MAX_SHIFT_DAYS} level-shift days, EUI '
                        f'{EUI_OK[0]:g}-{EUI_OK[1]:g} kWh/m2',
                pair=f'latest consecutive local years (Y, Y+1) both ok with Y+1/Y mean-load ratio on the same days '
                     f'within {RATIO_OK[0]}-{RATIO_OK[1]}; history = Y, forecast = Y+1'),
        weather=dict(how='T_out, dewpoint, wind_speed from NOAA ISD-Lite 421810-99999 (Delhi IGI, 14.8 km from campus); '
                         'gaps <= 6 h interpolated, longer gaps from NASA POWER hourly (MERRA-2) at the campus; GHI = '
                         'NASA POWER ALLSKY_SFC_SW_DWN (CERES); hour labels as BuildingData (POWER HR = UTC hour)',
                     coverage=wsrc),
        input=dict(call="LoadCast.forecast(weather, tz='Asia/Kolkata', lat=28.5463, use, area_m2, meter='electricity', "
                        "meters_present=['electricity'], start/end = the forecast year's local bounds, history = "
                        "None (cold) | the history year's kWh (warm) | its last 8 or 2 weeks (warm_w8 / warm_w2))",
                   gaps=['holiday table covers US/CA/UK/IE only: Asia/Kolkata falls back to US holidays; the I-BLEND '
                         'semester calendar cannot be given to the model',
                         'Delhi climate (hot, monsoon) is outside BDG2 / BuildingSim training weather',
                         'electricity only; year built unknown (None); hvac and schedule left unknown']),
        dropped_units=dropped,
        units=units,
        rows=[dict(id=f'{mode}_{agg}', metric=f'electricity {agg} CV(RMSE), median over buildings', better='lower',
                   mode=mode, unit_set='exam buildings, good forecast-year hours')
              for mode, agg in [('cold', 'hourly'), ('warm_year', 'hourly'), ('warm_w8', 'hourly'),
                                ('warm_w2', 'hourly'), ('cold', 'daily'), ('cold', 'monthly'),
                                ('warm_year', 'daily'), ('warm_year', 'monthly')]],
        metric_notes='ASHRAE G14 CV(RMSE); LoadCast and dev guesses p = 0, fitted baselines n - p. NMBE and 90 % '
                     'coverage reported without verdict. Daily and monthly rows on totals of good hours.',
        model_input='released loadcast_weights.npz with its shipped widening, no refit on these data; a round-2 '
                    'release is scored the same way with its own weights',
        baselines=[
            dict(name='dumb dev median-intensity x median time-of-week shape (BDG2 dev, per use, electricity)',
                 kind='dumb', rows=['cold_*', 'warm_w8_hourly', 'warm_w2_hourly']),
            dict(name='dev-calibrated BuildingSim twin (spec from I-BLEND metadata only; dev knobs and dev median a, b)',
                 kind='physics', rows=['cold_*', 'warm_w8_hourly', 'warm_w2_hourly'],
                 note='only if the integrity-pass twin builder accepts a non-BDG2 spec; otherwise the row says not run'),
            dict(name='dev TOWT-shape guess (dev median intensity x median dev TOWT shape)', kind='classical',
                 rows=['cold_*', 'warm_w8_hourly', 'warm_w2_hourly']),
            dict(name='TOWT fitted on the history year (BuildingData code)', kind='classical', rows=['warm_year_*']),
            dict(name='change-point model fitted on the history year (BuildingData G14 code; daily / monthly '
                      'change-point models for the daily / monthly rows)', kind='classical', rows=['warm_year_*']),
            dict(name="LoadCast's ridge TOWT on the same history (towt_lc, the network's anchor)", kind='classical',
                 rows=['warm_*']),
        ],
        interval=dict(method='95 % percentile bootstrap over exam buildings, paired', resamples=2000, seed=SEED,
                      statistic='median(LoadCast) minus median(best baseline) on the same resample (best = lowest '
                                'median on the full exam row)',
                      verdict='win if the interval is below zero, loss if above, tie if it spans zero; n stated'),
        scored_once_for=['LoadCast v1.0', 'any round-2 release'],
    )
    doc['split_sha256'] = hashlib.sha256(json.dumps(doc['units'], sort_keys=True, default=str).encode()).hexdigest()
    if os.path.exists(SPLIT):
        old = json.load(open(SPLIT, encoding='utf-8'))
        if check:
            print('rebuild matches frozen split' if old['split_sha256'] == doc['split_sha256'] else 'REBUILD DIFFERS')
            return
        raise SystemExit(f'{SPLIT} exists and is frozen; not rewriting it (use --check).')
    with open(SPLIT, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(doc, fh, indent=1, default=str)
        fh.write('\n')
    print(json.dumps({b: (u['history_year'], u['forecast_year'], u['forecast_good_hours']) for b, u in units.items()}))
    print('dropped:', json.dumps(dropped, indent=1))
    print('weather:', json.dumps(wsrc))
    print(f'wrote {SPLIT} sha256 {file_hash(SPLIT)}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if cmd == 'fetch':
        fetch()
    else:
        build(check='--check' in sys.argv)
