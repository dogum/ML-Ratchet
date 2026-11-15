"""GB frequency events: detection in NESO 1 s data, 1 s windows, metrics, loss attribution, inertia and mix context.

Stages (argv): detect | windows | attribute | build
 detect   : per-minute scan (raw/neso/scan) -> candidates where the largest 10 s fall <= -0.15 Hz (merged within 10 min)
 windows  : 1 s frequency for [-120 s, +480 s] around each candidate via the NESO datastore SQL API
            -> proc/frequency/GB/events/<event_id>.npz (+ .json meta, contract format: t [s from window start], f [Hz])
 attribute: Elexon REMIT messages (unplanned, event start within +-20 min, >= 100 MW unavailable, published within 6 h)
            and Elexon B1610 metered output of those BM units in the settlement period before the event
 build    : proc/frequency/gb_events.parquet (one row per event) with metrics, loss, inertia and generation mix;
            cited label corrections from data/sources/gb_event_label_corrections.csv (loss_MW_v1 = label before them)
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, sys, json, time, glob
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neso_sql import sql
RAW = f'{GRID_ROOT}/data/raw'; PROC = f'{GRID_ROOT}/data/proc'
EV = f'{PROC}/frequency/GB/events'; os.makedirs(EV, exist_ok=True); os.makedirs(f'{RAW}/elexon', exist_ok=True)
RES = json.load(open(f'{RAW}/neso/freq_resources.json'))
THR = -0.15


def minute_utc(m, tz):
    t = pd.to_datetime(m.replace('T', ' '))
    if tz and tz.strip():
        sign = 1 if tz.strip()[0] == '+' else -1
        off = int(tz.strip()[1:3]) * 60 + int(tz.strip()[3:5])
        t = t - pd.Timedelta(minutes=sign * off)
    return t.tz_localize('UTC')


def detect():
    d = pd.concat([pd.read_parquet(f).assign(ym=os.path.basename(f)[:7]) for f in sorted(glob.glob(f'{RAW}/neso/scan/*.parquet'))])
    d = d.dropna(subset=['d10min'])
    c = d[d.d10min <= THR].copy()
    c['t_utc'] = [minute_utc(m, z) for m, z in zip(c.m, c.tz)]
    c = c.sort_values('t_utc')
    grp = (c.t_utc.diff() > pd.Timedelta(minutes=10)).cumsum()
    ev = c.loc[c.groupby(grp).d10min.idxmin() if False else c.groupby(grp).d10min.apply(lambda s: s.idxmin()).values] if False else None
    rows = []
    for _, g in c.groupby(grp):
        r = g.sort_values('d10min').iloc[0]
        rows.append(dict(minute_key=r.m, tz=r.tz, ym=r.ym, t_minute_utc=r.t_utc, d10min=float(r.d10min), fmin_minute=float(g.fmin.min())))
    ev = pd.DataFrame(rows)
    ev.to_parquet(f'{RAW}/neso/candidates.parquet', index=False)
    # coverage: minutes scanned per month
    cov = d.groupby('ym').n.agg(['count', 'sum']).rename(columns={'count': 'minutes', 'sum': 'seconds'})
    cov.to_csv(f'{RAW}/neso/scan_coverage.csv')
    print(len(ev), 'candidates from', len(d), 'minutes')


def windows():
    ev = pd.read_parquet(f'{RAW}/neso/candidates.parquet')
    for _, r in ev.iterrows():
        rid = RES[r.ym]
        sep = 'T' if 'T' in r.minute_key else ' '
        t_loc = pd.to_datetime(r.minute_key.replace('T', ' '))
        a = (t_loc - pd.Timedelta(seconds=120)).strftime(f'%Y-%m-%d{sep}%H:%M:%S'); b = (t_loc + pd.Timedelta(seconds=540)).strftime(f'%Y-%m-%d{sep}%H:%M:%S')
        out = f'{RAW}/neso/win_{r.t_minute_utc:%Y%m%dT%H%M}.parquet'
        if os.path.exists(out): continue
        try:
            rec = sql(f'''SELECT dtm::text AS dtm, f FROM "{rid}" WHERE dtm >= '{a}' AND dtm < '{b}' ORDER BY dtm''', timeout=600)
        except Exception as e:
            print('FAIL', r.minute_key, e, flush=True); continue
        w = pd.DataFrame(rec)
        w.to_parquet(out, index=False); print('win', r.minute_key, len(w), flush=True)


def parse_win(w):
    s = w.dtm.astype(str)
    tz = s.str.extract(r'([+-]\d{4})$')[0]
    base = pd.to_datetime(s.str.replace(r'\s*[+-]\d{4}$', '', regex=True).str.replace('T', ' '))
    off = tz.fillna('+0000').apply(lambda z: (1 if z[0] == '+' else -1) * (int(z[1:3]) * 60 + int(z[3:5])))
    t = (base - pd.to_timedelta(off, unit='m')).dt.tz_localize('UTC')
    return pd.Series(pd.to_numeric(w.f, errors='coerce').values, index=t).sort_index()


def metrics(f):
    """f: 1 s series (UTC index). Onset = first second of the steepest 1-2 s fall region."""
    x = f.resample('1s').mean().interpolate(limit=3)
    d1 = x.diff()
    i = d1.idxmin()                              # steepest 1 s fall
    # walk back to the onset: last second before i with d1 > -0.01 Hz/s
    j = i
    while j - pd.Timedelta(seconds=1) in d1.index and d1.loc[j - pd.Timedelta(seconds=1)] <= -0.01 and (i - j).total_seconds() < 10:
        j = j - pd.Timedelta(seconds=1)
    t0 = j - pd.Timedelta(seconds=1)
    pre = x[t0 - pd.Timedelta(seconds=60): t0 - pd.Timedelta(seconds=1)]
    f0 = float(pre.mean()) if len(pre) else float('nan')
    post = x[t0: t0 + pd.Timedelta(seconds=90)]      # 90 s: catches the 9 Aug 2019 nadir at +76 s (second trip)
    tn = post.idxmin()
    nadir10 = float(x[t0: t0 + pd.Timedelta(seconds=15)].min())
    rocof1 = float(-d1[t0: t0 + pd.Timedelta(seconds=5)].min())
    d2 = x.diff(2) / 2
    rocof2 = float(-d2[t0: t0 + pd.Timedelta(seconds=6)].min())
    f60 = x[t0 + pd.Timedelta(seconds=55): t0 + pd.Timedelta(seconds=65)].mean()
    return dict(t0_utc=t0, f0_Hz=f0, nadir_Hz=float(post.min()), t_nadir_s=float((tn - t0).total_seconds()), nadir_15s_Hz=nadir10,
                rocof_1s_Hz_s=rocof1, rocof_2s_Hz_s=rocof2, f_60s_Hz=float(f60), df_nadir_Hz=float(post.min() - f0),
                n_missing_s=int(x[t0 - pd.Timedelta(seconds=60): t0 + pd.Timedelta(seconds=120)].isna().sum()))


def elexon(url, tries=4):
    for a in range(tries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200: return r.json()
        except Exception as e: pass
        time.sleep(10 * (a + 1))
    return None


def sp_of(t_utc):
    loc = t_utc.tz_convert('Europe/London')
    midnight = loc.normalize()
    sp = int((t_utc - midnight.tz_convert('UTC')).total_seconds() // 1800) + 1
    return str(loc.date()), sp


def attribute():
    rows = []
    for f in sorted(glob.glob(f'{RAW}/neso/win_*.parquet')):
        w = pd.read_parquet(f)
        if len(w) < 200: continue
        x = parse_win(w)
        try: m = metrics(x)
        except Exception as e: print('metric fail', f, e); continue
        t0 = m['t0_utc']
        cache = f'{RAW}/elexon/remit_{t0:%Y%m%dT%H%M%S}.json'
        if os.path.exists(cache): rem = json.load(open(cache))
        else:
            a = (t0 - pd.Timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%MZ'); b = (t0 + pd.Timedelta(hours=6)).strftime('%Y-%m-%dT%H:%MZ')
            js = elexon(f'https://data.elexon.co.uk/bmrs/api/v1/datasets/REMIT?publishDateTimeFrom={a}&publishDateTimeTo={b}&format=json')
            rem = js.get('data', []) if js else None
            json.dump(rem, open(cache, 'w')); time.sleep(1)
        match = []
        if rem:
            for r in rem:
                try:
                    es = pd.Timestamp(r.get('eventStartTime'))
                except Exception: continue
                if r.get('unavailabilityType') != 'Unplanned' or es is pd.NaT: continue
                if abs((es - t0).total_seconds()) > 20 * 60: continue
                lost = (r.get('normalCapacity') or 0) - (r.get('availableCapacity') or 0)
                if lost < 100: continue
                match.append(dict(asset=r.get('assetId'), fuel=r.get('fuelType'), lost_cap=lost, start=r.get('eventStartTime'),
                                  publish=r.get('publishTime'), mrid=r.get('mrid'), rev=r.get('revisionNumber')))
        mm = pd.DataFrame(match)
        if len(mm): mm = mm.sort_values(['asset', 'rev']).drop_duplicates('asset', keep='last')
        # B1610 pre-event output of matched units
        b1610 = {}
        if len(mm):
            d, sp = sp_of(t0 - pd.Timedelta(minutes=30))
            q = '&'.join(f'bmUnit={a}' for a in mm.asset)
            js = elexon(f'https://data.elexon.co.uk/bmrs/api/v1/datasets/B1610?settlementDate={d}&settlementPeriod={sp}&{q}&format=json')
            if js:
                for r in js.get('data', []): b1610[r['bmUnit']] = 2 * float(r['quantity'])
        mm_list = mm.to_dict('records') if len(mm) else []
        for r in mm_list: r['pre_output_MW_b1610'] = b1610.get(r['asset'])
        loss_b = sum(min(r['pre_output_MW_b1610'], r['lost_cap']) for r in mm_list if r['pre_output_MW_b1610'] is not None)
        loss_c = sum(r['lost_cap'] for r in mm_list)
        rows.append(dict(**m, window_file=os.path.basename(f), remit=json.dumps(mm_list, default=str),
                         n_remit=len(mm_list), loss_MW_remit_capacity=loss_c if mm_list else np.nan,
                         loss_MW_b1610=loss_b if any(r['pre_output_MW_b1610'] is not None for r in mm_list) else np.nan))
        print(f[-24:-8], round(m['df_nadir_Hz'], 3), len(mm_list), loss_b, flush=True)
    pd.DataFrame(rows).to_parquet(f'{RAW}/neso/attributed.parquet', index=False)


def interconnectors():
    """Elexon FUELINST (5-min generation by fuel type, incl. each interconnector's flow): an interconnector trip shows as
    an import dropping by >= 300 MW to <= 30% of its previous value between the 5-min slot ending at or before the onset
    and the first slot starting at or after it. Cached per event in raw/elexon/fuelinst_<t0>.json."""
    a = pd.read_parquet(f'{RAW}/neso/attributed.parquet')
    out = {}
    for t0 in a.t0_utc:
        cache = f'{RAW}/elexon/fuelinst_{t0:%Y%m%dT%H%M%S}.json'
        if os.path.exists(cache): js = json.load(open(cache))
        else:
            fa = (t0 - pd.Timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%MZ'); fb = (t0 + pd.Timedelta(minutes=20)).strftime('%Y-%m-%dT%H:%MZ')
            r = elexon(f'https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELINST?publishDateTimeFrom={fa}&publishDateTimeTo={fb}&format=json')
            js = r.get('data', []) if r else None
            json.dump(js, open(cache, 'w')); time.sleep(0.5)
        if not js: continue
        d = pd.DataFrame(js); d = d[d.fuelType.str.startswith('INT')]
        d['start'] = pd.to_datetime(d.startTime, utc=True)
        d = d.drop_duplicates(['start', 'fuelType'], keep='last')
        trips = []
        for ft, g in d.groupby('fuelType'):
            g = g.set_index('start').generation.sort_index()
            before = g[g.index + pd.Timedelta(minutes=5) <= t0]; after = g[g.index >= t0]
            if len(before) and len(after):
                b, a_ = float(before.iloc[-1]), float(after.iloc[0])
                if b - a_ >= 300 and a_ <= 0.3 * b: trips.append((ft, b - a_))
        if trips: out[str(t0)] = trips
    json.dump(out, open(f'{RAW}/neso/interconnector_trips.json', 'w'), indent=1)
    print(len(out), 'events with an interconnector trip in FUELINST')


REPORTED = {  # officially reported losses (NESO/Ofgem); value used as the loss label
    '2019-08-09T15:52': dict(loss_MW=1131.0, loss_MW_total=1878.0, loss_source='NESO (ESO) technical report 6 Sep 2019 / Ofgem report Jan 2020',
                             note='~150 MW embedded (vector shift) + 737 MW Hornsea + 244 MW Little Barford ST at 15:52:33 UTC; '
                                  '+350 MW embedded RoCoF trips within ~1 s (1,481 MW); +210 MW LB GT1a at 15:53:31 (1,691 MW); '
                                  'LFDD ~931 MW demand at 48.8 Hz; +187 MW LB GT1b (1,878 MW total).')}


def build():
    a = pd.read_parquet(f'{RAW}/neso/attributed.parquet')
    a = a.sort_values('t0_utc').drop_duplicates('t0_utc')
    a['event_id'] = 'GB_' + a.t0_utc.dt.strftime('%Y%m%dT%H%M%S')
    # loss label
    # Label rule (decided from the REMIT-vs-onset timing histogram, which peaks at -2..0 min; matches 10-20 min away
    # are mostly coincidental): keep REMIT unplanned outages that start within [-5, +8] min of the onset and whose BM unit
    # was producing >= 50 MW in the settlement period before (B1610), or has no B1610 record (e.g. interconnectors).
    # loss = sum over kept units of min(B1610 output, capacity lost) (capacity lost if no B1610).
    # confidence: 'medium' if every kept match starts within 2 min and has B1610 output, else 'low'.
    ic = json.load(open(f'{RAW}/neso/interconnector_trips.json')) if os.path.exists(f'{RAW}/neso/interconnector_trips.json') else {}
    L, C, S, K = [], [], [], []
    for _, r in a.iterrows():
        keep = []
        has_ic = str(r.t0_utc) in ic
        for m in json.loads(r.remit):
            dt = (pd.Timestamp(m['start']) - r.t0_utc).total_seconds() / 60
            out = m.get('pre_output_MW_b1610')
            if has_ic and out is None: continue        # interconnector BMU: counted once, from FUELINST below
            if -5 <= dt <= 8 and (out is None or out >= 50):
                keep.append((dt, out, m['lost_cap'], m['asset']))
        if keep:
            L.append(sum(min(o, c) if o is not None else c for _, o, c, _ in keep))
            med = all(abs(dt) <= 2 and o is not None for dt, o, _, _ in keep)
            C.append('medium' if med else 'low')
            S.append('Elexon REMIT unplanned outage(s) ' + ', '.join(x[3] for x in keep) + ' + B1610 output (SP before)')
            K.append(len(keep))
        else:
            L.append(np.nan); C.append(None); S.append(None); K.append(0)
    a['loss_MW'] = L; a['loss_confidence'] = C; a['loss_source'] = S; a['n_remit_kept'] = K
    # interconnector trips (FUELINST): add the import lost; REMIT messages without B1610 (interconnectors) were skipped above
    for i, r in a.iterrows():
        tr = ic.get(str(r.t0_utc))
        if not tr: continue
        add = sum(x[1] for x in tr); names = ', '.join(x[0] for x in tr)
        base = 0.0 if pd.isna(r.loss_MW) else r.loss_MW
        a.at[i, 'loss_MW'] = base + add
        a.at[i, 'loss_source'] = ((r.loss_source + ' + ') if isinstance(r.loss_source, str) else '') + f'Elexon FUELINST interconnector import lost ({names})'
        a.at[i, 'loss_confidence'] = 'medium'
    a['loss_note'] = None; a['loss_MW_total'] = np.nan
    for k, v in REPORTED.items():
        m = a.t0_utc.dt.strftime('%Y-%m-%dT%H:%M') == k
        for c, val in v.items():
            a.loc[m, 'loss_MW' if c == 'loss_MW' else ('loss_note' if c == 'note' else c)] = val
        a.loc[m, 'loss_confidence'] = 'high (official report)'
    # cited label corrections (data/sources/gb_event_label_corrections.csv; the same function griddata.events applies to
    # the shipped v1.0 file). loss_MW_v1 keeps the label the rules above give.
    sys.path.insert(0, GRID_CODE)
    from griddata import apply_label_corrections
    a = apply_label_corrections(a)
    # inertia (NESO outturn, GVA.s) per settlement period
    ine = pd.read_parquet(f'{RAW}/neso/inertia.parquet')
    ine['date'] = ine['Settlement Date'].astype(str).str[:10]; ine['sp'] = pd.to_numeric(ine['Settlement Period']).astype(int)
    ine = ine.drop_duplicates(['date', 'sp']).set_index(['date', 'sp'])
    a['inertia_outturn_GVAs'] = [pd.to_numeric(ine['Outturn Inertia'].get(sp_of(t), np.nan), errors='coerce') for t in a.t0_utc]
    a['inertia_market_GVAs'] = [pd.to_numeric(ine['Market Provided Inertia'].get(sp_of(t), np.nan), errors='coerce') for t in a.t0_utc]
    # generation mix (NESO historic mix, half-hourly; DATETIME = period start, UTC)
    gm = pd.read_parquet(f'{RAW}/neso/genmix.parquet')
    gm['t'] = pd.to_datetime(gm.DATETIME, utc=True); gm = gm.drop_duplicates('t').set_index('t').sort_index()
    cols = ['GAS', 'COAL', 'NUCLEAR', 'WIND', 'WIND_EMB', 'HYDRO', 'IMPORTS', 'BIOMASS', 'OTHER', 'SOLAR', 'STORAGE', 'GENERATION']
    for c in cols: gm[c] = pd.to_numeric(gm[c], errors='coerce')
    mix = gm[cols].reindex(a.t0_utc.dt.floor('30min').values)
    for c in cols: a[f'mix_{c.lower()}_MW'] = mix[c].values
    a['year'] = a.t0_utc.dt.year
    for _, r in a.iterrows():
        w = parse_win(pd.read_parquet(f'{RAW}/neso/{r.window_file}'))
        t = (w.index - w.index[0]).total_seconds().values.astype('float64')
        np.savez_compressed(f'{EV}/{r.event_id}.npz', t=t, f=w.values.astype('float64'))
        json.dump(dict(region='GB', nominal_hz=50.0, start_utc=str(w.index[0]), event_t0_utc=str(r.t0_utc),
                       source='NESO system frequency data (1 s), via CKAN datastore API', event_id=r.event_id),
                  open(f'{EV}/{r.event_id}.json', 'w'))
    a.to_parquet(f'{PROC}/frequency/gb_events.parquet', index=False)
    print(len(a), 'events;', int(a.loss_MW.notna().sum()), 'with a documented loss')
    print(a[['event_id', 'df_nadir_Hz', 'rocof_1s_Hz_s', 'loss_MW', 'inertia_outturn_GVAs']].sort_values('df_nadir_Hz').head(25).to_string())


if __name__ == '__main__':
    for s in sys.argv[1:]: globals()[s]()
