"""Freeze the Wildfire splits (seed 20261118) -> proc/splits.json, proc/fires.parquet, proc/guard/*.

Unit = the physical fire (an MTBS event). Every source (FEDS-MTBS 12-hourly perimeters, GOFER hourly perimeters,
WildfireSpreadTS daily VIIRS rasters) inherits the split of the MTBS fire(s) it matches.
Run once. Re-running rewrites only if splits.json does not exist (use --force to rebuild before Phase 2 starts).
"""
import os, sys, json, glob, hashlib, datetime as dt
import numpy as np, pandas as pd, pyarrow.parquet as pq, shapely
from pyproj import Transformer
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root

SEED = 20261118
P = f'{_DATA}/data/proc'
OUT = f'{P}/splits.json'
REGION = {'CA': 'CA'}
for s in ['OR', 'WA']: REGION[s] = 'NW'
for s in ['ID', 'MT', 'ND']: REGION[s] = 'NR'
for s in ['NV', 'UT']: REGION[s] = 'GB'
for s in ['AZ', 'NM']: REGION[s] = 'SW'
for s in ['CO', 'WY', 'SD', 'NE', 'KS']: REGION[s] = 'RM'
for s in ['TX', 'OK']: REGION[s] = 'SP'
for s in ['FL', 'GA', 'AL', 'MS', 'LA', 'AR', 'SC', 'NC', 'TN', 'KY', 'VA', 'WV']: REGION[s] = 'SE'
SIZE_EDGES = [0, 5000, 50000, 1e12]          # acres (MTBS BurnBndAc)
SIZE_NAMES = ['1k-5k ac', '5k-50k ac', '>50k ac']
N_DEV, N_DEV_GOFER = 40, 4
DEV_YEARS = (2017, 2021)
RECENT_YEARS = (2022, 2024)


def window(r):
    w, h = r.xmax - r.xmin, r.ymax - r.ymin
    b = max(3000.0, 0.3 * max(w, h))
    # snap outward to the 90 m grid aligned with LANDFIRE 30 m edges (x, y = 15 + 90 k)
    x0 = np.floor((r.xmin - b - 15) / 90) * 90 + 15; x1 = np.ceil((r.xmax + b - 15) / 90) * 90 + 15
    y0 = np.floor((r.ymin - b - 15) / 90) * 90 + 15; y1 = np.ceil((r.ymax + b - 15) / 90) * 90 + 15
    return x0, y0, x1, y1


def fire_table():
    f = pd.read_parquet(f'{P}/feds/fires_raw.parquet')
    f['state'] = f.Event_ID.str[:2]
    f['region'] = f.state.map(REGION).fillna(pd.Series(np.where(f.state.isin(['AK', 'HI', 'PR']), 'nonCONUS', 'NE_MW'), index=f.index))
    f['year'] = f.Ig_Date.dt.year
    f['size_class'] = pd.cut(f.BurnBndAc, SIZE_EDGES, labels=SIZE_NAMES).astype(str)
    fg = pd.read_parquet(f'{P}/landfire/fuelgroups.parquet')
    f = f.merge(fg, on='Event_ID', how='left')
    # FEDS progression summary
    rows = []
    for fn in sorted(glob.glob(f'{P}/feds/perims/*.parquet')):
        d = pq.read_table(fn, columns=['Event_ID', 't', 'farea']).to_pandas()
        for ev, g in d.groupby('Event_ID'):
            g = g.sort_values('t'); a = g.farea.values / 1e6  # FEDS farea is m2
            pm = g[g.t.dt.hour == 12]
            grow = np.diff(pm.farea.values / 1e6) if len(pm) > 1 else np.array([])
            rows.append(dict(Event_ID=ev, feds_nsteps=len(g), feds_t0=g.t.iloc[0], feds_t1=g.t.iloc[-1],
                             feds_area_km2=float(a[-1]), feds_npm=len(pm), feds_growth_days=int((grow > 0.1).sum())))
    s = pd.DataFrame(rows)
    f = f.merge(s, on='Event_ID', how='left')
    f['feds_ratio'] = f.feds_area_km2 / f.mtbs_area_km2
    W = np.array([window(r) for r in f.itertuples()])
    f['wx0'], f['wy0'], f['wx1'], f['wy1'] = W.T
    return f


def footprint_overlap(f, geo):
    """Fraction of each fire's window covered by OTHER fires' final perimeters (any year)."""
    gs = shapely.from_wkb([geo[e] for e in f.Event_ID])
    tree = shapely.STRtree(gs)
    boxes = shapely.box(f.wx0, f.wy0, f.wx1, f.wy1)
    out = np.zeros(len(f))
    for i, b in enumerate(boxes):
        idx = [j for j in tree.query(b) if j != i]
        if idx:
            u = shapely.union_all(shapely.intersection(gs[idx], b))
            out[i] = shapely.area(u) / shapely.area(b)
    return out


def systematic(df, n, rng, keys):
    if n <= 0 or len(df) == 0: return []
    df = df.sort_values(keys)
    step = len(df) / n; start = rng.uniform(0, step)
    idx = sorted(set(int(start + k * step) for k in range(n)))
    return list(df.Event_ID.iloc[idx])


def match_gofer(f, geo):
    pr = pd.read_parquet(f'{P}/gofer/fireProg.parquet')
    out = {}
    for (name, yr), g in pr.groupby(['fname', 'fyear']):
        last = shapely.from_wkb(g.sort_values('tUTC').wkb.iloc[-1])
        t0, t1 = g.tUTC.min(), g.tUTC.max()
        c = f[(f.year == yr) & (f.state == 'CA')]
        hits = []
        for r in c.itertuples():
            gg = shapely.from_wkb(geo[r.Event_ID])
            inter = shapely.area(shapely.intersection(gg, last))
            if inter > 0.2 * min(shapely.area(gg), shapely.area(last)) or (inter > 0 and shapely.area(gg) > 20e6 and inter > 0.5 * shapely.area(gg)):
                hits.append((r.Event_ID, inter / shapely.area(gg)))
        out[f'{name}|{yr}'] = dict(events=[h[0] for h in hits], t0=str(t0), t1=str(t1))
    return out


def match_wsts(f, geo):
    tr = {}
    gs = shapely.from_wkb([geo[e] for e in f.Event_ID]); tree = shapely.STRtree(gs)
    out = {}
    for fn in sorted(glob.glob(f'{P}/wsts/meta/*.json')):
        m = json.load(open(fn))
        if m['epsg'] not in tr: tr[m['epsg']] = Transformer.from_crs(m['epsg'], 5070, always_xy=True)
        a, _, c, _, e, f0 = m['transform']
        xs = [c, c + a * m['width']]; ys = [f0, f0 + e * m['height']]
        X, Y = np.meshgrid(np.linspace(*xs, 9), np.linspace(*ys, 9))
        px, py = tr[m['epsg']].transform(X.ravel(), Y.ravel())
        tile = shapely.convex_hull(shapely.multipoints(np.column_stack([px, py])))
        d0 = pd.Timestamp(m['days'][0]) - pd.Timedelta(days=15); d1 = pd.Timestamp(m['days'][-1])
        idx = tree.query(tile)
        hits = [f.Event_ID.iloc[j] for j in idx if d0 <= f.Ig_Date.iloc[j] <= d1]
        cxy = shapely.centroid(tile)
        prim = None
        if hits:
            dd = [shapely.distance(cxy, gs[f.index.get_loc(f.index[f.Event_ID == h][0])]) for h in hits]
            prim = hits[int(np.argmin(dd))]
        out[m['fire']] = dict(year=m['year'], days=[m['days'][0], m['days'][-1]], events=hits, primary=prim,
                              tile_wkb_hex=shapely.to_wkb(tile, hex=True))
    return out


def main():
    if os.path.exists(OUT) and '--force' not in sys.argv:
        print('splits.json exists; frozen. Use --force only before Phase 2 starts.'); return
    rng = np.random.default_rng(SEED)
    t = pq.read_table(f'{P}/feds/final_perims.parquet')
    geo = dict(zip(t.column('Event_ID').to_pylist(), t.column('wkb').to_pylist()))
    f = fire_table()
    conus = f.region != 'nonCONUS'
    wild = f.Incid_Type.isin(['Wildfire', 'Unknown'])
    f['universe'] = np.where(~conus, 'nonCONUS', np.where(wild, 'spread', 'rx'))
    f['other_fire_overlap'] = 0.0
    sp = f.universe == 'spread'
    f.loc[sp, 'other_fire_overlap'] = footprint_overlap(f[sp].reset_index(drop=True), geo)
    gof = match_gofer(f, geo)
    gofer_events = {e for v in gof.values() for e in v['events']}
    f['gofer'] = f.Event_ID.isin(gofer_events)
    elig = (sp & f.year.between(*DEV_YEARS) & (f.Incid_Type == 'Wildfire') & (f.feds_npm >= 4)
            & (f.feds_growth_days >= 2) & f.feds_ratio.between(0.5, 1.5) & (f.other_fire_overlap < 0.10)
            & f.fuel_group.notna())
    f['dev_eligible'] = elig
    E = f[elig]
    # 1) GOFER dev fires (hourly truth for diurnal calibration): seeded draw, one per year first
    ge = E[E.gofer].sample(frac=1.0, random_state=SEED)
    dev = []
    for y in sorted(ge.year.unique()):
        if len(dev) < N_DEV_GOFER: dev.append(ge[ge.year == y].Event_ID.iloc[0])
    for e in ge.Event_ID:
        if len(dev) < N_DEV_GOFER and e not in dev: dev.append(e)
    # 2) stratified by region; within region systematic over (fuel group, size class, year)
    rest = E[~E.Event_ID.isin(dev)]
    n_rest = N_DEV - len(dev)
    cnt = rest.region.value_counts()
    alloc = {r: max(1, int(round(n_rest * c / len(rest)))) for r, c in cnt.items()}
    while sum(alloc.values()) > n_rest:
        r = max(alloc, key=lambda k: alloc[k] - n_rest * cnt[k] / len(rest)); alloc[r] -= 1
    while sum(alloc.values()) < n_rest:
        r = max(alloc, key=lambda k: n_rest * cnt[k] / len(rest) - alloc[k]); alloc[r] += 1
    for r in sorted(alloc):
        dev += systematic(rest[rest.region == r], alloc[r], rng, ['fuel_group', 'size_class', 'year', 'Event_ID'])
    f['split'] = 'exam'
    f.loc[f.universe == 'rx', 'split'] = 'aux_rx'
    f.loc[f.universe == 'nonCONUS', 'split'] = 'excluded_nonCONUS'
    f.loc[f.Event_ID.isin(dev), 'split'] = 'dev'
    f['exam_period'] = np.where(f.year >= RECENT_YEARS[0], 'recent', np.where(f.year >= 2017, 'main', 'early'))
    f.loc[f.split != 'exam', 'exam_period'] = ''
    # matched products
    for k, v in gof.items():
        ss = set(f.set_index('Event_ID').loc[v['events'], 'split']) if v['events'] else {'exam'}
        v['split'] = 'dev' if ss == {'dev'} else ('exam' if 'dev' not in ss else 'excluded_mixed')
    ws = match_wsts(f, geo)
    for k, v in ws.items():
        ss = set(f.set_index('Event_ID').loc[v['events'], 'split']) if v['events'] else set()
        if not ss: v['split'] = 'exam'; v['note'] = 'no MTBS match (small fire); exam'
        elif ss == {'dev'}: v['split'] = 'dev'
        elif 'dev' in ss: v['split'] = 'excluded_mixed'
        else: v['split'] = 'exam'
    os.makedirs(f'{P}/guard', exist_ok=True)
    f.to_parquet(f'{P}/fires.parquet')
    rules = {
        'unit': 'the fire (MTBS Event_ID). All time steps of a fire share its split; GOFER and WildfireSpreadTS fires '
                'inherit the split of the MTBS fires they overlap (tile and date match); mixed matches are excluded.',
        'dev': 'Only dev fires may be used to calibrate, gate, tune or choose anything (incl. simulator calibration, '
               'replay factors, thresholds, sigma widening). aux_rx (prescribed fires) may be used like dev for '
               'simulator calibration only; never scored.',
        'exam': 'Exam fires are scored once, at the end. Exam progression is never a training target, never used to '
                'choose anything. exam_period: early 2012-2016 (LANDFIRE only post-fire, no HRRR), main 2017-2021, '
                'recent 2022-2024 (temporal exam: whole seasons held out).',
        'simulation_inputs': (
            'Simulated training data may use real landscapes and weather ONLY if (a) the landscape window does not '
            'intersect any exam fire footprint (MTBS final perimeter buffered by 2 km; any year, because the landscape '
            'IS the fire site), and (b) weather is not taken from within 50 km of an exam fire footprint during that '
            'fire\'s active window [ignition - 3 d, last FEDS step + 3 d] (fire-day weather). Dev fires\' own '
            'landscapes and weather are free. Random non-fire places at any time are free. Enforce with '
            'firedata.guard.check_landscape(bbox) and firedata.guard.check_weather(bbox, t0, t1), which raise '
            'ExamLeakError. Footprints: proc/guard/exam_footprints.parquet.'),
        'firesize': ('FireSize universe = FPA-FOD 6th ed. CONUS fires 2012-2020 (Short 2022). A FOD fire with an '
                     'MTBS_ID inherits that MTBS fire\'s split. Other FOD fires: dev = seeded 1.5% sample of 2012-2019 '
                     '(list in firesize.dev_fod_ids); exam = all the rest, with exam_period recent = FIRE_YEAR 2020. '
                     'Per-fire table: proc/firesize_split.parquet.'),
    }
    meta = dict(seed=SEED, frozen=dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'), n_dev_target=N_DEV,
                dev_years=DEV_YEARS, recent_years=RECENT_YEARS, region_map=REGION, size_classes=SIZE_NAMES,
                dev_rule='eligible = CONUS Wildfire 2017-2021, >=4 FEDS PM steps, >=2 growth days, FEDS/MTBS area '
                         '0.5-1.5, <10% of window covered by other fires, fuel group known; 4 GOFER fires (seeded, '
                         'one per year first), then per-region n = round(36 x share) (min 1), systematic sampling '
                         'with seeded start over (fuel group, size class, year)',
                window_rule='final MTBS perimeter bbox + max(3 km, 0.3 x max side), snapped to the 90 m grid '
                            'x,y = 15 + 90k (EPSG:5070), aligned with LANDFIRE 30 m cells')
    S = dict(meta=meta, rules=rules,
             fires={k: sorted(f.Event_ID[f.split == k]) for k in ['dev', 'exam', 'aux_rx', 'excluded_nonCONUS']},
             exam_periods={k: sorted(f.Event_ID[(f.split == 'exam') & (f.exam_period == k)]) for k in ['early', 'main', 'recent']},
             gofer=gof, wsts={k: {kk: vv for kk, vv in v.items() if kk != 'tile_wkb_hex'} for k, v in ws.items()})
    # ---- FireSize split (FPA-FOD) ----
    fod = pd.read_parquet(f'{P}/fpafod/fod.parquet', columns=['FOD_ID', 'FIRE_YEAR', 'MTBS_ID'])
    fod = fod[fod.FIRE_YEAR >= 2012].sort_values('FOD_ID').reset_index(drop=True)
    msplit = dict(zip(f.Event_ID, f.split))
    inh = fod.MTBS_ID.map(msplit)
    r2 = np.random.default_rng(SEED + 1).random(len(fod))
    fsplit = np.where(inh.notna(), inh.fillna(''), np.where((fod.FIRE_YEAR <= 2019) & (r2 < 0.015), 'dev', 'exam'))
    fsplit = np.where(fsplit == 'excluded_nonCONUS', 'exam', fsplit)
    fs = pd.DataFrame({'FOD_ID': fod.FOD_ID.values, 'FIRE_YEAR': fod.FIRE_YEAR.values, 'MTBS_ID': fod.MTBS_ID.values,
                       'split': fsplit})
    fs['exam_period'] = np.where(fs.split == 'exam', np.where(fs.FIRE_YEAR == 2020, 'recent', 'main'), '')
    fs.to_parquet(f'{P}/firesize_split.parquet')
    S['firesize'] = dict(universe='FPA-FOD CONUS 2012-2020', n=int(len(fs)), counts=fs.split.value_counts().to_dict(),
                         dev_fod_ids=sorted(int(x) for x in fs.FOD_ID[fs.split == 'dev']))
    # ---- exam footprints for the guard ----
    rows = []
    ex = f[f.split == 'exam']
    for r in ex.itertuples():
        t1 = r.feds_t1 if pd.notna(r.feds_t1) else r.Ig_Date + pd.Timedelta(days=30)
        rows.append(dict(fire_id=r.Event_ID, source='mtbs', wkb=geo[r.Event_ID], t0=r.Ig_Date, t1=pd.Timestamp(t1)))
    tiles = {k: v['tile_wkb_hex'] for k, v in ws.items()}
    for k, v in ws.items():
        if v['split'] == 'exam' and not v['events']:
            rows.append(dict(fire_id=k, source='wsts_tile', wkb=bytes.fromhex(tiles[k]), t0=pd.Timestamp(v['days'][0]),
                             t1=pd.Timestamp(v['days'][1])))
    for k, v in gof.items():
        if v['split'] == 'exam' and not v['events']:
            g = pd.read_parquet(f'{P}/gofer/fireProg.parquet')
            n, y = k.split('|'); g = g[(g.fname == n) & (g.fyear == int(y))].sort_values('tUTC')
            rows.append(dict(fire_id=k, source='gofer', wkb=g.wkb.iloc[-1], t0=g.tUTC.min().tz_localize(None),
                             t1=g.tUTC.max().tz_localize(None)))
    fp = pd.DataFrame(rows); fp['t0'] = pd.to_datetime(fp.t0); fp['t1'] = pd.to_datetime(fp.t1)
    fp.to_parquet(f'{P}/guard/exam_footprints.parquet')
    S['guard'] = dict(n_footprints=int(len(fp)), by_source=fp.source.value_counts().to_dict(),
                      file='proc/guard/exam_footprints.parquet', land_buffer_m=2000, weather_buffer_m=50000,
                      weather_days=3)
    body = json.dumps(S, sort_keys=True, default=str)
    S['sha256'] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(S, open(OUT, 'w'), indent=1, default=str)
    json.dump({k: v['tile_wkb_hex'] for k, v in ws.items()}, open(f'{P}/guard/wsts_tiles.json', 'w'))
    print(f.split.value_counts()); print(f[f.split == 'dev'][['Event_ID', 'Incid_Name', 'region', 'fuel_group', 'size_class', 'year', 'gofer']].to_string())
    print('gofer', {k: v['split'] for k, v in gof.items()})
    print('wsts', pd.Series([v['split'] for v in ws.values()]).value_counts().to_dict())


if __name__ == '__main__':
    main()
