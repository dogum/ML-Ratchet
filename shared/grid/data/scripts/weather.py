"""Weather for the focus BAs and GB.

Stages (argv): points | power | asos | build
 points: weighted k-means points per BA
   pop   (k=8): counties of the BA's EIA-861 service territory (PUDL), weight = county population (EIA-861 /
                Census), point = population-weighted centroid of county internal points (Census DP1)
   wind  (k=6): EIA-860 wind generators in the BA (existing, our fleet table), weight = capacity
   solar (k=6): EIA-860 solar PV generators, weight = capacity
   GB: fixed points (see GB_POINTS; weights documented there)
 power: NASA POWER hourly point data (MERRA-2 met + CERES irradiance), 2018-01-01..2026-10-31, UTC
 asos:  IEM ASOS routine hourly METARs at the station nearest each US population point
 build: BA-level hourly series (proc/weather/<BA>.parquet), weights in proc/weather/points.json
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import os, sys, json, time, io
import numpy as np, pandas as pd, requests
RAW = f'{GRID_ROOT}/data/raw'; PROC = f'{GRID_ROOT}/data/proc'
WR = f'{RAW}/weather'; WP = f'{PROC}/weather'
for d in (WR, f'{WR}/power', f'{WR}/asos', WP): os.makedirs(d, exist_ok=True)
FOCUS = ['ERCO', 'CISO', 'PJM', 'MISO', 'SWPP', 'ISNE', 'NYIS', 'BPAT']
BA_ID = {'ERCO': 5723, 'CISO': 2775, 'PJM': 14725, 'MISO': 56669, 'SWPP': 59504, 'ISNE': 13434, 'NYIS': 13501, 'BPAT': 1738}
# GB: approximate built-up-area populations (ONS 2021 Census / NRS Scotland, rounded, millions): used as relative
# weights only. Wind: main offshore/onshore clusters, equal weights. Solar: southern England / Midlands, equal weights.
GB_POINTS = {
    'pop': [('London', 51.51, -0.13, 8.8), ('Birmingham', 52.48, -1.90, 2.9), ('Manchester', 53.48, -2.24, 2.7),
            ('Leeds', 53.80, -1.55, 1.8), ('Glasgow', 55.86, -4.25, 1.0), ('Liverpool', 53.41, -2.98, 0.9),
            ('Southampton', 50.91, -1.40, 0.9), ('Newcastle', 54.98, -1.61, 0.8), ('Nottingham', 52.95, -1.15, 0.8),
            ('Bristol', 51.45, -2.59, 0.7), ('Edinburgh', 55.95, -3.19, 0.5), ('Cardiff', 51.48, -3.18, 0.45)],
    'wind': [('Hornsea-DoggerBank', 54.0, 1.8, 1), ('EastAnglia', 52.6, 2.2, 1), ('ThamesEstuary', 51.6, 1.5, 1),
             ('MorayFirth', 58.1, -3.0, 1), ('IrishSea-Walney', 54.0, -3.6, 1), ('Highlands-onshore', 57.3, -4.5, 1),
             ('SouthernUplands-onshore', 55.4, -3.8, 1)],
    'solar': [('Cornwall', 50.4, -4.9, 1), ('Hampshire', 51.0, -1.3, 1), ('EastAnglia-land', 52.3, 1.0, 1),
              ('Midlands', 52.5, -1.5, 1), ('Kent', 51.2, 0.8, 1)]}


def wkmeans(xy, w, k, seed=20261113, iters=50):
    rng = np.random.default_rng(seed)
    k = min(k, len(xy))
    c = xy[rng.choice(len(xy), k, replace=False, p=w / w.sum())]
    for _ in range(iters):
        lab = np.argmin(((xy[:, None, :] - c[None]) ** 2).sum(-1), 1)
        for j in range(k):
            m = lab == j
            if m.any(): c[j] = (xy[m] * w[m, None]).sum(0) / w[m].sum()
    lab = np.argmin(((xy[:, None, :] - c[None]) ** 2).sum(-1), 1)
    ws = np.array([w[lab == j].sum() for j in range(k)])
    keep = ws > 0
    return c[keep], ws[keep] / ws[keep].sum()


def stage_points():
    t = pd.read_parquet(f'{RAW}/pudl/out_eia861__yearly_balancing_authority_service_territory.parquet')
    t = t[t.report_date == t.report_date.max()]
    cty = pd.read_parquet(f'{RAW}/pudl/out_censusdp1tract__counties.parquet',
                          columns=['county_id_fips', 'internal_point_latitude', 'internal_point_longitude'])
    pts = {}
    for ba in FOCUS:
        c = t[t.balancing_authority_id_eia == BA_ID[ba]].drop_duplicates('county_id_fips').merge(cty, on='county_id_fips')
        c = c.dropna(subset=['internal_point_latitude', 'population'])
        xy = c[['internal_point_latitude', 'internal_point_longitude']].astype(float).values
        cen, w = wkmeans(xy, c.population.astype(float).values, 8)
        P = [dict(kind='pop', lat=round(float(a), 3), lon=round(float(b), 3), w=round(float(x), 4)) for (a, b), x in zip(cen, w)]
        fl = pd.read_parquet(f'{PROC}/fleet/fleet_{ba}.parquet')
        fl = fl[fl.retired_date.isna() & (fl.status_latest == 'existing') & fl.lat.notna()]
        for kind, tech, k in (('wind', 'wind', 6), ('solar', 'solar_pv', 6)):
            f = fl[fl.tech == tech]
            if f.pmax_MW.sum() < 50: continue
            cen, w = wkmeans(f[['lat', 'lon']].values.astype(float), f.pmax_MW.fillna(0).values.astype(float) + 1e-6, k)
            P += [dict(kind=kind, lat=round(float(a), 3), lon=round(float(b), 3), w=round(float(x), 4)) for (a, b), x in zip(cen, w)]
        pts[ba] = dict(points=P, n_counties=int(len(c)), population=int(c.population.sum()),
                       wind_MW=float(fl[fl.tech == 'wind'].pmax_MW.sum()), solar_MW=float(fl[fl.tech == 'solar_pv'].pmax_MW.sum()))
    G = []
    for kind, L in GB_POINTS.items():
        tot = sum(x[3] for x in L)
        G += [dict(kind=kind, name=n, lat=a, lon=b, w=round(v / tot, 4)) for n, a, b, v in L]
    pts['GB'] = dict(points=G, note='fixed points; pop weights = approx. built-up-area population; wind/solar equal weights')
    for ba, d in pts.items():
        for i, p in enumerate(d['points']): p['pid'] = f'{ba}_{p["kind"]}_{i}'
    json.dump(pts, open(f'{WP}/points.json', 'w'), indent=1)
    print({k: len(v['points']) for k, v in pts.items()})


def stage_power():
    pts = json.load(open(f'{WP}/points.json'))
    for ba, d in pts.items():
        for p in d['points']:
            f = f'{WR}/power/{p["pid"]}.parquet'
            if os.path.exists(f): continue
            url = ('https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M,QV2M,WS10M,WS50M,WD50M,PS,'
                   f'ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN&community=RE&longitude={p["lon"]}&latitude={p["lat"]}'
                   '&start=20180101&end=20261031&format=CSV&time-standard=UTC')
            for a in range(4):
                try:
                    r = requests.get(url, timeout=600)
                    if r.status_code == 200 and 'YEAR,MO' in r.text: break
                except Exception as e: print('err', e)
                time.sleep(30 * (a + 1))
            txt = r.text; body = txt[txt.index('YEAR,MO'):]
            x = pd.read_csv(io.StringIO(body)).replace(-999.0, np.nan)
            x['time_utc'] = pd.to_datetime(dict(year=x.YEAR, month=x.MO, day=x.DY, hour=x.HR), utc=True)
            x = x.drop(columns=['YEAR', 'MO', 'DY', 'HR'])
            x = x.astype({c: 'float32' for c in x.columns if c != 'time_utc'})
            x.to_parquet(f, index=False); print('power', p['pid'], len(x), flush=True); time.sleep(1)


def nearest_asos(lat, lon, stations):
    d = (stations.lat - lat) ** 2 + ((stations.lon - lon) * np.cos(np.radians(lat))) ** 2
    return stations.iloc[np.argsort(d.values)[:5]]


def stage_asos():
    pts = json.load(open(f'{WP}/points.json'))
    states = {'ERCO': ['TX'], 'CISO': ['CA'], 'PJM': ['PA', 'NJ', 'MD', 'VA', 'OH', 'IL', 'DC', 'DE', 'WV', 'MI', 'KY', 'IN', 'NC'],
              'MISO': ['MN', 'WI', 'MI', 'IL', 'IN', 'IA', 'MO', 'AR', 'LA', 'MS', 'TX', 'ND', 'SD', 'KY', 'MT'],
              'SWPP': ['KS', 'OK', 'NE', 'TX', 'NM', 'MO', 'AR', 'LA', 'SD', 'ND', 'IA', 'MN', 'MT', 'WY'],
              'ISNE': ['MA', 'CT', 'RI', 'NH', 'VT', 'ME'], 'NYIS': ['NY'], 'BPAT': ['WA', 'OR', 'ID', 'MT']}
    cache = f'{WR}/asos_stations.parquet'
    if os.path.exists(cache): st = pd.read_parquet(cache)
    else:
        L = []
        for s in sorted({x for v in states.values() for x in v}):
            g = None
            for a in range(5):
                try:
                    g = requests.get(f'https://mesonet.agron.iastate.edu/geojson/network/{s}_ASOS.geojson', timeout=60).json(); break
                except Exception as e:
                    print('retry geojson', s, e, flush=True); time.sleep(20 * (a + 1))
            for f in g['features']:
                pr = f['properties']; lon, lat = f['geometry']['coordinates'][:2]
                L.append(dict(sid=pr['sid'], name=pr.get('sname'), state=s, lat=lat, lon=lon,
                              archive_begin=pr.get('archive_begin'), archive_end=pr.get('archive_end')))
            time.sleep(1)
        st = pd.DataFrame(L).drop_duplicates('sid'); st.to_parquet(cache)
    st = st[(pd.to_datetime(st.archive_begin, errors='coerce') < '2017-01-01') & st.archive_end.isna()]
    chosen = {}
    for ba, d in pts.items():
        if ba == 'GB': continue
        for p in d['points']:
            if p['kind'] != 'pop': continue
            cand = nearest_asos(p['lat'], p['lon'], st[st.state.isin(states[ba])])
            for _, s in cand.iterrows():
                f = f'{WR}/asos/{s.sid}.parquet'
                if not os.path.exists(f):
                    url = ('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=' + s.sid +
                           '&data=tmpf&data=dwpf&data=relh&data=sknt&data=drct&data=skyc1&data=alti'
                           '&year1=2018&month1=1&day1=1&year2=2026&month2=11&day2=1&tz=Etc/UTC&format=onlycomma'
                           '&latlon=no&missing=empty&trace=empty&report_type=3')
                    r = None
                    for a in range(4):
                        try:
                            r = requests.get(url, timeout=600)
                            if r.status_code == 200: break
                        except Exception as e: print('err', e)
                        time.sleep(30 * (a + 1))
                    x = pd.read_csv(io.StringIO(r.text), low_memory=False)
                    x['time_utc'] = pd.to_datetime(x['valid'], utc=True)
                    for c in ['tmpf', 'dwpf', 'relh', 'sknt', 'drct', 'alti']: x[c] = pd.to_numeric(x[c], errors='coerce')
                    h = pd.DataFrame({'time_utc': x.time_utc.dt.round('h'), 'T_C': (x.tmpf - 32) / 1.8, 'Td_C': (x.dwpf - 32) / 1.8,
                                      'RH': x.relh, 'wind_ms': x.sknt * 0.514444, 'wind_dir': x.drct})
                    h = h.groupby('time_utc').mean().astype('float32').reset_index()
                    h.to_parquet(f, index=False); print('asos', s.sid, len(h), flush=True); time.sleep(2)
                cov = pd.read_parquet(f).T_C.notna().sum() / (8.8 * 8760)
                if cov > 0.85:
                    chosen[p['pid']] = dict(sid=s.sid, name=s['name'], lat=float(s.lat), lon=float(s.lon), coverage=round(float(cov), 3)); break
    json.dump(chosen, open(f'{WP}/asos_choice.json', 'w'), indent=1)


def stage_build():
    pts = json.load(open(f'{WP}/points.json'))
    ch = json.load(open(f'{WP}/asos_choice.json')) if os.path.exists(f'{WP}/asos_choice.json') else {}
    for ba, d in pts.items():
        series = {}
        for kind in ('pop', 'wind', 'solar'):
            P = [p for p in d['points'] if p['kind'] == kind]
            if not P: continue
            acc = None; wsum = 0
            for p in P:
                x = pd.read_parquet(f'{WR}/power/{p["pid"]}.parquet').set_index('time_utc')
                acc = x * p['w'] if acc is None else acc.add(x * p['w'], fill_value=np.nan)
                wsum += p['w']
            acc = acc / wsum
            if kind == 'pop':
                series['T2m_C'] = acc.T2M; series['RH2m_pct'] = acc.RH2M; series['q2m_g_kg'] = acc.QV2M
                series['wind10m_pop_ms'] = acc.WS10M; series['ghi_pop_Wm2'] = acc.ALLSKY_SFC_SW_DWN; series['ps_kPa'] = acc.PS
            elif kind == 'wind':
                series['wind50m_ms'] = acc.WS50M; series['wind10m_ms'] = acc.WS10M; series['T2m_wind_C'] = acc.T2M
            else:
                series['ghi_Wm2'] = acc.ALLSKY_SFC_SW_DWN; series['ghi_clear_Wm2'] = acc.CLRSKY_SFC_SW_DWN; series['T2m_solar_C'] = acc.T2M
        o = pd.DataFrame(series)
        # ASOS population-weighted temperature / dew point / wind (station truth), weights renormalised over chosen stations
        P = [p for p in d['points'] if p['kind'] == 'pop' and p['pid'] in ch]
        if P:
            num = None; den = None
            for p in P:
                a = pd.read_parquet(f'{WR}/asos/{ch[p["pid"]]["sid"]}.parquet').set_index('time_utc')[['T_C', 'Td_C', 'RH', 'wind_ms']]
                a = a.reindex(o.index)
                wv = a.notna().astype(float) * p['w']
                num = a.fillna(0) * p['w'] if num is None else num + a.fillna(0) * p['w']
                den = wv if den is None else den + wv
            asos = num / den.replace(0, np.nan)
            o['T_asos_C'] = asos.T_C; o['Td_asos_C'] = asos.Td_C; o['RH_asos_pct'] = asos.RH; o['wind_asos_ms'] = asos.wind_ms
            o['asos_weight_cov'] = den.T_C / sum(p['w'] for p in P)
        o = o.astype('float32'); o.index.name = 'time_utc'
        o.reset_index().to_parquet(f'{WP}/{ba}.parquet', index=False, compression='zstd')
        print(ba, o.shape, o.index.min(), o.index.max(), flush=True)


if __name__ == '__main__':
    {'points': stage_points, 'power': stage_power, 'asos': stage_asos, 'build': stage_build}[sys.argv[1]]()
