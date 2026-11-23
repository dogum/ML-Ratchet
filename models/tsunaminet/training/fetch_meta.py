"""Event list (NGDC/WDS tsunami events), DART observations listed by NGDC (runups with type 3 = deep-ocean gauge),
USGS W-phase moment tensors and finite-fault models, and NDBC DART station position history.
Writes data/events_raw.json. Inclusion rule (fixed before any DART waveform was looked at):
  NGDC earthquake-caused tsunami, 2006-2025, eq magnitude >= 7.5, epicentre within 200 km of a Pacific SIFT unit source."""
import os, json, time, urllib.request, urllib.parse, numpy as np
from _paths import DATA
UA = {'User-Agent': 'tsunaminet-research-script'}
def get(url, binary=False, tries=4):
    for k in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
            return r if binary else r.decode('utf-8', 'replace')
        except Exception as e:
            print('retry', url[:100], e, flush=True); time.sleep(5 * (k + 1))
    return None

PAC = ('acsz', 'cssz', 'epsz', 'kbsz', 'kisz', 'mosz', 'ngsz', 'ntsz', 'nvsz', 'nzsz', 'rnsz', 'wpsz')
def unit_sources():
    U = []
    for l in open(os.path.join(DATA, 'sift', 'info_sz.dat')):
        if l.startswith('#'): continue
        t = [x.strip() for x in l.split(',')]
        if t[0][:4] in PAC:
            U.append(dict(name=t[0], lon=float(t[2]) % 360, lat=float(t[3]), strike=float(t[5]), dip=float(t[6]),
                          depth=float(t[7]), length=float(t[8]), width=float(t[9]), rake=float(t[10])))
    return U

def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2); dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def usgs(ev):
    t = '%04d-%02d-%02dT%02d:%02d:%02d' % (ev['year'], ev['month'], ev['day'], ev['hour'] or 0, ev['minute'] or 0, int(ev['second'] or 0))
    import datetime as dt
    t0 = dt.datetime.fromisoformat(t)
    q = dict(format='geojson', starttime=(t0 - dt.timedelta(minutes=3)).isoformat(), endtime=(t0 + dt.timedelta(minutes=3)).isoformat(),
             minmagnitude=7.0, latitude=ev['latitude'], longitude=ev['longitude'], maxradiuskm=300)
    d = json.loads(get('https://earthquake.usgs.gov/fdsnws/event/1/query?' + urllib.parse.urlencode(q)))
    if not d['features']: return None
    f = max(d['features'], key=lambda f: f['properties']['mag'])
    det = json.loads(get(f['properties']['detail']))
    P = det['properties']['products']; g = det['geometry']['coordinates']
    out = dict(usgs_id=det['id'], time_ms=det['properties']['time'], mag=det['properties']['mag'], magType=det['properties']['magType'],
               hypo_lon=g[0] % 360, hypo_lat=g[1], hypo_depth=g[2])
    for mt in P.get('moment-tensor', []):
        p = mt['properties']
        if (p.get('derived-magnitude-type') or '').lower() in ('mww',) and 'wphase' in (p.get('beachball-type', '') + p.get('derived-magnitude-type', '').lower() + 'wphase'):
            out['wphase'] = dict(mw=float(p['derived-magnitude']), lat=float(p['derived-latitude']), lon=float(p['derived-longitude']) % 360,
                                 depth=float(p['derived-depth']), np1=[float(p['nodal-plane-1-' + k]) for k in ('strike', 'dip', 'rake')],
                                 np2=[float(p['nodal-plane-2-' + k]) for k in ('strike', 'dip', 'rake')], source=mt['source'])
            break
    for ff in P.get('finite-fault', []):
        c = ff['contents'].get('basic_inversion.param')
        if not c: continue
        txt = get(c['url'])
        rows = []
        for line in txt.splitlines():
            s = line.split()
            if len(s) >= 11:
                try:
                    v = [float(x) for x in s[:11]]; rows.append(v)
                except ValueError:
                    pass
        if rows:
            a = np.array(rows)  # lat lon depth slip(cm) rake strike dip t_rup t_ris t_fal mo
            mo = a[:, 10]; w = mo / mo.sum()
            lonc = np.degrees(np.angle((w * np.exp(1j * np.radians(a[:, 1]))).sum())) % 360
            out['ffm'] = dict(centroid_lat=float((w * a[:, 0]).sum()), centroid_lon=float(lonc), centroid_depth=float((w * a[:, 2]).sum()),
                              mw=float(ff['properties'].get('derived-magnitude', 'nan')), n_sub=len(a), moment_sum=float(mo.sum()),
                              strike=float(ff['properties'].get('model-strike', 'nan')), dip=float(ff['properties'].get('model-dip', 'nan')),
                              length=float(ff['properties'].get('model-length', 'nan')), width=float(ff['properties'].get('model-width', 'nan')),
                              sub=a[:, [0, 1, 2, 3, 10]].round(4).tolist())
        break
    return out

if __name__ == '__main__':
    U = unit_sources(); ulat = np.array([u['lat'] for u in U]); ulon = np.array([u['lon'] for u in U])
    json.dump(U, open(os.path.join(DATA, 'unit_sources_pacific.json'), 'w'))
    d = json.loads(get('https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/events?minYear=2006&maxYear=2025&minEqMagnitude=7.5'))
    E = []
    for e in d['items']:
        if e.get('causeCode') not in (1, 3) or e.get('latitude') is None: continue
        dmin = hav(e['latitude'], e['longitude'] % 360, ulat, ulon).min()
        if dmin > 200: continue
        rec = {k: e.get(k) for k in ('id', 'year', 'month', 'day', 'hour', 'minute', 'second', 'eqMagnitude', 'eqDepth', 'latitude', 'longitude', 'locationName', 'maxWaterHeight', 'numRunups', 'causeCode')}
        rec['dist_unit_km'] = float(dmin)
        ru = json.loads(get(f'https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/runups?tsunamiEventId={e["id"]}'))
        rec['ngdc_dart'] = [{k: r.get(k) for k in ('locationName', 'latitude', 'longitude', 'runupHt', 'period', 'arrDay', 'arrHour', 'arrMin')} for r in ru['items'] if r.get('typeMeasurementId') == 3]
        rec['usgs'] = usgs(e)
        print(rec['year'], rec['month'], rec['day'], rec['eqMagnitude'], rec['locationName'], 'darts', len(rec['ngdc_dart']),
              'wphase' if rec['usgs'] and 'wphase' in rec['usgs'] else '-', 'ffm' if rec['usgs'] and 'ffm' in rec['usgs'] else '-', flush=True)
        E.append(rec)
    json.dump(E, open(os.path.join(DATA, 'events_raw.json'), 'w'), indent=1)
    sm = get('https://www.ndbc.noaa.gov/metadata/stationmetadata.xml')
    open(os.path.join(DATA, 'ndbc_stationmetadata.xml'), 'w').write(sm)
    print('events', len(E))
