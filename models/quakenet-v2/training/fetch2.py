"""Extra station records for QuakeNet v2: for the 40 dev events and the 400-event exam subset in splits.json, every IU/II
station at 25-95 degrees that was operating, in a seeded random order, until the event has 10 stations (QuakeNet's
original records count). Same processing as QuakeNet v1.2's fetch.py: EarthScope FDSN LH channels, response removed to
velocity (m/s), rotated to ZNE, 0.008-0.35 Hz, kept from 300 s before the ak135 P time to 2,700 s after.
Writes the planned order to data/station_order.json first (fixed before any record is read)."""
import sys, os, json, time, random, pickle, concurrent.futures as cf, numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
from _paths import H; from _paths import Q; OUT = H + 'data/wf/'; INVD = H + 'data/inv/'
os.makedirs(OUT, exist_ok=True); os.makedirs(INVD, exist_ok=True)
TP = TauPyModel('ak135')

def inv_of(cl, net, sta):
    f = f'{INVD}{net}.{sta}.pkl'
    if os.path.exists(f): return pickle.load(open(f, 'rb'))
    inv = cl.get_stations(network=net, station=sta, channel='LH?', level='response'); pickle.dump(inv, open(f + '.tmp', 'wb')); os.replace(f + '.tmp', f); return inv

def get(job):
    key, net, sta, loc, t0, extra = job; out = f'{OUT}ev_{key}_{net}.{sta}.npz'
    if os.path.exists(out) or os.path.exists(out + '.fail'): return 'have'
    cl = Client('EARTHSCOPE', timeout=120); err = ''
    for a in range(3):
        try:
            st = cl.get_waveforms(net, sta, loc, 'LH?', UTCDateTime(t0) - 600, UTCDateTime(t0) + 3600); inv = inv_of(cl, net, sta); break
        except Exception as e:
            err = str(e)[:80]
            if 'No data' in err or '204' in err: break
            time.sleep(3 + 5 * a)
    else:
        open(out + '.fail', 'w').write(err); return 'fail'
    if err and 'st' not in dir(): open(out + '.fail', 'w').write(err); return 'nodata'
    try:
        st.merge(fill_value=0); st.detrend('linear'); st.taper(0.02)
        st.remove_response(inventory=inv, output='VEL', pre_filt=(0.003, 0.005, 0.4, 0.48), water_level=None)
        st.rotate('->ZNE', inventory=inv); st.filter('bandpass', freqmin=0.008, freqmax=0.35, corners=4, zerophase=True)
        st.trim(UTCDateTime(t0) - 300, UTCDateTime(t0) + 2700, pad=True, fill_value=0)
        tr = {t.stats.channel[-1]: t for t in st}
        if not all(c in tr for c in 'ZNE'): open(out + '.fail', 'w').write('missing comps'); return 'missing'
        L = 3000; X = np.stack([np.asarray(tr[c].data[:L], np.float32) for c in 'ZNE'])
        if X.shape[1] < L or not np.isfinite(X).all() or np.abs(X).max() == 0: open(out + '.fail', 'w').write('bad'); return 'bad'
        np.savez_compressed(out, x=X, meta=json.dumps(extra)); return 'ok'
    except Exception as e:
        open(out + '.fail', 'w').write('proc ' + str(e)[:80]); return 'proc'

def plan():
    S = json.load(open(Q + 'stations.json')); SP = json.load(open(H + 'splits.json')); rng = random.Random(11); order = {}; J = []
    for k in SP['dev'] + SP['exam_multi']:
        e = SP['events'][k]; iso = UTCDateTime(e['time']).isoformat(); cand = []
        for s, v in sorted(S.items()):
            if s in e['stations']: continue
            if not (v['start'] <= iso and (v['end'] in ('None', '') or v['end'] >= iso)): continue
            d = locations2degrees(e['lat'], e['lon'], v['lat'], v['lon'])
            if 25 <= d <= 95: cand.append(s)
        rng.shuffle(cand); order[k] = list(e['stations']) + cand
        jobs = []
        for s in cand:     # tried in this order until the event has 10 usable records (or 14 tries)
            v = S[s]; net, sta = s.split('.'); loc = '00' if '00' in v['locs'] else v['locs'][0]
            d = locations2degrees(e['lat'], e['lon'], v['lat'], v['lon'])
            arr = TP.get_travel_times(source_depth_in_km=max(e['depth'], 0), distance_in_degree=d, phase_list=['P'])
            if not arr: continue
            _, az, baz = gps2dist_azimuth(e['lat'], e['lon'], v['lat'], v['lon'])
            ev = dict(id=k, time=e['time'], lat=e['lat'], lon=e['lon'], depth=e['depth'], mag=e['mw'], magType=e['magType'], place=e['place'])
            jobs.append((k, net, sta, loc, e['time'] + arr[0].time, dict(event=ev, station=s, dist=d, baz=baz, az=az, p_time=e['time'] + arr[0].time)))
        J.append((k, len(e['stations']), jobs))
    if not os.path.exists(H + 'data/station_order.json'): json.dump(order, open(H + 'data/station_order.json', 'w'))
    return J

def event(job):
    k, have, jobs = job; out = {}; tries = 0
    for j in jobs:
        if have >= 10 or tries >= 14: break
        r = get(j); tries += r != 'have'; out[r] = out.get(r, 0) + 1
        if r == 'ok' or (r == 'have' and os.path.exists(f"{OUT}ev_{j[0]}_{j[1]}.{j[2]}.npz")): have += 1
    return out

if __name__ == '__main__':
    J = plan(); print(len(J), 'events', flush=True); t = time.time(); res = {}
    with cf.ThreadPoolExecutor(3) as ex:
        for i, rr in enumerate(ex.map(event, J)):
          for r, n in rr.items():
            res[r] = res.get(r, 0) + n
          if True:
            if i % 20 == 0: print(i, res, '%.0fs' % (time.time() - t), flush=True)
    print('done', res, flush=True)
