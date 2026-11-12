"""Real seismograms for QuakeNet: event-station pairs at 25-95 degrees and quiet-time noise windows, three LH channels
(1 sample/s) from EarthScope, instrument response removed to ground velocity (m/s), rotated to Z/N/E, kept from 300 s
before the ak135 P arrival (or the window start, for noise) to 2,700 s after."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, time, random, concurrent.futures as cf, numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
D = _P.WORK + '/data/'; OUT = D + 'wf/'; os.makedirs(OUT, exist_ok=True)
TP = TauPyModel('ak135'); INV = {}

def inv_of(cl, net, sta):
    k = f'{net}.{sta}'
    if k not in INV: INV[k] = cl.get_stations(network=net, station=sta, channel='LH?', level='response')
    return INV[k]

def get(job):
    kind, key, net, sta, loc, t0, extra = job; out = f'{OUT}{kind}_{key}_{net}.{sta}.npz'
    if os.path.exists(out): return 'have'
    cl = Client('EARTHSCOPE', timeout=120)
    for a in range(3):
        try:
            st = cl.get_waveforms(net, sta, loc, 'LH?', UTCDateTime(t0) - 600, UTCDateTime(t0) + 3600); inv = inv_of(cl, net, sta); break
        except Exception as e:
            err = str(e)[:80]; time.sleep(3 + 5 * a)
    else:
        return 'fail ' + err
    try:
        st.merge(fill_value=0); st.detrend('linear'); st.taper(0.02)
        st.remove_response(inventory=inv, output='VEL', pre_filt=(0.003, 0.005, 0.4, 0.48), water_level=None)
        st.rotate('->ZNE', inventory=inv); st.filter('bandpass', freqmin=0.008, freqmax=0.35, corners=4, zerophase=True)
        st.trim(UTCDateTime(t0) - 300, UTCDateTime(t0) + 2700, pad=True, fill_value=0)
        tr = {t.stats.channel[-1]: t for t in st}
        if not all(c in tr for c in 'ZNE'): return 'missing comps'
        L = 3000; X = np.stack([np.asarray(tr[c].data[:L], np.float32) for c in 'ZNE'])
        if X.shape[1] < L or not np.isfinite(X).all() or np.abs(X).max() == 0: return 'bad'
        np.savez_compressed(out, x=X, meta=json.dumps(extra)); return 'ok'
    except Exception as e:
        return 'proc ' + str(e)[:80]

def jobs(n_ev=1800, per_ev=2, n_noise=1500, seed=0):
    rng = random.Random(seed); E = [e for e in json.load(open(D + 'events.json')) if e['magType'].startswith('mw')]; S = json.load(open(D + 'stations.json'))
    deep = [e for e in E if e['depth'] > 70]; shallow = [e for e in E if e['depth'] <= 70]
    pick = rng.sample(deep, min(len(deep), 700)) + rng.sample(shallow, n_ev - min(len(deep), 700)); J = []
    for e in pick:
        cand = []
        for k, s in S.items():
            if not (s['start'] <= UTCDateTime(e['time']).isoformat() and (s['end'] in ('None', '') or s['end'] >= UTCDateTime(e['time']).isoformat())): continue
            d = locations2degrees(e['lat'], e['lon'], s['lat'], s['lon'])
            if 25 <= d <= 95: cand.append((k, d))
        for k, d in rng.sample(cand, min(per_ev, len(cand))):
            s = S[k]; net, sta = k.split('.'); loc = '00' if '00' in s['locs'] else s['locs'][0]
            arr = TP.get_travel_times(source_depth_in_km=max(e['depth'], 0), distance_in_degree=d, phase_list=['P'])
            if not arr: continue
            _, az, baz = gps2dist_azimuth(e['lat'], e['lon'], s['lat'], s['lon'])
            J.append(('ev', e['id'], net, sta, loc, e['time'] + arr[0].time, dict(event=e, station=k, dist=d, baz=baz, az=az, p_time=e['time'] + arr[0].time)))
    times = np.array(sorted(x['time'] for x in E)); big = np.array(sorted(x['time'] for x in E if x['mag'] >= 6.5))
    keys = list(S.keys())
    while sum(j[0] == 'no' for j in J) < n_noise:
        t = rng.uniform(UTCDateTime('2012-06-01').timestamp, UTCDateTime('2025-09-01').timestamp); k = rng.choice(keys); s = S[k]
        if not (s['start'] <= UTCDateTime(t).isoformat() and (s['end'] in ('None', '') or s['end'] >= UTCDateTime(t).isoformat())): continue
        i = np.searchsorted(times, t); 
        if (i > 0 and t - times[i - 1] < 4 * 3600) or (i < len(times) and times[i] - t < 3000): continue
        j = np.searchsorted(big, t)
        if j > 0 and t - big[j - 1] < 12 * 3600: continue
        net, sta = k.split('.'); loc = '00' if '00' in s['locs'] else s['locs'][0]; J.append(('no', f'{int(t)}', net, sta, loc, t, dict(station=k, t=t)))
    return J

if __name__ == '__main__':
    J = jobs(); print(len(J), 'jobs', flush=True); t = time.time(); res = {}
    with cf.ThreadPoolExecutor(3) as ex:
        for i, r in enumerate(ex.map(get, J)):
            res[r.split(' ')[0]] = res.get(r.split(' ')[0], 0) + 1
            if i % 200 == 0: print(i, res, '%.0fs' % (time.time() - t), flush=True)
    print('done', res, flush=True)
