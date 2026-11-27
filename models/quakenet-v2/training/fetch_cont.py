"""Continuous records for the independent-pick supplementary exam (27 Nov 2026; audit queue).

The released exam cut every record around the ak135 P time predicted from the USGS catalogue (origin, epicentre), so the
P search (+-30 s around sample 300) was catalogue-assisted. Here each record is a fixed block of continuous data that
knows only the UTC hour of the origin: from 10 minutes before that hour to 2 hours after it (7,800 samples at 1 sps).
The P wave of any origin in the hour at 25-95 degrees reaches the station 5-74 minutes after the hour, so it lies in the
block, but where is not given to the reader. Picks are made later by indep_exam.py.

Stations per event: the same fixed order as the exam (splits.json originals first, then data/station_order.json),
tried in turn until the event has 10 usable records (extra stations: at most 14 tries, as fetch2.py). Processing is
fetch2.py's: EarthScope FDSN LH channels, response removed to velocity (m/s), rotated to ZNE, 0.008-0.35 Hz.
The catalogue P time is stored in the file only for grading (key 'grading_p_time'); the reader never sees it.
Writes data/wf_cont/ev_<id>_<NET.STA>.npz (x: 3 x 7800 float32, meta json). Restart-safe.
Usage: fetch_cont.py dev|exam"""
import sys, os, json, time, pickle, concurrent.futures as cf, numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from _paths import H, Q
OUT = H + 'data/wf_cont/'; INVD = H + 'data/inv/'
os.makedirs(OUT, exist_ok=True); os.makedirs(INVD, exist_ok=True)
TP = TauPyModel('ak135'); PRE, LEN = 600, 7800

def hour_of(t):
    T = UTCDateTime(t); return UTCDateTime(T.year, T.month, T.day, T.hour)

def inv_of(cl, net, sta):
    f = f'{INVD}{net}.{sta}.pkl'
    if os.path.exists(f): return pickle.load(open(f, 'rb'))
    inv = cl.get_stations(network=net, station=sta, channel='LH?', level='response'); pickle.dump(inv, open(f + '.tmp', 'wb')); os.replace(f + '.tmp', f); return inv

def get(k, e, s, S):
    out = f'{OUT}ev_{k}_{s}.npz'
    if os.path.exists(out): return 'have'
    if os.path.exists(out + '.fail'): return 'havefail'
    v = S[s]; net, sta = s.split('.'); loc = '00' if '00' in v['locs'] else v['locs'][0]; t0 = hour_of(e['time']) - PRE
    cl = Client('EARTHSCOPE', timeout=120); st = None; err = ''
    for a in range(3):
        try:
            st = cl.get_waveforms(net, sta, loc, 'LH?', t0 - 300, t0 + LEN + 300); inv = inv_of(cl, net, sta); break
        except Exception as ex:
            err = str(ex)[:80]
            if 'No data' in err or '204' in err: break
            time.sleep(3 + 5 * a)
    if st is None: open(out + '.fail', 'w').write(err or 'no data'); return 'fail'
    try:
        st.merge(fill_value=0); st.detrend('linear'); st.taper(0.02)
        st.remove_response(inventory=inv, output='VEL', pre_filt=(0.003, 0.005, 0.4, 0.48), water_level=None)
        st.rotate('->ZNE', inventory=inv); st.filter('bandpass', freqmin=0.008, freqmax=0.35, corners=4, zerophase=True)
        st.trim(t0, t0 + LEN, pad=True, fill_value=0); tr = {t.stats.channel[-1]: t for t in st}
        if not all(c in tr for c in 'ZNE'): open(out + '.fail', 'w').write('missing comps'); return 'missing'
        X = np.stack([np.asarray(tr[c].data[:LEN], np.float32) for c in 'ZNE'])
        if X.shape[1] < LEN or not np.isfinite(X).all() or (X.std(1) == 0).any(): open(out + '.fail', 'w').write('bad'); return 'bad'
        d = locations2degrees(e['lat'], e['lon'], v['lat'], v['lon'])
        arr = TP.get_travel_times(source_depth_in_km=max(e['depth'], 0), distance_in_degree=d, phase_list=['P'])
        meta = dict(event=k, station=s, lat=v['lat'], lon=v['lon'], t_start=float(t0.timestamp), grading_p_time=float(e['time'] + arr[0].time) if arr else None)
        np.savez_compressed(out, x=X, meta=json.dumps(meta)); return 'ok'
    except Exception as ex:
        open(out + '.fail', 'w').write('proc ' + str(ex)[:80]); return 'proc'

def event(args):
    k, e, order, S = args; have, tries, res = 0, 0, {}
    for s in order:
        if have >= 10: break
        orig = s in e['stations']
        if not orig and tries >= 14: break
        if s not in S: continue
        r = get(k, e, s, S); res[r] = res.get(r, 0) + 1
        if not orig: tries += 1
        if r in ('ok', 'have'): have += 1
    return res

if __name__ == '__main__':
    which = sys.argv[1]; SP = json.load(open(H + 'splits.json')); S = json.load(open(Q + 'stations.json'))
    order = json.load(open(H + 'data/station_order.json')); keys = SP['dev'] if which == 'dev' else SP['exam_multi']
    J = [(k, SP['events'][k], order.get(k, SP['events'][k]['stations']), S) for k in keys]; t = time.time(); tot = {}
    print(len(J), 'events', flush=True)
    with cf.ThreadPoolExecutor(3) as ex:
        for i, rr in enumerate(ex.map(event, J)):
            for r, n in rr.items(): tot[r] = tot.get(r, 0) + n
            if i % 20 == 0: print(i, tot, '%.0fs' % (time.time() - t), flush=True)
    print('done', tot, '%.0fs' % (time.time() - t), flush=True)
