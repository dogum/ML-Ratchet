"""Real against simulated for dev events with USGS moment tensors: the same source, distance and back-azimuth, synthesized
from the Green's functions, next to the recorded seismogram."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, urllib.request, numpy as np
sys.path.insert(0, _P.SRC); import quakesim as QS, pipeline as PL
from scipy.signal import sosfiltfilt
D = _P.WORK + '/data/'

def usgs_mt(eid):
    js = json.load(urllib.request.urlopen(f'https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={eid}&format=geojson', timeout=60))
    for p in js['properties']['products'].get('moment-tensor', []):
        q = p['properties']
        if 'tensor-mrr' in q:
            return np.array([float(q['tensor-' + k]) for k in ('mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp')]), float(q.get('derived-depth', js['geometry']['coordinates'][2]))
    return None, None

def synth_for(m, M):
    from obspy.geodetics import gps2dist_azimuth
    z = min(QS.DEPTHS, key=lambda v: abs(v - m['depth'])); d = int(round(m['dist'])); d = min(max(d, 25), 95)
    S = json.load(open(D + 'stations.json'))[m['station']]; _, az, baz = gps2dist_azimuth(m['lat'], m['lon'], S['lat'], S['lon'])
    G = QS.gf(z, d); Z, R, T = QS.zrt(G, M, np.radians(az)); m0 = np.sqrt(0.5 * np.sum(M[:3] ** 2) + np.sum(M[3:] ** 2)); k = QS.stf(QS.half_duration(m0))
    Z, R, T = (np.convolve(x, k)[:len(x)] for x in (Z, R, T)); X = QS.zne(Z, R, T, np.radians(baz))
    X = sosfiltfilt(PL.SOS_BAND, np.pad(X, ((0, 0), (200, 200))), axis=1)[:, 200:-200][:, ::2]; return X

if __name__ == '__main__':
    J = json.load(open(D + 'real_meta.json')); z = np.load(D + 'real.npz'); out = []
    idx = [i for i, s in enumerate(J['split']) if s == 'dev']
    for i in idx[:40]:
        m = J['meta'][i]; M, _ = usgs_mt(m['id'])
        if M is None: continue
        X = synth_for(m, M); real = z['x'][i].astype(float) * 10 ** z['ls'][i]; L = min(X.shape[1], real.shape[1])
        cc = [float(np.corrcoef(X[j, :L], real[j, :L])[0, 1]) for j in range(3)]; amp = float(np.log10(np.std(real[:, :L]) / np.std(X[:, :L])))
        # best lag of the vertical (real minus synthetic), +-60 s
        lags = range(-60, 61, 1); c = [np.corrcoef(np.roll(X[0, :L], l), real[0, :L])[0, 1] for l in lags]; bl = list(lags)[int(np.argmax(c))]
        out.append(dict(i=int(i), id=m['id'], mw=m['mw'], depth=m['depth'], dist=m['dist'], station=m['station'], cc=cc, log_amp_ratio=amp, best_lag=bl, best_cc=float(max(c))))
        print(out[-1], flush=True)
    json.dump(out, open(D + 'compare_dev.json', 'w'), indent=1)
