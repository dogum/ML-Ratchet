"""One-station location: walk dist degrees from the station along the back-azimuth to place the epicenter, and compare with
the USGS epicenter (great-circle km)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC)
D = _P.WORK + '/data/'

def destination(lat, lon, baz, dist):
    p1, l1, b, d = np.radians(lat), np.radians(lon), np.radians(baz), np.radians(dist)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(b)); l2 = l1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 180) % 360 - 180

def gc_km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, (a1, o1, a2, o2)); return 6371 * np.arccos(np.clip(np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(o1 - o2), -1, 1))

def locate(P, meta):
    S = json.load(open(D + 'stations.json')); sl = np.array([S[m['station']]['lat'] for m in meta]); so = np.array([S[m['station']]['lon'] for m in meta])
    la, lo = destination(sl, so, P['baz'], P['dist']); return la, lo, gc_km(la, lo, np.array([m['lat'] for m in meta]), np.array([m['lon'] for m in meta]))

if __name__ == '__main__':
    import qnet as Q; net = Q.load_net(sys.argv[1]); J = json.load(open(D + 'real_meta.json'))
    for which in ('dev', 'exam'):
        s, idx, P = Q.real(net, which); _, _, e = locate(P, [J['meta'][i] for i in idx])
        print(which, 'location error km: median %.0f, 68%% %.0f, within 500 km %.2f, within 1000 km %.2f' % (np.median(e), np.percentile(e, 68), np.mean(e < 500), np.mean(e < 1000)))
