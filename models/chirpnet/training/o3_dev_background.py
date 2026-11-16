"""O3 dev background for ChirpNet's release threshold (15 Nov 2026). Real O3 noise that is NOT exam data.

The O3 exam found one time-slid background window (7.24) above the 7.0 threshold set on O1/O2 noise. This script asks
the same question on dev data first: real O3 H1 + L1 strain from GWOSC bulk files (4096 s, 4 kHz) at times drawn by a
fixed seed, cut into 32 s pieces and scored with exactly the exam's background protocol (o3_exam.py): each piece is
whitened on its own, gated, and L1 is slid against H1 by 3-25 s within the piece's central 28 s.

Pieces are kept only if both detectors have data with no gaps, pass DATA and CBC_CAT1, hold no CBC hardware injection,
and lie more than 600 s from every O3 event GWOSC lists (any catalogue, marginal ones included), so none of the exam's
eight event files can be touched. Nothing here changes the network.

    python o3_dev_background.py fetch      # ~2 GB of downloads (deleted after cutting), pieces to work/data_o3dev/
    python o3_dev_background.py score      # writes logs/o3_dev_background.json and work/o3_dev_bg.npy
"""
import _paths as _P  # portable paths (see _paths.py)
import calendar
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, _P.MODEL)
import chirpnet as C  # noqa: E402

API = 'https://gwosc.org/api/v2'
UA = {'User-Agent': 'ml-ratchet-dev/1.0'}
DATA = os.path.join(_P.WORK, 'data_o3dev')
RUNS = {'O3a': (1238166018, 1253977218), 'O3b': (1256655618, 1269363618)}
SEED, N_PAIRS, PER_PAIR, PIECE = 2026, 8, 8, 32
GPS_UNIX = 315964800 - 18      # GPS = unix - 315964800 + 18 leap seconds (valid 2017-2026)


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def event_times():
    """Every O3 event GWOSC lists, as GPS intervals to avoid: the named second when the name carries a time,
    the whole UTC day otherwise."""
    out = []
    for run in RUNS:
        url = f'{API}/runs/{run}/events'
        while url:
            page = get(url)
            for e in page['results']:
                m = re.match(r'GW(\d{2})(\d{2})(\d{2})(?:_(\d{2})(\d{2})(\d{2}))?', e['name'])
                if not m: continue
                y, mo, d = 2000 + int(m[1]), int(m[2]), int(m[3])
                if m[4]:
                    t = calendar.timegm((y, mo, d, int(m[4]), int(m[5]), int(m[6]))) - GPS_UNIX; out.append((t - 1, t + 1))
                else:
                    t = calendar.timegm((y, mo, d, 0, 0, 0)) - GPS_UNIX; out.append((t, t + 86400))
            url = page.get('next')
    return out


def url_of(run, det, gps):
    return f'https://gwosc.org/archive/data/{run}_4KHZ_R1/{gps // 1048576 * 1048576}/{det[0]}-{det}_GWOSC_{run}_4KHZ_R1-{gps}-4096.hdf5'


def download(url, path):
    for a in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(path + '.part', 'wb') as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
            os.replace(path + '.part', path); return True
        except urllib.error.HTTPError as e:
            if e.code == 404: return False
            time.sleep(5 * (a + 1))
        except Exception:
            time.sleep(5 * (a + 1))
    return False


def read(path):
    import h5py
    with h5py.File(path, 'r') as h:
        return (h['strain/Strain'][:], int(h['meta/GPSstart'][()]), h['quality/simple/DQmask'][:], h['quality/injections/Injmask'][:])


def fetch():
    os.makedirs(os.path.join(DATA, 'pieces'), exist_ok=True)
    avoid = event_times(); print(len(avoid), 'O3 events to avoid', flush=True)
    rng = np.random.default_rng(SEED); got = sorted(f for f in os.listdir(os.path.join(DATA, 'pieces')) if f.endswith('.npz'))
    pairs = {f.split('_')[1] for f in got}; tries = 0
    want = {'O3a': N_PAIRS // 2, 'O3b': N_PAIRS // 2}
    for run, n_want in want.items():
        lo, hi = RUNS[run]; have = sum(1 for p in pairs if lo <= int(p) < hi)
        cand = rng.permutation(np.arange(lo // 4096 + 1, hi // 4096 - 1)) * 4096
        for gps in cand:
            if have >= n_want: break
            gps = int(gps); tries += 1
            if str(gps) in pairs: continue
            paths = {d: os.path.join(DATA, f'{d}-{gps}.hdf5') for d in ('H1', 'L1')}
            if not all(os.path.exists(p) or download(url_of(run, d, gps), p) for d, p in paths.items()):
                print(run, gps, 'missing file', flush=True); [os.remove(p) for p in paths.values() if os.path.exists(p)]; continue
            (h, t0, dqh, ih), (l, _, dql, il) = read(paths['H1']), read(paths['L1'])
            ok = []
            for k in range(0, 4096 // PIECE):
                a, b = k * PIECE, (k + 1) * PIECE; ta, tb = t0 + a, t0 + b
                if any(ta - 600 < e1 and tb + 600 > e0 for e0, e1 in avoid): continue
                if not (np.all(dqh[a:b] & 3 == 3) and np.all(dql[a:b] & 3 == 3) and np.all(ih[a:b] & 1) and np.all(il[a:b] & 1)): continue
                xs = slice(a * 4096, b * 4096)
                if np.isfinite(h[xs]).all() and np.isfinite(l[xs]).all(): ok.append(k)
            if len(ok) >= PER_PAIR:
                for k in sorted(rng.choice(ok, PER_PAIR, replace=False)):
                    xs = slice(k * PIECE * 4096, (k + 1) * PIECE * 4096)
                    np.savez(os.path.join(DATA, 'pieces', f'{run}_{gps}_{t0 + k * PIECE}.npz'), H1=h[xs], L1=l[xs], gps=t0 + k * PIECE)
                have += 1; pairs.add(str(gps))
            print(run, gps, 'clean pieces', len(ok), 'kept' if len(ok) >= PER_PAIR else 'skipped', flush=True)
            for p in paths.values(): os.remove(p)
    print('tries', tries, 'pairs', len(pairs))


def score():
    import chirpnet
    net = chirpnet.ChirpNet(os.path.join(_P.MODEL, 'chirpnet_weights.npz')); thr = net.meta['threshold']
    files = sorted(f for f in os.listdir(os.path.join(DATA, 'pieces')) if f.endswith('.npz'))
    per, allb = {}, []
    for i, fn in enumerate(files):
        z = np.load(os.path.join(DATA, 'pieces', fn))
        H, _ = C.gate(C.whiten(z['H1'], C.FS_IN)); L, _ = C.gate(C.whiten(z['L1'], C.FS_IN))
        core = slice(2 * C.FS, 30 * C.FS); hc, lc = H[core], L[core]; m = len(hc); b = []
        for ks in range(3, 26):
            ls = np.roll(lc, ks * C.FS); starts = range(0, m - C.WIN, 256)
            x = np.stack([np.stack([hc[s:s + C.WIN], ls[s:s + C.WIN]]) for s in starts]).astype(np.float32)
            b.append(np.concatenate([net.statistic(x[j:j + 64])[0][:, 0] for j in range(0, len(x), 64)]))
        b = np.concatenate(b); allb.append(b); per[fn[:-4]] = {'max': float(b.max()), 'n': int(len(b))}
        print(i, fn, 'max %.2f' % b.max(), flush=True)
    b = np.concatenate(allb); live = len(b) * 256 / C.FS
    o12 = net.bg; o12_live = net.bg_live_s
    both = np.concatenate([o12, b]); new_thr = float(np.ceil((both.max() + 0.5) * 2) / 2)
    res = {'protocol': __doc__.split('\n\n')[0], 'n_pieces': len(files), 'n_windows': int(len(b)), 'livetime_s': float(live),
           'max': float(b.max()), 'quantiles': {q: float(np.quantile(b, float(q))) for q in ('0.5', '0.99', '0.999', '0.9999')},
           'top10': np.sort(b)[-10:][::-1].round(3).tolist(), 'n_above_7': int(np.sum(b >= thr)), 'threshold_now': thr,
           'o1o2': {'n_windows': int(len(o12)), 'livetime_s': float(o12_live), 'max': float(o12.max())},
           'threshold_rule': 'loudest of (O1/O2 background + O3 dev background) + 0.5, rounded up to the next 0.5 (the rule that set 7.0)',
           'threshold_new': new_thr, 'pieces': per}
    json.dump(res, open(os.path.join(_P.SRC, 'logs', 'o3_dev_background.json'), 'w'), indent=1)
    np.save(os.path.join(_P.WORK, 'o3_dev_bg.npy'), b.astype(np.float32))
    print(json.dumps({k: v for k, v in res.items() if k != 'pieces'}, indent=1))


if __name__ == '__main__':
    {'fetch': fetch, 'score': score}[sys.argv[1]]()
