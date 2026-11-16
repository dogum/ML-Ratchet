"""Dev check for the neutron-star chirp mass (15 Nov 2026): simulated inspirals injected into real O2 noise that is NOT
exam data, read by the released network (ChirpNetBNS.scan) and by the matched-filter refinement (BNSMatchedFilter).

GW170817's exam reading (1.136 vs LIGO's 1.1975, 5% low, outside a narrow 68% bar) is one event. This asks whether the
same failure shows on dev: do the network's chirp masses miss by percents, and do its 68% bars cover 68%, for
GW170817-like injections in real O2 noise? And does the matched filter fix it?

Noise: GWOSC O2 bulk files (4096 s, 4 kHz, H1 + L1) at times drawn by a fixed seed, more than one day from every O2
event GWOSC lists (GW170817 included), passing DATA and CBC_CAT1, with no hardware injection and no gaps; one 2048 s
piece per file pair. Injections: TaylorF2 with aligned spins (|chi| <= 0.05) and tides (Lambda-tilde 0-1000), so the
templates (no spin, no tides) do not match them exactly; detector-frame chirp mass 0.9-1.8, eta 0.2-0.25, network SNR
12-35 split at random between the detectors, H1-L1 delay within 10 ms, random phases. Odd-numbered injections fit the
matched filter's widen-only interval factor; even-numbered ones check it.

    python bns_mf_dev.py fetch                 # ~1 GB downloads, deleted after cutting; pieces to work/data_o2dev/
    python bns_mf_dev.py run 0 3 &  python bns_mf_dev.py run 1 3 &  python bns_mf_dev.py run 2 3
    python bns_mf_dev.py report                # writes logs/bns_mf_dev.json
"""
import _paths as _P  # portable paths (see _paths.py)
import calendar
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, _P.MODEL); sys.path.insert(0, _P.SRC)
import chirpnet as C  # noqa: E402
import o3_dev_background as O3  # noqa: E402

DATA = os.path.join(_P.WORK, 'data_o2dev'); OUT = os.path.join(_P.WORK, 'bns_mf_dev')
O2 = (1164556817, 1187733618); SEED, N_PIECES, PIECE, N_INJ = 2027, 4, 2048, 120


def o2_events():
    out = []; url = f'{O3.API}/runs/O2/events'
    while url:
        page = O3.get(url)
        for e in page['results']:
            m = re.match(r'GW(\d{2})(\d{2})(\d{2})', e['name'])
            if m:
                t = calendar.timegm((2000 + int(m[1]), int(m[2]), int(m[3]), 0, 0, 0)) - O3.GPS_UNIX; out.append((t - 86400, t + 2 * 86400))
        url = page.get('next')
    return out


def fetch():
    os.makedirs(DATA, exist_ok=True); avoid = o2_events(); rng = np.random.default_rng(SEED)
    have = len([f for f in os.listdir(DATA) if f.endswith('.npz')])
    for gps in rng.permutation(np.arange(O2[0] // 4096 + 1, O2[1] // 4096 - 1)) * 4096:
        if have >= N_PIECES: break
        gps = int(gps)
        if any(gps - 100 < e1 and gps + 4196 > e0 for e0, e1 in avoid) or os.path.exists(os.path.join(DATA, f'O2_{gps}.npz')): continue
        paths = {d: os.path.join(DATA, f'{d}-{gps}.hdf5') for d in ('H1', 'L1')}
        url = lambda d: f'https://gwosc.org/archive/data/O2_4KHZ_R1/{gps // 1048576 * 1048576}/{d[0]}-{d}_GWOSC_O2_4KHZ_R1-{gps}-4096.hdf5'
        if not all(os.path.exists(p) or O3.download(url(d), p) for d, p in paths.items()):
            print(gps, 'missing file', flush=True); [os.remove(p) for p in paths.values() if os.path.exists(p)]; continue
        (h, t0, dqh, ih), (l, _, dql, il) = O3.read(paths['H1']), O3.read(paths['L1'])
        for a in (0, 1024, 2048):
            b = a + PIECE; xs = slice(a * 4096, b * 4096)
            if (np.all(dqh[a:b] & 3 == 3) and np.all(dql[a:b] & 3 == 3) and np.all(ih[a:b] & 1) and np.all(il[a:b] & 1)
                    and np.isfinite(h[xs]).all() and np.isfinite(l[xs]).all()):
                np.savez(os.path.join(DATA, f'O2_{gps}.npz'), H1=h[xs], L1=l[xs], gps=t0 + a); have += 1; print(gps, 'piece at', a, flush=True); break
        else:
            print(gps, 'no clean 2048 s piece', flush=True)
        for p in paths.values(): os.remove(p)


def draw(i):
    r = np.random.default_rng(SEED * 1000 + i)
    return dict(i=i, piece=int(r.integers(N_PIECES)), tc=float(r.uniform(600, 1900)), mc=float(r.uniform(0.9, 1.8)), eta=float(r.uniform(0.2, 0.25)),
                chi=float(r.uniform(-0.05, 0.05)), lam=float(r.uniform(0, 1000)), snr=float(r.uniform(12, 35)), frac=float(r.uniform(0.2, 0.8)),
                dt=float(r.uniform(-0.01, 0.01)), ph=r.uniform(0, 2 * np.pi, 2).tolist())


def inject(x, fs, p, k, snr):
    """Add a TaylorF2 inspiral (spins, tides) to raw strain x with coalescence at p['tc'] (+dt for L1) and the given SNR."""
    n = len(x); f = np.fft.rfftfreq(n, 1 / fs); fp, pp = C.psd(x, fs, seg=8.0); S = np.interp(f, fp, pp)
    h = C.taylorf2(f, p['mc'], p['eta'], p['chi'], p['lam']); taper = np.clip((f - 18) / 4, 0, 1); h = h * taper
    tc = p['tc'] + (p['dt'] if k == 1 else 0.0); h = h * np.exp(-2j * np.pi * f * tc + 1j * p['ph'][k])
    m = (f >= 20) & (f <= 1000); rho = np.sqrt(4 * (f[1] - f[0]) * np.sum(np.abs(h[m]) ** 2 / S[m]))
    return x + np.fft.irfft(h * snr / rho, n) * fs          # continuous FT -> samples: irfft(H) * fs


def run(worker, nw):
    os.makedirs(OUT, exist_ok=True)
    bns = C.ChirpNetBNS(os.path.join(_P.MODEL, 'chirpnet_bns_weights.npz')); pieces = sorted(f for f in os.listdir(DATA) if f.endswith('.npz'))
    for i in range(worker, N_INJ, nw):
        out = os.path.join(OUT, f'{i:04d}.json')
        if os.path.exists(out): continue
        t = time.time(); p = draw(i); z = np.load(os.path.join(DATA, pieces[p['piece']])); g0 = float(z['gps'])
        rs = np.sqrt([p['frac'], 1 - p['frac']]) * p['snr']
        x = [inject(z[d].astype(np.float64), C.FS_IN, p, k, rs[k]) for k, d in enumerate(('H1', 'L1'))]
        W = [C.whiten_bns(v) for v in x]; k0 = int((p['tc'] - 60) * C.FSB); k1 = k0 + 120 * C.FSB
        hits = bns.scan(W[0][k0:k1], W[1][k0:k1], gps_start=g0 + k0 / C.FSB, whitened=True)
        hit = next((h for h in hits if abs(h['gps'] - (g0 + p['tc'])) < 2.0), None); rec = dict(p, found=hit is not None)
        if hit:
            rec.update(nn_mc=hit['chirp_mass'], nn_68=hit['chirp_mass_68'], nn_score=hit['score'], nn_dt=hit['gps'] - g0 - p['tc'])
            mf = C.BNSMatchedFilter(x[0], x[1], g0, hit['gps']).fit(hit['chirp_mass']); rec.update(mf_mc=mf['chirp_mass'], mf_68=mf['chirp_mass_68'], mf_snr=mf['network_snr'], mf_eta=mf['eta'])
        rec['seconds'] = time.time() - t; json.dump(rec, open(out, 'w'))
        print(i, 'mc %.4f' % p['mc'], 'nn %.4f' % rec.get('nn_mc', np.nan), 'mf %.4f' % rec.get('mf_mc', np.nan), '%.0fs' % rec['seconds'], flush=True)


def summary(R, widen=1.0):
    mc = np.array([r['mc'] for r in R]); out = {'n': len(R)}
    for k in ('nn', 'mf'):
        e = np.array([r[k + '_mc'] for r in R]) / mc - 1; lo = np.array([r[k + '_68'][0] for r in R]); hi = np.array([r[k + '_68'][1] for r in R]); est = np.array([r[k + '_mc'] for r in R])
        if k == 'mf': lo, hi = est * (lo / est) ** widen, est * (hi / est) ** widen
        out[k] = {'med_abs_err': float(np.median(np.abs(e))), 'bias': float(np.median(e)), 'p90_abs_err': float(np.quantile(np.abs(e), 0.9)),
                  'cov68': float(np.mean((lo <= mc) & (mc <= hi))), 'half_width_med': float(np.median((hi - lo) / 2 / est))}
    return out


def report():
    R = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith('.json')]; F = [r for r in R if r['found']]
    fitset, check = [r for r in F if r['i'] % 2 == 1], [r for r in F if r['i'] % 2 == 0]
    w = 1.0
    while w < 50:          # widen-only factor for the matched filter's 68% range, fitted on the odd injections
        mc = np.array([r['mc'] for r in fitset]); est = np.array([r['mf_mc'] for r in fitset])
        lo = est * (np.array([r['mf_68'][0] for r in fitset]) / est) ** w; hi = est * (np.array([r['mf_68'][1] for r in fitset]) / est) ** w
        if np.mean((lo <= mc) & (mc <= hi)) >= 0.683: break
        w *= 1.05
    like = [r for r in F if 1.1 <= r['mc'] <= 1.3]
    res = {'protocol': __doc__.split('\n\n')[0], 'n_injected': len(R), 'n_found': len(F), 'mf_widen': float(w),
           'all': summary(F), 'check_raw': summary(check), 'check_widened': summary(check, w), 'gw170817_like_mc_1.1_1.3': summary(like, w),
           'rows': [{k: r.get(k) for k in ('i', 'mc', 'snr', 'found', 'nn_mc', 'nn_68', 'mf_mc', 'mf_68', 'mf_snr')} for r in R]}
    json.dump(res, open(os.path.join(_P.SRC, 'logs', 'bns_mf_dev.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'fetch': fetch()
    elif a[0] == 'run': run(int(a[1]), int(a[2]))
    elif a[0] == 'report': report()
