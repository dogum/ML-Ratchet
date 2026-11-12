"""TransitNet's reading pipeline, identical for simulated and real stars: clean and normalize each sector, remove the star's
slow variability with a running median that skips the candidate's transits, refine the ephemeris, fold, and bin into views:
  local     +-2.5 transit durations around the transit, 201 bins
  secondary +-2.5 durations around phase 0.5, 101 bins (an eclipsing binary's second eclipse)
  global    the whole orbit, 401 bins
  odd/even  odd and even transits separately, +-1.5 durations, 101 bins each (a binary found at half its period)
Each view is (flux - 1) / scale, where scale is the measured depth (or three times the noise, if larger), with a channel
for which bins have data. Scalars: period, duration, depth scale, noise, number of transits, and the star's TIC values."""
import json, glob, os, numpy as np
# Light-curve store used by the training scripts (files_of); not needed for inference.
D = os.path.join(os.environ.get('ML_RATCHET_WORK') and os.path.join(os.environ['ML_RATCHET_WORK'], 'transitnet')
                 or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'work'), 'data', ''); LC = D + 'lc/'
NL, NS, NG, NO, NC = 201, 101, 401, 101, 101

_INDEX = None
def files_of(tic):
    global _INDEX
    if _INDEX is None:
        import os; _INDEX = {}
        for n in sorted(os.listdir(LC)):
            if n.endswith('.npz'): _INDEX.setdefault(n.split('_')[0], []).append(LC + n)
    return _INDEX.get(str(tic), [])

def load(tic, sectors=None):
    out = []
    for p in files_of(tic):
        z = np.load(p); s = int(p.split('_s')[-1][:3])
        if sectors is not None and s not in sectors: continue
        t, f, e, q = z['time'], z['flux'].astype(np.float64), z['err'].astype(np.float64), z['q']
        m = (q == 0) & np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (f > 0)
        if m.sum() < 500: continue
        c1, c2 = z['c1'].astype(np.float64)[m], z['c2'].astype(np.float64)[m]
        med = np.median(f[m]); out.append(dict(t=t[m], f=f[m] / med, e=e[m] / med, c1=c1, c2=c2, sector=s, hdr=json.loads(str(z['hdr']))))
    return out

def segments(t, gap=0.5):
    br = np.where(np.diff(t) > gap)[0] + 1; return np.split(np.arange(len(t)), br)

def detrend(t, f, intr, window=1.0, binw=1 / 48):
    """running median over `window` days of 30-min bin means of the out-of-transit points, per continuous segment"""
    from scipy.ndimage import median_filter
    trend = np.ones_like(f)
    for idx in segments(t):
        tt, ff, oo = t[idx], f[idx], ~intr[idx]
        if oo.sum() < 20: trend[idx] = np.median(ff); continue
        b = np.floor((tt - tt[0]) / binw).astype(int); nb = b.max() + 1
        cnt = np.bincount(b[oo], minlength=nb); sm_ = np.bincount(b[oo], ff[oo], minlength=nb)
        ok = cnt >= 3
        if ok.sum() < 3: trend[idx] = np.median(ff[oo]); continue
        cen = (np.arange(nb) + 0.5) * binw + tt[0]; bm = np.interp(cen, cen[ok], sm_[ok] / cnt[ok])
        sz = max(3, int(round(window / binw)) | 1); sm = median_filter(bm, size=sz, mode='nearest')
        trend[idx] = np.interp(tt, cen, sm)
    return f / trend

def detrend_c(t, c, intr, window=1.0):
    """centroid (pixels) minus its running median, out-of-transit points only; NaN where the centroid is missing"""
    ok = np.isfinite(c); out = np.full_like(c, np.nan)
    if ok.sum() < 50: return out
    y = c[ok] + 1000.0; out[ok] = y - y / detrend(t[ok], y, intr[ok], window); return out      # offset keeps the ratio filter positive

def phase_dt(t, P, T0):
    return (np.mod(t - T0 + 0.5 * P, P) - 0.5 * P)

def box_snr(t, f, P, T0, dur):
    dt = phase_dt(t, P, T0); inn = np.abs(dt) < 0.4 * dur; out = (np.abs(dt) > 1.0 * dur) & (np.abs(dt) < 3 * dur)
    if inn.sum() < 3 or out.sum() < 10: return -np.inf
    d = np.mean(f[out]) - np.mean(f[inn]); s = np.std(f[out]) / np.sqrt(inn.sum()); return d / max(s, 1e-9)

def refine(t, f, P, T0, dur, sigP=None):
    """grid search around the candidate ephemeris for the strongest box signal (near-transit points only): T0 within half a
    duration, P within 3 sigma of its catalogue error, in steps that move the transits at the data's far end by 1/8 duration"""
    span = t.max() - t.min(); dP = min(3 * (sigP if sigP and np.isfinite(sigP) else 0), 0.002 * P); step = 0.125 * dur * P / max(span, 1)
    nP = int(np.clip(2 * dP / step, 1, 81)) | 1; grid = P + (np.linspace(-dP, dP, nP) if nP > 1 else np.zeros(1))
    tc = T0 + np.round((np.median(t) - T0) / P) * P          # reference epoch in the middle of the data
    near = np.abs(phase_dt(t, P, tc)) < 4 * dur + span * dP / P; tn, fn = t[near], f[near]
    best = (box_snr(tn, fn, P, tc, dur), P, tc)
    for pp in grid:
        for tt in tc + np.linspace(-0.5, 0.5, 11) * dur:
            sv = box_snr(tn, fn, pp, tt, dur)
            if sv > best[0]: best = (sv, pp, tt)
    return best[1], best[2], best[0]

def binned(x, y, lo, hi, n):
    b = np.floor((x - lo) / (hi - lo) * n).astype(int); m = (b >= 0) & (b < n)
    cnt = np.bincount(b[m], minlength=n).astype(float); s = np.bincount(b[m], y[m], minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'): mu = s / cnt
    return mu, cnt

def views(dt, n, f, P, dur):
    """dt: time from the nearest predicted transit centre (days), n: transit number, per point"""
    x = dt / dur
    loc, cl = binned(x, f - 1, -2.5, 2.5, NL)
    dt2 = np.mod(dt, P) - 0.5 * P; sec, cs = binned(dt2 / dur, f - 1, -2.5, 2.5, NS)
    ph = dt / P; glo, cg = binned(ph, f - 1, -0.5, 0.5, NG)
    odd = (n % 2) == 1
    ov, co = binned(x[odd], f[odd] - 1, -1.5, 1.5, NO); ev, ce = binned(x[~odd], f[~odd] - 1, -1.5, 1.5, NO)
    cen = np.abs(np.linspace(-2.5, 2.5, NL)) < 0.3
    depth = -np.nanmean(loc[cen & (cl > 0)]) if (cen & (cl > 0)).any() else 0.0
    noise_pt = 1.4826 * np.median(np.abs(f - np.median(f))); pts_per_bin = max(np.median(cl[cl > 0]) if (cl > 0).any() else 1, 1)
    noise_bin = noise_pt / np.sqrt(pts_per_bin); scale = max(depth if np.isfinite(depth) else 0, 3 * noise_bin, 2e-5)
    nz = lambda v: np.nan_to_num(v / scale, nan=0.0)
    ntr = len(np.unique(n[np.abs(x) < 0.5]))
    V = dict(local=np.stack([nz(loc), (cl > 0).astype(float)]), secondary=np.stack([nz(sec), (cs > 0).astype(float)]), glob=np.stack([nz(glo), (cg > 0).astype(float)]),
             oddeven=np.stack([nz(ov), nz(ev), (co > 0).astype(float), (ce > 0).astype(float)]))
    return V, dict(depth=float(depth if np.isfinite(depth) else 0), scale=float(scale), noise_bin=float(noise_bin), noise_pt=float(noise_pt), ntr=int(ntr))

def star_scalars(hdr):
    g = lambda k, d: float(hdr.get(k)) if hdr.get(k) not in (None, '', 'None') and np.isfinite(float(hdr.get(k))) else d
    teff, logg, rad, tmag = g('TEFF', np.nan), g('LOGG', np.nan), g('RADIUS', np.nan), g('TESSMAG', 11.0)
    have = np.isfinite([teff, logg, rad]).all()
    rho = (10 ** logg / 6.674e-8) * 3 / (4 * np.pi * (rad * 6.957e10)) if have else np.nan      # g/cm^3 from logg and R
    return dict(teff=teff, logg=logg, rad=rad, tmag=tmag, rho=rho, have=float(have))

def scalars(P, dur, info, star):
    ok = star['have'] > 0
    return np.array([np.log10(P), np.log10(dur * 24), np.log10(info['scale']), np.log10(info['noise_bin'] / info['scale']), np.log10(max(info['ntr'], 1)),
                     (star['tmag'] - 10) / 3, (star['teff'] - 5500) / 1500 if ok else 0, (star['logg'] - 4.3) / 0.5 if ok else 0,
                     np.log10(star['rho']) if ok else 0, star['have']], np.float32)

def align(secs_fd, P, T0, dur, sigP):
    """per-sector transit-time search around the catalogue prediction, then a linear ephemeris through the sectors where the
    signal is clear (box SNR >= 7). Catalogue periods drift: a 1-minute period error is hours after a few hundred orbits."""
    hits = []
    for t, fd in secs_fd:
        tm = np.median(t); n = np.round((tm - T0) / P); tc = T0 + n * P
        W = float(np.clip(max(2 * dur, abs(n) * 5 * (sigP if sigP and np.isfinite(sigP) else 1e-5 * P)), 2 * dur, 0.15 * P))
        offs = np.arange(-W, W + 1e-9, 0.1 * dur); near = np.abs(phase_dt(t, P, tc)) < W + 3 * dur; tn, fn = t[near], fd[near]
        if len(tn) < 50: continue
        sn = np.array([box_snr(tn, fn, P, tc + o, dur) for o in offs]); k = int(np.argmax(sn))
        if sn[k] >= 7: hits.append((n, tc + offs[k], sn[k]))
    if len(hits) >= 2:
        nn, tt, ww = (np.array(v) for v in zip(*hits)); A = np.stack([np.ones_like(nn), nn], 1) * ww[:, None]; c = np.linalg.lstsq(A, tt * ww, rcond=None)[0]
        if np.ptp(nn) > 0: return float(c[1]), float(c[0]), len(hits)
    if len(hits) == 1: return P, float(hits[0][1] - hits[0][0] * P), 1
    return P, T0, 0

def read(secs, P, T0, dur, sigP=None, others=(), do_align=True):
    """sectors (from load or a simulation) and a candidate ephemeris -> views and info (refined ephemeris, depth, noise).
    others: (P, T0, dur) of other known transiting bodies of the same star, whose transits are cut out first"""
    win = float(np.clip(3 * dur, 0.75, 2.5)); parts = []; cparts = []
    for s in secs:
        t, f = s['t'], s['f']; c1, c2 = s.get('c1', np.full(len(t), np.nan)), s.get('c2', np.full(len(t), np.nan))
        for (Po, To, do) in others:
            keep = np.abs(phase_dt(t, Po, To)) > 0.6 * do + 0.01; t, f, c1, c2 = t[keep], f[keep], c1[keep], c2[keep]
        if len(t) > 200: parts.append((t, f)); cparts.append((c1, c2))
    first = [(t, detrend(t, f, np.abs(phase_dt(t, P, T0)) < 0.75 * dur + 0.02, window=win)) for t, f in parts]
    nal = 0
    if do_align: P, T0, nal = align(first, P, T0, dur, sigP)
    t = np.concatenate([p[0] for p in parts]); f = np.concatenate([p[1] for p in parts]); o = np.argsort(t); t, f = t[o], f[o]
    fd = np.concatenate([detrend(tt, ff, np.abs(phase_dt(tt, P, T0)) < 0.75 * dur + 0.02, window=win) for tt, ff in parts])[o]
    P, T0, snr = refine(t, fd, P, T0, dur, sigP=1e-6 * P if nal >= 2 else sigP)
    cd = [np.concatenate([detrend_c(tt, cc[j], np.abs(phase_dt(tt, P, T0)) < 0.75 * dur + 0.02, window=win) for (tt, ff), cc in zip(parts, cparts)])[o] for j in (0, 1)]
    mad = 1.4826 * np.median(np.abs(fd - np.median(fd))); up = fd - np.median(fd) > 6 * mad; t, fd = t[~up], fd[~up]; cd = [c[~up] for c in cd]   # flares, cosmic rays
    dt = phase_dt(t, P, T0); n = np.round((t - T0) / P).astype(int)
    V, info = views(dt, n, fd, P, dur); info.update(P=P, T0=T0, dur=dur, n_pts=len(t), n_aligned=nal, snr=float(snr))
    V['cent'], info['cent_noise'] = centroid_view(dt / dur, cd, info['scale'])
    return V, info

def centroid_view(x, cd, scale):
    """in-transit motion of the flux-weighted centroid, column and row, +-2.5 durations, 101 bins, divided by the depth scale:
    a dip on a star d pixels from the light's centre moves the centroid by about depth x d, so the view reads d in pixels"""
    rows = []; noise = []
    for c in cd:
        ok = np.isfinite(c); mu, cnt = binned(x[ok], c[ok], -2.5, 2.5, NC); rows.append(np.nan_to_num(mu / scale)); rows.append((cnt > 0).astype(float))
        oo = ok & (np.abs(x) > 1.5); npb = max(np.median(cnt[cnt > 0]) if (cnt > 0).any() else 1, 1)
        noise.append(1.4826 * np.median(np.abs(c[oo] - np.median(c[oo]))) / np.sqrt(npb) if oo.sum() > 20 else 1.0)
    return np.stack([rows[0], rows[2], rows[1], rows[3]]), float(np.mean(noise) / scale)
