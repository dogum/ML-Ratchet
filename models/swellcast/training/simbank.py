"""Simulated training/validation banks: synthetic storm swell injected into REAL NDBC background windows.

Background windows: 240-h windows from the 27 buoys (2008-2025) with >= 70% data, that do not overlap any candidate
storm-event window at that buoy (so no exam or dev window is ever used as background), and whose long-period energy is
below the buoy's 60th percentile ('quiet-ish'; they still hold real wind sea, small swells and instrument noise).
Storm: random distance (log-uniform 700-11,000 km) and bearing such that the great circle to the buoy is open water."""
import os, sys, json, time, argparse, numpy as np
import geo, catalog, fetch, realdata as RD, swellsim as SS, events as EV, tracks as TR
ROOT = catalog.ROOT; BANK = os.path.join(ROOT, 'bank')
TC = (np.arange(SS.NT) * SS.DT_H - SS.HALF_H + SS.DT_H / 2)

class Backgrounds:
    def __init__(self, quiet_pct=60):
        cands = [c for c in catalog.candidates(min_wind=64, dmin=800e3, dmax=10000e3) if c['buoy'] in fetch.BUOYS]
        busy = {}
        for c in cands: busy.setdefault(c['buoy'], []).append(c['t0'] + c['D'] / SS.cg(SS.F_REF) / 3600)
        self.B = {}; nw = SS.NT
        for b in fetch.BUOYS:
            if not os.path.exists(os.path.join(RD.BG, f'{b}.npz')): continue
            h0, E, C, S = RD.load_bg(b); ok = np.isfinite(E[:, 4].astype(np.float32)).astype(float)
            cov = np.convolve(ok, np.ones(nw), 'valid') / nw; starts = np.arange(len(cov))
            sw = np.nansum(np.where(np.isfinite(E[:, 1:11].astype(np.float32)), E[:, 1:11].astype(np.float32), 0), 1) * SS.DF   # m0 0.0375-0.0825 Hz
            idx = np.convolve(sw, np.ones(nw), 'valid') / np.maximum(np.convolve(ok, np.ones(nw), 'valid'), 1)
            centre = h0 + (starts + nw // 2) * SS.DT_H; good = cov >= 0.7
            for t in busy.get(b, []): good &= np.abs(centre - t) > SS.HALF_H + 36 + SS.HALF_H
            if good.sum() < 50: continue
            thr = np.percentile(idx[good], quiet_pct); good &= idx <= thr
            self.B[b] = dict(h0=h0, E=E, C=C, S=S, starts=starts[good][::4])   # every 12 h
        self.buoys = sorted(self.B); S_ = catalog.stations(); self.pos = {b: S_[b][:2] for b in self.buoys}

    def draw(self, rng):
        b = self.buoys[rng.integers(len(self.buoys))]; d = self.B[b]; s = int(rng.choice(d['starts']))
        sl = slice(s, s + SS.NT); return b, d['h0'] + s * SS.DT_H + SS.HALF_H, d['E'][sl].astype(np.float32), d['C'][sl].astype(np.float32), d['S'][sl].astype(np.float32)

def place_storm(rng, blat, blon, dmin=700e3, dmax=11000e3):
    for _ in range(200):
        D = 10 ** rng.uniform(np.log10(dmin), np.log10(dmax)); brg = rng.uniform(0, 360); la, lo = geo.dest(blat, blon, brg, D)
        if abs(la) > 62: continue
        if geo.on_land(la, lo) or not geo.ocean_path(float(la), float(lo), blat, blon): continue
        return float(la), float(lo), float(geo.dist(la, lo, blat, blon)), brg
    return None

def lowcut(rng, Eb, E3, Ec3, Es3):
    thr = 10 ** rng.normal(-1.0, 0.35); Ebz = np.nan_to_num(Eb, nan=1.0)
    for i in range(SS.NT):
        nz = np.where(Ebz[i] >= 0.005)[0]; jc = nz[0] if len(nz) else 0
        if jc == 0: continue
        keep = E3[i, :jc] >= thr; low = np.where(~keep)[0]; cut = low[-1] + 1 if len(low) else 0
        E3[i, :cut] = 0; Ec3[i, :cut] = 0; Es3[i, :cut] = 0

def make_one(rng, BG, version='v1'):
    while True:
        b, hc, Eb, Cb, Sb = BG.draw(rng); blat, blon = BG.pos[b]; st = place_storm(rng, blat, blon)
        if st is not None: break
    la, lo, D, brg = st; p = SS.sample_storm(rng, prior=version)
    trk = None
    if version >= 'v3' and rng.random() < 0.65:
        # v3: a REAL historical hurricane track (IBTrACS 1985-2007, never dev/exam) at its real place relative to this buoy;
        # the label is its peak-wind point and time, the same rule as the real truth
        TK = TR.load(); cand = TK['pairs'].get(b, [])
        if cand:
            trk = TK['tracks'][cand[rng.integers(len(cand))]]; la, lo = trk['lat'][trk['k']], trk['lon'][trk['k']]
            D = float(geo.dist(la, lo, blat, blon)); brg = float(geo.bearing(blat, blon, la, lo))
    jit = rng.uniform(-36, 36); t0_rel = -(D / SS.cg(SS.F_REF) / 3600) - jit
    ws = dict(tc=rng.uniform(-120, 120), sig=rng.uniform(6, 24), hs=rng.uniform(1, 4), tp=rng.uniform(5, 10), dir=rng.uniform(0, 360)) if rng.random() < 0.3 else None
    th = np.arange(-SS.HALF_H, SS.HALF_H, 1.0)
    if trk is not None:
        pt = SS.sample_track_params(rng); E, Ec, Es, lab = SS.simulate_track(blat, blon, trk, t0_rel, pt, th, rng=rng)
        if ws is not None: W, Wc, Wsn = SS.windsea(th, ws); E, Ec, Es = E + W, Ec + Wc, Es + Wsn
        p = dict(p, hs=lab['hs'], tp=lab['tp'], R=lab['R'], speed=lab['speed'], fe=pt['fe'], sig1=np.nan, sig2=np.nan, w_iso=pt['w_iso'])
    else:
        E, Ec, Es = SS.simulate(blat, blon, la, lo, t0_rel, p, th, ws=ws, rng=rng)
    q = np.round(E, 2); r = np.where(E > 0, q / np.maximum(E, 1e-12), 0); E, Ec, Es = q, Ec * r, Es * r          # NDBC 0.01 m^2/Hz resolution
    E3, Ec3, Es3 = SS.to_coarse(E, Ec, Es)
    if version >= 'v2':
        # crossing swells: 0-2 other storms whose 0.06 Hz arrival is >= 48 h from the window centre (the label stays with
        # the centred one); real windows often hold a second swell from another ocean basin
        for _ in range(rng.choice([0, 1, 1, 2])):
            st2 = place_storm(rng, blat, blon)
            if st2 is None: continue
            p2 = SS.sample_storm(rng, prior=version); off = rng.choice([-1, 1]) * rng.uniform(48, 150)
            E2, Ec2, Es2 = SS.simulate(blat, blon, st2[0], st2[1], -(st2[2] / SS.cg(SS.F_REF) / 3600) + off, p2, th, rng=rng)
            q2 = np.round(E2, 2); r2 = np.where(E2 > 0, q2 / np.maximum(E2, 1e-12), 0); a, b_, c_ = SS.to_coarse(q2, Ec2 * r2, Es2 * r2)
            E3, Ec3, Es3 = E3 + a, Ec3 + b_, Es3 + c_
        # NDBC low-frequency cutoff: below the background's first non-zero bin, added energy survives only above a
        # threshold (median 0.1 m^2/Hz, from the first non-zero values in background spectra), and the cut stays contiguous
        lowcut(rng, Eb, E3, Ec3, Es3)
    Et = Eb + E3; okd = np.isfinite(Cb) & np.isfinite(Sb)
    Ct = np.where(okd, (np.nan_to_num(Eb) * np.nan_to_num(Cb) + Ec3) / np.maximum(Et, 1e-9), np.nan)
    St = np.where(okd, (np.nan_to_num(Eb) * np.nan_to_num(Sb) + Es3) / np.maximum(Et, 1e-9), np.nan)
    Ct = np.where(np.isfinite(Et), Ct, np.nan); St = np.where(np.isfinite(Et), St, np.nan)
    x = np.stack([Et, Ct, St]).astype(np.float16)
    contrast = EV.ridge_contrast(Et, t0_rel, D, TC)
    y = np.array([D / 1e3, t0_rel, geo.bearing(blat, blon, la, lo), p['hs'], contrast if np.isfinite(contrast) else -9, float(fetch.BUOYS.index(b)),
                  p['tp'], p['R'] / 1e3, p['speed'], p['fe'], p['sig1'], p['sig2'], p['w_iso'], hc], np.float64)
    return x, y
YCOLS = ['D_km', 't0_rel_h', 'dir_deg', 'hs_src', 'contrast', 'buoy_idx', 'tp', 'R_km', 'speed', 'fe', 'sig1', 'sig2', 'w_iso', 'bg_centre_h']

def build(name, n, seed, chunk=1000, version='v1'):
    os.makedirs(BANK, exist_ok=True); BG = None; t0 = time.time()
    for c in range(0, n, chunk):
        fn = os.path.join(BANK, f'{name}_c{c // chunk:03d}.npz')
        if os.path.exists(fn): continue
        if BG is None: BG = Backgrounds(); print('backgrounds', {b: len(BG.B[b]['starts']) for b in BG.buoys}, flush=True)
        rng = np.random.default_rng([seed, c // chunk]); X = np.zeros((min(chunk, n - c), 3, SS.NT, 34), np.float16); Y = np.zeros((len(X), len(YCOLS)))
        for i in range(len(X)): X[i], Y[i] = make_one(rng, BG, version)
        np.savez(fn, x=X, y=Y); print(name, c + len(X), round(time.time() - t0), flush=True)
    # consolidate into memory-mappable arrays
    fs = [os.path.join(BANK, f'{name}_c{c // chunk:03d}.npz') for c in range(0, n, chunk)]
    X = np.lib.format.open_memmap(os.path.join(BANK, f'{name}_x.npy'), 'w+', np.float16, (n, 3, SS.NT, 34)); Y = []; k = 0
    for f in fs:
        z = np.load(f); X[k:k + len(z['x'])] = z['x']; Y.append(z['y']); k += len(z['x'])
    X.flush(); np.save(os.path.join(BANK, f'{name}_y.npy'), np.concatenate(Y)); json.dump(dict(cols=YCOLS, version=version, n=n, seed=seed), open(os.path.join(BANK, f'{name}.json'), 'w'))
    for f in fs: os.remove(f)
    print('done', name, n)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('n', type=int); ap.add_argument('seed', type=int); ap.add_argument('--version', default='v1')
    a = ap.parse_args(); build(a.name, a.n, a.seed, version=a.version)
