"""Simulated training banks and frozen validation sets.
usage: python simbank.py <physics> <name> <n> <seed> <noise: train|val>
Each example: a random interface earthquake (tsim.random_source), the DART network of a random date 2006-2025 (NDBC
deployment history, sites with Green's functions), waveforms by superposition, laid into real quiet DART noise windows
(noise_bank.py), then the same station selection and features as real events (features.py)."""
import os, sys, json, time, datetime as dt, numpy as np
from _paths import DATA, BANK
import tsim, features as F
PHYS = {
    'p0': dict(delay_frac=0.0, delay_sd=0.0, amp_sd=0.0, rupture_vel=None),      # textbook: linear SWE, instantaneous rupture
}
def load_phys():
    fn = os.path.join(DATA, 'physics_versions.json')
    if not os.path.exists(fn):   # a fresh work folder: the shipped record of the physics versions (p1 from the twin test)
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'physics_versions.json')
    if os.path.exists(fn): PHYS.update(json.load(open(fn)))
    return PHYS

def active_sites(sites, gf, day):
    return [s for s in sites if s['site'] in gf.G and any(a <= day and (b == '' or day <= b) for a, b in s['spans'])]

def make(phys, n, seed, noise_split, gf=None, geo=None, verbose=True):
    P = load_phys()[phys]; rng = np.random.default_rng(seed)
    geo = geo or tsim.Geometry(); gf = gf or tsim.GFBank()
    sites = json.load(open(os.path.join(DATA, 'dart_sites.json')))
    nz = np.load(os.path.join(BANK, 'noise_windows.npz')); sel = nz['is_val'] if noise_split == 'val' else ~nz['is_val']
    NE, NM = nz['eta'][sel].astype(np.float32), nz['mask'][sel]
    days = [(dt.date(2006, 1, 1) + dt.timedelta(days=int(d))).isoformat() for d in rng.integers(0, 7300, 400)]
    X = np.zeros((n, F.K, F.NCH, tsim.NW), np.float16); S = np.zeros((n, F.K, F.NSC), np.float32); M = np.zeros((n, F.K), bool)
    TH = np.zeros((n, 4), np.float32); INFO = []
    ta_cache = {}
    i = 0; tries = 0; t0 = time.time()
    while i < n:
        tries += 1
        src = tsim.random_source(geo, rng); fr = src['frame']
        day = days[rng.integers(len(days))]
        act = active_sites(sites, gf, day)
        cen = [u for u in (fr['units'][tsim.HALF], fr['units'][tsim.NCOL + tsim.HALF]) if u >= 0]
        delays = None
        if P.get('rupture_vel'):
            delays = np.abs(src['col_s']) / P['rupture_vel'] / 60.0
        cands = []
        for s in act:
            key = (s['site'], cen[0]); ta = ta_cache.get(key)
            if key not in ta_cache: ta = ta_cache[key] = gf.arrival(s['site'], cen)
            if ta is None or ta > F.TA_MAX: continue
            k = rng.integers(len(NE)); ne, nm = NE[k], NM[k]
            eta = gf.wave(s['site'], src['units'], src['slips'], delays)
            # nuisance physics (p1+): travel-time delay proportional to arrival time, station amplitude scatter
            if P['delay_frac'] or P['delay_sd']:
                d = ta * P['delay_frac'] + P['delay_sd'] * rng.standard_normal()
                tt = np.arange(len(eta)) - d; eta = np.interp(tt, np.arange(len(eta)), eta, left=0.0)
            if P['amp_sd']: eta = eta * np.exp(P['amp_sd'] * rng.standard_normal())
            if P.get('amp_gain') or P.get('amp_sd_log10'):
                eta = eta * P.get('amp_gain', 1.0) * 10 ** (P.get('amp_sd_log10', 0.0) * rng.standard_normal())
            obs = (eta + ne).astype(np.float32)
            x, sc, cov, scale = F.station_block(gf, s['site'], fr['units'], ta, obs, nm)
            if x is None: continue
            cands.append((ta, cov, scale, (x, sc, s['site'])))
        chosen = F.select(cands)
        if not chosen: continue
        for k, c in enumerate(chosen):
            X[i, k] = c[3][0]; S[i, k] = c[3][1]; M[i, k] = True
        TH[i] = src['theta']; INFO.append(dict(day=day, sites=[c[3][2] for c in chosen], lon=src['lon'], lat=src['lat'], zone=geo.chains[fr['chain']][0]['zone']))
        i += 1
        if verbose and i % 2000 == 0: print(i, n, 'tries', tries, '%.0fs' % (time.time() - t0), flush=True)
    return X, S, M, TH, INFO

if __name__ == '__main__':
    phys, name, n, seed, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    out = os.path.join(BANK, f'bank_{name}.npz')
    X, S, M, TH, INFO = make(phys, n, seed, ns)
    np.savez(out, X=X, S=S, M=M, TH=TH, info=json.dumps(INFO), phys=phys)
    print('saved', out, X.shape)
