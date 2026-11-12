"""TransitNet training data: real TESS light curves of ordinary stars (the noise pool), with a simulated planet, eclipsing
binary or blended (background) eclipsing binary injected, or nothing at all, then read by the same pipeline as the real
exam stars from a catalogue-like candidate ephemeris (period and epoch with realistic errors, a duration good to ~15%).
Classes: 0 planet, 1 eclipsing binary on the target, 2 blended eclipsing binary (diluted by the target's light), 3 nothing.
Planet labels: log10 radius ratio, impact parameter, log10 a/R*, log10 duration (hours)."""
import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, glob, time, numpy as np
sys.path.insert(0, _P.SRC); import pipeline as PL, transitsim as TS
D = _P.WORK + '/data/'; CLS_P = np.array([0.45, 0.2, 0.2, 0.15])

def pool_index():
    M = json.load(open(D + 'targets.json')); pool = []
    for tic in M['pool']:
        for p in glob.glob(f'{PL.LC}{tic}_s*.npz'): pool.append(p)
    return sorted(set(pool))

class Pool:
    def __init__(self, files):
        self.files = files; self.cache = {}; self.tmag = np.full(len(files), np.nan)
    def get(self, i):
        if i not in self.cache:
            tic = os.path.basename(self.files[i]).split('_')[0]; s = int(self.files[i].split('_s')[-1][:3]); L = PL.load(tic, {s})
            self.cache[i] = L[0] if L else None
            if L: self.tmag[i] = float(L[0]['hdr'].get('TESSMAG') or np.nan)
        return self.cache[i]

LD_V = 2   # v1: q1 0.25-0.70 (u1+u2 0.50-0.84); v2, calibrated on the dev planets' free-limb-darkening fits (median u1+u2 0.54): q1 0.15-0.55
def ld(rng):
    q1, q2 = (rng.uniform(0.25, 0.7), rng.uniform(0.15, 0.45)) if LD_V == 1 else (rng.uniform(0.15, 0.55), rng.uniform(0.2, 0.45))
    return 2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)

def centroid_shift(m, phi, sep, pa, eff):
    """flux-weighted centroid motion (pixels) when the source holding a fraction phi of the light dims so that the total is m:
    dx = (1 - m) / m * (1 - phi) * (target - source) for a dimming companion; for a dimming target with neighbours holding
    a fraction 1 - phi of the light, the same formula with the neighbours as the reference. eff < 1: centroids measured inside
    an aperture move less than the light does."""
    a = (1 - m) / np.maximum(m, 1e-3) * (1 - phi) * sep * eff; return a * np.cos(pa), a * np.sin(pa)

def draw_rho(rng, star):
    return (star['rho'] if star['have'] > 0 and np.isfinite(star['rho']) else np.exp(rng.normal(np.log(1.4), 0.7))) * np.exp(rng.normal(0, 0.2))

def candidate(rng, P, T0, T14, half=False):
    rel = np.exp(rng.uniform(np.log(1e-7), np.log(3e-5))); sig = rel * P; true_err = sig * (5 if rng.random() < 0.15 else 1)
    Pc = (P / 2 if half else P) + rng.normal(0, true_err); return Pc, T0 + rng.normal(0, 0.15 * T14), T14 * np.exp(rng.normal(0, 0.15)), sig

def sample(rng, pool, cls):
    i = rng.integers(len(pool.files)); s0 = pool.get(i)
    if s0 is None: return None
    secs = [s0]; nsec = rng.choice([1, 2, 3], p=[0.35, 0.35, 0.3]); tm = float(s0['hdr'].get('TESSMAG') or 11)
    tries = 0
    while len(secs) < nsec and tries < 30:
        tries += 1; j = rng.integers(len(pool.files)); s = pool.get(j)
        if s is not None and abs(float(s['hdr'].get('TESSMAG') or 11) - tm) < 1.0 and all(abs(np.median(s['t']) - np.median(x['t'])) > 20 for x in secs): secs.append(s)
    star = PL.star_scalars(s0['hdr']); t_all = np.concatenate([s['t'] for s in secs]); lab = dict(cls=cls)
    crowd = float(s0['hdr'].get('CROWDSAP') or 1.0); eff = rng.uniform(0.3, 1.0); pa = rng.uniform(0, 2 * np.pi)
    sep = np.exp(rng.uniform(np.log(0.03), np.log(3.0))) if cls == 2 else rng.uniform(0.5, 3.0)   # pixels: binary to target, or target to its neighbours' light
    P = np.exp(rng.uniform(np.log(0.4), np.log(25))); T0 = rng.uniform(secs[0]['t'].min(), secs[0]['t'].min() + P)
    if cls == 0:
        rho = draw_rho(rng, star); aR = TS.a_over_R(P, rho); k = np.exp(rng.uniform(np.log(0.01), np.log(0.3)))
        e, w = (min(rng.beta(0.867, 3.03), 0.8), rng.uniform(0, 2 * np.pi)) if P > 4 else (0.0, np.pi / 2)
        if aR * (1 - e) < 2.5: e = 0.0
        b = rng.uniform(0, 1) if rng.random() < 0.9 else rng.uniform(1, 1 + k)
        cosi = b * (1 + e * np.sin(w)) / ((1 - e * e) * aR)
        if cosi >= 1: return None
        inc = np.arccos(cosi); u1, u2 = ld(rng); third = rng.uniform(0, 0.05); T14 = TS.t14(P, aR, k, b, inc, e, w)
        if not T14 > 0: return None
        def model(t, T0=T0, P=P, k=k, aR=aR, inc=inc, u1=u1, u2=u2, e=e, w=w, third=third, T14=T14):
            out = np.ones_like(t); m = np.abs(PL.phase_dt(t, P, T0)) < 0.6 * T14 + 0.01        # only near transit (conjunction at T0)
            if m.any(): out[m] = (TS.planet_flux(t[m], T0, P, k, aR, inc, u1, u2, e, w) + third) / (1 + third)
            return out
        lab.update(y=[np.log10(k), b, np.log10(aR), np.log10(T14 * 24)], depth=float(1 - model(np.array([T0]))[0])); half = False
    elif cls in (1, 2):
        rho1 = draw_rho(rng, star); q = rng.uniform(0.15, 1); aR = TS.a_over_R(P, rho1 * (1 + q)); k = rng.uniform(0.2, 1.0); b = rng.uniform(0, 1 + k)
        e, w = (rng.uniform(0, 0.4), rng.uniform(0, 2 * np.pi)) if (P > 3 and rng.random() < 0.5) else (0.0, np.pi / 2)
        if aR * (1 - e) < 1.5 * (1 + k): return None
        cosi = b * (1 + e * np.sin(w)) / ((1 - e * e) * aR)
        if cosi >= 1: return None
        inc = np.arccos(cosi); J = np.exp(rng.uniform(np.log(0.03), 0)); ua, ub = ld(rng), ld(rng); A = 1.2 * q * aR ** -3 * np.sin(inc) ** 2
        third = rng.uniform(0, 0.2) if cls == 1 else np.exp(rng.uniform(np.log(3), np.log(3000)))
        T14 = TS.t14(P, aR, k, b, inc, e, w)
        if not T14 > 0: return None
        model = lambda t: TS.eb_flux(t, T0, P, k, aR, inc, ua, ub, J, e, w, A, third)
        lab.update(y=[np.nan] * 4, depth=float(1 - model(np.array([T0]))[0])); half = rng.random() < 0.15
    else:
        model = None; rho = draw_rho(rng, star); aR = TS.a_over_R(P, rho); T14 = P / np.pi * np.arcsin(min(1, 1 / aR)) * rng.uniform(0.3, 1.0)
        lab.update(y=[np.nan] * 4, depth=0.0); half = False
    if model is not None:
        sim = []
        for s in secs:
            mm = model(s['t'])
            # share of the light held by the source that dims: the background binary's for a blend, the target's otherwise
            ph = 1 / (1 + third) if cls == 2 else 1 - min(0.9, (1 - crowd) + third / (1 + third))
            dx, dy = centroid_shift(mm, ph, sep, pa, eff)
            sim.append(dict(t=s['t'], f=s['f'] * mm, c1=s['c1'] + dx, c2=s['c2'] + dy, hdr=s['hdr']))
        if not (np.abs(PL.phase_dt(t_all, P, T0)) < 0.3 * T14).any(): return None          # no transit falls in the data
    else:
        sim = secs
    Pc, T0c, durc, sig = candidate(rng, P, T0, T14, half)
    V, info = PL.read(sim, Pc, T0c, durc, sigP=sig)
    lab.update(info={k: info[k] for k in ('depth', 'scale', 'noise_bin', 'ntr', 'snr', 'n_aligned', 'cent_noise')}, half=half, P=P, nsec=len(secs))
    return V, PL.scalars(info['P'], durc, info, star), lab

def make(seed, n, out):
    rng = np.random.default_rng(seed); pool = Pool(pool_index()); R = {k: [] for k in ('local', 'secondary', 'glob', 'oddeven', 'cent', 's', 'sc', 'cls', 'y', 'snr', 'depth')}; t = time.time(); done = 0
    while done < n:
        cls = int(rng.choice(4, p=CLS_P)); r = sample(rng, pool, cls)
        if r is None: continue
        V, s, lab = r
        for k in ('local', 'secondary', 'glob', 'oddeven', 'cent'): R[k].append(V[k].astype(np.float16))
        R['sc'].append(np.float32(np.log10(max(lab['info']['cent_noise'], 1e-4))))
        R['s'].append(s); R['cls'].append(cls); R['y'].append(np.array(lab['y'], np.float32)); R['snr'].append(lab['info']['snr']); R['depth'].append(lab['depth']); done += 1
        if len(pool.cache) > 600: pool.cache.clear()
        if done % 500 == 0: print(seed, done, '%.0fs' % (time.time() - t), flush=True)
    np.savez_compressed(out, **{k: np.stack(v) if k not in ('cls', 'snr', 'depth', 'sc') else np.array(v) for k, v in R.items()}, ld_v=LD_V)

if __name__ == '__main__':
    make(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
