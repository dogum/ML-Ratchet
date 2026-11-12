"""Null windows for the detection head: REAL quiet buoy windows with nothing injected (the same 'quiet-ish' pool the
injection banks draw from: no >= 64 kt storm window nearby, long-period energy below the buoy's 60th percentile).
Split by year so the real false-alarm test is never seen: null_train = 2008-2023, null_val = 2024, quiet test = 2025.
v2 option: inject 0-2 crossing swells whose 0.06 Hz arrival is >= 48 h from the window centre (still 'no centred swell')."""
import os, sys, json, numpy as np
import simbank as SB, swellsim as SS, fetch, geo

def year_of(h): return int(np.datetime64(int(h), 'h').astype('datetime64[Y]').astype(int) + 1970)

def crossing(rng, blat, blon, th):
    E3 = np.zeros((SS.NT, 34)); C3 = E3.copy(); S3 = E3.copy()
    for _ in range(rng.choice([0, 1, 1, 2])):
        st = SB.place_storm(rng, blat, blon)
        if st is None: continue
        p = SS.sample_storm(rng, prior='v2'); off = rng.choice([-1, 1]) * rng.uniform(48, 150)
        E2, Ec2, Es2 = SS.simulate(blat, blon, st[0], st[1], -(st[2] / SS.cg(SS.F_REF) / 3600) + off, p, th, rng=rng)
        q = np.round(E2, 2); r = np.where(E2 > 0, q / np.maximum(E2, 1e-12), 0); a, b, c = SS.to_coarse(q, Ec2 * r, Es2 * r)
        E3 += a; C3 += b; S3 += c
    return E3, C3, S3

def build(name, n, seed, years, version='v1'):
    BG = SB.Backgrounds(); rng = np.random.default_rng(seed); pool = []
    for b in BG.buoys:
        d = BG.B[b]
        for s in d['starts']:
            if year_of(d['h0'] + s * SS.DT_H + SS.HALF_H) in years: pool.append((b, int(s)))
    print(name, 'pool', len(pool), flush=True)
    idx = rng.choice(len(pool), min(n, len(pool)), replace=False); X = np.zeros((len(idx), 3, SS.NT, 34), np.float16); Y = np.full((len(idx), len(SB.YCOLS)), np.nan)
    th = np.arange(-SS.HALF_H, SS.HALF_H, 1.0)
    for k, i in enumerate(idx):
        b, s = pool[i]; d = BG.B[b]; sl = slice(s, s + SS.NT); Eb, Cb, Sb = (d[c][sl].astype(np.float32) for c in 'ECS')
        if version >= 'v2':
            blat, blon = BG.pos[b]; E3, C3, S3 = crossing(rng, blat, blon, th); SB.lowcut(rng, Eb, E3, C3, S3); Et = Eb + E3
            Cb = np.where(np.isfinite(Cb), (np.nan_to_num(Eb) * np.nan_to_num(Cb) + C3) / np.maximum(Et, 1e-9), np.nan)
            Sb = np.where(np.isfinite(Sb), (np.nan_to_num(Eb) * np.nan_to_num(Sb) + S3) / np.maximum(Et, 1e-9), np.nan); Eb = Et
        X[k] = np.stack([Eb, Cb, Sb]); Y[k, 5] = fetch.BUOYS.index(b); Y[k, 13] = d['h0'] + s * SS.DT_H + SS.HALF_H
    np.save(os.path.join(SB.BANK, f'{name}_x.npy'), X); np.save(os.path.join(SB.BANK, f'{name}_y.npy'), Y)
    json.dump(dict(cols=SB.YCOLS, version=version, n=len(X), seed=seed, years=sorted(years), null=True), open(os.path.join(SB.BANK, f'{name}.json'), 'w'))
    print('done', name, len(X), flush=True)

if __name__ == '__main__':
    name, n, seed, y0, y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    build(name, n, seed, set(range(y0, y1 + 1)), sys.argv[6] if len(sys.argv) > 6 else 'v1')
