"""Read the truth catalogue (Yu et al. 2018, ApJS 236, 42; journal machine-readable tables 1 and 2) and pick the real stars:
- exam: a fixed random draw (seed 2026) of 4-year red giants within six log-numax strata, 70 per stratum;
- dev: 3 more per stratum (18 stars), never in the exam;
- noise pool: quiet Kepler dwarfs (DR25 stellar table: logg > 4.25, no KOI, no TCE, >= 1400 d of data), matched to the
  giants' Kepler-magnitude distribution. Dwarfs oscillate above 1000 uHz, far beyond the 283 uHz long-cadence Nyquist
  frequency, so their light curves are real Kepler noise (photon noise, systematics, gaps) with nothing in our band."""
import _paths as _P  # portable paths (see _paths.py)
import json, csv, numpy as np
D = _P.WORK + '/data/'
STRATA = [(8, 15), (15, 25), (25, 40), (40, 70), (70, 130), (130, 250)]

def fw(line, a, b, typ=float):
    s = line[a - 1:b].strip(); return typ(s) if s else np.nan

def read_yu():
    T1 = {}
    for ln in open(D + 'yu2018_t1.mrt.txt'):
        if not ln[:9].strip().isdigit(): continue
        k = int(ln[0:9]); T1[k] = dict(kic=k, kp=fw(ln, 11, 16), qlen=int(fw(ln, 18, 19)), dlen=fw(ln, 21, 27), numax=fw(ln, 29, 34), e_numax=fw(ln, 36, 40),
                                       dnu=fw(ln, 42, 47), e_dnu=fw(ln, 49, 53), amp=fw(ln, 55, 59), e_amp=fw(ln, 61, 65), width=fw(ln, 67, 71), gran=fw(ln, 78, 86))
    for ln in open(D + 'yu2018_t2.mrt.txt'):
        if not ln[:8].strip().isdigit(): continue
        k = int(ln[0:8])
        if k in T1: T1[k].update(teff=fw(ln, 10, 13), e_teff=fw(ln, 15, 17), logg=fw(ln, 19, 23), feh=fw(ln, 31, 35), M=fw(ln, 42, 45), e_M=fw(ln, 47, 50), R=fw(ln, 52, 56), e_R=fw(ln, 58, 61),
                                 M_rgb=fw(ln, 63, 66), R_rgb=fw(ln, 73, 77), M_cl=fw(ln, 84, 87), R_cl=fw(ln, 94, 98), phase=int(fw(ln, 106, 106)))
    return T1

if __name__ == '__main__':
    Y = read_yu(); print('Yu 2018 stars', len(Y))
    long = [r for r in Y.values() if r['dlen'] >= 1200 and np.isfinite(r['teff'])]
    print('>= 1200 d of data', len(long))
    rng = np.random.default_rng(2026); exam, dev = [], []
    for lo, hi in STRATA:
        s = sorted([r['kic'] for r in long if lo <= r['numax'] < hi]); pick = rng.permutation(s)[:73].tolist()
        dev += pick[:3]; exam += pick[3:]; print(f'stratum {lo}-{hi} uHz: {len(s)} stars')
    kp = np.array([r['kp'] for r in long]); kp = kp[np.isfinite(kp)]
    W = [r for r in csv.DictReader(open(D + 'dwarfs_dr25.csv')) if r['st_quarters'].count('1') >= 16 and int(r['kepid']) not in Y]
    wk = np.array([float(r['kepmag']) for r in W]); print('quiet dwarfs', len(W))
    # magnitude-matched pool: for each of 420 random giant magnitudes take the nearest unused dwarf
    used = set(); pool = []
    for m in rng.choice(kp, 420, replace=False):
        o = np.argsort(np.abs(wk - m))
        for j in o:
            if j not in used: used.add(j); pool.append(int(W[j]['kepid'])); break
    pool = rng.permutation(pool).tolist(); split = dict(exam=exam, dev=dev, noise_train=pool[:360], noise_val=pool[360:])
    json.dump(split, open(D + 'split.json', 'w')); json.dump({str(k): v for k, v in Y.items()}, open(D + 'yu2018.json', 'w'))
    print({k: len(v) for k, v in split.items()}, 'giant Kp median %.2f, pool Kp median %.2f' % (np.median(kp), np.median([float(W[j]['kepmag']) for j in used])))
