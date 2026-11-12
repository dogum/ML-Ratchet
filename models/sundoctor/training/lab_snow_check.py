"""Does the lab's snowstorm (as app.js builds it) read as snow? Same transform in numpy, on the three lab fortnights."""
import _paths as _P  # portable paths (see _paths.py)
import sys, numpy as np
sys.path.insert(0, _P.SRC)
import realexam as RX, realexam2 as RX2, evaluate2 as E2, simbatch2 as SB2, inject as IJ

def storm(p, pu, g, ta, day, D=14):
    cover = np.zeros(D); v = 0
    for d in range(D): v = 0.95 if d == day - 1 else max(0, v - 0.4); cover[d] = v
    P, G, T, tr = p.copy(), g.copy(), ta.copy(), []
    for d in range(D):
        s = 0.3 if d == day - 1 else 1.0; tsn = 1 - min(1, 1.3 * cover[d]); cold = cover[d] > 0 or s < 1
        P[d] = pu[d] * s * tsn; G[d] = g[d] * s
        if cold: T[d] = ta[d] - np.nanmean(ta[d]) - 1
        tr.append(1 - tsn)
    return P, G, T, np.array(tr)

if __name__ == '__main__':
    net = E2.load(sys.argv[1]); days, Pall, Gall, Tall, fr, geo = RX.load(); di = {str(x.date()): i for i, x in enumerate(days)}
    for home in (False, True):
        o = RX2.diagnose(net, Pall[:, :, 10].astype(float), Gall, Tall, geo, home)
        for nm, dt in (('Spring', '2019-04-08'), ('Summer', '2019-07-08'), ('Winter', '2019-12-04')):
            a = di[dt]; sl = slice(a, a + 14); p = np.nan_to_num(Pall[sl, :, 10].astype(float)); g = Gall[sl].astype(float); ta = np.nan_to_num(Tall[sl], nan=15).astype(float)
            pe = SB2.expected_power(np.nan_to_num(np.maximum(g, 0)), ta, geo['ca'][sl]); pu = IJ.unclipped(p, pe)
            P, G, T, tr = storm(p, pu, g, ta, 5)
            X = SB2.inputs(P, None if home else G, T, geo['cz'][sl], geo['ca'][sl], geo['poa_clear'][sl] / 1000, not home)
            r = o['ref'][a + 7]; X = np.concatenate([X, np.full((1, 14, 96), r, np.float32), np.full((1, 14, 96), 1.0, np.float32)])[None]
            q = E2.predict(net, X)
            print('home' if home else 'poa ', nm, 'snow pred', np.round(q['snow'][0][3:8], 2), 'true', np.round(tr[3:8], 2), 'frac', np.round(6 * (1 - q['frac'][0][3:8]), 1), 'outage', np.round(q['outage'][0][3:8], 2))
