"""CellSense lifetime data: each simulated cell gets a fast-charging rate and its own aging rates, its whole trajectory is
run to end of life (agesim), and then it is read once at a random cycle: the 4C curve at that cycle, its curve when new,
the cycle number and its charging time 0-80%. Labels: everything the health bank has, plus the cycle at which it crosses
0.88 Ah (log10) and the cycles it has left."""
import _paths as _P  # portable paths (see _paths.py)
import sys, numpy as np
sys.path.insert(0, _P.SRC); import cellsim as CS, simbank as SB, agesim as AS
LABELS = SB.LABELS + ['log_life', 'left']
V2 = True   # early-life physics (agesim v2): slow graphite from the first cycles, tied to the later graphite loss

def make(seed, B, out=None, chunk=500):
    rng = np.random.default_rng(seed); R_ = {k: [] for k in ('Q0', 'T0c', 'Q1', 'T1c', 'y', 's')}; done = 0
    while done < B:
        n = chunk; P0 = SB.fresh(rng, n); Rt = AS.rates(rng, n, v2=V2)
        Q0, T0, cc0, tot0, ir0 = SB.curves(P0); q_eq0, _ = AS.window(P0['Q_p'], P0['Q_n'], P0['Q_Li'], P0['sig_p'], P0['sig_n'])
        tr = AS.trajectory(P0, Rt, kappa=tot0 / q_eq0); life = tr['life']; ok = np.isfinite(life) & (life >= 60)
        k = np.where(ok)[0]
        if not len(k): continue
        sub = lambda d: {a: (v[k] if isinstance(v, np.ndarray) else v) for a, v in d.items()}; P0k, Rk = sub(P0), sub(Rt); trk = {a: (v[k] if a != 'n' else v) for a, v in tr.items()}
        # read cycle: mostly through life, some a little past end of life (as the real records run)
        frac = np.where(rng.random(len(k)) < 0.15, rng.uniform(0.0, 0.15, len(k)), rng.uniform(0.0, 1.05, len(k)))
        nread = np.clip(np.round(frac * life[k] / 10) * 10, 10, trk['n'][-1])
        if V2: nread = np.where(rng.random(len(k)) < 0.25, rng.integers(6, 16, len(k)) * 10, nread)     # v2: a quarter read around cycle 100
        i = (nread / 10).astype(int)
        P1 = AS.state_at(P0k, Rk, trk, i); Q1, T1, cc1, tot1, ir1 = SB.curves(P1)
        if V2:   # v2: the reference curve is the cell at cycle 10, as the real reference is cycles 5-10; fade and growth relative to it
            Q0, T0, _, tot0k, ir0k = SB.curves(AS.state_at(P0k, Rk, trk, np.ones(len(k), int))); Q0, T0 = Q0, T0
        else:
            Q0, T0, tot0k, ir0k = Q0[k], T0[k], tot0[k], ir0[k]
        y = np.stack([tot1, cc1, ir1 * 1000, trk['lli'][np.arange(len(k)), i], trk['lam_ne'][np.arange(len(k)), i], trk['lam_pe'][np.arange(len(k)), i], ir1 / ir0k - 1, 1 - tot1 / tot0k,
                      np.log10(life[k]), life[k] - nread], 1)
        s = np.stack([nread, 48.0 / Rk['c']], 1)
        for a, v in (('Q0', Q0), ('T0c', T0), ('Q1', Q1), ('T1c', T1), ('y', y), ('s', s)): R_[a].append(v.astype(np.float32))
        done += len(k); print(seed, done, flush=True)
    R_ = {a: np.concatenate(v)[:B] for a, v in R_.items()}
    if out: np.savez_compressed(out, **R_, labels=np.array(LABELS), vgrid=SB.VG)
    return R_

if __name__ == '__main__':
    make(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
