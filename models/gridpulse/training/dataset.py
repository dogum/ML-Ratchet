"""Turn a bank (noise-free 0.1 s deviations + labels) into network examples: NESO-like 1 s measurement with a random
sub-second phase, real quiet GB noise, griddata's onset rule, the -20..+60 s window, truncation after N seconds, and
targets. Selection mimics the real event catalogues: GB = 10 s fall <= -0.15 Hz (griddata detection),
ERCOT = drop >= 60 mHz or loss >= 350 MW (ERCOT operations-report criteria)."""
from common import *
import measure as M

NS = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45, 60])
NP = np.array([1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 4], float); NP /= NP.sum()
TARGETS = ['log_loss', 'log_E', 'log_beta', 'log_head', 'log_ffr', 'nadir', 'settle']
_NOISE = None


def noise_split(kind):
    """train: 2018-07, 2020-07 and 2022-01 days 0-20; val: 2022-01 days 21+ (disjoint days)."""
    global _NOISE
    if _NOISE is None:
        _NOISE = np.load(f'{DATA}/noise.npz')
    m, d = _NOISE['month'], _NOISE['day']
    sel = ~((m == 202201) & (d >= 21)) if kind == 'train' else ((m == 202201) & (d >= 21))
    return _NOISE['seg'][sel]


def targets(L, meas):
    f0 = L.f0.values
    beta = L.loss_MW.values / np.maximum(np.abs(L.pointB_true.values), 0.005)
    return np.c_[np.log(L.loss_MW.values), np.log(L.inertia_MWs.values), np.log(beta),
                 np.log(np.maximum(L.headroom_MW.values, 50.0)), np.log(L.ffr_MW.values + 100.0),
                 meas['nadir'] * 50.0 / f0 * 10, meas['settle'] * 50.0 / f0 * 10].astype(np.float32)


def selected(dev, L):
    """Catalogue selection on the noise-free 1 s mean trace."""
    d1 = dev[:, ::10][:, :69]
    fall10 = (d1[:, 10:] - d1[:, :-10]).min(1)
    gb = L.system.values == 'GB'
    return np.where(gb, fall10 <= -0.15, (L.nadir_true.values <= -0.06) | (L.loss_MW.values >= 350))


def make(dev, L, rng, noise, mode='mean', tau=0.0, Nfix=None, ercot_noise=(0.8, 1.4)):
    B = len(dev)
    ph = rng.uniform(0, 1, B)
    d1 = M.to_1s(dev, ph, mode, tau)
    gb = L.system.values == 'GB'
    sc = np.where(gb, 1.0, rng.uniform(*ercot_noise, B))
    nz = noise[rng.integers(len(noise), size=B)] - 0.0
    nz = nz - nz[:, :60].mean(1, keepdims=True)           # centre (frequency offset carries no information)
    x = M.compose(d1, nz * sc[:, None])
    t0 = M.onset(x)
    meas = M.window(x, t0)
    N = np.full(B, Nfix) if Nfix is not None else rng.choice(NS, size=B, p=NP)
    return meas, N, targets(L, meas)


def features(w, N, is_gb, f0, demand=None, ctx=False):
    """Network input: window (pu-scaled: 0.1 Hz at 50 Hz -> 1.0) masked after onset+N, plus [is_gb, N/60, log N]
    and, if ctx, [log(demand / 30 GW), demand known] (demand NaN -> unknown)."""
    w = np.asarray(w, np.float32) * (50.0 / np.asarray(f0, np.float32))[:, None] * 10
    tt = np.arange(-M.PRE, M.POST + 1)[None, :]
    m = tt <= np.asarray(N)[:, None]
    x = np.where(m, w, 0.0)
    F = np.c_[x, np.asarray(is_gb, np.float32), np.asarray(N, np.float32) / 60, np.log(np.asarray(N, np.float32)) / 4]
    if ctx:
        d = np.asarray(demand, np.float64); known = np.isfinite(d)
        F = np.c_[F, np.where(known, np.log(np.where(known, d, 30000.0) / 30000.0), 0.0), known.astype(np.float32)]
    return F.astype(np.float32)
