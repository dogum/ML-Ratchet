"""Loading simulated banks and making training windows (with the physics-version target transforms)."""
import glob, os, json
import numpy as np
from common import ROOT, PHYS, dyn_features, N_DYN_FEAT

WIN, WARM = 730, 365


def load_phys(ver):
    with open(f'{PHYS}/{ver}.json') as f:
        return json.load(f)


def load_bank(ver, split):
    """ver = physics version; its json may point at another version's bank dir ('bank': 'p0')."""
    ph = load_phys(ver)
    d = ph.get('bank', ver)
    fs = sorted(glob.glob(f'{ROOT}/bank/{d}/{split}_*.npz'))
    fs = [f for f in fs if not f.endswith('.tmp.npz')]
    if not fs:
        raise FileNotFoundError(f'no bank {d}/{split}')
    keys = [k for k in np.load(fs[0]).files if k != 'q']       # true Q is not needed for training/scoring
    B = {}
    for k in keys:
        B[k] = np.concatenate([np.load(f)[k] for f in fs])
    n = len(B['qo'])
    rng = np.random.default_rng(12345 + (0 if split == 'train' else 1))
    tz = ph.get('tz_frac')
    B['tzf'] = rng.uniform(tz[0], tz[1], n).astype(np.float32) if tz else np.zeros(n, np.float32)
    # target in units of p_mean (static index 11 = log p_mean)
    B['pm'] = np.exp(B['st'][:, 11]).astype(np.float32)
    B['ph'] = ph
    return B


def target(B, i, a, b):
    """Gauge-like target for sample i, days [a, b): local-day mixing with the next UTC day if tz is on."""
    f = B['tzf'][i]
    q = B['qo'][i]
    y = (1 - f) * q[a:b] + f * q[a + 1:b + 1]
    return y / B['pm'][i]


def inputs(F, doy0_shifted, nextday):
    """F: (L+1, 8) raw forcing (one extra day at the end); returns (L, n_feat)."""
    L = F.shape[0] - 1
    doy = (doy0_shifted + np.arange(L)) % 365 + 1
    x = dyn_features(F[:L], doy)
    if nextday:
        nd = dyn_features(F[1:L + 1], doy)[:, :2]   # next-day P (scaled and log)
        x = np.concatenate([x, nd], 1)
    return x


def n_feat(nextday, nowcast=False):
    return N_DYN_FEAT + (2 if nextday else 0) + (3 if nowcast else 0)


def nowcast_channels(qobs_norm, lead):
    """Gauged/nowcast mode: observed flow known up to `lead` days before each target day.
    qobs_norm: (L,) observed Q / p_mean with NaN gaps. Returns (L, 3): lagged flow (log1p), availability, lead/7."""
    L = len(qobs_norm)
    lag = np.full(L, np.nan, np.float32)
    lag[lead:] = qobs_norm[:L - lead]
    ok = np.isfinite(lag)
    out = np.zeros((L, 3), np.float32)
    out[ok, 0] = np.log1p(np.maximum(lag[ok], 0) * 5.0)
    out[:, 1] = ok
    out[:, 2] = lead / 7.0
    return out


def batch(B, idx, offs, nextday, nowcast=False, rng=None, leads=None):
    X, Y, S, SIG = [], [], [], []
    for n, (i, o) in enumerate(zip(idx, offs)):
        F = B['F'][i, o:o + WIN + 1].astype(np.float32)
        x = inputs(F, B['doy0'][i] - 1 + o, nextday)
        y = target(B, i, o, o + WIN)
        if nowcast:
            lead = int(leads[n]) if leads is not None else int(rng.integers(1, 8))
            yo = y.copy()
            if rng is not None and rng.random() < 0.3:          # gauge gaps
                a = int(rng.integers(0, WIN)); yo[a:a + int(rng.integers(5, 120))] = np.nan
            x = np.concatenate([x, nowcast_channels(yo, lead)], 1)
        X.append(x)
        Y.append(y)
        S.append(B['st'][i])
        SIG.append(B['sig'][i])
    return np.stack(X), np.stack(Y), np.stack(S), np.array(SIG, np.float32)


def fu_runoff_ratio(log_aridity, w=2.6):
    phi = np.exp(log_aridity)
    return np.clip((1 + phi ** w) ** (1 / w) - phi, 0.0, 1.0)


def add_sigma(B, mode='target_std'):
    """Per-sample loss scale. 'target_std' (Kratzert NSE*: std of the target series) biases an UNGAUGED sim-trained
    network low: the scale depends on hidden parameters, so wet/flashy parameter draws get down-weighted.
    'budyko' uses only inputs: Fu's runoff ratio from the forcing aridity (w = 2.6)."""
    n = len(B['qo'])
    if mode == 'budyko':
        B['sig'] = fu_runoff_ratio(B['st'][:, 13]).astype(np.float32)
        return B
    sig = np.empty(n, np.float32)
    for i in range(n):
        sig[i] = np.std(target(B, i, 0, B['qo'].shape[1] - 1))
    B['sig'] = sig
    return B
