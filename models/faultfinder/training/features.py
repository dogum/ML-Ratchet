"""Raw day [13, 96] (NaN = missing) -> network input [NF, 96]. Pure numpy (shared by training, release, web port).

Channels: 6 temperatures (T - 20)/10; 5 physics differences (MA-RA, SA-MA, SA-SP, OA-RA, ZONE-RA);
estimated outdoor-air fraction (MA-RA)/(OA-RA) where |OA-RA| > 3 K; 7 actuator/state signals (damper, valves,
fan speed, fan power / its daily 95th percentile, fan on, occupied); time of day (sin, cos); 14 availability masks.
Missing samples are filled by linear interpolation along time (or 0) and flagged in the masks.
"""
import numpy as np

RAWC = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone', 'damper_oa', 'valve_cc', 'valve_hc', 'fan_speed',
        'fan_power', 'fan_on', 'occ']
R = {c: i for i, c in enumerate(RAWC)}
NT = 96
NF = 6 + 5 + 1 + 7 + 2 + 14
_TT = np.arange(NT) / NT * 2 * np.pi
SIN, COS = np.sin(_TT), np.cos(_TT)


def _fill(v):
    m = np.isfinite(v)
    if m.all():
        return v
    if not m.any():
        return np.zeros_like(v)
    i = np.arange(len(v))
    return np.interp(i, i[m], v[m])


def featurize(x):
    x = np.asarray(x, np.float64)
    M = np.isfinite(x)
    avail = M.mean(axis=1) > 0
    f = np.zeros((NF, NT), np.float32)
    X = np.stack([_fill(x[i]) for i in range(len(RAWC))])
    fp = x[R['fan_power']]
    if np.isfinite(fp).any():
        p = np.nanpercentile(fp, 95)
        X[R['fan_power']] = X[R['fan_power']] / p if p > 1e-9 else 0.0
    k = 0
    for c in ('T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone'):
        f[k] = (X[R[c]] - 20.0) / 10.0 if avail[R[c]] else 0.0
        k += 1
    for a, b, s in (('T_ma', 'T_ra', 5.0), ('T_sa', 'T_ma', 5.0), ('T_sa', 'T_sa_sp', 3.0), ('T_oa', 'T_ra', 10.0),
                    ('T_zone', 'T_ra', 3.0)):
        f[k] = np.clip((X[R[a]] - X[R[b]]) / s, -5, 5) if (avail[R[a]] and avail[R[b]]) else 0.0
        k += 1
    if avail[R['T_oa']] and avail[R['T_ma']] and avail[R['T_ra']]:
        den = X[R['T_oa']] - X[R['T_ra']]
        ok = np.abs(den) > 3.0
        fr = np.where(ok, (X[R['T_ma']] - X[R['T_ra']]) / np.where(ok, den, 1.0), 0.0)
        f[k] = np.clip(fr, -0.5, 1.5)
        frm = ok.astype(np.float32)
    else:
        frm = np.zeros(NT, np.float32)
    k += 1
    for c in ('damper_oa', 'valve_cc', 'valve_hc', 'fan_speed', 'fan_power', 'fan_on', 'occ'):
        f[k] = np.clip(X[R[c]], -0.2, 2.0) if avail[R[c]] else 0.0
        k += 1
    f[k] = SIN
    f[k + 1] = COS
    k += 2
    f[k:k + 13] = M.astype(np.float32)
    f[k + 13] = frm
    return f


def featurize_batch(xs):
    return np.stack([featurize(x) for x in xs])
