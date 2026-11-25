"""Law variables for the vehicle trip-energy law (agent C, Vehicle Bake-off). numpy only.

The same named variables are computed from the TABLE track (EnergyNet's 151 inputs, exact) and from the TRACE track
(1 Hz trace + spec + rl, through EnergyNet's own physics integrals features.phys_ints / e_phys: equal to the table to
~2e-4 in log), so one law serves both tracks.

Variables (intensive, per km of the trip; road-load terms at the tier's road load, i.e. what E_phys uses):
  p  positive wheel work per km            MJ/km   (sum of max(P_wheel, 0) dt / distance)
  n  negative wheel work per km (magnitude) MJ/km  (braking / regen opportunity)
  t  time per km                            min/km  (60 / mean speed in km/h)
  m  vehicle mass used by the physics       t       (exact, reported or class mass, by tier)
  T  mean ambient temperature               C
  c  thermal state at the start             -1 warm, 0 unknown, +1 cold soak   (tier flags V[43], V[44])
  d  trip distance                          km      (per-trip costs such as a cold start show up as k / d)
Plus, for plug-ins, f = EnergyNet's charge-depleting share (F[11]: from start SOC and battery size, 0.5 if unknown).
Target: e = trip energy / distance (MJ/km); E = e x distance.
"""
import numpy as np

NR, NF = 90, 12
V0 = NR + NF
VARS = ['p', 'n', 't', 'm', 'T', 'c', 'd']
VAR_INFO = {
    'p': {'label': 'positive wheel work per km (road load: aero + rolling + climbing + accelerating)', 'symbol': 'p',
          'unit': 'MJ/km', 'range': [0.05, 3.0]},
    'n': {'label': 'negative wheel work per km (braking / descending: what regen could recover)', 'symbol': 'n',
          'unit': 'MJ/km', 'range': [0.0, 1.5]},
    't': {'label': 'time per km (60 / mean speed in km/h)', 'symbol': 't', 'unit': 'min/km', 'range': [0.5, 6.0]},
    'm': {'label': 'vehicle mass', 'symbol': 'm', 'unit': 't', 'range': [1.0, 40.0]},
    'T': {'label': 'mean ambient temperature', 'symbol': 'T', 'unit': '°C', 'range': [-25, 40]},
    'c': {'label': 'start state: -1 warm, 0 unknown, +1 cold soak', 'symbol': 'c', 'unit': '-', 'range': [-1, 1]},
    'd': {'label': 'trip distance', 'symbol': 'd', 'unit': 'km', 'range': [0.5, 100.0]},
}
GROUPS = ['ice', 'hev', 'bev']          # laws; plug-ins mix hev and bev with EnergyNet's share f
ARCH_GROUP = ['ice', 'hev', 'hev', 'hev', 'phev', 'phev', 'bev']


def from_table(X):
    """X (n, 151) -> dict of law variables + dist_km, dur_s, f, E_phys (all float64 arrays)."""
    X = np.asarray(X, np.float64)
    dur = 600.0 * np.exp(X[:, 0]); dist = 5000.0 * np.exp(X[:, 1])
    E = np.exp(X[:, NR]); Ep = E * np.exp(X[:, NR + 1]); En = np.maximum(E * np.exp(X[:, NR + 2]) - 1e-4, 0.0)
    m = 1500.0 * np.exp(X[:, V0 + 14])
    km = np.maximum(dist, 1.0) / 1000.0
    return dict(p=Ep / km, n=En / km, t=dur / 60.0 / km, m=m / 1000.0, T=X[:, 76] * 20.0,
                c=X[:, V0 + 44] - X[:, V0 + 43], f=X[:, NR + 11], dist_km=km, dur_s=dur, E_phys=E,
                arch=np.argmax(X[:, V0:V0 + 7], 1), cls=np.argmax(X[:, V0 + 7:V0 + 14], 1))


def from_trace(batch, F=None):
    """trace-track batch -> the same dict (physics integrals recomputed with EnergyNet's features module F)."""
    if F is None:
        import features as F
    n = batch['n']
    out = {k: np.zeros(n) for k in ('p', 'n', 't', 'm', 'T', 'c', 'f', 'dist_km', 'dur_s', 'E_phys')}
    for i, (tr, sp, rl) in enumerate(zip(batch['traces'], batch['spec'], batch['rl'])):
        tr = np.asarray(tr, np.float64)
        v = tr[:, 0]; N = len(v)
        P = dict(v=v, a=np.gradient(v) if N > 1 else np.zeros(N), grade=tr[:, 1], T=tr[:, 3], wh=tr[:, 4],
                 wc=tr[:, 5], ws=np.hypot(tr[:, 4], tr[:, 5]), rho=tr[:, 6],
                 surf=np.clip(np.rint(tr[:, 7]), 0, 4).astype(int), N=N)
        arch = F.ARCHS[int(batch['arch'][i])]; cls = F.CLASSES[int(batch['cls'][i])]; hd = cls in F.HD
        m, cda, crr, cv, rot = (float(x) for x in rl)
        ph = F.phys_ints(P, m, cda, crr, cv, rot, hd)
        soc = float(sp[41]) if sp[42] > 0.5 else None
        cap = float(np.exp(sp[33]) * 10.0) if sp[34] > 0.5 else None
        E, Eh, Eb, f = F.e_phys(ph, arch, m, float(N), soc, cap, hd)
        dist = max(float(v.sum()), 1.0); km = dist / 1000.0
        out['p'][i] = ph[0] / km; out['n'][i] = max(-ph[1], 0.0) / km; out['t'][i] = N / 60.0 / km
        out['m'][i] = m / 1000.0; out['T'][i] = float(tr[:, 3].mean()); out['c'][i] = float(sp[44]) - float(sp[43])
        out['f'][i] = f; out['dist_km'][i] = km; out['dur_s'][i] = N; out['E_phys'][i] = E
    out['arch'] = np.asarray(batch['arch'], int); out['cls'] = np.asarray(batch['cls'], int)
    return out


def matrix(d):
    d = dict(d); d['d'] = d['dist_km']
    return np.stack([d[k] for k in VARS], 1)
