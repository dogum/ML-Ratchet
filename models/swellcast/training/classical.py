"""Classical method (Munk et al. 1963): find the straight dispersive ridge f = g (t - t0) / (4 pi D) in the spectrogram.
1. log energy minus each frequency's 25th percentile over the window (anomaly).
2. Grid search over D (700-11,000 km) and t0: score = mean anomaly along the line for 0.04-0.10 Hz (>= 6 points).
   Lines are kept only if they cross 0.06 Hz within +-60 h of the window centre (the window rule, known to every method).
3. Refine: least-squares line through the peak frequency of each time step within +-0.01 Hz of the best line, giving
   the slope df/dt -> D = g / (4 pi slope) and t0 from the intercept.
4. Direction: energy-weighted mean of (r1 cos, r1 sin) on ridge points. Strength proxy: storm Hs from the ridge energy,
   undoing spherical spreading for a 500-km storm (no dissipation)."""
import numpy as np
import swellsim as SS
G = 9.81; TC = (np.arange(SS.NT) * SS.DT_H - SS.HALF_H + SS.DT_H / 2)
DG = 10 ** np.linspace(np.log10(700), np.log10(11000), 70)   # the same distance range the network was trained on
TG = np.arange(-480, 121, 3.0)
FSEL = (SS.FGRID >= 0.04) & (SS.FGRID <= 0.10); FIDX = np.where(FSEL)[0]

def read(x):
    """x: (3, 80, 34) raw window -> dict(D_km, t0, dir, hs, score)"""
    E = x[0].astype(np.float64); m = np.isfinite(E)
    L = np.log10(np.maximum(np.where(m, E, 0.005), 0.005)); base = np.nanpercentile(np.where(m, L, np.nan), 25, axis=0)
    A = np.where(m, L - base[None], np.nan)
    best = (-np.inf, None, None)
    f = SS.FGRID[FIDX]
    for D in DG:
        k = 4 * np.pi * D * 1e3 / G / 3600   # h per Hz: t = t0 + k f
        for t0 in TG:
            if abs(t0 + k * 0.06) > 60: continue
            t = t0 + k * f; it = np.round((t - TC[0]) / SS.DT_H).astype(int); ok = (it >= 0) & (it < SS.NT)
            if ok.sum() < 6: continue
            v = A[it[ok], FIDX[ok]]; v = v[np.isfinite(v)]
            if len(v) < 6: continue
            s = v.mean()
            if s > best[0]: best = (s, D, t0)
    s, D, t0 = best
    if D is None: return dict(D_km=np.nan, t0=np.nan, dir=np.nan, hs=np.nan, score=-9)
    # refine with a least-squares fit of ridge peak frequencies (Munk's df/dt)
    k = 4 * np.pi * D * 1e3 / G / 3600; tt, ff, ww = [], [], []
    for i in range(SS.NT):
        fl = (TC[i] - t0) / k
        if not (0.04 <= fl <= 0.10): continue
        jj = np.where(np.abs(SS.FGRID - fl) <= 0.0101)[0]; a = A[i, jj]
        if not np.isfinite(a).any() or np.nanmax(a) < 0.2: continue
        j = jj[np.nanargmax(a)]; tt.append(TC[i]); ff.append(SS.FGRID[j]); ww.append(np.nanmax(a))
    D2, t02 = D, t0
    if len(tt) >= 6:
        sl, ic = np.polyfit(np.array(tt), np.array(ff), 1, w=np.sqrt(ww))
        if sl > 0:
            Dn = G / (4 * np.pi * sl / 3600) / 1e3; tn = -ic / sl
            if 700 <= Dn <= 11000 and abs(tn + 4 * np.pi * Dn * 1e3 / G / 3600 * 0.06) < 90: D2, t02 = Dn, tn
    # direction + strength on ridge points
    k = 4 * np.pi * D2 * 1e3 / G / 3600; cs = [0, 0]; ep = 0
    for i in range(SS.NT):
        fl = (TC[i] - t02) / k
        if not (0.04 <= fl <= 0.10): continue
        j = int(np.argmin(np.abs(SS.FGRID - fl)))
        if np.isfinite(A[i, j]) and A[i, j] > 0.2:
            e = E[i, j]; ep = max(ep, e - 10 ** base[j])
            if np.isfinite(x[1, i, j]) and np.isfinite(x[2, i, j]): cs[0] += e * x[1, i, j]; cs[1] += e * x[2, i, j]
    dr = np.degrees(np.arctan2(cs[1], cs[0])) % 360 if (cs[0] or cs[1]) else np.nan
    spread = 500e3 / (2 * np.pi) / (6371e3 * abs(np.sin(D2 * 1e3 / 6371e3)) + 1e5)
    hs = 4 * np.sqrt(max(ep, 1e-3) * 0.02 / spread) if ep > 0 else np.nan     # ~0.02 Hz swell bandwidth
    return dict(D_km=float(D2), t0=float(t02), dir=float(dr), hs=float(hs), score=float(s))
