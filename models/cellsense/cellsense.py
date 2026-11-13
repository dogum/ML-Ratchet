"""CellSense: battery health from one fast discharge, learned from physics alone.

Reads one 4C constant-current discharge (to 2.0 V) of an LFP/graphite cell (trained on the A123 APR18650M1A, 1.1 Ah, the
cell of Severson et al. 2019) as capacity Q against voltage V, with the can temperature, optionally alongside the same
cell's discharge from when it was new, and returns, each with a 68% error bar:

  q_total   capacity including a hold at 2.0 V down to C/50 (Ah): what is really left, not just what 4C delivers
  q_cc      capacity of the 4C constant-current part (Ah)
  ir        30 ms pulse resistance at 80% SOC (milliohm)
  lli       lithium lost (fraction of the new cell's cyclable lithium)
  lam_ne    graphite lost (fraction); lam_pe: LFP lost (fraction)
  r_growth  pulse-resistance growth since new; fade: capacity fade since new
A discharge cut short (from full, stopping before 2.0 V) goes to a third network; it reads well down to about 0.75 Ah.
And, with the cycle number and charging time, a second network's life (the cycle at which q_total reaches 0.88 Ah, with
life_lo / life_hi) and left (cycles left).

numpy only. Example:
    import cellsense, numpy as np
    cs = cellsense.CellSense()
    d = np.load('examples/b2c43.npz')
    r = cs.diagnose(d['Q'][3], d['T'][3], d['Q'][0], d['T'][0], cycle=int(d['cycle'][3]), charge_time_min=float(d['charge_time']))
    r['q_total'], r['q_total_sig'], r['life'], r['life_lo'], r['life_hi']
"""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
VGRID = np.linspace(3.5, 2.0, 1000)[::4].copy()            # 250 voltage levels, 3.5 V down to 2.0 V (Severson's Vdlin grid, every 4th)

def q_of_v(V, Q, T=None, vgrid=VGRID):
    """capacity (and temperature) at each grid voltage from a discharge recorded in time: V falling, Q rising"""
    V, Q = np.asarray(V, float), np.asarray(Q, float); Vm = np.minimum.accumulate(V); ok = (vgrid <= Vm[0]) & (vgrid >= Vm[-1])
    q = np.full(len(vgrid), np.nan); q[ok] = np.interp(-vgrid[ok], -Vm, Q); q[vgrid > Vm[0]] = 0.0
    t = None
    if T is not None:
        t = np.full(len(vgrid), np.nan); t[ok] = np.interp(-vgrid[ok], -Vm, np.asarray(T, float)); t[vgrid > Vm[0]] = float(T[0])
    return q, t

def gsmooth(a, sigma):
    """Gaussian smoothing along the voltage axis with the edges held (the lifetime network reads curves smoothed over 24 mV)"""
    if not sigma: return a
    r = int(4.0 * sigma + 0.5); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    p = np.concatenate([np.repeat(a[:, :1], r, 1), a, np.repeat(a[:, -1:], r, 1)], 1)
    return np.stack([np.convolve(row, k[::-1], 'valid') for row in p])

def features(Q1, T1, Q0=None, T0=None, smooth=0):
    """missing points (NaN), such as the end of a partial discharge, are carried forward and masked"""
    Q1, T1 = np.atleast_2d(Q1).astype(float), np.atleast_2d(T1).astype(float); B, L = Q1.shape
    has = np.zeros(B) if Q0 is None else np.isfinite(np.atleast_2d(Q0)).any(1).astype(float)
    if Q0 is None: Q0, T0 = np.zeros_like(Q1), np.full_like(T1, 30.0)
    Q0, T0 = np.atleast_2d(Q0).astype(float), np.atleast_2d(T0).astype(float)
    def fill(a, default):
        m = np.isfinite(a); out = np.where(m, a, np.nan)
        for b in range(len(a)):
            r = out[b]; ok = np.isfinite(r)
            if not ok.any(): r[:] = default; continue
            idx = np.where(ok, np.arange(L), 0); np.maximum.accumulate(idx, out=idx); r[:] = r[idx]; r[:np.argmax(ok)] = r[np.argmax(ok)]
        return out, m.astype(float)
    q1, m1 = fill(Q1, 0.0); t1, _ = fill(T1, 30.0); q0, m0 = fill(Q0, 0.0); t0, _ = fill(T0, 30.0); h = has[:, None] * np.ones((1, L))
    if smooth: q1, t1, q0, t0 = (gsmooth(a, smooth) for a in (q1, t1, q0, t0))
    return np.nan_to_num(np.stack([q1, q0 * h, (q1 - q0) * 10 * h, (t1 - 30) / 10, (t0 - 30) / 10 * h, m1, m0 * h, h], 1)).astype(np.float32)

def _gelu(x):
    a = np.abs(x) * 0.7071067811865476; t = 1 / (1 + 0.3275911 * a)
    erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return 0.5 * x * (1 + np.sign(x) * erf)

def _conv(x, w, b, stride=1, pad=0):
    k = w.shape[2]; xp = np.pad(x, ((0, 0), (0, 0), (pad, pad))); L = (xp.shape[2] - k) // stride + 1
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)[:, :, ::stride][:, :, :L]
    return np.einsum('bclk,ock->bol', win, w, optimize=True) + b[None, :, None]

class CellSenseNet:
    def __init__(self, path=os.path.join(HERE, 'cellsense_weights.npz')):
        z = np.load(path); self.meta = json.loads(str(z['meta'])); self.W = {k: z[k] for k in z.files if k != 'meta'}
    def forward(self, X, S=None):
        W = self.W; h = _gelu(_conv(X, W['stem_w'], W['stem_b'], 1, 3))
        for i, (st, p1, p2) in enumerate(self.meta['blocks']):
            y = _gelu(_conv(h, W[f'b{i}c1_w'], W[f'b{i}c1_b'], st, p1)); y = _conv(y, W[f'b{i}c2_w'], W[f'b{i}c2_b'], 1, p2)
            h = _gelu(y + _conv(h, W[f'b{i}sk_w'], W[f'b{i}sk_b'], st, 0))
        z = np.concatenate([h.mean(-1), h.max(-1)] + ([S] if self.meta['extra'] else []), 1)
        return _gelu(z @ W['h0_w'].T + W['h0_b']) @ W['h1_w'].T + W['h1_b']

def _run(net, X, S):
    m = net.meta; lab = m['labels']; mu_y, sd_y = np.array(m['mu_y']), np.array(m['sd_y']); o = net.forward(X, S if m['extra'] else None); k = len(lab)
    mu = o[:, :k] * sd_y + mu_y; sg = np.exp(np.clip(o[:, k:], -7, 3)) * sd_y
    return {n_: mu[:, j] for j, n_ in enumerate(lab)} | {n_ + '_sig': sg[:, j] for j, n_ in enumerate(lab)}

class CellSense:
    """three networks: health (capacity, resistance, degradation modes) for full discharges, a partial-discharge network for
    curves that stop short of 2.0 V, and lifetime (cycle life, cycles left). CellSense(chemistry='nmc811') returns the
    separate NMC811 reader instead (CellSenseNMC)."""
    def __new__(cls, *args, chemistry='lfp', **kw):
        if str(chemistry).lower() in ('nmc811', 'nmc'): return CellSenseNMC(*args, **kw)
        return super().__new__(cls)

    def __init__(self, health=os.path.join(HERE, 'cellsense_weights.npz'), life=os.path.join(HERE, 'cellsense_life_weights.npz'),
                 partial=os.path.join(HERE, 'cellsense_partial_weights.npz'), chemistry='lfp'):
        self.health = CellSenseNet(health); self.life = CellSenseNet(life) if life and os.path.exists(life) else None
        self.partial = CellSenseNet(partial) if partial and os.path.exists(partial) else None
    def diagnose(self, Q_now, T_now, Q_new=None, T_new=None, cycle=None, charge_time_min=None):
        """Q_now, T_now: capacity (Ah) and can temperature (C) on VGRID for the discharge being read (use q_of_v for raw data;
        a discharge that stops before 2.0 V leaves NaN below where it stopped and is read by the partial-discharge network);
        Q_new, T_new: the same cell when new (optional, improves everything relative); cycle and charge_time_min (charging time
        from 0 to 80% SOC) turn on the lifetime forecast."""
        X = features(Q_now, T_now, Q_new, T_new); B = len(X)
        S0 = np.zeros((B, 3), np.float32); Qn = np.atleast_2d(Q_now)
        part = ~np.isfinite(Qn[:, -1]) if self.partial is not None else np.zeros(B, bool)    # stopped before 2.0 V: a partial discharge
        r = _run(self.health, X, S0)
        if part.any():
            sm = self.partial.meta.get('smooth', 0); rp = _run(self.partial, features(Q_now, T_now, Q_new, T_new, smooth=sm), S0)
            for k in rp: r[k] = np.where(part, rp[k], r[k])
        r['partial'] = part.astype(float)
        if self.life is not None and cycle is not None and charge_time_min is not None:
            S = np.array([[np.log10(max(cycle, 1)) / 3.5, charge_time_min / 15.0, 1.0]], np.float32).repeat(B, 0)
            sm = self.life.meta.get('smooth', 0); L = _run(self.life, features(Q_now, T_now, Q_new, T_new, smooth=sm) if sm else X, S)
            for k in ('log_life', 'log_life_sig', 'left', 'left_sig'): r[k] = L[k]
            r['life'] = 10 ** r['log_life']; r['life_lo'] = 10 ** (r['log_life'] - r['log_life_sig']); r['life_hi'] = 10 ** (r['log_life'] + r['log_life_sig'])
        return {k: (v if B > 1 else float(v[0])) for k, v in r.items()}


# ---- NMC811: the LG INR21700-M50T (NMC811 / graphite-SiOx, 4.85 Ah), a second, separate network ----------------------------
VGRID_NMC = np.linspace(4.15, 2.50, 250)                    # 250 voltage levels, 4.15 V down to 2.50 V (6.6 mV apart)


def q_of_v_nmc(V, Q, tol=0.001):
    """a C/20 discharge recorded in time (V falling from the rest after full charge, Q counted from the start of the discharge)
    -> capacity at each VGRID_NMC voltage. A test that stopped within tol of 2.50 V counts as complete (cyclers stop a hair
    above the cut-off); one that stopped earlier is a partial discharge and leaves NaN below where it stopped."""
    V, Q = np.asarray(V, float), np.asarray(Q, float)
    if V[-1] - VGRID_NMC[-1] < tol: V, Q = np.append(V, VGRID_NMC[-1]), np.append(Q, Q[-1])
    return q_of_v(V, Q, vgrid=VGRID_NMC)[0]


def features_nmc(Q1, Q0=None, smooth=0):
    """(B, 250) capacity-vs-voltage curves on VGRID_NMC -> (B, 6, 250) channels; NaN (a curve that stopped early) is masked,
    with the last value carried forward; smooth: Gaussian width in grid points applied to the filled curves"""
    def fill(a):
        a = np.atleast_2d(np.asarray(a, np.float64)); m = np.isfinite(a); out = np.where(m, a, 0.0); B, L = a.shape
        idx = np.where(m, np.arange(L)[None], 0); np.maximum.accumulate(idx, axis=1, out=idx); out = np.take_along_axis(out, idx, 1)
        first = np.argmax(m, 1); out = np.where(np.arange(L)[None] < first[:, None], out[np.arange(B), first][:, None], out)
        out[~m.any(1)] = 0.0; return out, m.astype(np.float32)
    q1, m1 = fill(Q1); B, L = q1.shape
    has = np.zeros(B, np.float32) if Q0 is None else np.isfinite(np.atleast_2d(Q0)).any(1).astype(np.float32)
    q0, m0 = (np.zeros_like(q1), np.zeros_like(m1)) if Q0 is None else fill(Q0); h = has[:, None] * np.ones((1, L), np.float32)
    if smooth: q1, q0 = gsmooth(q1, smooth), gsmooth(q0, smooth)
    return np.nan_to_num(np.stack([q1, q0 * h, (q1 - q0) * 10 * h, m1, m0 * h, h], 1)).astype(np.float32)


class CellSenseNMC:
    """Reads one slow (C/20, 0.24 A) discharge from full of an LG INR21700-M50T NMC811 cell, as capacity discharged Q (Ah)
    on VGRID_NMC (use q_of_v_nmc(V, Q) for raw data), complete to 2.5 V or stopped early (a partial discharge:
    NaN below where it stopped), optionally beside the same cell's first full test. Returns, each with a 68% error bar
    (already widened by the dev-fitted factor; raw=True gives the network's own):
      q_total   capacity to 2.5 V at C/20 (Ah)
      fade      capacity fade since the reference test (only meaningful with a reference)
      lli       lithium lost; lam_ne: negative electrode (graphite-SiOx) lost; lam_pe: NMC811 lost (fractions of new)
    The degradation modes have no answer key in the data it was examined on: they are the physics model's explanation."""
    def __init__(self, weights=os.path.join(HERE, 'cellsense_nmc811_weights.npz')):
        self.net = CellSenseNet(weights); self.widen = self.net.meta.get('widen', {})

    def diagnose(self, Q_now, Q_new=None, raw=False):
        X = features_nmc(Q_now, Q_new, self.net.meta.get('smooth', 0)); r = _run(self.net, X, None)
        Qn = np.atleast_2d(np.asarray(Q_now, float))
        if self.net.meta.get('target') == 'rest':          # the network gives the charge still to come below where the curve stops
            r['q_total'] = r['q_total'] + np.nanmax(np.where(np.isfinite(Qn), Qn, -np.inf), 1)
        if self.net.meta.get('endrule'):                   # a curve that reached 2.5 V: its capacity is its end (coulomb counting)
            full = np.isfinite(Qn[:, -1]); r['q_total'] = np.where(full, Qn[:, -1], r['q_total']); r['q_total_sig'] = np.where(full, 0.002, r['q_total_sig'])
            if Q_new is not None:
                Q0 = np.atleast_2d(np.asarray(Q_new, float)); ok = full & np.isfinite(Q0[:, -1])
                r['fade'] = np.where(ok, 1 - Qn[:, -1] / np.where(ok, Q0[:, -1], 1.0), r['fade']); r['fade_sig'] = np.where(ok, 0.002 / 4.8, r['fade_sig'])
        if not raw:
            for k, f in self.widen.items(): r[k + '_sig'] = r[k + '_sig'] * f
        r['partial'] = (~np.isfinite(np.atleast_2d(Q_now)[:, -1])).astype(float); B = len(X)
        return {k: (v if B > 1 else float(v[0])) for k, v in r.items()}
