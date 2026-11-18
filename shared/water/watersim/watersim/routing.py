"""Routing: unit hydrographs, Muskingum, Muskingum-Cunge, kinematic wave, and multi-reach networks.

Unit hydrographs (ordinates sum to 1, input at step t contributes ord[0] at step t):
  triangle(MAXBAS)   HBV-light triangular weighting function (Seibert & Vis 2012)
  gr4j(X4)           GR4J S-curves SH1 = (t/X4)^2.5 and SH2 (Perrin et al. 2003; airGR)
  gamma(shape, scale) two-parameter gamma (Nash 1957 cascade generalised), ordinates from S-curve differences
Channel routing (daily mean inflows, substeps inside the day, exact mass bookkeeping):
  muskingum(I, K, X)          McCarthy (1938); C0, C1, C2 from the trapezoidal storage equation S = K[X I + (1-X) O]
  muskingum_cunge(I, L, ...)  Cunge (1969) / Ponce & Yevjevich (1978) variable-parameter form: c = 5/3 V (wide
                              channel, Manning), K = dx/c, X = 1/2 (1 - Q/(B S0 c dx)) from a 3-point reference flow
  kinematic(q_lat, I, L, ...) kinematic wave with uniform lateral inflow, Manning A = alpha' Q^0.6, implicit
                              backward-difference box scheme solved by Newton iteration (Chow, Maidment & Mays 1988 ch. 9)
  route_network(...)          sub-basin flows routed through a tree of Muskingum-Cunge reaches in topological order
Muskingum/Muskingum-Cunge treat the daily values as end-of-day samples with linear inflow in between (the textbook
convention); the kinematic wave holds the lateral inflow constant through the day and returns day-mean outflow. In
both, the reported channel storage is the running sum of inflow minus outflow, so the ledger closes to round-off.
"""
import math

import numpy as np
from scipy.special import gammainc

from ._nb import njit


# --------------------------------------------------------------------------------------- unit hydrographs
def triangle(maxbas, dt=1.0):
    """HBV-light triangular weights for base MAXBAS (days); continuous MAXBAS allowed."""
    m = max(float(maxbas) / dt, 1.0)
    n = int(math.ceil(m))
    t = np.arange(n + 1, dtype=float)

    def cum(x):
        x = np.clip(x, 0, m)
        h = m / 2.0
        a = np.where(x <= h, x ** 2 / (m * h), 0.0)
        b = np.where(x > h, 1.0 - (m - x) ** 2 / (m * h), 0.0)
        return a + b
    o = np.diff(cum(t))
    return o / o.sum()


def gr4j_ordinates(x4, nh=20):
    """airGR UH1 (length nh) and UH2 (length 2 nh) ordinates."""
    t = np.arange(0, 2 * nh + 1, dtype=float)
    ss1 = np.where(t < x4, (t / x4) ** 2.5, 1.0)
    ss1[0] = 0.0
    tt = t / x4
    ss2 = np.where(t < x4, 0.5 * tt ** 2.5, np.where(t < 2 * x4, 1.0 - 0.5 * np.clip(2.0 - tt, 0, None) ** 2.5, 1.0))
    ss2[0] = 0.0
    uh1 = np.diff(ss1)[: nh]
    uh2 = np.diff(ss2)[: 2 * nh]
    out1 = np.zeros(2 * nh)
    out1[:nh] = uh1
    return out1, uh2


def gamma_uh(shape, scale, dt=1.0, tol=1e-6, max_len=120):
    """Gamma unit hydrograph ordinates (per step dt), truncated where the S-curve reaches 1 - tol; renormalised."""
    shape = max(float(shape), 1e-3)
    scale = max(float(scale), 1e-6)
    n = 1
    while n < max_len and gammainc(shape, n * dt / scale) < 1 - tol:
        n += 1
    t = np.arange(n + 1) * dt
    o = np.diff(gammainc(shape, t / scale))
    return o / o.sum()


def nash_uh(n_res, k, dt=1.0, **kw):
    """Nash (1957) cascade of n linear reservoirs with constant k (days) = gamma(n, k)."""
    return gamma_uh(n_res, k, dt, **kw)


def model_uh(model, p, dt=1.0):
    """(n_comp, L) ordinate matrix for a model's internal routing, from its parameter dict."""
    if model == 'hbv':
        o = triangle(p['MAXBAS'], dt)
        return np.vstack([o, o, o])
    if model == 'gr4j':
        a, b = gr4j_ordinates(p['X4'])
        return np.vstack([a, b, np.zeros_like(a)])
    if model in ('sacsma', 'vic_lite'):
        o = gamma_uh(p['UH_SHAPE'], p['UH_SCALE'], dt)
        return np.vstack([o, o, o])
    return np.ones((3, 1))


def stack_uh(mats):
    """Pad a list of (3, L_i) matrices to one (N, 3, L) array."""
    L = max(m.shape[1] for m in mats)
    out = np.zeros((len(mats), 3, L))
    for i, m in enumerate(mats):
        out[i, :, : m.shape[1]] = m
    return out


@njit
def convolve_uh(x, ordinates):
    """Discrete convolution y[t] = sum_k ord[k] x[t-k] (same length as x)."""
    n = x.shape[0]
    L = ordinates.shape[0]
    y = np.zeros(n)
    for t in range(n):
        acc = 0.0
        for k in range(L):
            if t - k < 0:
                break
            acc += ordinates[k] * x[t - k]
        y[t] = acc
    return y


# --------------------------------------------------------------------------------------- Muskingum
@njit
def _musk_core(I, K, X, dt, O0):
    """Daily values are treated as end-of-step samples with the inflow linear between them (the usual Muskingum
    convention). Substeps h <= K/4 (but h >= 2KX so that C0 >= 0). Storage is the running sum of (in - out) per
    step, so the channel ledger closes exactly with the daily series."""
    n = I.shape[0]
    O = np.zeros(n)
    S = np.zeros(n)
    nsub = 1
    if K > 0.0:
        nsub = int(math.ceil(10.0 * dt / K))
        if nsub < 4:
            nsub = 4
        if X > 0.0:
            nmax = int(math.floor(dt / (2.0 * K * X)))
            if nmax < 1:
                nmax = 1
            if nsub > nmax:
                nsub = nmax
        if nsub > 200:
            nsub = 200
        if nsub < 1:
            nsub = 1
    h = dt / nsub
    den = 2.0 * K * (1.0 - X) + h
    c0 = (h - 2.0 * K * X) / den
    c1 = (h + 2.0 * K * X) / den
    c2 = (2.0 * K * (1.0 - X) - h) / den
    o = O0
    ip = I[0] if n > 0 else 0.0
    s = 0.0
    for t in range(n):
        it = I[t]
        for j in range(nsub):
            a = ip + (it - ip) * j / nsub
            b = ip + (it - ip) * (j + 1) / nsub
            o = c0 * b + c1 * a + c2 * o
        O[t] = o
        s += (it - o) * dt
        S[t] = s
        ip = it
    return O, S


def muskingum(I, K, X, dt=1.0, O0=None):
    """Route daily-mean inflow I (any flow unit) through one Muskingum reach (K and dt in days).
    Returns (O daily-mean outflow, S storage at the end of each step in flow-unit x days)."""
    I = np.asarray(I, float)
    if O0 is None:
        O0 = float(I[0]) if len(I) else 0.0
    if K <= 0:
        return I.copy(), np.zeros_like(I)
    return _musk_core(I, float(K), float(X), float(dt), float(O0))


def muskingum_coefficients(K, X, dt):
    den = 2 * K * (1 - X) + dt
    return (dt - 2 * K * X) / den, (dt + 2 * K * X) / den, (2 * K * (1 - X) - dt) / den


@njit
def _mc_reach(I, dx, S0, B, n_man, dt_s, nsub, O0, constant, qref_c):
    """Variable-parameter Muskingum-Cunge for one reach (wide rectangular channel, Manning). I in m3/s as end-of-step
    samples, linear between; dx in m; dt_s seconds per step. Returns (O samples, running storage of in - out in m3)."""
    n = I.shape[0]
    O = np.zeros(n)
    S = np.zeros(n)
    h = dt_s / nsub
    o = O0
    s = 0.0
    ip = I[0] if n > 0 else 0.0
    for t in range(n):
        it = I[t]
        for j in range(nsub):
            a = ip + (it - ip) * j / nsub
            b = ip + (it - ip) * (j + 1) / nsub
            if constant:
                qr = qref_c
            else:
                qr = (a + b + o) / 3.0
            if qr < 1e-6:
                qr = 1e-6
            y = (qr * n_man / (B * math.sqrt(S0))) ** 0.6
            v = qr / (B * y)
            c = 5.0 / 3.0 * v
            K = dx / c
            X = 0.5 * (1.0 - qr / (B * S0 * c * dx))
            if X < 0.0:
                X = 0.0
            den = 2.0 * K * (1.0 - X) + h
            c0 = (h - 2.0 * K * X) / den
            c1 = (h + 2.0 * K * X) / den
            c2 = (2.0 * K * (1.0 - X) - h) / den
            o = c0 * b + c1 * a + c2 * o
            if o < 0.0:
                o = 0.0
        O[t] = o
        s += (it - o) * dt_s
        S[t] = s
        ip = it
    return O, S


@njit
def _mc_cascade(I, n_reach, dx, S0, B, n_man, dt_s, nsub, constant, qref_c):
    """Muskingum-Cunge through n_reach identical reaches, all reaches advanced together at each substep (so the
    intermediate hydrographs keep substep resolution). I: end-of-step samples (m3/s), linear between."""
    n = I.shape[0]
    O = np.zeros(n)
    S = np.zeros(n)
    h = dt_s / nsub
    o = np.empty(n_reach)
    for r in range(n_reach):
        o[r] = I[0] if n > 0 else 0.0
    s = 0.0
    ip = I[0] if n > 0 else 0.0
    for t in range(n):
        it = I[t]
        for j in range(nsub):
            a = ip + (it - ip) * j / nsub
            b = ip + (it - ip) * (j + 1) / nsub
            for r in range(n_reach):
                if constant:
                    qr = qref_c
                else:
                    qr = (a + b + o[r]) / 3.0
                if qr < 1e-6:
                    qr = 1e-6
                y = (qr * n_man / (B * math.sqrt(S0))) ** 0.6
                c = 5.0 / 3.0 * qr / (B * y)
                K = dx / c
                X = 0.5 * (1.0 - qr / (B * S0 * c * dx))
                if X < 0.0:
                    X = 0.0
                den = 2.0 * K * (1.0 - X) + h
                c0 = (h - 2.0 * K * X) / den
                c1 = (h + 2.0 * K * X) / den
                c2 = (2.0 * K * (1.0 - X) - h) / den
                on = c0 * b + c1 * a + c2 * o[r]
                if on < 0.0:
                    on = 0.0
                a = o[r]
                b = on
                o[r] = on
        O[t] = o[n_reach - 1]
        s += (it - O[t]) * dt_s
        S[t] = s
        ip = it
    return O, S


def channel_geometry(area_km2, slope=None, length_km=None, width_m=None, manning_n=0.035):
    """Default main-channel geometry from basin area. Hack (1957) length L = 1.4 A^0.6 (miles, sq. miles);
    bankfull width B = 1.5 A^0.4 m (a downstream hydraulic-geometry fit, Leopold & Maddock 1953 style)."""
    a_mi2 = area_km2 / 2.58999
    L = length_km if length_km else 1.4 * a_mi2 ** 0.6 * 1.609344
    B = width_m if width_m else max(1.5 * area_km2 ** 0.4, 2.0)
    S0 = slope if slope else 0.002
    return dict(length_km=float(L), width_m=float(B), slope=float(max(S0, 1e-5)), manning_n=float(manning_n))


def muskingum_cunge(I, length_km, slope, width_m, manning_n=0.035, dt_days=1.0, n_reach=None, qref=None, nsub=None):
    """Route inflow I (m3/s, daily values) down a channel of `length_km` split into reaches (Cunge 1969).
    qref=None: variable-parameter (3-point reference flow, Ponce & Yevjevich 1978); a number: constant-parameter.
    Grid: substep h <= 6 h and <= a quarter of the travel time; reach length dx ~ c h (Courant ~ 1) at the reference
    flow (qref, else the 90th percentile flow), at most 100 reaches. All reaches advance together each substep.
    Returns (outflow samples m3/s, running channel storage of in - out in m3/s x days)."""
    I = np.asarray(I, float)
    L = length_km * 1000.0
    dt_s = dt_days * 86400.0
    pos = I[I > 0]
    qm = float(qref) if qref is not None else (float(np.percentile(pos, 90)) if len(pos) else 1.0)
    y = (qm * manning_n / (width_m * math.sqrt(slope))) ** 0.6
    c = 5.0 / 3.0 * qm / (width_m * y)
    travel = L / c
    if nsub is None:
        h = min(6 * 3600.0, max(travel / 4.0, 600.0), dt_s)
        nsub = max(1, int(math.ceil(dt_s / h)))
    h = dt_s / nsub
    if n_reach is None:
        n_reach = max(1, min(100, int(round(L / (c * h)))))
    dx = L / n_reach
    O, S = _mc_cascade(I, n_reach, dx, slope, width_m, manning_n, dt_s, nsub, qref is not None,
                       float(qref) if qref is not None else 0.0)
    return O, S / 86400.0


# --------------------------------------------------------------------------------------- kinematic wave
@njit
def _kw_core(qlat, Iup, L, S0, B, n_man, dt_s, ncell, nsub, Q0):
    """Implicit backward box scheme, wide rectangular channel: A = a Q^0.6, a = (n B^(2/3)/sqrt(S0))^0.6.
    qlat: lateral inflow per unit length (m2/s) per day-step; Iup: upstream inflow (m3/s)."""
    n = qlat.shape[0]
    dx = L / ncell
    h = dt_s / nsub
    alpha = (n_man * B ** (2.0 / 3.0) / math.sqrt(S0)) ** 0.6
    beta = 0.6
    Q = np.empty(ncell)
    for i in range(ncell):
        Q[i] = Q0[i]
    A = np.empty(ncell)
    for i in range(ncell):
        A[i] = alpha * Q[i] ** beta if Q[i] > 0 else 0.0
    out = np.zeros(n)
    stor = np.zeros(n)
    for t in range(n):
        acc = 0.0
        for _ in range(nsub):
            qin = Iup[t]
            for i in range(ncell):
                rhs = dx * A[i] + h * qin + h * qlat[t] * dx
                # solve dx*alpha*x^beta + h*x = rhs for x >= 0 (Newton, monotone)
                x = Q[i] if Q[i] > 1e-9 else rhs / h
                for _it in range(50):
                    if x < 1e-12:
                        x = 1e-12
                    fx = dx * alpha * x ** beta + h * x - rhs
                    dfx = dx * alpha * beta * x ** (beta - 1.0) + h
                    xn = x - fx / dfx
                    if xn <= 0.0:
                        xn = 0.5 * x
                    if abs(xn - x) < 1e-12 * (1.0 + x):
                        x = xn
                        break
                    x = xn
                if rhs <= 0.0:
                    x = 0.0
                Q[i] = x
                A[i] = (rhs - h * x) / dx   # exact continuity of the cell
                qin = x
            acc += qin * h
        out[t] = acc / dt_s
        s = 0.0
        for i in range(ncell):
            s += A[i] * dx
        stor[t] = s
    return out, stor


def kinematic(qlat_m3s, length_km, slope, width_m, manning_n=0.035, dt_days=1.0, ncell=None, inflow_up=None,
              nsub=None):
    """Kinematic-wave routing of a basin's runoff entering uniformly along a channel of `length_km`.
    qlat_m3s: total lateral inflow (m3/s, daily means) spread evenly along the channel. Returns (outflow m3/s,
    channel storage at each day end in m3 / 86400, i.e. m3/s x days, absolute)."""
    q = np.asarray(qlat_m3s, float)
    L = length_km * 1000.0
    if ncell is None:
        ncell = int(min(max(4, L / 2000.0), 60))
    if nsub is None:
        nsub = 24
    Iup = np.zeros_like(q) if inflow_up is None else np.asarray(inflow_up, float)
    # steady initial state for the first day's inflow: Q(x) = Iup + q x / L
    x = (np.arange(ncell) + 1) / ncell
    Q0 = Iup[0] + q[0] * x if len(q) else np.zeros(ncell)
    out, stor = _kw_core(q / L, Iup, L, slope, width_m, manning_n, dt_days * 86400.0, ncell, nsub, Q0)
    return out, stor / 86400.0


def kinematic_initial_storage(q0, length_km, slope, width_m, manning_n=0.035, ncell=None):
    L = length_km * 1000.0
    if ncell is None:
        ncell = int(min(max(4, L / 2000.0), 60))
    alpha = (manning_n * width_m ** (2.0 / 3.0) / math.sqrt(slope)) ** 0.6
    x = (np.arange(ncell) + 1) / ncell
    Q = q0 * x
    return float(np.sum(alpha * Q ** 0.6) * L / ncell) / 86400.0


# --------------------------------------------------------------------------------------- networks
def route_network(local_m3s, topology, dt_days=1.0):
    """Route sub-basin flows through a tree of Muskingum-Cunge reaches.
    local_m3s: {node_id: array of local runoff (m3/s) entering at the node}
    topology:  {node_id: {'down': downstream node id or None, 'length_km', 'slope', 'width_m', 'manning_n'}}
               (reach = channel from this node to its downstream node; the outlet has down=None).
    Returns ({node_id: total flow at the node (m3/s)}, {node_id: reach storage series}, outlet id)."""
    ids = list(local_m3s)
    ups = {i: [] for i in ids}
    for i in ids:
        d = topology[i].get('down')
        if d is not None:
            ups[d].append(i)
    order, seen = [], set()

    def visit(i):
        for u in ups[i]:
            if u not in seen:
                visit(u)
        seen.add(i)
        order.append(i)
    outlets = [i for i in ids if topology[i].get('down') is None]
    for o in outlets:
        visit(o)
    at_node, routed, storage = {}, {}, {}
    for i in order:
        q = np.asarray(local_m3s[i], float).copy()
        for u in ups[i]:
            q = q + routed[u]
        at_node[i] = q
        t = topology[i]
        if t.get('down') is not None:
            routed[i], storage[i] = muskingum_cunge(q, t['length_km'], t['slope'], t['width_m'],
                                                    t.get('manning_n', 0.035), dt_days)
    return at_node, storage, outlets[0] if outlets else None
