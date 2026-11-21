"""Suppression and reporting model applied to free-burn growth curves (Fried & Fried 1996-style containment race).

For a scenario: line building starts at t_a = tau0 + T_resp (h since ignition), at rate r(t) = r0 (1 + (t - t_a)/tau_e)
(extended attack escalates), so the cumulative line is Lc(t) = r0 [(t - t_a) + (t - t_a)^2 / (2 tau_e)]. The fire is
contained at the first t >= t_a with Lc(t) >= f_perim * P(t), where P is the perimeter of an ellipse of area A(t) and
length-to-breadth LW(t). Final size = A(t_c) (A(96 h), a lower bound, if never contained). Reported size = max(size,
floor). The sim-level draws tau0 (pre-discovery burn time) and ros_mult (ROS adjustment) are reweighted to the
calibrated distributions by importance weights.
"""
from common import *

# size bins in acres (FPA-FOD classes: A < 0.25, B < 10, C < 100, D < 300, E < 1000, F < 5000, G >= 5000)
EDGES_AC = np.array([0.095, 0.255, 0.995, 2.995, 9.995, 29.995, 99.995, 299.995, 999.995, 4999.995, 19999.995])
EDGES = np.log10(EDGES_AC * ACRE_HA)      # log10 ha
NB = len(EDGES) + 1
ESC_BIN = 7                               # bins >= 7 are >= 100 acres
LARGE_BIN = 9
# representative log10 ha per bin (for medians); open ends extended by 0.5 decade
_E = np.r_[EDGES[0] - 0.6, EDGES, EDGES[-1] + 0.6]

FGROUPS = ['GR', 'GS', 'SH', 'TU', 'TL', 'SB', 'NB']

# literature-default suppression model (rung 0): IA arrival ~1 h after discovery; initial line production by fuel group
# (NWCG Fireline Handbook / Fried & Fried orders of magnitude, m/h for a typical IA force); escalation doubling in 4 h.
THETA0 = dict(mu_resp=np.log(1.0), mu_resp_fed=0.0, sd_resp=0.8,
              r0=dict(GR=1500.0, GS=1000.0, SH=500.0, TU=400.0, TL=400.0, SB=200.0, NB=1000.0), r0_mult=1.0, sd_r0=0.7,
              tau_e=4.0, f_perim=1.0, floor_ac=0.1,
              # sim-level reweighting targets (= the sim priors at rung 0, so weights are 1)
              tau0=dict(mu=np.log(1.0), sd=1.0, mu_nat=0.0, mu_miss=0.0), ros=None)


def fgroup(code):
    c = int(code)
    if 101 <= c <= 109 or 1 <= c <= 3: return 'GR'
    if 121 <= c <= 124: return 'GS'
    if 141 <= c <= 149 or 4 <= c <= 7: return 'SH'
    if 161 <= c <= 165: return 'TU'
    if 181 <= c <= 189 or 8 <= c <= 10: return 'TL'
    if 201 <= c <= 204 or 11 <= c <= 13: return 'SB'
    return 'NB'


def contain(TG, A, lw, tau0, t_resp, r0, tau_e, f_perim=1.0):
    """Vectorised containment for S scenarios on one growth curve. Returns final size (ha) and contained flag."""
    t_a = tau0 + np.asarray(t_resp)[:, None]                     # S x 1
    dt = np.maximum(TG[None, :] - t_a, 0.0)
    Lc = np.asarray(r0)[:, None] * (dt + dt ** 2 / (2.0 * np.asarray(tau_e)[:, None] if np.ndim(tau_e) else 2.0 * tau_e))
    lam = np.clip(lw, 1.0, 8.0)
    P = f_perim * np.pi * np.sqrt(2.0 * A * 1e4 / np.pi * (lam + 1.0 / lam))      # m
    ok = (Lc >= P[None, :]) & (TG[None, :] >= t_a)
    hit = ok.any(1)
    k = np.argmax(ok, 1)
    # interpolate between k-1 and k
    k0 = np.maximum(k - 1, 0)
    g0 = Lc[np.arange(len(k)), k0] - P[k0]
    g1 = Lc[np.arange(len(k)), k] - P[k]
    f = np.where((g1 - g0) > 0, np.clip(-g0 / np.maximum(g1 - g0, 1e-12), 0, 1), 1.0)
    f = np.where(k == 0, 1.0, f)
    la = np.log(np.maximum(A, 1e-6))
    size = np.exp(la[k0] + f * (la[k] - la[k0]))
    size = np.where(hit, size, A[-1])
    return size, hit


def draws(theta, n, fgrp, fed, rs):
    th = theta
    mu = th['mu_resp'] + th['mu_resp_fed'] * fed
    t_resp = np.exp(rs.normal(mu, th['sd_resp'], n))
    r0 = th['r0'][fgrp] * th['r0_mult'] * np.exp(rs.normal(0, th['sd_r0'], n))
    return t_resp, r0


def size_hist(rec, theta, n=64, rs=None, fed=0, fgrp=None):
    """Histogram over size bins of the reported size for one simulated ignition (dict from sim.run_one)."""
    rs = rs if rs is not None else np.random.default_rng(0)
    from sim import TG
    fg = fgrp or fgroup(rec['fuel0'])
    t_resp, r0 = draws(theta, n, fg, fed, rs)
    size, hit = contain(TG, rec['A'].astype(float), rec['lw'].astype(float), rec['tau0'], t_resp, r0, theta['tau_e'],
                        theta['f_perim'])
    size = np.maximum(size, theta['floor_ac'] * ACRE_HA)
    b = np.searchsorted(EDGES, np.log10(size), side='right')
    return np.bincount(b, minlength=NB) / n, size


def imp_weight(tau0, ros_mult, cause, theta, fgroups=None):
    """Importance weight of a sim's (tau0, ros_mult) draw under the calibrated distributions vs the sim priors
    (tau0 ~ LN(log 1, 1); ros_mult ~ log-uniform[0.2, 1.5])."""
    t = theta['tau0']
    cause = np.asarray(cause, dtype=str)
    mu = t['mu'] + t['mu_nat'] * (cause == 'Natural') + t['mu_miss'] * np.char.startswith(cause, 'Missing')
    lt = np.log(tau0)
    w = np.exp(-0.5 * ((lt - mu) / t['sd']) ** 2) / t['sd'] / np.exp(-0.5 * (lt / 1.0) ** 2)
    if theta.get('ros') is not None:
        r = theta['ros']
        lr = np.log(ros_mult)
        mu = r['mu']
        if theta.get('ros_g') and fgroups is not None:     # per-fuel-group centre (ignition cell's group)
            mu = np.array([theta['ros_g'].get(g, r['mu']) for g in fgroups])
        w = w * np.exp(-0.5 * ((lr - mu) / r['sd']) ** 2) / r['sd'] * (np.log(1.5) - np.log(0.2)) / np.sqrt(2 * np.pi)
    return w


def median_log10(p):
    """Median log10 ha of a binned distribution (linear inside the bin)."""
    p = np.atleast_2d(p)
    c = np.cumsum(p, 1)
    k = np.argmax(c >= 0.5, 1)
    cprev = np.where(k > 0, c[np.arange(len(k)), np.maximum(k - 1, 0)], 0.0)
    pk = np.maximum(p[np.arange(len(k)), k], 1e-12)
    f = np.clip((0.5 - cprev) / pk, 0, 1)
    return _E[k] + f * (_E[k + 1] - _E[k])
