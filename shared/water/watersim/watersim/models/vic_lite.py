"""VIC-lite: a two-layer variable-infiltration-capacity model in the spirit of VIC (Liang et al. 1994) and
the ARNO/Xinanjiang runoff curve (Zhao 1992; Todini 1996). Structurally different from HBV/GR4J/SAC-SMA: saturation
excess is generated from a statistical distribution of point storage capacities, drainage follows Brooks-Corey, and
baseflow follows the ARNO nonlinear curve.

Upper layer W1 (capacity W1M):  point capacity curve i = i_m [1 - (1 - A)^(1/b)], i_m = W1M (1 + b);
    i0 = i_m [1 - (1 - W1/W1M)^(1/(1+b))]
    R = P - (W1M - W1)                                    if i0 + P >= i_m
      = P - (W1M - W1) + W1M [1 - (i0 + P)/i_m]^(1+b)    otherwise              (Liang et al. 1994, eq. 13)
ET: E1 = ROOT1 PET min(1, W1/(0.7 W1M)), E2 = (1-ROOT1) PET min(1, W2/(0.7 W2M))   (linear stress below 70 %)
Drainage W1 -> W2: KSAT (W1/W1M)^EXPT (Brooks-Corey, 4 sub-steps per day, limited by room in W2)
Baseflow from W2 (ARNO, Todini 1996; VIC):  linear DS DSMAX/(WS W2M) W2 below WS W2M, plus
    (DSMAX - DS DSMAX/WS) ((W2 - WS W2M)/(W2M - WS W2M))^2 above.
Routing: gamma unit hydrograph (UH_SHAPE, UH_SCALE) on each component (Lohmann et al. 1996 use a similar UH).
Components: quick = R, inter = 0, base = baseflow. Storage: soil = W1, groundwater = W2.
"""
from .._nb import njit

NAME = 'vic_lite'
PARAMS = [
    ('W1M', 150.0, 30.0, 500.0, 'mm', 'upper layer storage capacity'),
    ('B_INF', 0.2, 0.01, 1.0, '-', 'variable infiltration curve shape'),
    ('ROOT1', 0.6, 0.2, 1.0, '-', 'share of ET demand served by the upper layer'),
    ('KSAT', 15.0, 0.5, 200.0, 'mm/day', 'saturated drainage rate upper -> lower layer'),
    ('EXPT', 8.0, 3.0, 25.0, '-', 'Brooks-Corey drainage exponent'),
    ('W2M', 400.0, 50.0, 2000.0, 'mm', 'lower (baseflow) layer capacity'),
    ('DSMAX', 5.0, 0.1, 40.0, 'mm/day', 'maximum baseflow'),
    ('DS', 0.1, 0.001, 1.0, '-', 'fraction of DSMAX where nonlinear baseflow begins'),
    ('WS', 0.7, 0.2, 1.0, '-', 'fraction of W2M where nonlinear baseflow begins'),
    ('UH_SHAPE', 2.0, 1.0, 5.0, '-', 'gamma unit hydrograph shape'),
    ('UH_SCALE', 0.7, 0.1, 3.0, 'day', 'gamma unit hydrograph scale'),
]
STATES = ['W1', 'W2']
UH_KIND = 'gamma'
N_COMPONENTS = 3


def init_state(p, init=None):
    s = {'W1': 0.5 * p['W1M'], 'W2': 0.5 * p['W2M']}
    if init:
        s.update({k: v for k, v in init.items() if k in s})
    return [s[k] for k in STATES]


@njit
def _baseflow(w2, W2M, DSMAX, DS, WS):
    lin = DS * DSMAX / (WS * W2M)
    q = lin * w2
    thr = WS * W2M
    if w2 > thr and W2M > thr:
        x = (w2 - thr) / (W2M - thr)
        q += (DSMAX - DS * DSMAX / WS) * x * x
    return q


@njit
def step(s, p, w, pet, dt, uhord, uhbuf, f):
    W1M = p[0]
    b = p[1]
    root1 = p[2]
    KSAT = p[3]
    EXPT = p[4]
    W2M = p[5]
    DSMAX = p[6]
    DS = p[7]
    WS = p[8]
    w1 = s[0]
    w2 = s[1]
    # saturation-excess runoff from the variable infiltration curve
    r = 0.0
    if w > 0.0:
        im = W1M * (1.0 + b)
        rat = 1.0 - w1 / W1M
        if rat < 0.0:
            rat = 0.0
        i0 = im * (1.0 - rat ** (1.0 / (1.0 + b)))
        if i0 + w >= im:
            r = w - (W1M - w1)
        else:
            r = w - (W1M - w1) + W1M * (1.0 - (i0 + w) / im) ** (1.0 + b)
        if r < 0.0:
            r = 0.0
        if r > w:
            r = w
        w1 += w - r
        if w1 > W1M:
            r += w1 - W1M
            w1 = W1M
    # evapotranspiration
    l1 = w1 / (0.7 * W1M)
    if l1 > 1.0:
        l1 = 1.0
    e1 = root1 * pet * l1
    if e1 > w1:
        e1 = w1
    w1 -= e1
    l2 = w2 / (0.7 * W2M)
    if l2 > 1.0:
        l2 = 1.0
    e2 = (1.0 - root1) * pet * l2
    if e2 > w2:
        e2 = w2
    w2 -= e2
    # drainage and baseflow, 4 sub-steps
    qb = 0.0
    dr_tot = 0.0
    nsub = 4
    h = dt / nsub
    for _ in range(nsub):
        dr = KSAT * h * (w1 / W1M) ** EXPT
        if dr > w1:
            dr = w1
        room = W2M - w2
        if dr > room:
            dr = room if room > 0.0 else 0.0
        w1 -= dr
        w2 += dr
        dr_tot += dr
        q = _baseflow(w2, W2M, DSMAX, DS, WS) * h
        if q > w2:
            q = w2
        w2 -= q
        qb += q
    s[0] = w1
    s[1] = w2
    L = uhord.shape[1]
    comp = (r, 0.0, qb)
    outs = 0.0
    store = 0.0
    for c in range(3):
        x = comp[c]
        for k in range(L):
            uhbuf[c, k] += x * uhord[c, k]
        o = uhbuf[c, 0]
        for k in range(L - 1):
            uhbuf[c, k] = uhbuf[c, k + 1]
        uhbuf[c, L - 1] = 0.0
        f[2 + c] = o
        outs += o
        for k in range(L):
            store += uhbuf[c, k]
    f[0] = e1 + e2
    f[1] = outs
    f[5] = 0.0
    f[6] = w1
    f[7] = w2
    f[8] = store
    f[9] = dr_tot
    f[10] = r
