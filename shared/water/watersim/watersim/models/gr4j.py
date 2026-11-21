"""GR4J (Perrin, Michel & Andreassian 2003), following the airGR Fortran code (frun_GR4J.f, Coron et al. 2017).

Net rainfall / evaporation:  if P >= E: Pn = P - E, En = 0 (AE = E) else Pn = 0, En = E - P (AE = Es + P)
Production store S (capacity X1):
    Ps = X1 (1 - (S/X1)^2) tanh(Pn/X1) / (1 + S/X1 tanh(Pn/X1))
    Es = S (2 - S/X1) tanh(En/X1) / (1 + (1 - S/X1) tanh(En/X1))
    Perc = S (1 - (1 + (4/9 S/X1)^4)^(-1/4))
Pr = Pn - Ps + Perc, split 90 % to UH1 (base X4 days) and 10 % to UH2 (base 2 X4).
Exchange F = X2 (R/X3)^3.5 (gain if X2 > 0) applied to both branches (limited so stores stay >= 0).
Routing store R (capacity X3): R += Q9 + F ; Qr = R (1 - (1 + (R/X3)^4)^(-1/4)); Qd = max(0, Q1 + F); Q = Qr + Qd.
Components: quick = Qd, inter = 0, base = Qr (routing store outflow). Storage: soil = S, groundwater = R, routing = UHs.
"""
import math

from .._nb import njit

NAME = 'gr4j'
PARAMS = [
    ('X1', 350.0, 20.0, 1500.0, 'mm', 'production store capacity'),
    ('X2', 0.0, -5.0, 3.0, 'mm/day', 'groundwater exchange coefficient (+ gain, - loss)'),
    ('X3', 90.0, 10.0, 500.0, 'mm', 'routing store capacity (1-day-ahead)'),
    ('X4', 1.7, 0.5, 6.0, 'day', 'unit hydrograph time base'),
]
STATES = ['S', 'R']
UH_KIND = 'gr4j'
N_COMPONENTS = 2


def init_state(p, init=None):
    s = {'S': 0.5 * p['X1'], 'R': 0.4 * p['X3']}   # airGR CreateIniStates defaults (0.3, 0.5) are also common
    if init:
        s.update({k: v for k, v in init.items() if k in s})
    return [s[k] for k in STATES]


@njit
def step(s, p, w, pet, dt, uhord, uhbuf, f):
    X1 = p[0]
    X2 = p[1]
    X3 = p[2]
    S = s[0]
    R = s[1]
    if w <= pet:
        en = pet - w
        ws = en / X1
        if ws > 13.0:
            ws = 13.0
        tws = math.tanh(ws)
        sr = S / X1
        er = S * (2.0 - sr) * tws / (1.0 + (1.0 - sr) * tws)
        ae = er + w
        S -= er
        pr = 0.0
    else:
        ae = pet
        pn = w - pet
        ws = pn / X1
        if ws > 13.0:
            ws = 13.0
        tws = math.tanh(ws)
        sr = S / X1
        ps = X1 * (1.0 - sr * sr) * tws / (1.0 + sr * tws)
        pr = pn - ps
        S += ps
    if S < 0.0:
        S = 0.0
    perc = S * (1.0 - (1.0 + (S / (2.25 * X1)) ** 4) ** (-0.25))
    S -= perc
    pr += perc
    L = uhord.shape[1]
    x1 = 0.9 * pr
    x2 = 0.1 * pr
    for k in range(L):
        uhbuf[0, k] += x1 * uhord[0, k]
        uhbuf[1, k] += x2 * uhord[1, k]
    q9 = uhbuf[0, 0]
    q1 = uhbuf[1, 0]
    for k in range(L - 1):
        uhbuf[0, k] = uhbuf[0, k + 1]
        uhbuf[1, k] = uhbuf[1, k + 1]
    uhbuf[0, L - 1] = 0.0
    uhbuf[1, L - 1] = 0.0
    rr = R / X3
    exch = X2 * rr ** 3.5
    aexch1 = exch
    if R + q9 + exch < 0.0:
        aexch1 = -R - q9
    R = R + q9 + exch
    if R < 0.0:
        R = 0.0
    rr = (R / X3) ** 4
    qr = R * (1.0 - 1.0 / math.sqrt(math.sqrt(1.0 + rr)))
    R -= qr
    aexch2 = exch
    if q1 + exch < 0.0:
        aexch2 = -q1
    qd = q1 + exch
    if qd < 0.0:
        qd = 0.0
    s[0] = S
    s[1] = R
    store = 0.0
    for k in range(L):
        store += uhbuf[0, k] + uhbuf[1, k]
    f[0] = ae
    f[1] = qr + qd
    f[2] = qd
    f[3] = 0.0
    f[4] = qr
    f[5] = aexch1 + aexch2
    f[6] = S
    f[7] = R
    f[8] = store
    f[9] = perc
    f[10] = 0.0
