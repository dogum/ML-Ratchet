"""HBV (Bergstrom 1976, 1992) in the HBV-light formulation of Seibert & Vis (2012), lumped soil.

Soil routine (input I = rain + snowmelt outflow, mm/day):
    recharge = I (SM/FC)^BETA ;  SM += I - recharge ;  excess above FC joins recharge
    Ea = PET min(1, SM / (LP FC)) (limited by SM)
Response routine (two boxes):
    SUZ += recharge ; PERC' = min(PERC, SUZ) moves to SLZ
    Q0 = K0 max(SUZ - UZL, 0), Q1 = K1 SUZ, Q2 = K2 SLZ   (explicit daily step; K0 + K1 <= 0.9 by bounds)
Routing: triangular weighting function with base MAXBAS days (HBV-light), applied to each component.
Components: quick = Q0, inter = Q1, base = Q2 (lower groundwater box). Storage: soil = SM, groundwater = SUZ + SLZ.
"""
from .._nb import njit

NAME = 'hbv'
PARAMS = [
    ('FC', 250.0, 50.0, 700.0, 'mm', 'maximum soil moisture storage (field capacity)'),
    ('LP', 0.7, 0.3, 1.0, '-', 'fraction of FC above which ET is not limited'),
    ('BETA', 2.0, 1.0, 6.0, '-', 'shape of the recharge curve'),
    ('PERC', 1.5, 0.0, 6.0, 'mm/day', 'maximum percolation from upper to lower box'),
    ('UZL', 20.0, 0.0, 100.0, 'mm', 'threshold for the quick-flow outlet Q0'),
    ('K0', 0.2, 0.05, 0.5, '1/day', 'quick-flow recession coefficient'),
    ('K1', 0.08, 0.01, 0.4, '1/day', 'upper-box recession coefficient'),
    ('K2', 0.02, 0.0005, 0.15, '1/day', 'lower-box (baseflow) recession coefficient'),
    ('MAXBAS', 2.0, 1.0, 7.0, 'day', 'base of the triangular routing function'),
]
STATES = ['SM', 'SUZ', 'SLZ']
UH_KIND = 'triangle'   # ordinates from MAXBAS, applied after the response routine
N_COMPONENTS = 3


def init_state(p, init=None):
    s = {'SM': 0.6 * p['FC'], 'SUZ': 5.0, 'SLZ': 30.0}
    if init:
        s.update({k: v for k, v in init.items() if k in s})
    return [s[k] for k in STATES]


@njit
def step(s, p, w, pet, dt, uhord, uhbuf, f):
    FC = p[0]
    LP = p[1]
    BETA = p[2]
    PERC = p[3] * dt
    UZL = p[4]
    K0 = 1.0 - (1.0 - p[5]) ** dt
    K1 = 1.0 - (1.0 - p[6]) ** dt
    K2 = 1.0 - (1.0 - p[7]) ** dt
    sm = s[0]
    suz = s[1]
    slz = s[2]
    # soil routine
    rech = 0.0
    if w > 0.0:
        r = sm / FC
        if r > 1.0:
            r = 1.0
        rech = w * r ** BETA
        sm += w - rech
        if sm > FC:
            rech += sm - FC
            sm = FC
    lim = sm / (LP * FC)
    if lim > 1.0:
        lim = 1.0
    ea = pet * lim
    if ea > sm:
        ea = sm
    sm -= ea
    # response routine
    suz += rech
    perc = PERC if PERC < suz else suz
    suz -= perc
    slz += perc
    q0 = K0 * (suz - UZL) if suz > UZL else 0.0
    q1 = K1 * suz
    suz -= q0 + q1
    if suz < 0.0:  # cannot happen within bounds; guards mass
        q1 += suz
        suz = 0.0
    q2 = K2 * slz
    slz -= q2
    s[0] = sm
    s[1] = suz
    s[2] = slz
    # triangular routing of each component
    L = uhord.shape[1]
    comp = (q0, q1, q2)
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
    f[0] = ea
    f[1] = outs
    f[5] = 0.0
    f[6] = sm
    f[7] = suz + slz
    f[8] = store
    f[9] = perc
    f[10] = q0
