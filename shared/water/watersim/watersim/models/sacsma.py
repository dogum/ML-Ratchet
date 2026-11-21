"""Sacramento Soil Moisture Accounting model (SAC-SMA; Burnash, Ferral & McGuire 1973; Burnash 1995).

Follows the NWS operational subroutine FLAND1 (as distributed with NWSRFS and ported in hydromad's sma_sac.c):
upper/lower zone tension and free water stores (UZTW, UZFW, LZTW, LZFS, LZFP), an additional impervious area
(ADIMP, store ADIMC), permanently impervious fraction PCTIM, the percolation demand curve
PERC = (LZFPM LZPK + LZFSM LZSK)(UZFWC/UZFWM)(1 + ZPERC DEFR^REXP), interflow, primary and supplemental baseflow,
a fraction PFREE of percolation going directly to free lower-zone water, SIDE (deep recharge loss fraction of
baseflow), RSERV (lower-zone free water unavailable to ET) and riparian ET RIVA. Inner time increments: 1 + 0.2 (UZFWC + excess) per day.
The channel inflow is routed with a two-parameter gamma unit hydrograph (UH_SHAPE, UH_SCALE), as in the CAMELS
SAC-SMA benchmark (Newman et al. 2015).
Components: quick = impervious + direct + surface runoff, inter = interflow, base = primary + supplemental baseflow
to the channel; deep loss (baseflow not to channel) is reported as a negative exchange.
Storage: soil = tension water (UZTW + LZTW over the pervious area, ADIMC over ADIMP), groundwater = free water.
"""
from .._nb import njit

NAME = 'sacsma'
PARAMS = [
    ('UZTWM', 50.0, 5.0, 150.0, 'mm', 'upper zone tension water capacity'),
    ('UZFWM', 40.0, 5.0, 150.0, 'mm', 'upper zone free water capacity'),
    ('UZK', 0.3, 0.1, 0.75, '1/day', 'upper zone free water depletion (interflow) rate'),
    ('PCTIM', 0.01, 0.0, 0.1, '-', 'permanently impervious fraction'),
    ('ADIMP', 0.05, 0.0, 0.4, '-', 'additional (variable) impervious fraction'),
    ('RIVA', 0.0, 0.0, 0.1, '-', 'riparian vegetation fraction'),
    ('ZPERC', 40.0, 1.0, 250.0, '-', 'maximum percolation rate multiplier'),
    ('REXP', 2.0, 1.0, 5.0, '-', 'shape of the percolation curve'),
    ('LZTWM', 130.0, 10.0, 500.0, 'mm', 'lower zone tension water capacity'),
    ('LZFSM', 25.0, 5.0, 400.0, 'mm', 'lower zone supplemental free water capacity'),
    ('LZFPM', 60.0, 10.0, 1000.0, 'mm', 'lower zone primary free water capacity'),
    ('LZSK', 0.08, 0.01, 0.35, '1/day', 'supplemental baseflow depletion rate'),
    ('LZPK', 0.008, 0.0005, 0.03, '1/day', 'primary baseflow depletion rate'),
    ('PFREE', 0.1, 0.0, 0.8, '-', 'fraction of percolation going directly to lower-zone free water'),
    ('SIDE', 0.0, 0.0, 0.5, '-', 'ratio of deep recharge to channel baseflow (loss)'),
    ('RSERV', 0.3, 0.0, 0.4, '-', 'fraction of lower-zone free water not available to ET'),
    ('UH_SHAPE', 2.0, 1.0, 5.0, '-', 'gamma unit hydrograph shape'),
    ('UH_SCALE', 0.7, 0.1, 3.0, 'day', 'gamma unit hydrograph scale'),
]
STATES = ['UZTWC', 'UZFWC', 'LZTWC', 'LZFSC', 'LZFPC', 'ADIMC']
UH_KIND = 'gamma'
N_COMPONENTS = 3


def init_state(p, init=None):
    s = {'UZTWC': 0.5 * p['UZTWM'], 'UZFWC': 0.2 * p['UZFWM'], 'LZTWC': 0.6 * p['LZTWM'],
         'LZFSC': 0.3 * p['LZFSM'], 'LZFPC': 0.5 * p['LZFPM'], 'ADIMC': 0.5 * (p['UZTWM'] + p['LZTWM'])}
    if init:
        s.update({k: v for k, v in init.items() if k in s})
    return [s[k] for k in STATES]


@njit
def step(s, p, w, pet, dt, uhord, uhbuf, f):
    uztwm = p[0]
    uzfwm = p[1]
    uzk = 1.0 - (1.0 - p[2]) ** dt
    pctim = p[3]
    adimp = p[4]
    riva = p[5]
    zperc = p[6]
    rexp = p[7]
    lztwm = p[8]
    lzfsm = p[9]
    lzfpm = p[10]
    lzsk = 1.0 - (1.0 - p[11]) ** dt
    lzpk = 1.0 - (1.0 - p[12]) ** dt
    pfree = p[13]
    side = p[14]
    rserv = p[15]
    uztwc = s[0]
    uzfwc = s[1]
    lztwc = s[2]
    lzfsc = s[3]
    lzfpc = s[4]
    adimc = s[5]
    parea = 1.0 - pctim - adimp
    edmnd = pet
    pxv = w
    # ---- evapotranspiration from the upper zone
    e1 = edmnd * (uztwc / uztwm)
    red = edmnd - e1
    uztwc -= e1
    e2 = 0.0
    skip_transfer = False
    if uztwc < 0.0:
        e1 += uztwc
        uztwc = 0.0
        red = edmnd - e1
        if uzfwc < red:
            e2 = uzfwc
            uzfwc = 0.0
            red -= e2
            skip_transfer = True
        else:
            e2 = red
            uzfwc -= e2
            red = 0.0
    if not skip_transfer:
        if uztwc / uztwm < uzfwc / uzfwm:
            uzrat = (uztwc + uzfwc) / (uztwm + uzfwm)
            uztwc = uztwm * uzrat
            uzfwc = uzfwm * uzrat
    # FLAND1 zeroes tiny stores; the removed water is booked as ET so the ledger stays exact
    tiny = 0.0
    if uztwc < 1e-5:
        tiny += uztwc
        uztwc = 0.0
    if uzfwc < 1e-5:
        tiny += uzfwc
        uzfwc = 0.0
    # ---- lower zone tension ET
    e3 = red * (lztwc / (uztwm + lztwm))
    lztwc -= e3
    if lztwc < 0.0:
        e3 += lztwc
        lztwc = 0.0
    ratlzt = lztwc / lztwm
    saved = rserv * (lzfpm + lzfsm)
    ratlz = (lztwc + lzfpc + lzfsc - saved) / (lztwm + lzfpm + lzfsm - saved)
    if ratlzt < ratlz:
        dl = (ratlz - ratlzt) * lztwm
        lztwc += dl
        lzfsc -= dl
        if lzfsc < 0.0:
            lzfpc += lzfsc
            lzfsc = 0.0
    if lztwc < 1e-5:
        tiny += lztwc
        lztwc = 0.0
    # ---- ET from the ADIMP area
    e5 = e1 + (red + e2) * ((adimc - e1 - uztwc) / (uztwm + lztwm))
    if e5 < 0.0:   # guard (not in FLAND1): never condense water onto the ADIMP area
        e5 = 0.0
    adimc -= e5
    if adimc < 0.0:
        e5 += adimc
        adimc = 0.0
    e5 *= adimp
    # ---- input to upper zone tension water
    twx = pxv + uztwc - uztwm
    if twx < 0.0:
        uztwc += pxv
        twx = 0.0
    else:
        uztwc = uztwm
    adimc += pxv - twx
    roimp = pxv * pctim
    sbf = 0.0
    ssur = 0.0
    sif = 0.0
    sperc = 0.0
    sdro = 0.0
    spbf = 0.0
    tiny_ad = 0.0
    ninc = int(1.0 + 0.2 * (uzfwc + twx))
    dinc = 1.0 / ninc
    pinc = twx / ninc
    duz = 1.0 - (1.0 - uzk) ** dinc
    dlzp = 1.0 - (1.0 - lzpk) ** dinc
    dlzs = 1.0 - (1.0 - lzsk) ** dinc
    for _ in range(ninc):
        adsur = 0.0
        ratio = (adimc - uztwc) / lztwm
        if ratio < 0.0:
            ratio = 0.0
        addro = pinc * ratio * ratio
        # baseflow
        bf = lzfpc * dlzp
        lzfpc -= bf
        if lzfpc <= 1e-4:
            bf += lzfpc
            lzfpc = 0.0
        sbf += bf
        spbf += bf
        bf = lzfsc * dlzs
        lzfsc -= bf
        if lzfsc <= 1e-4:
            bf += lzfsc
            lzfsc = 0.0
        sbf += bf
        if pinc + uzfwc <= 0.01:
            uzfwc += pinc
        else:
            # percolation
            percm = lzfpm * dlzp + lzfsm * dlzs
            perc = percm * uzfwc / uzfwm
            defr = 1.0 - (lztwc + lzfpc + lzfsc) / (lztwm + lzfpm + lzfsm)
            if defr < 0.0:
                defr = 0.0
            perc = perc * (1.0 + zperc * defr ** rexp)
            if perc >= uzfwc:
                perc = uzfwc
            uzfwc -= perc
            check = lztwc + lzfpc + lzfsc + perc - lztwm - lzfpm - lzfsm
            if check > 0.0:
                perc -= check
                uzfwc += check
            sperc += perc
            # interflow
            dl = uzfwc * duz
            sif += dl
            uzfwc -= dl
            # distribute percolation
            perct = perc * (1.0 - pfree)
            if perct + lztwc <= lztwm:
                lztwc += perct
                percf = 0.0
            else:
                percf = perct + lztwc - lztwm
                lztwc = lztwm
            percf += perc * pfree
            if percf != 0.0:
                hpl = lzfpm / (lzfpm + lzfsm)
                ratlp = lzfpc / lzfpm
                ratls = lzfsc / lzfsm
                den = (1.0 - ratlp) + (1.0 - ratls)
                if den > 0.0:
                    fracp = hpl * 2.0 * (1.0 - ratlp) / den
                else:
                    fracp = hpl
                if fracp > 1.0:
                    fracp = 1.0
                percp = percf * fracp
                percs = percf - percp
                lzfsc += percs
                if lzfsc > lzfsm:
                    percs = percs - lzfsc + lzfsm
                    lzfsc = lzfsm
                lzfpc += percf - percs
                if lzfpc > lzfpm:
                    excess = lzfpc - lzfpm
                    lztwc += excess
                    lzfpc = lzfpm
            # split pinc between upper free water and surface runoff
            if pinc != 0.0:
                if pinc + uzfwc <= uzfwm:
                    uzfwc += pinc
                else:
                    sur = pinc + uzfwc - uzfwm
                    uzfwc = uzfwm
                    ssur += sur * parea
                    adsur = sur * (1.0 - addro / pinc)
                    ssur += adsur * adimp
        adimc += pinc - addro - adsur
        if adimc > uztwm + lztwm:
            addro += adimc - (uztwm + lztwm)
            adimc = uztwm + lztwm
        sdro += addro * adimp
        if adimc < 1e-5:
            tiny_ad += adimc
            adimc = 0.0
    eused = e1 + e2 + e3
    sif *= parea
    tbf = sbf * parea
    bfcc = tbf / (1.0 + side)
    bfncc = tbf - bfcc
    tci = roimp + sdro + ssur + sif + bfcc
    e4 = (edmnd - eused) * riva
    if e4 < 0.0:
        e4 = 0.0
    tci -= e4
    if tci < 0.0:
        e4 += tci
        tci = 0.0
    eused *= parea
    tet = eused + e5 + e4 + tiny * parea + tiny_ad * adimp
    s[0] = uztwc
    s[1] = uzfwc
    s[2] = lztwc
    s[3] = lzfsc
    s[4] = lzfpc
    s[5] = adimc
    # components (riparian ET is taken from the fast part first, then interflow, then baseflow)
    cq = roimp + sdro + ssur
    ci = sif
    cb = bfcc
    rem = e4
    a = rem if rem < cq else cq
    cq -= a
    rem -= a
    a = rem if rem < ci else ci
    ci -= a
    rem -= a
    cb -= rem
    if cb < 0.0:
        cb = 0.0
    L = uhord.shape[1]
    comp = (cq, ci, cb)
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
    f[0] = tet
    f[1] = outs
    f[5] = -bfncc
    f[6] = (uztwc + lztwc) * parea + adimc * adimp
    f[7] = (uzfwc + lzfsc + lzfpc) * parea
    f[8] = store
    f[9] = sperc * parea
    f[10] = ssur + sdro + roimp
