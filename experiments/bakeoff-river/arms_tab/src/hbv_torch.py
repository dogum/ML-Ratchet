"""A torch (differentiable) re-implementation of WaterSim's HBV + degree-day snow + gamma channel UH + irrigation,
exactly as harness/vendor/watersim runs them for simulate(model='hbv', snow='degree_day', routing='uh') with PET
taken from the forcing, 5 equal-area elevation bands (or 1 band when the range is <= 100 m), no reservoirs,
spin-up = the window's first n_warm rows. Used to TRAIN the physics-in-the-loop arms; inference runs WaterSim itself.
Verified against WaterSim in verify_hbv.py (float64).

Batch of N basins, T rows each. Equations: watersim/snow.py dd_step, models/hbv.py step, engine.run_core (bands,
snow-covered-area PET reduction), routing.triangle, routing.gamma_uh(3, travel/3), simulate._uh_route (steady
history), simulate._irrigation.
"""
import math

import torch

# name, lo, hi, log-scale (WaterSim bounds; PMULT is this arm's precipitation multiplier on the forcing P)
PARAMS = [
    ('FC', 50.0, 700.0, True), ('LP', 0.3, 1.0, False), ('BETA', 1.0, 6.0, False), ('PERC', 0.0, 6.0, False),
    ('UZL', 0.0, 100.0, False), ('K0', 0.05, 0.5, True), ('K1', 0.01, 0.4, True), ('K2', 0.0005, 0.15, True),
    ('MAXBAS', 1.0, 7.0, False),
    ('TT', -2.5, 2.5, False), ('TTI', 0.0, 4.0, False), ('CFMAX', 1.0, 8.0, False), ('SFCF', 0.7, 1.4, False),
    ('TLAPSE', -9.0, -4.0, False), ('PGRAD', 0.0, 1.0, False),
    ('CH_VEL', 0.3, 3.0, True), ('PMULT', 0.5, 2.0, True),
]
NAMES = [p[0] for p in PARAMS]
DEFAULTS = {'FC': 250.0, 'LP': 0.7, 'BETA': 2.0, 'PERC': 1.5, 'UZL': 20.0, 'K0': 0.2, 'K1': 0.08, 'K2': 0.02,
            'MAXBAS': 2.0, 'TT': 0.0, 'TTI': 2.0, 'CFMAX': 3.5, 'SFCF': 1.0, 'TLAPSE': -6.5, 'PGRAD': 0.3,
            'CH_VEL': 1.0, 'PMULT': 1.0}
FIXED = {'CFR': 0.05, 'CWH': 0.1}
LUH = 7          # triangle length (MAXBAS <= 7)
LCH = 120        # gamma channel UH max length (WaterSim max_len 120 days)
LIRR = 700       # irrigation return-store kernel length (0.95^700 ~ 3e-16)


def to_params(z):
    """Unbounded (N, 17) -> dict of bounded parameter tensors (sigmoid into the bounds, log-scale where flagged)."""
    s = torch.sigmoid(z)
    out = {}
    for j, (n, lo, hi, lg) in enumerate(PARAMS):
        if lg:
            out[n] = torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * s[:, j])
        else:
            out[n] = lo + (hi - lo) * s[:, j]
    return out


def from_params(d):
    """Inverse of to_params for a dict of floats (for initial biases)."""
    z = []
    for n, lo, hi, lg in PARAMS:
        v = float(d[n])
        u = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if lg else (v - lo) / (hi - lo)
        u = min(max(u, 1e-4), 1 - 1e-4)
        z.append(math.log(u / (1 - u)))
    return z


def bands(elev_range_m):
    """(frac (N, 5), dz_km (N, 5)) as watersim.spec.build_bands from elev_min/max = mean -+ range/2."""
    r = torch.as_tensor(elev_range_m, dtype=torch.float64)
    k = (torch.arange(5, dtype=torch.float64) + 0.5) / 5 - 0.5
    many = (r > 100.0)[:, None]
    dz = torch.where(many, r[:, None] * k[None] / 1000.0, torch.zeros(1, dtype=torch.float64))
    frac = torch.where(many, torch.full((1, 5), 0.2, dtype=torch.float64),
                       torch.tensor([[1.0, 0, 0, 0, 0]], dtype=torch.float64))
    return frac, dz


def channel_length_km(area_km2):
    a = torch.as_tensor(area_km2, dtype=torch.float64) / 2.58999
    return 1.4 * a ** 0.6 * 1.609344


def triangle_ord(maxbas):
    m = torch.clamp(maxbas, min=1.0)[:, None]
    h = m / 2.0
    t = torch.arange(LUH + 1, dtype=maxbas.dtype)[None]
    x = torch.minimum(t, m)
    cum = torch.where(x <= h, x ** 2 / (m * h), 1.0 - (m - x) ** 2 / (m * h))
    o = cum[:, 1:] - cum[:, :-1]
    return o / o.sum(1, keepdim=True)


def _p3(x):
    """Regularised lower incomplete gamma P(3, x) = 1 - exp(-x) (1 + x + x^2 / 2)."""
    return -torch.expm1(-x) - torch.exp(-x) * (x + 0.5 * x * x)


def gamma3_ord(scale):
    """watersim.routing.gamma_uh(3, scale): ordinates truncated where P(3, n/scale) >= 1 - 1e-6 (n <= 120)."""
    k = torch.arange(LCH + 1, dtype=scale.dtype)[None]
    c = _p3(k / scale[:, None])
    with torch.no_grad():
        ok = (c >= 1 - 1e-6)
        ok[:, 0] = False
        ok[:, LCH] = True
        n = torch.argmax(ok.to(torch.int64), 1)              # first k >= 1 reaching the tolerance (or 120)
        mask = (torch.arange(LCH)[None] < n[:, None]).to(scale.dtype)
    o = (c[:, 1:] - c[:, :-1]) * mask
    return o / o.sum(1, keepdim=True)


def causal_conv(x, o, pad_value):
    """y[t] = sum_k o[k] x[t-k] with x[t<0] = pad_value. x (N, T), o (N, L), pad_value (N,)."""
    N, T = x.shape
    L = o.shape[1]
    xx = torch.cat([pad_value[:, None].expand(N, L), x], 1)
    y = torch.nn.functional.conv1d(xx[None], o.flip(1)[:, None, :], groups=N)[0]
    return y[:, 1:T + 1]


def run(P, PET, Tm, p, frac, dz, area_km2, irr_frac, n_warm, grad_from=None):
    """Simulate N basins. P, PET, Tm (N, T) float64 = forcing rows 0 .. n_warm + n_target - 1 (no extra row);
    p: dict of (N,) parameter tensors (to_params); frac, dz (N, 5); area_km2, irr_frac (N,).
    Returns Q (N, n_target) mm/day at the outlet (after channel UH and irrigation), = WaterSim ts['Q'].
    grad_from: rows before this index run under no_grad (default n_warm: the spin-up carries no gradient)."""
    N, T = P.shape
    dt = P.dtype
    grad_from = n_warm if grad_from is None else grad_from
    P = P * p['PMULT'][:, None]
    PET = torch.clamp(PET, min=0.0)
    # band precipitation multipliers and temperature offsets
    m = torch.clamp(1.0 + p['PGRAD'][:, None] * dz, min=0.0)
    norm = (frac * m).sum(1, keepdim=True)
    pmb = m / norm
    tlz = p['TLAPSE'][:, None] * dz
    TT, TTI, CFMAX, SFCF = p['TT'][:, None], p['TTI'][:, None], p['CFMAX'][:, None], p['SFCF'][:, None]
    CFR, CWH = FIXED['CFR'], FIXED['CWH']
    FC, LP, BETA, PERC, UZL = p['FC'], p['LP'], p['BETA'], p['PERC'], p['UZL']
    K0, K1, K2 = p['K0'], p['K1'], p['K2']
    tri = triangle_ord(p['MAXBAS'])
    # states
    ice = torch.zeros(N, 5, dtype=dt); liq = torch.zeros(N, 5, dtype=dt)
    sm = 0.6 * FC; suz = torch.full((N,), 5.0, dtype=dt); slz = torch.full((N,), 30.0, dtype=dt)
    buf = torch.zeros(N, LUH, dtype=dt)
    zero_col = torch.zeros(N, 1, dtype=dt)
    qm, rain_rec = [], []

    def step(t, ice, liq, sm, suz, slz, buf):
        Pb = P[:, t:t + 1] * pmb
        Tb = Tm[:, t:t + 1] + tlz
        fs = torch.clamp((TT + 0.5 * TTI - Tb) / TTI, 0.0, 1.0)
        snow = Pb * fs * SFCF
        rain = Pb * (1.0 - fs)
        ice = ice + snow
        warm = Tb > TT
        melt = torch.where(warm, torch.minimum(CFMAX * (Tb - TT), ice), torch.zeros_like(ice))
        refr = torch.where(warm, torch.zeros_like(liq), torch.minimum(CFR * CFMAX * (TT - Tb), liq))
        ice = ice - melt + refr
        liq = liq + melt - refr
        pos = ice > 0.0
        liq2 = liq + rain
        outp = torch.clamp(liq2 - CWH * ice, min=0.0)
        out = torch.where(pos, outp, liq2)
        liq = torch.where(pos, liq2 - outp, torch.zeros_like(liq))
        ice = torch.where(pos, ice, torch.zeros_like(ice))
        sca = torch.clamp((ice + liq) / 10.0, max=1.0)
        w = (frac * out).sum(1)
        tsca = (frac * sca).sum(1)
        train = (frac * rain).sum(1)
        pet = PET[:, t] * (1.0 - tsca)
        # HBV soil
        r = torch.clamp(sm / FC, min=1e-12, max=1.0)
        rech = w * r ** BETA
        sm = sm + w - rech
        exc = torch.clamp(sm - FC, min=0.0)
        rech = rech + exc
        sm = sm - exc
        lim = torch.clamp(sm / (LP * FC), max=1.0)
        ea = torch.minimum(pet * lim, sm)
        sm = sm - ea
        # response
        suz = suz + rech
        perc = torch.minimum(PERC, suz)
        suz = suz - perc
        slz = slz + perc
        q0 = torch.where(suz > UZL, K0 * (suz - UZL), torch.zeros_like(suz))
        q1 = K1 * suz
        suz = suz - (q0 + q1)
        neg = suz < 0.0
        q1 = torch.where(neg, q1 + suz, q1)
        suz = torch.where(neg, torch.zeros_like(suz), suz)
        q2 = K2 * slz
        slz = slz - q2
        buf = buf + (q0 + q1 + q2)[:, None] * tri
        o = buf[:, 0]
        buf = torch.cat([buf[:, 1:], zero_col], 1)
        return ice, liq, sm, suz, slz, buf, o, train

    with torch.no_grad():
        for t in range(grad_from):
            ice, liq, sm, suz, slz, buf, o, tr = step(t, ice, liq, sm, suz, slz, buf)
            if t >= n_warm:
                qm.append(o); rain_rec.append(tr)
    if grad_from > 0:
        ice, liq, sm, suz, slz, buf = [x.detach() for x in (ice, liq, sm, suz, slz, buf)]
    for t in range(grad_from, T):
        ice, liq, sm, suz, slz, buf, o, tr = step(t, ice, liq, sm, suz, slz, buf)
        if t >= n_warm:
            qm.append(o); rain_rec.append(tr)
    qm = torch.stack(qm, 1)
    rain = torch.stack(rain_rec, 1)
    # channel routing: gamma(3, travel / 3) UH with a steady history equal to the first value
    travel = channel_length_km(area_km2) * 1000.0 / p['CH_VEL'] / 86400.0
    och = gamma3_ord(torch.clamp(travel / 3.0, min=1e-3))
    q = causal_conv(qm, och, qm[:, 0])
    # irrigation (simulate._irrigation with the default settings), on the irrigated share
    irr = torch.as_tensor(irr_frac, dtype=dt)
    if bool((irr > 0).any()):
        Tt = Tm[:, n_warm:]
        deficit = torch.clamp(PET[:, n_warm:] - rain, min=0.0) * (Tt > 10.0).to(dt)
        demand = irr[:, None] * deficit / 0.6
        wdr = torch.minimum(demand, 0.6 * q)
        Lk = min(LIRR, q.shape[1])
        kern = (0.3 * 0.95 ** torch.arange(1, Lk + 1, dtype=dt))[None].expand(N, Lk)
        S = causal_conv(wdr, kern, torch.zeros(N, dtype=dt))          # store after the step
        ret = S * (0.05 / 0.95)
        q = q - wdr + ret
    return q
