"""EnergyNet features (numpy; ported 1:1 to web/enet.js).

A trip is described by
  route vector R (90): duration, distance, speeds, stops, speed x accel histogram, grade histogram and climb, weather
  physics integrals (8) for a set of road-load parameters (m, CdA, Crr, c_v): positive / negative wheel energy and
      the work of aero, rolling, grade up/down, inertia, from the same road-load physics as RoadSim (station wind
      along the heading, ISO 28580 tyre temperature factor, surface factors)
  vehicle vector V (47): architecture, class, mass, road load, engine, transmission, battery, label energy, start SOC,
      thermal state, label kind (OBD MAF estimate), dynamometer test, each unknown field masked (value 0, flag 0)
The network predicts log(E / E_phys), E_phys being a generic model-free energy estimate from the physics integrals.
"""
import numpy as np

ARCHS = ['conv', 'hev_parallel_p2', 'hev_series', 'hev_powersplit', 'phev_powersplit', 'phev_series', 'bev']
CLASSES = ['car', 'suv', 'pickup', 'van', 'bus', 'truck_md', 'truck_hd']
TRANS = ['AT', 'DCT', 'CVT', 'AMT', 'single', 'eCVT']
HD = ('bus', 'truck_md', 'truck_hd')
# class road-load priors (RoadSim CLASS_DEFAULTS): mass kg, CdA m2, Crr, Crr speed coef, rotating mass fraction
CLS = {'car': (1650, 0.62, 0.0085, 0.004, 0.025), 'suv': (2000, 0.85, 0.0090, 0.004, 0.025),
       'pickup': (2450, 1.10, 0.0095, 0.004, 0.03), 'van': (2600, 1.25, 0.0090, 0.004, 0.03),
       'bus': (15000, 6.0, 0.0070, 0.002, 0.05), 'truck_md': (10000, 5.0, 0.0072, 0.002, 0.05),
       'truck_hd': (30000, 5.9, 0.0062, 0.0015, 0.04)}
SURF = np.array([1.0, 0.95, 1.12, 1.6, 2.2])
SPD_E = [0.5, 4.17, 8.33, 13.89, 19.44, 25.0, 30.56]           # idle | <15 | 15-30 | 30-50 | 50-70 | 70-90 | 90-110 | >110 km/h
ACC_E = [-1.5, -0.5, -0.1, 0.1, 0.5, 1.5]
GRD_E = [-0.06, -0.03, -0.01, 0.01, 0.03, 0.06]
NR, NP, NV, NF = 90, 8, 47, 12
# generic physics-estimate constants (fitted on the simulated training bank only; see fit_phys in train.py)
PHYS_K = {'ice': [0.2845, 4642.0], 'hev': [0.3023, 0.3757, 1662.0], 'bev': [0.8928, 0.6305, 565.9]}


def prep(trip):
    """trip dict of 1 Hz arrays -> float64 arrays used by every feature."""
    v = np.nan_to_num(np.asarray(trip['v'], float)); N = len(v)
    def g(k, d):
        x = trip.get(k)
        if x is None: return np.full(N, d, float)
        x = np.asarray(x, float)
        if np.isnan(x).all(): return np.full(N, d, float)
        if np.isnan(x).any():
            i = np.arange(N); ok = ~np.isnan(x); x = np.interp(i, i[ok], x[ok])
        return x
    grade = np.clip(g('grade', 0.0), -0.2, 0.2)
    T = g('T_amb', 20.0); ws = g('wind_speed', 0.0); wd = g('wind_dir', 0.0); hd = g('heading', 0.0)
    rho = g('rho_air', 0.0)
    if (rho <= 0).all():
        elev = g('elev', 0.0)
        rho = 101325.0 * (1 - 2.25577e-5 * elev) ** 5.25588 / (287.05 * (T + 273.15))
    rel = np.radians(wd - hd)
    surf = np.clip(np.nan_to_num(g('surface', 0.0)).astype(int), 0, 4)
    a = np.gradient(v) if N > 1 else np.zeros(N)
    return dict(v=v, a=a, grade=grade, T=T, wh=ws * np.cos(rel), wc=np.abs(ws * np.sin(rel)), ws=ws, rho=rho, surf=surf, N=N)


def route_vec(P):
    v, a, gr, N = P['v'], P['a'], P['grade'], P['N']
    dist = max(v.sum(), 1.0); dur = float(N)
    mov = v > 0.5
    stops = np.sum((v[1:] < 0.5) & (v[:-1] >= 0.5))
    R = np.zeros(NR)
    R[0] = np.log(dur / 600.0); R[1] = np.log(dist / 5000.0); R[2] = dist / dur / 20.0; R[3] = 1 - mov.mean()
    R[4] = min(stops / (dist / 1000.0), 10.0) / 3.0; R[5] = v[0] / 20.0; R[6] = v[-1] / 20.0
    R[7] = np.sqrt(np.mean(a[mov] ** 2)) if mov.any() else 0.0
    R[8] = np.sum(np.clip(np.diff(v ** 2), 0, None)) / 2.0 / dist
    R[9] = np.sum(v ** 3) / dist / 400.0
    si = np.digitize(v, SPD_E); ai = np.digitize(a, ACC_E)
    H = np.bincount(si * 7 + ai, minlength=56)[:56] / dur
    R[10:66] = H
    gi = np.digitize(gr, GRD_E)
    R[66:73] = np.bincount(gi, weights=v, minlength=7)[:7] / dist
    dh = v * gr
    km = dist / 1000.0
    R[73] = np.sum(np.clip(dh, 0, None)) / km / 20.0; R[74] = np.sum(np.clip(-dh, 0, None)) / km / 20.0; R[75] = np.sum(dh) / km / 20.0
    T = P['T']
    R[76] = T.mean() / 20.0; R[77] = T.min() / 20.0; R[78] = T.max() / 20.0; R[79] = (T < 0).mean(); R[80] = (T > 25).mean()
    R[81] = np.sum(P['wh'] * v) / dist / 5.0; R[82] = np.sum(P['wc'] * v) / dist / 5.0; R[83] = P['ws'].mean() / 5.0
    R[84] = (P['rho'].mean() - 1.2) * 5.0
    s = P['surf']
    R[85] = (s == 2).mean(); R[86] = (s == 4).mean(); R[87] = (s == 3).mean(); R[88] = (s == 1).mean()
    R[89] = (v > 25).mean()
    return R


def wheel_power(P, m, CdA, Crr, cv, rot, hd=False):
    """per-second road-load terms (W)."""
    v, a = P['v'], P['a']
    Kt = 0.010 if hd else 0.008
    fT = 1.0 / np.maximum(0.6, 1.0 + Kt * (P['T'] - 20.0))
    vr = v + P['wh']
    Fa = 0.5 * P['rho'] * CdA * vr * np.abs(vr)
    th = np.arctan(P['grade'])
    Fr = m * 9.81 * np.cos(th) * Crr * (1 + cv * v) * fT * SURF[P['surf']] * (v > 0.1)
    Fg = m * 9.81 * np.sin(th)
    Fi = m * (1 + rot) * a
    return Fa * v, Fr * v, Fg * v, Fi * v


def phys_ints(P, m, CdA, Crr, cv, rot, hd=False):
    pa, pr, pg, pi = wheel_power(P, m, CdA, Crr, cv, rot, hd)
    p = pa + pr + pg + pi
    return np.array([p.clip(0).sum(), p.clip(None, 0).sum(), pa.sum(), pr.sum(), pg.clip(0).sum(), pg.clip(None, 0).sum(),
                     pi.clip(0).sum(), pi.clip(None, 0).sum()]) / 1e6        # MJ


def variants(P, cls, m_true, CdA, Crr, cv, m_rep):
    """physics integrals for (true m, true RL), (true m, imputed RL), (reported m, imputed RL), (class m, imputed RL)."""
    mc, Cc, Rc, cvc, rot = CLS[cls]; hd = cls in HD
    imp = lambda m: (Cc * (m / mc) ** 0.5, Rc, cvc)
    out = [phys_ints(P, m_true, CdA, Crr, cv, rot, hd)]
    for m in (m_true, m_rep, mc):
        c, r, q = imp(m); out.append(phys_ints(P, m, c, r, q, rot, hd))
    return np.stack(out)


def e_phys(ph, arch, m, dur, soc=None, cap=None, hd=False):
    """generic energy estimate (MJ) from physics integrals ph (8,) -> (E, E_hev_form, E_bev_form, frac_cd)."""
    Ep, En = ph[0], -ph[1]
    s = (m / 1500.0) ** 0.7
    ki, kh, kb = PHYS_K['ice'], PHYS_K['hev'], PHYS_K['bev']
    e_ice = Ep / ki[0] + ki[1] * s * dur / 1e6
    e_hev = max((Ep - kh[1] * En) / kh[0], 0.2 * Ep / kh[0]) + kh[2] * s * dur / 1e6
    e_bev = max(Ep / kb[0] - kb[1] * En, 0.3 * Ep) + kb[2] * s * dur / 1e6
    e_ice, e_hev, e_bev = max(e_ice, 1e-3), max(e_hev, 1e-3), max(e_bev, 1e-3)
    if arch == 'conv': return e_ice, e_hev, e_bev, 0.0
    if arch == 'bev': return e_bev, e_hev, e_bev, 1.0
    if arch.startswith('phev'):
        if soc is None: f = 0.5
        else:
            usable = 0.8 * cap if cap else (100.0 if hd else 8.0)
            f = float(np.clip(soc * usable * 3.6 / e_bev, 0, 1))
        return f * e_bev + (1 - f) * e_hev, e_hev, e_bev, f
    return e_hev, e_hev, e_bev, 0.0


def phys_vec(ph, E, Eh, Eb, f, m, dist):
    Ep = max(ph[0], 1e-6)
    F = np.zeros(NF)
    F[0] = np.log(E); F[1] = np.log(Ep / E); F[2] = np.log((-ph[1] + 1e-4) / E)
    F[3] = ph[2] / Ep; F[4] = ph[3] / Ep; F[5] = ph[4] / Ep; F[6] = ph[5] / Ep; F[7] = ph[6] / Ep
    F[8] = np.log(Ep * 1e6 / (m * max(dist, 1.0)) / 0.1)
    F[9] = np.log(Eh / E); F[10] = np.log(Eb / E); F[11] = f
    return F


def vehicle_vec(arch, cls, m_used, m_level, cda, crr, rl_known, disp, power, diesel, trans, ngears, cap, lab_fuel, lab_elec,
                soc, warm, cold, maf, dyno, E, vmean):
    """lab_fuel / lab_elec: (city MJ/km, hwy MJ/km) or None. m_level: 2 exact, 1 rounded/reported, 0 unknown."""
    V = np.zeros(NV); i = 0
    V[ARCHS.index(arch)] = 1; i = 7
    V[i + CLASSES.index(cls)] = 1; i = 14
    V[i] = np.log(m_used / 1500.0); V[i + 1] = float(m_level == 2); V[i + 2] = float(m_level == 1); i = 17
    if rl_known: V[i] = np.log(cda / 0.7); V[i + 1] = np.log(crr / 0.009); V[i + 2] = 1
    i = 20
    if disp: V[i] = np.log(disp / 2.0); V[i + 1] = 1
    if power: V[i + 2] = np.log(power / 100.0); V[i + 3] = 1
    V[i + 4] = float(bool(diesel)); i = 25
    if trans in TRANS: V[i + TRANS.index(trans)] = 1
    if ngears: V[i + 6] = ngears / 10.0; V[i + 7] = 1
    i = 33
    if cap: V[i] = np.log(cap / 10.0); V[i + 1] = 1
    i = 35
    w = float(np.clip((48.3 - vmean / 0.44704) / (48.3 - 21.2), 0, 1))
    for lab in (lab_fuel, lab_elec):
        if lab is not None and lab[0] and lab[1]:
            mix = w * lab[0] + (1 - w) * lab[1]
            V[i] = np.log(mix * E[1] / E[0]); V[i + 1] = np.log(lab[0] / lab[1]); V[i + 2] = 1
        i += 3
    if soc is not None: V[41] = soc; V[42] = 1
    V[43] = float(bool(warm)); V[44] = float(bool(cold)); V[45] = float(bool(maf)); V[46] = float(bool(dyno))
    return V
