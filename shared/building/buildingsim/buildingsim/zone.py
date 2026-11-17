"""Envelope geometry, boundary conditions and the ISO 13790 5R1C zone model.

Geometry: a rectangular footprint (aspect ratio, n floors, floor-to-floor height); long facades face south and
north when orientation = 0. Windows = WWR x gross facade area on each facade.

Boundary temperatures (all vectorised before the time loop):
- windows: outdoor air T_out (U-value includes films);
- opaque walls and roof: sol-air temperature (ASHRAE Fundamentals ch. 18):
  T_sa = T_out + a I / h_o - eps F_sky sigma (T_out^4 - T_sky^4) / h_o, h_o = 8 + 4 v_local W/m2K;
  a vented attic sees T_out + 0.35 (T_sa,roof - T_out) (assumption);
- ground: slab perimeter F-factor x perimeter to the Kusuda ground temperature at 1 m.
All opaque paths meet at the mass node (ISO 13790 H_tr,em) through one weighted boundary temperature
theta_em = sum_j H_j theta_j / sum_j H_j, exact for parallel paths.

Solar through windows: per facade, beam x SHGC x angular modifier (pvlib.iam.physical, Fresnel + absorption, raised
to the number of panes) x overhang sunlit fraction, plus sky and ground diffuse x the hemispherical modifiers
(pvlib.iam.marion_diffuse), x a shading factor (frames, screens, surroundings).

Infiltration: LBL / Sherman-Grimsrud model, ASHRAE Fundamentals 2017 ch. 16:
Q [L/s] = A_L [cm2] sqrt(C_s |dT| + C_w U^2), C_s, C_w by number of stories and shelter class.
`infiltration_ach` in the spec is the rate at dT = 15 K and U = 4 m/s; `ach50` (blower door) is used instead
when given (ELA at 4 Pa from n = 0.65). Commercial infiltration drops to 25 % while the supply fan runs (as in the
DOE prototypes).

Zone: ISO 13790:2008 Annex C 5R1C network (air node, central/surface node, one mass node), Crank-Nicolson on the
mass node, generalised to any step length. HVAC heat can be injected at the air, surface or mass node.
"""
import math
import numpy as np
import pvlib

from . import psychro as ps
from .spec import MASS_CLASSES

CP_AIR = 1020.0      # J/kg K, moist air at a typical indoor humidity ratio (sensible heat of air streams)
SIGMA = 5.670374e-8

# ASHRAE Fundamentals 2017 ch.16 tables 4-5 (LBL model): stack coefficient per number of stories and wind
# coefficient [(L/s)^2 / (cm^4 (m/s)^2)] by shelter class (rows) and stories (columns 1,2,3)
_CS = {1: 0.000145, 2: 0.000290, 3: 0.000435}
_CW = {1: (0.000319, 0.000420, 0.000494), 2: (0.000246, 0.000325, 0.000382), 3: (0.000174, 0.000231, 0.000271),
       4: (0.000104, 0.000137, 0.000161), 5: (0.000032, 0.000042, 0.000049)}


def geometry(spec):
    g = spec['geometry']
    A = float(spec['floor_area_m2'])
    nf = max(1, int(spec['n_floors']))
    bg = int(g.get('below_grade_floors', 0) or 0)       # e.g. a conditioned basement level
    fp = A / (nf + bg)
    ar = max(1.0, float(g['aspect_ratio']))
    L = math.sqrt(fp * ar)
    W = fp / L
    H = nf * float(g['floor_to_floor_m'])            # above-grade wall height
    ex = float(g.get('wall_exposed_frac', 1.0))
    gross = {'S': L * H * ex, 'N': L * H * ex, 'E': W * H * ex, 'W': W * H * ex}
    wwr = float(spec['window_wwr'])
    win = {k: v * wwr for k, v in gross.items()}
    if 'window_area_by_facade' in g:
        win = {k: float(g['window_area_by_facade'].get(k, 0.0)) for k in gross}
    opq = {k: max(gross[k] - win[k], 0.0) for k in gross}
    return dict(A_f=A, n_floors=nf, footprint=fp, L=L, W=W, H=H, gross=gross, win=win, opaque=opq,
                roof=fp if g.get('roof_exposed', True) else 0.0, ground=fp, perimeter=2 * (L + W), volume=A * float(g['ceiling_m']),
                ceiling=float(g['ceiling_m']))


def _panes(window_U):
    return 1 if window_U > 4.0 else (2 if window_U > 1.6 else 3)


def window_solar(spec, geo, wx):
    """Transmitted solar gain through all windows [W] per facade (dict) and total, vectorised."""
    g = spec['geometry']
    shgc = float(spec['SHGC'])
    npanes = _panes(float(spec['window_U']))
    diff = pvlib.iam.marion_diffuse('physical', surface_tilt=90.0)
    iam_sky, iam_gnd = float(diff['sky']) ** npanes, float(diff['ground']) ** npanes
    sf = float(g['shading_factor'])
    ov = float(g.get('window_overhang', 0.0))
    gap = float(g.get('overhang_gap', 0.0))      # height of the overhang above the window head / window height
    alt = np.radians(90.0 - wx['zenith'])
    # interior shading (blinds/curtains) by season: (summer, winter) multipliers; summer = 30-day running mean
    # outdoor temperature >= 15 C. ResStock default for every dwelling: 0.7 / 0.85.
    ish = g.get('interior_shading')
    if ish:
        import pandas as pd
        rm = pd.Series(wx['T_out']).rolling(int(30 * 86400 / wx['dt_s']), min_periods=1).mean().values
        season = np.where(rm >= 15.0, float(ish[0]), float(ish[1]))
    else:
        season = 1.0
    out, tot = {}, 0.0
    for k in ('N', 'E', 'S', 'W'):
        a = geo['win'][k]
        s = wx['surf'][k]
        if a <= 0:
            out[k] = np.zeros(wx['n'])
            continue
        aoi = np.clip(s['aoi'], 0, 90)
        iam_b = np.nan_to_num(pvlib.iam.physical(aoi)) ** npanes
        beam = s['beam'] * iam_b
        if ov > 0:
            gam = np.radians(wx['azimuth'] - s['az'])
            prof = np.tan(np.clip(alt, 0, None)) / np.maximum(np.cos(gam), 1e-3)
            sunlit = np.clip(1.0 - np.clip(ov * prof - gap, 0, None), 0, 1)
            beam = beam * sunlit
        q = a * sf * shgc * season * (beam + iam_sky * s['sky'] + iam_gnd * s['ground'])
        out[k] = q
        tot = tot + q
    return out, tot


def boundary(spec, geo, wx):
    """Opaque boundary temperatures [C] (walls, roof/attic, ground), their conductances [W/K]."""
    g = spec['geometry']
    T = wx['T_out']
    v_loc = 0.7 * wx['wind']
    ho = 8.0 + 4.0 * v_loc
    Tk4 = (T + 273.15) ** 4 - (wx['T_sky'] + 273.15) ** 4
    aw, ar = float(g['wall_absorptance']), float(g['roof_absorptance'])
    Uw, Ur = float(spec['wall_U']), float(spec['roof_U'])
    Hw = {k: Uw * geo['opaque'][k] for k in ('N', 'E', 'S', 'W')}
    Hw_tot = sum(Hw.values())
    Tw = np.zeros(wx['n'])
    for k in ('N', 'E', 'S', 'W'):
        s = wx['surf'][k]
        I = s['beam'] + s['sky'] + s['ground']
        Tsa = T + aw * I / ho - 0.9 * 0.5 * SIGMA * Tk4 / ho
        Tw += (Hw[k] / max(Hw_tot, 1e-9)) * Tsa
    sh = wx['surf']['H']
    Ir = sh['beam'] + sh['sky']
    Tsa_r = T + ar * Ir / ho - 0.9 * SIGMA * Tk4 / ho
    if g.get('attic'):
        Tr = T + float(g.get('attic_frac', 0.35)) * (Tsa_r - T)
    else:
        Tr = Tsa_r
    Hr = Ur * geo['roof']
    fnd = g.get('foundation', 'slab')
    fp, P = geo['ground'], geo['perimeter'] * float(g.get('wall_exposed_frac', 1.0))
    Tg1, Tg3 = wx['T_ground'], wx['T_ground_deep']
    U_bw = 1.0 / (1.0 / float(g.get('basement_U', 1.43)) + 1.5)   # below-grade wall incl. ~1.5 m2K/W of soil path
    if fnd == 'slab':
        Hg = float(g['slab_F']) * P
        Tg = Tg1
    elif fnd == 'basement':      # conditioned basement level: below-grade walls (2.1 m) + floor to deep ground
        Hg = U_bw * 2.1 * P + 0.12 * fp
        Tg = Tg3
    elif fnd in ('crawlspace', 'crawlspace_unvented', 'unconditioned_basement', 'ambient'):
        # unconditioned buffer space under the floor, steady-state heat balance (ISO 13789 b-factor):
        # H = H_iu H_ue / (H_iu + H_ue), boundary = conductance-weighted outdoor / ground temperature
        H_iu = float(g.get('floor_U', 1.0)) * fp
        if fnd == 'crawlspace':            # vented, ~1.5 ACH of a 0.9 m space + short walls
            H_out, H_gr = 0.45 * fp + 2.0 * 0.6 * P, 0.3 * fp
        elif fnd == 'crawlspace_unvented':
            H_out, H_gr = 0.05 * fp + 1.0 * 0.6 * P, 0.3 * fp
        elif fnd == 'unconditioned_basement':   # 0.2 ACH of a 2.4 m basement + 0.3 m above grade walls
            H_out, H_gr = 0.16 * fp + 1.5 * 0.3 * P, U_bw * 2.1 * P + 0.1 * fp
        else:                               # 'ambient': raised floor / skirted mobile home, well ventilated
            H_out, H_gr = 3.0 * fp, 0.1 * fp
        H_ue = H_out + H_gr
        Hg = H_iu * H_ue / (H_iu + H_ue) if H_iu > 0 else 0.0
        Tg = (H_out * T + H_gr * Tg1) / H_ue
    else:   # 'adiabatic' (mid-floor / BESTEST-style raised floor uses spec floor_U to outdoor)
        Hg = float(g.get('floor_U', 0.0)) * geo['ground']
        Tg = T.copy()
    return dict(H_wall=Hw_tot, H_roof=Hr, H_ground=Hg, T_wall=Tw, T_roof=Tr, T_ground=np.asarray(Tg, float))


def infiltration_coeffs(spec, geo, wx):
    """Returns (ELA_cm2, C_s, C_w U^2 array) so that Q[L/s] = ELA sqrt(C_s |dT| + CwU2)."""
    nf = geo['n_floors']
    res = spec['primary_use'].startswith('residential')
    shelter = int(spec.get('geometry', {}).get('shelter_class') or (3 if res else 2))
    n_eff = min(nf, 3)
    Cs = _CS[n_eff] * (min(geo['H'], 4 * 3.5) / (n_eff * (2.5 if res else 3.5)) if nf > 3 else 1.0)
    Cw = _CW[shelter][n_eff - 1] * ((geo['H'] / 7.5) ** 0.4 if nf > 3 else 1.0)
    V = geo['volume']
    if spec.get('ach50'):
        Q50 = float(spec['ach50']) * V / 3600.0
        C = Q50 / 50 ** 0.65
        Q4 = C * 4 ** 0.65
        ela = Q4 * math.sqrt(1.2 / (2 * 4.0)) * 1e4      # cm2
    else:
        q_ref = float(spec['infiltration_ach']) * V / 3.6         # L/s
        ela = q_ref / math.sqrt(Cs * 15.0 + Cw * 16.0)
    return ela, Cs, Cw * wx['wind'] ** 2


class Zone5R1C:
    """Single thermal zone, ISO 13790 5R1C. Call prepare(k, ...) then advance(q_air, q_surf, q_mass)."""

    def __init__(self, spec, geo, bnd, n, dt_s, theta0=20.0):
        A_f = geo['A_f']
        tm = spec['thermal_mass']
        if isinstance(tm, (int, float)):
            Cm = float(tm) * 1000.0 * A_f
            am = 2.5 if tm < 200 else (3.0 if tm < 300 else 3.5)
        else:
            am, cm = MASS_CLASSES[tm]
            Cm = cm * A_f
        if spec['geometry'].get('C_m_J_K'):
            Cm = float(spec['geometry']['C_m_J_K'])
        if spec['geometry'].get('A_m_m2'):
            self.A_m = float(spec['geometry']['A_m_m2'])
        else:
            self.A_m = am * A_f
        self.A_t = float(spec['geometry'].get('A_t_m2') or 4.5 * A_f)
        self.C_m = Cm
        self.H_tr_is = 3.45 * self.A_t
        self.H_tr_ms = 9.1 * self.A_m
        self.H_tr_w = float(spec['window_U']) * sum(geo['win'].values())
        Hop = bnd['H_wall'] + bnd['H_roof'] + bnd['H_ground']
        for k_, v_ in (('H_wall', bnd['H_wall']), ('H_roof', bnd['H_roof']), ('H_ground', bnd['H_ground']),
                       ('window_U', float(spec['window_U'])), ('C_m', Cm)):
            if not (v_ >= 0 and math.isfinite(v_)):
                raise ValueError(f'invalid envelope parameter {k_} = {v_} (spec {spec.get("id")})')
        if Hop >= 9.1 * self.A_m:
            raise ValueError('opaque conductance exceeds the 5R1C mass coupling (H_op >= 9.1 A_m); check U-values')
        self.H_op = Hop
        self.H_tr_em = 1.0 / (1.0 / Hop - 1.0 / self.H_tr_ms) if Hop > 0 else 0.0
        self.w_wall = bnd['H_wall'] / Hop if Hop > 0 else 0.0
        self.w_roof = bnd['H_roof'] / Hop if Hop > 0 else 0.0
        self.w_gnd = bnd['H_ground'] / Hop if Hop > 0 else 0.0
        self.dt = dt_s
        self.cm = Cm / dt_s
        self.f_st = 1.0 - self.A_m / self.A_t - self.H_tr_w / (9.1 * self.A_t)
        self.f_m = self.A_m / self.A_t
        self.theta_m = theta0
        self.theta_s = theta0
        self.theta_air = theta0
        self.UA = self.H_tr_w + Hop     # envelope conductance without air exchange
        # time weighting of the mass node: Crank-Nicolson (ISO 13790) unless the mass time constant is shorter
        # than two steps, where CN oscillates; then implicit Euler (L-stable). Ledgers stay exact either way.
        H_ve_typ = 0.3 * 1.2 * 1020.0 * geo['volume'] / 3600.0 + 1.0
        self.tau_h = self.time_constant_h(H_ve_typ)
        self.w = 0.5 if self.tau_h * 3600.0 > 2.0 * dt_s else 1.0

    def time_constant_h(self, H_ve):
        H1 = 1 / (1 / H_ve + 1 / self.H_tr_is)
        H3 = 1 / (1 / (H1 + self.H_tr_w) + 1 / self.H_tr_ms)
        return self.C_m / (H3 + self.H_tr_em) / 3600.0

    def solve(self, theta_m_prev, th_e, th_em, H_ve, th_sup, phi_int, phi_sol, qa, qs, qm):
        """One step of the ISO 13790 C.3 equations. Returns (theta_m_t, theta_m_avg, theta_s, theta_air)."""
        H1 = 1.0 / (1.0 / H_ve + 1.0 / self.H_tr_is)
        H2 = H1 + self.H_tr_w
        H3 = 1.0 / (1.0 / H2 + 1.0 / self.H_tr_ms)
        g = 0.5 * phi_int + phi_sol
        p_ia = 0.5 * phi_int + qa
        p_st = self.f_st * g + qs
        p_m = self.f_m * g + qm
        pm_tot = p_m + self.H_tr_em * th_em + H3 * (p_st + self.H_tr_w * th_e + H1 * (p_ia / H_ve + th_sup)) / H2
        Hm = H3 + self.H_tr_em
        w = self.w
        tm_t = (theta_m_prev * (self.cm - (1.0 - w) * Hm) + pm_tot) / (self.cm + w * Hm)
        tm = w * tm_t + (1.0 - w) * theta_m_prev
        ts = (self.H_tr_ms * tm + p_st + self.H_tr_w * th_e + H1 * (th_sup + p_ia / H_ve)) / (self.H_tr_ms +
                                                                                            self.H_tr_w + H1)
        ta = (self.H_tr_is * ts + H_ve * th_sup + p_ia) / (self.H_tr_is + H_ve)
        return tm_t, tm, ts, ta
