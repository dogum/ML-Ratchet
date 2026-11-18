"""ThermoPilot features (numpy only; vendored into the release).

A 'building card' describes what the thermostat knows about the house (from its parameters, an audit, or an
online estimate): UA (W/K), thermal capacitance C (J/K), floor area, heating kind, fan mode, a heating table
(T_out -> max heating W, heat-pump electric W at full load), a cooling table (T_out -> sensible W, electric W),
and the expected non-HVAC electricity per hour. Features at hour k use only what a thermostat can see then:
the indoor temperature history, its own past HVAC heat/power, the 24 h weather FORECAST, the tariff and the
occupant comfort band schedule.
"""
import math
import numpy as np

KINDS = ['hp', 'furnace', 'elec', 'baseboard', 'none']
H = 24
NB = 11          # per-hour channels
NS = 34          # scalar features
NF = NS + NB * H


def card_from_model(M, hi0=24.0):
    """Building card from a tp.mpc.Model (the NOMINAL house)."""
    Tg = np.arange(-40.0, 50.01, 1.0)
    if M.kind == 'hp':
        Qh = np.interp(Tg, M.Tg, M.Qh_tab); Ph = np.interp(Tg, M.Tg, M.Ph_tab)
        Qmax = Qh + M.backup
    else:
        Qh = np.zeros_like(Tg); Ph = np.zeros_like(Tg)
        Qmax = np.full_like(Tg, M.Qheat if M.kind != 'none' else 0.0)
    Qs = np.zeros_like(Tg); Pc = np.zeros_like(Tg)
    if M.cool is not None:
        for i, T in enumerate(Tg):
            _, _, _, q_s, p_ = M.cool.full(float(hi0), 0.0095, float(T), M.m)
            Qs[i], Pc[i] = q_s, p_
    if M.kind == 'hp':
        eff = np.where(Ph > 0, Qh / np.maximum(Ph, 1e-9), 1.0)
    elif M.kind in ('elec', 'baseboard'):
        eff = np.ones_like(Tg) * (1.0 if M.bb_fuel == 'elec' or M.kind == 'elec' else M.eff)
    else:
        eff = np.full_like(Tg, M.eff)
    return dict(UA=float(M.UA), C=float(M.C), area=float(M.sim.spec['floor_area_m2']), kind=M.kind,
                gas_heat=bool(M.kind == 'furnace' or (M.kind == 'baseboard' and M.bb_fuel != 'elec')),
                fan_cont=bool(M.fan_cont), res=bool(M.res), Tg=Tg, Qmax=Qmax, Qh=Qh, Ph=Ph, eff=eff, Qs=Qs, Pc=Pc,
                P_fan=float(M.P_fan), backup=float(M.backup),
                # R8 card fields (the card MPC's plant): heater size, scalar efficiency, outdoor-air conductance of a
                # constant-volume fan, compressor speeds
                Qheat=float(M.Qheat), eff0=float(M.eff), gx=float(M.gx), hp_speed=getattr(M, 'hp_speed', 'single'),
                cool_speed=M.cool_speed)


def card_map(M, ctx):
    """R8 card house: the 12-number linear map of tp.card_mpc.CardMPC, fitted by least squares to the nominal
    twin's hourly 5R1C one-step maps (tp.mpc.Model.coeffs at the middle of the band, the episode's weather):
    At, Aq, Ct, Cq are their means; A and C are regressed on [1, T_out, GHI, base load kW]."""
    n = M.n
    ia = np.arange(n)
    lo = ctx['lo'][np.minimum(ia, len(ctx['lo']) - 1)]; hi = ctx['hi'][np.minimum(ia, len(ctx['hi']) - 1)]
    c = M.coeffs(ia, M.To[ia], M.ghi[ia], 0.5 * (lo + hi))
    X = np.stack([np.ones(n), M.To[ia], M.ghi[ia], M.base_W[ia] / 1000.0], 1)
    out = {k: float(np.mean(c[k])) for k in ('At', 'Aq', 'Ct', 'Cq')}
    # A, C are the maps at T_mass = 0; with At, Ct fixed at their means the regression absorbs the hourly variation
    for k, kt in (('A', 'At'), ('C', 'Ct')):
        y = np.asarray(c[k], float) + (np.asarray(c[kt], float) - out[kt]) * 20.0   # re-centre at T_mass = 20 C
        out[k.lower()] = [float(v) for v in np.linalg.lstsq(X, y, rcond=None)[0]]
    return out


def full_card(M, ctx, hi0):
    """The R8 building card: the v1.0 card + the card house map."""
    card = card_from_model(M, hi0)
    card['map'] = card_map(M, ctx)
    return card


def pref_kW(card, base_W):
    """Reference electric power (kW) for normalising: full-load HVAC electric power at -5 / 35 C + mean base load."""
    ph = np.interp(-5.0, card['Tg'], card['Ph']) if card['kind'] == 'hp' else (
        np.interp(-5.0, card['Tg'], card['Qmax']) if card['kind'] in ('elec', 'baseboard') and not card['gas_heat'] else 0.0)
    pc = np.interp(35.0, card['Tg'], card['Pc'])
    return max(0.5, (max(ph, pc) + card['P_fan'] + float(np.mean(base_W))) / 1000.0)


class Featurizer:
    """Stateful: call step(k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G, ctx) every hour."""

    def __init__(self, card, ctx, lam, base_W):
        self.card, self.ctx, self.lam = card, ctx, float(lam)
        self.base_W = np.asarray(base_W, float)
        self.P = pref_kW(card, self.base_W)
        self.Th = []
        self.qh = []
        self.peak = {}
        self.lo0, self.hi0 = float(ctx['pref_lo']), float(ctx['pref_hi'])
        c = card
        self.ua = max(c['UA'], 1.0)
        self.desc = np.zeros(13, np.float32)
        self.desc[0] = math.log(self.ua / 150.0)
        self.desc[1] = math.log(max(c['C'] / self.ua / 3600.0, 1.0) / 30.0)
        self.desc[2] = math.log(max(c['area'], 10.0) / 150.0)
        self.desc[3 + KINDS.index(c['kind'])] = 1.0
        self.desc[8] = float(c['fan_cont']); self.desc[9] = float(c['res']); self.desc[10] = float(c['backup'] > 0)
        self.desc[11] = float(c['gas_heat']); self.desc[12] = math.log(self.P)

    def observe(self, k, elec_prev_kW):
        ctx = self.ctx
        if k <= 0:
            return
        j = k - 1
        if ctx['dperiod'][j] >= 0:
            key = (ctx['ym'][j], int(ctx['dperiod'][j]))
            self.peak[key] = max(self.peak.get(key, 0.0), elec_prev_kW)
        if ctx['frate'][j] > 0:
            key = (ctx['ym'][j], 'flat')
            self.peak[key] = max(self.peak.get(key, 0.0), elec_prev_kW)

    def step(self, k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G):
        c, ctx = self.card, self.ctx
        self.observe(k, elec_prev_kW)
        self.Th.append(float(T_zone)); self.qh.append(float(q_prev_W))
        n = len(ctx['price'])
        idx = np.minimum(np.arange(k, k + H), n - 1)
        price = ctx['price'][idx]
        pm = max(float(np.mean(np.abs(price))), 0.02)
        lo, hi = ctx['lo'][idx], ctx['hi'][idx]
        x = np.zeros(NF, np.float32)
        Tz = float(T_zone)
        x[0] = (Tz - lo[0]) / 3; x[1] = (hi[0] - Tz) / 3; x[2] = (Tz - self.lo0) / 3; x[3] = (self.hi0 - Tz) / 3
        for i in range(1, 4):
            x[3 + i] = (Tz - self.Th[-1 - i]) if len(self.Th) > i else 0.0
        for i in range(1, 4):
            x[6 + i] = (self.qh[-i] / self.ua / 10.0) if len(self.qh) >= i else 0.0
        x[10] = elec_prev_kW / self.P
        kk = k
        dp = int(ctx['dperiod'][kk])
        x[11] = self.peak.get((ctx['ym'][kk], dp), 0.0) / self.P if dp >= 0 else 0.0
        x[12] = self.peak.get((ctx['ym'][kk], 'flat'), 0.0) / self.P if ctx['frate'][kk] > 0 else 0.0
        h = ctx['hour'][kk]
        x[13] = math.sin(2 * math.pi * h / 24); x[14] = math.cos(2 * math.pi * h / 24); x[15] = float(ctx['wd'][kk])
        x[16] = math.log(max(price[0], 0.005) / 0.15)
        x[17] = math.log(self.lam / (pm * self.P))
        x[18] = ctx['drate'][kk] / max(ctx['drem'][kk], 1) / pm / 10.0
        x[19] = ctx['frate'][kk] / max(ctx['frem'][kk], 1) / pm / 10.0
        x[20] = float(ctx['state'][kk] == 0)
        x[21:34] = self.desc
        Tg = c['Tg']
        To = np.asarray(fc_T, float)[:H]; G = np.asarray(fc_G, float)[:H]
        b = NS
        x[b:b + H] = (To - Tz) / 10.0; b += H
        x[b:b + H] = G / 800.0; b += H
        x[b:b + H] = price / pm - 1.0; b += H
        x[b:b + H] = (lo - self.lo0) / 3.0; b += H
        x[b:b + H] = (hi - self.hi0) / 3.0; b += H
        x[b:b + H] = np.interp(To, Tg, c['eff']) / 4.0; b += H
        pc = np.interp(To, Tg, c['Pc']); qs = np.interp(To, Tg, c['Qs'])
        x[b:b + H] = np.where(pc > 0, qs / np.maximum(pc, 1e-9), 0.0) / 4.0; b += H
        x[b:b + H] = self.base_W[idx] / 1000.0 / self.P; b += H
        x[b:b + H] = (ctx['drate'][idx] > 0) + 0.5 * (ctx['frate'][idx] > 0); b += H
        x[b:b + H] = np.interp(To, Tg, c['Qmax']) / self.ua / 20.0; b += H
        x[b:b + H] = qs / self.ua / 20.0; b += H
        return x
