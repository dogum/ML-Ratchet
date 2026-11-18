"""ThermoPilot inference, numpy only.

A thermostat controller: every hour it picks a heating and a cooling setpoint from a menu of 36 combinations, using
the indoor temperature, its own past HVAC output, a 24 h weather forecast, the tariff and the occupant comfort band,
plus a 'building card' that describes the house.

v1.1 (default, mode='mpc') is classical, with no learned parameters: a 24 h dynamic programme over the house's mass
temperature on a 12-number linear house map carried by the card (card['map'], see CardMPC and fit_map), behind a
rule-based 6 h comfort shield. v1.0 (mode='network') is the MLP trained in BuildingSim by imitating a
perfect-foresight dynamic-programming controller (weights thermopilot_v1.npz).

    import numpy as np, thermopilot_core as tp
    ctx = tp.make_ctx(local_times, price, lo, hi, pref_lo, pref_hi, state)      # per-hour arrays for the period
    pilot = tp.pilot(card, ctx, lam=3.0, base_W=np.zeros(len(price)))         # v1.1; mode='network', W=tp.load(...) for v1.0
    for k in range(len(price)):
        sp = pilot.step(k, T_zone, q_hvac_prev_W, elec_prev_kW, forecast_T_out[24], forecast_GHI[24])
        # -> {'heat_sp': .., 'cool_sp': ..}  (degC; +-100 means that side is off)
"""
import math
import numpy as np

# ------------------------------------------------------------------------------------------- action menu
DH = [None, 'edge', 0.0, 1.0, 2.0, 3.0]      # heating: off, band edge, occupied low + 0..3 K (pre-heat)
DC = [None, 'edge', 0.0, -1.0, -2.0, -3.0]   # cooling: off, band edge, occupied high - 0..3 K (pre-cool)
ACTIONS = [(i, j) for i in range(len(DH)) for j in range(len(DC))]
SHIELD_RATE = 1.0


def setpoints(lo_k, hi_k, lo, hi, ih, ic):
    d = DH[ih]
    h = -100.0 if d is None else (lo_k if d == 'edge' else max(lo + d, lo_k))
    d = DC[ic]
    c = 100.0 if d is None else (hi_k if d == 'edge' else min(hi + d, hi_k))
    if c < h + 0.5:
        c = h + 0.5
    return h, c


# ------------------------------------------------------------------------------------------- features
KINDS = ['hp', 'furnace', 'elec', 'baseboard', 'none']
H = 24
NB = 11
NS = 34
NF = NS + NB * H


def make_ctx(local_times, price, lo, hi, pref_lo, pref_hi, state, dperiod=None, drate=None, frate=None):
    """Per-hour context for a period. local_times: pandas DatetimeIndex in local time (hourly).
    price $/kWh; lo/hi comfort band (degC, may widen when asleep/away); state 0 home, 1 asleep, 2 away;
    dperiod demand period id (-1 none) with drate $/kW-month; frate flat demand $/kW-month."""
    n = len(price)
    dperiod = np.full(n, -1) if dperiod is None else np.asarray(dperiod, int)
    drate = np.zeros(n) if drate is None else np.asarray(drate, float)
    frate = np.zeros(n) if frate is None else np.asarray(frate, float)
    ym = local_times.year.values * 12 + local_times.month.values
    doy = local_times.dayofyear.values
    drem, frem = np.ones(n), np.ones(n)
    for m in np.unique(ym):
        sel = np.flatnonzero(ym == m)
        for p in np.unique(dperiod[sel]):
            if p < 0:
                continue
            days = np.unique(doy[sel][dperiod[sel] == p])
            for i in sel[dperiod[sel] == p]:
                drem[i] = np.sum(days >= doy[i])
        days = np.unique(doy[sel])
        for i in sel:
            frem[i] = np.sum(days >= doy[i])
    return dict(price=np.asarray(price, float), lo=np.asarray(lo, float), hi=np.asarray(hi, float),
                state=np.asarray(state, int), pref_lo=float(pref_lo), pref_hi=float(pref_hi), dperiod=dperiod,
                drate=drate, frate=frate, hour=local_times.hour.values, wd=local_times.dayofweek.values < 5, ym=ym,
                doy=doy, drem=drem, frem=frem)


def pref_kW(card, base_W):
    ph = np.interp(-5.0, card['Tg'], card['Ph']) if card['kind'] == 'hp' else (
        np.interp(-5.0, card['Tg'], card['Qmax']) if card['kind'] in ('elec', 'baseboard') and not card['gas_heat'] else 0.0)
    pc = np.interp(35.0, card['Tg'], card['Pc'])
    return max(0.5, (max(ph, pc) + card['P_fan'] + float(np.mean(base_W))) / 1000.0)


class Featurizer:
    def __init__(self, card, ctx, lam, base_W):
        self.card, self.ctx, self.lam = card, ctx, float(lam)
        self.base_W = np.asarray(base_W, float)
        self.P = pref_kW(card, self.base_W)
        self.Th, self.qh, self.peak = [], [], {}
        self.lo0, self.hi0 = float(ctx['pref_lo']), float(ctx['pref_hi'])
        c = card
        self.ua = max(c['UA'], 1.0)
        d = np.zeros(13, np.float32)
        d[0] = math.log(self.ua / 150.0)
        d[1] = math.log(max(c['C'] / self.ua / 3600.0, 1.0) / 30.0)
        d[2] = math.log(max(c['area'], 10.0) / 150.0)
        d[3 + KINDS.index(c['kind'])] = 1.0
        d[8] = float(c['fan_cont']); d[9] = float(c['res']); d[10] = float(c['backup'] > 0)
        d[11] = float(c['gas_heat']); d[12] = math.log(self.P)
        self.desc = d

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
        dp = int(ctx['dperiod'][k])
        x[11] = self.peak.get((ctx['ym'][k], dp), 0.0) / self.P if dp >= 0 else 0.0
        x[12] = self.peak.get((ctx['ym'][k], 'flat'), 0.0) / self.P if ctx['frate'][k] > 0 else 0.0
        h = ctx['hour'][k]
        x[13] = math.sin(2 * math.pi * h / 24); x[14] = math.cos(2 * math.pi * h / 24); x[15] = float(ctx['wd'][k])
        x[16] = math.log(max(price[0], 0.005) / 0.15)
        x[17] = math.log(self.lam / (pm * self.P))
        x[18] = ctx['drate'][k] / max(ctx['drem'][k], 1) / pm / 10.0
        x[19] = ctx['frate'][k] / max(ctx['frem'][k], 1) / pm / 10.0
        x[20] = float(ctx['state'][k] == 0)
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


# ------------------------------------------------------------------------------------------- network + controller
def load(path):
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in z.files}


def mlp_forward(W, x):
    h = (x - W['mu']) / W['sd']
    i = 0
    while f'W{i}' in W:
        h = h @ W[f'W{i}'] + W[f'b{i}']
        if f'W{i + 1}' in W:
            h = np.maximum(h, 0.0)
        i += 1
    return h


class ThermoPilot:
    def __init__(self, W, card, ctx, lam, base_W):
        self.W, self.ctx = W, ctx
        self.shield = int(W['shield']) if 'shield' in W else 0
        self.feat = Featurizer(card, ctx, lam, base_W)
        self.lo0, self.hi0 = float(ctx['pref_lo']), float(ctx['pref_hi'])
        self.last = None

    def step(self, k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G):
        x = self.feat.step(k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G)
        a = int(np.argmax(mlp_forward(self.W, x.astype(np.float64))))
        self.last = a
        ih, ic = ACTIONS[a]
        ctx = self.ctx
        h, c = setpoints(ctx['lo'][k], ctx['hi'][k], self.lo0, self.hi0, ih, ic)
        if self.shield:
            lo, hi = float(ctx['lo'][k]), float(ctx['hi'][k])
            n = len(ctx['lo'])
            if self.shield == 2:
                k1 = min(k + 1, n - 1)
                lo, hi = max(lo, float(ctx['lo'][k1])), min(hi, float(ctx['hi'][k1]))
            elif self.shield >= 4:
                # heating can recover ~1 K/h, cooling ~0.5 K/h (conservative); look 6 h ahead
                n_ = len(self.ctx['lo']) if hasattr(self, 'ctx') else n
                for j in range(1, 7):
                    kj = min(k + j, n_ - 1)
                    lo = max(lo, float(self.ctx['lo'][kj]) - SHIELD_RATE * (j - 1))
                    hi = min(hi, float(self.ctx['hi'][kj]) + 0.5 * SHIELD_RATE * (j - 1))
            elif self.shield >= 3:
                for j in range(1, 5):
                    kj = min(k + j, n - 1)
                    lo = max(lo, float(ctx['lo'][kj]) - SHIELD_RATE * (j - 1))
                    hi = min(hi, float(ctx['hi'][kj]) + SHIELD_RATE * (j - 1))
            h = max(h, lo); c = min(c, hi)
            if c < h + 0.5:
                c = h + 0.5
        return dict(heat_sp=h, cool_sp=c)


# ------------------------------------------------------------------------------------------- v1.1: the card MPC
# The anchor is a receding-horizon DP (the same one as the classical MPC in training/tp/mpc.py: 0.25 K grid over the
# mass temperature, 24 h, hourly replanning, the 36 actions above, demand charges amortised over the running monthly
# peaks, lambda x K h outside the band) whose house is a 12-number linear map on the building card, not BuildingSim:
#     T_air = A + At T_mass + Aq q,      T_mass' = C + Ct T_mass + Cq q,       q = HVAC heat (W)
#     A = a0 + aT T_out + aG GHI + aB base_kW,   C = c0 + cT T_out + cG GHI + cB base_kW
# with forecast T_out and GHI and the expected non-HVAC load. card['map'] = dict(At, Aq, Ct, Cq, a=[4], c=[4]).
DX_PLF = (0.85, 0.15, 0.0)
DX_PLF_TWO = (0.92, 0.08, 0.0)
GAS_RES = 15.34 / 303.6        # $/kWh of gas (the training episodes' residential price)


def _plf(f, c):
    return c[0] + f * (c[1] + f * c[2])


def part_power(P_on, f, speed):
    """Electric power of a DX coil / heat pump at part load f (0..1) of full-load power P_on."""
    f = np.asarray(f, float)
    if speed == 'single':
        P = P_on * f / (DX_PLF[0] + f * DX_PLF[1])
    else:
        f = np.clip(f, 0.0, 1.0)
        if speed == 'variable':
            hi = P_on * f * (0.75 + 0.25 * f)
            f3 = np.maximum(f / 0.3, 1e-9)
            lo = P_on * 0.3 * (0.75 + 0.25 * 0.3) * f3 / _plf(np.minimum(f3, 1.0), DX_PLF)
            P = np.where(f >= 0.3, hi, lo)
        elif speed == 'two':
            P = P_on * f / _plf(f, DX_PLF_TWO)
        else:
            P = P_on * f / _plf(f, DX_PLF)
    return np.where(f > 0, P, 0.0)


class CardMPC:
    """The anchor. values(k, ...) -> (action, 36 action values in $); step(...) -> setpoints."""

    def __init__(self, card, ctx, lam, base_W, gas_price=GAS_RES, H=24):
        self.card, self.ctx, self.lam, self.H = card, ctx, float(lam), H
        self.base_W = np.asarray(base_W, float)
        self.gas_price = float(gas_price)
        mp = card['map']
        self.At, self.Aq, self.Ct, self.Cq = (float(mp[k]) for k in ('At', 'Aq', 'Ct', 'Cq'))
        self.a, self.c = np.asarray(mp['a'], float), np.asarray(mp['c'], float)
        self.lo0, self.hi0 = float(ctx['pref_lo']), float(ctx['pref_hi'])
        self.aih = np.array([a[0] for a in ACTIONS]); self.aic = np.array([a[1] for a in ACTIONS])
        self.kind = card['kind']
        self.Qheat = float(card.get('Qheat', np.max(card['Qmax'])))
        self.eff = float(card.get('eff0', np.median(card['eff'])))
        self.gx = float(card.get('gx', 0.0))
        self.hp_speed = card.get('hp_speed', 'single')
        self.cool_speed = card.get('cool_speed', 'single')
        self.tm = None
        self.prev = None          # (T_out, GHI, base_kW) of the last hour, for the observer
        self.peak = {}
        self.last = None

    def maps(self, To, G, base_kW):
        X = np.stack([np.ones_like(To), To, G, base_kW], 1)
        return X @ self.a, X @ self.c

    def observe(self, k, T_zone, q_prev_W, elec_prev_kW):
        ctx = self.ctx
        if k > 0:
            j = k - 1
            if ctx['dperiod'][j] >= 0:
                key = (int(ctx['ym'][j]), int(ctx['dperiod'][j]))
                self.peak[key] = max(self.peak.get(key, 0.0), float(elec_prev_kW))
            if ctx['frate'][j] > 0:
                key = (int(ctx['ym'][j]), 'flat')
                self.peak[key] = max(self.peak.get(key, 0.0), float(elec_prev_kW))
        if self.prev is None or self.tm is None:
            self.tm = float(T_zone)            # first hour: mass at the air temperature
            return
        A, C = self.maps(*(np.array([v]) for v in self.prev))
        tm_prev = (float(T_zone) - A[0] - self.Aq * q_prev_W) / self.At if abs(self.At) > 1e-3 else self.tm
        tm_prev = float(np.clip(tm_prev, -10.0, 50.0))
        self.tm = float(C[0] + self.Ct * tm_prev + self.Cq * q_prev_W)

    def _sp(self, lo, hi):
        Hn = len(lo)
        hs = np.empty((Hn, len(DH))); cs = np.empty((Hn, len(DC)))
        for i, d in enumerate(DH):
            hs[:, i] = -100.0 if d is None else (lo if d == 'edge' else np.maximum(self.lo0 + d, lo))
        for i, d in enumerate(DC):
            cs[:, i] = 100.0 if d is None else (hi if d == 'edge' else np.minimum(self.hi0 + d, hi))
        hsp = hs[:, self.aih]; csp = cs[:, self.aic]
        return hsp, np.maximum(csp, hsp + 0.5)

    def values(self, k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G):
        card, ctx, H = self.card, self.ctx, self.H
        self.observe(k, T_zone, q_prev_W, elec_prev_kW)
        n = len(ctx['price'])
        idx = np.minimum(np.arange(k, k + H), n - 1)
        To = np.asarray(fc_T, float)[:H]; G = np.asarray(fc_G, float)[:H]
        base = self.base_W[np.minimum(idx, len(self.base_W) - 1)]
        self.prev = (float(To[0]), float(G[0]), float(base[0]) / 1000.0)
        A, C = self.maps(To, G, base / 1000.0)
        lo, hi = ctx['lo'][idx], ctx['hi'][idx]
        Tg = card['Tg']
        Qmax = np.interp(To, Tg, card['Qmax']); Qh = np.interp(To, Tg, card['Qh']); Ph = np.interp(To, Tg, card['Ph'])
        Qs = np.interp(To, Tg, card['Qs']); Pc = np.interp(To, Tg, card['Pc'])
        hsp, csp = self._sp(lo, hi)
        price, drate, frate = ctx['price'][idx], ctx['drate'][idx].astype(float), ctx['frate'][idx].astype(float)
        nwin = max(int(np.sum(drate > 0)), 1)
        drate = drate / (max(1.0, nwin / 3.0) * np.maximum(ctx['drem'][idx], 1))
        nfl = max(int(np.sum(frate > 0)), 1)
        frate = frate / (max(1.0, nfl / 3.0) * np.maximum(ctx['frem'][idx], 1))
        pref = np.array([self.peak.get((int(ctx['ym'][i]), int(ctx['dperiod'][i])), 0.0) for i in idx])
        fref = np.array([self.peak.get((int(ctx['ym'][i]), 'flat'), 0.0) for i in idx])
        occ = ctx['state'][idx] == 0
        fan_cont = bool(card['fan_cont'])
        tm0 = float(np.clip(self.tm, -5.0, 45.0))
        g0 = math.floor(min(tm0, lo.min()) - 3.0)
        grid = np.arange(g0, max(tm0, hi.max()) + 3.01, 0.25)
        NG, dg = len(grid), 0.25
        kind, gas_heat = self.kind, bool(card['gas_heat'])
        V = np.zeros(NG)
        first = None
        for j in range(H - 1, -1, -1):
            Tm = grid[:, None] if j > 0 else np.array([[tm0]])
            fc_ = fan_cont and bool(occ[j])
            gx = self.gx if fc_ else 0.0
            q0 = card['P_fan'] if fc_ else 0.0
            b = self.Aq
            Tb = (A[j] + self.At * Tm + b * (gx * To[j] + q0)) / (1 + b * gx)
            be = b / (1 + b * gx)
            hs, cs = hsp[j][None, :], csp[j][None, :]
            Qhd = np.minimum(np.where(Tb < hs, (hs - Tb) / be, 0.0), Qmax[j])
            Qcd = np.minimum(np.where((Tb > cs) & (Qs[j] > 0), (Tb - cs) / be, 0.0), Qs[j])
            T = Tb + be * (Qhd - Qcd)
            qtot = (gx * (To[j] - T) + q0 if fc_ else 0.0) + Qhd - Qcd
            Tm1 = C[j] + self.Ct * Tm + self.Cq * qtot
            gas = 0.0
            if kind == 'hp':
                f = np.where(Qh[j] > 0, np.minimum(Qhd, Qh[j]) / max(Qh[j], 1e-9), 0.0)
                Pel = part_power(Ph[j], f, self.hp_speed) + np.maximum(Qhd - Qh[j], 0.0)
                run_h = np.where(Qhd > Qh[j], 1.0, f) if Qh[j] > 0 else np.where(Qhd > 0, 1.0, 0.0)
            elif kind == 'furnace':
                run_h = Qhd / max(self.Qheat, 1e-9)
                gas = Qhd / self.eff
                Pel = 0.004 * self.Qheat * run_h
            elif kind == 'baseboard' and gas_heat:
                run_h = np.zeros_like(Qhd); gas = Qhd / self.eff; Pel = 0.01 * Qhd
            elif kind == 'baseboard':
                run_h = np.zeros_like(Qhd); Pel = Qhd
            else:
                run_h = Qhd / max(self.Qheat, 1e-9); Pel = Qhd
            fcl = np.where(Qs[j] > 0, Qcd / max(Qs[j], 1e-9), 0.0)
            Pel = Pel + part_power(Pc[j], fcl, self.cool_speed)
            Pel = Pel + (card['P_fan'] if fc_ else card['P_fan'] * np.clip(run_h + fcl, 0, 1))
            elec = (base[j] + Pel) / 1000.0
            dev = np.maximum(lo[j] - T, 0.0) + np.maximum(T - hi[j], 0.0)
            cost = price[j] * elec + self.gas_price * gas / 1000.0 + self.lam * dev
            if drate[j] > 0:
                cost = cost + drate[j] * np.maximum(elec - pref[j], 0.0)
            if frate[j] > 0:
                cost = cost + frate[j] * np.maximum(elec - fref[j], 0.0)
            pos = np.clip((Tm1 - g0) / dg, 0, NG - 1.000001)
            i0 = pos.astype(int)
            w = pos - i0
            Q = cost + V[i0] * (1 - w) + V[i0 + 1] * w
            if j > 0:
                V = Q.min(axis=1)
            else:
                first = Q[0]
        a = int(np.argmin(first))
        self.last = a
        return a, first

    def step(self, k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G):
        a, _ = self.values(k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G)
        ih, ic = ACTIONS[a]
        h, c = setpoints(self.ctx['lo'][k], self.ctx['hi'][k], self.lo0, self.hi0, ih, ic)
        return dict(heat_sp=h, cool_sp=c)


def shield_sp(ctx, k, h, c, shield):
    """The comfort shield of v1.0 (mode 4: look 6 h ahead; heating recovers ~1 K/h, cooling ~0.5 K/h)."""
    if not shield:
        return h, c
    lo, hi = float(ctx['lo'][k]), float(ctx['hi'][k])
    n = len(ctx['lo'])
    if shield == 2:
        k1 = min(k + 1, n - 1)
        lo, hi = max(lo, float(ctx['lo'][k1])), min(hi, float(ctx['hi'][k1]))
    elif shield >= 4:
        for j in range(1, 7):
            kj = min(k + j, n - 1)
            lo = max(lo, float(ctx['lo'][kj]) - SHIELD_RATE * (j - 1))
            hi = min(hi, float(ctx['hi'][kj]) + 0.5 * SHIELD_RATE * (j - 1))
    elif shield >= 3:
        for j in range(1, 5):
            kj = min(k + j, n - 1)
            lo = max(lo, float(ctx['lo'][kj]) - SHIELD_RATE * (j - 1))
            hi = min(hi, float(ctx['hi'][kj]) + SHIELD_RATE * (j - 1))
    h = max(h, lo); c = min(c, hi)
    if c < h + 0.5:
        c = h + 0.5
    return h, c


def price_scale(ctx, k, P_kW):
    """S: mean |price| over the next 24 h (>= 0.02 $/kWh) x the reference power; the unit of the action values."""
    n = len(ctx['price'])
    idx = np.minimum(np.arange(k, k + H), n - 1)
    return max(float(np.mean(np.abs(ctx['price'][idx]))), 0.02) * P_kW



SHIELD_MPC = 4                 # v1.1: the card MPC runs behind the v1.0 6 h comfort shield (mode 4)


class CardPilot:
    """ThermoPilot v1.1 (default, mode='mpc'): the card MPC behind the 6 h comfort shield. No learned parameters."""

    def __init__(self, card, ctx, lam, base_W, gas_price=GAS_RES, shield=SHIELD_MPC):
        self.ctx, self.shield = ctx, int(shield)
        self.mpc = CardMPC(card, ctx, lam, base_W, gas_price)
        self.lo0, self.hi0 = float(ctx['pref_lo']), float(ctx['pref_hi'])
        self.last = self.Q = None

    def step(self, k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G):
        a, self.Q = self.mpc.values(k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G)
        self.last = a
        ih, ic = ACTIONS[a]
        h, c = setpoints(self.ctx['lo'][k], self.ctx['hi'][k], self.lo0, self.hi0, ih, ic)
        h, c = shield_sp(self.ctx, k, h, c, self.shield)
        return dict(heat_sp=h, cool_sp=c)


def fit_map(T_out, GHI, base_W, A, At, C, Ct, Aq, Cq, T_mass_ref=20.0):
    """The card's house map from a house model's hourly one-step maps (T_air = A + At T_mass + Aq q,
    T_mass' = C + Ct T_mass + Cq q, e.g. BuildingSim's 5R1C zone over a month): At, Aq, Ct, Cq are the means; A and C
    (re-centred at T_mass_ref) are least-squares fits on [1, T_out, GHI, base load kW]."""
    X = np.stack([np.ones(len(T_out)), np.asarray(T_out, float), np.asarray(GHI, float),
                  np.asarray(base_W, float) / 1000.0], 1)
    out = dict(At=float(np.mean(At)), Aq=float(np.mean(Aq)), Ct=float(np.mean(Ct)), Cq=float(np.mean(Cq)))
    for key, y, yt in (('a', A, At), ('c', C, Ct)):
        yy = np.asarray(y, float) + (np.asarray(yt, float) - out['At' if key == 'a' else 'Ct']) * T_mass_ref
        out[key] = [float(v) for v in np.linalg.lstsq(X, yy, rcond=None)[0]]
    return out


def pilot(card, ctx, lam, base_W, gas_price=GAS_RES, mode='mpc', W=None):
    """The controller: mode='mpc' (v1.1 default: card MPC + shield; card needs 'map') or mode='network' (v1.0: the
    MLP in W, e.g. load('thermopilot_v1.npz'))."""
    if mode == 'mpc':
        return CardPilot(card, ctx, lam, base_W, gas_price)
    if mode == 'network':
        if W is None:
            raise ValueError("mode='network' needs the v1.0 weights W")
        return ThermoPilot(W, card, ctx, lam, base_W)
    raise ValueError(f'unknown mode {mode!r}')
