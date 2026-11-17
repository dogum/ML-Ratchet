"""ThermoPilot inference, numpy only.

A learned thermostat: every hour it picks a heating and a cooling setpoint from a menu of 36 combinations, using
the indoor temperature history, its own past HVAC output, a 24 h weather forecast, the tariff and the occupant
comfort band, plus a 'building card' that describes the house. It was trained only in BuildingSim by imitating a
perfect-foresight dynamic-programming controller.

    import numpy as np, thermopilot_core as tp
    W = tp.load('thermopilot_v1.npz')
    ctx = tp.make_ctx(local_times, price, lo, hi, pref_lo, pref_hi, state)      # per-hour arrays for the period
    pilot = tp.ThermoPilot(W, card, ctx, lam=3.0, base_W=np.zeros(len(price)))
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
