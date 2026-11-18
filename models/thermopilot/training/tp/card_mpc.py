"""ThermoPilot rung R8 (17 Nov 2026, not released): the card MPC anchor and the residual pilot, numpy only.

Written as an extension of the released inference module (thermopilot_core.py, v1.0), which is unchanged: R8 was
rejected (it did not beat the card MPC alone on val), so this code stays with the training code. It imports the v1.0
action menu, features, shield rate and MLP from thermopilot_core.
"""
import math
import os
import sys

import numpy as np

_MODEL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _MODEL not in sys.path:
    sys.path.insert(0, _MODEL)
from thermopilot_core import (ACTIONS, DC, DH, H, SHIELD_RATE, Featurizer, ThermoPilot, mlp_forward,  # noqa: E402,F401
                              setpoints)


# ------------------------------------------------------------------------------------------- the card MPC anchor
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


class ResidualPilot:
    """R8: the card MPC's 36 action values Q plus a learned correction:
        logits = -(Q - min Q) / (S tau) + MLP([features, (Q - min Q) / S]);   action = argmax, then the shield.
    The MLP's last layer started at zero, so before training this is exactly the card MPC."""

    def __init__(self, W, card, ctx, lam, base_W, gas_price=GAS_RES):
        self.W, self.ctx = W, ctx
        self.shield = int(W['shield']) if 'shield' in W else 0
        self.tau = float(W['tau_a'])
        self.feat = Featurizer(card, ctx, lam, base_W)
        self.anchor = CardMPC(card, ctx, lam, base_W, gas_price)
        self.lo0, self.hi0 = float(ctx['pref_lo']), float(ctx['pref_hi'])
        self.last = self.last_anchor = None
        self.x = self.qa = None

    def step(self, k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G):
        x = self.feat.step(k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G)
        a0, Q = self.anchor.values(k, T_zone, q_prev_W, elec_prev_kW, fc_T, fc_G)
        qa = (Q - Q.min()) / price_scale(self.ctx, k, self.feat.P)
        self.x, self.qa, self.last_anchor = x, qa, a0
        logits = -qa / self.tau + mlp_forward(self.W, np.concatenate([x.astype(np.float64), qa]))
        a = int(np.argmax(logits))
        self.last = a
        ih, ic = ACTIONS[a]
        ctx = self.ctx
        h, c = setpoints(ctx['lo'][k], ctx['hi'][k], self.lo0, self.hi0, ih, ic)
        h, c = shield_sp(ctx, k, h, c, self.shield)
        return dict(heat_sp=h, cool_sp=c)


def pilot(W, card, ctx, lam, base_W, gas_price=GAS_RES):
    """The controller the weights describe: R8 residual (W has 'tau_a') or the v1.0 network."""
    if 'tau_a' in W:
        return ResidualPilot(W, card, ctx, lam, base_W, gas_price)
    return ThermoPilot(W, card, ctx, lam, base_W)
