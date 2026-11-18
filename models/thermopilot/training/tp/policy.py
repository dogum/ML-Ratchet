"""ThermoPilot policy: a small MLP (numpy inference) over tp.features, choosing one of 36 setpoint actions
(heating x cooling menu of tp.mpc.ACTIONS) every hour."""
import numpy as np
from .features import Featurizer
from . import mpc as M


def mlp_forward(W, x):
    h = (x - W['mu']) / W['sd']
    i = 0
    while f'W{i}' in W:
        h = h @ W[f'W{i}'] + W[f'b{i}']
        if f'W{i + 1}' in W:
            h = np.maximum(h, 0.0)
        i += 1
    return h


def load(path):
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in z.files}


SHIELD_RATE = 1.0     # K/h the shield assumes the HVAC can move the zone during recovery


class ThermoPilot:
    """Controller for an Episode: act(k, obs, info_prev) -> {'heat_sp', 'cool_sp'}."""

    def __init__(self, W, card, ctx, lam, base_W, forecaster, shield=None):
        self.W, self.ctx, self.fc = W, ctx, forecaster
        # comfort shield: 0 none; 1 the thermostat never floats outside the current band (an 'off' side becomes the
        # band edge); 2 = 1 + start recovery one hour before the band tightens. Stored with the weights.
        self.shield = int(W['shield']) if shield is None and 'shield' in W else int(shield or 0)
        self.feat = Featurizer(card, ctx, lam, base_W)
        self.lo0, self.hi0 = float(ctx['pref_lo']), float(ctx['pref_hi'])
        self.last = None
        self.x = None

    def features(self, k, obs, info_prev):
        q = info_prev['q_hvac_kW'] * 1000.0 if info_prev is not None else 0.0
        e = info_prev['electricity_kW'] if info_prev is not None else 0.0
        To, G = self.fc.get(k)
        return self.feat.step(k, obs['T_zone'], q, e, To, G)

    def act(self, k, obs, info_prev, x=None):
        self.x = self.features(k, obs, info_prev) if x is None else x
        logits = mlp_forward(self.W, self.x.astype(np.float64))
        a = int(np.argmax(logits))
        self.last = a
        ih, ic = M.ACTIONS[a]
        h, c = M.setpoints(self.ctx['lo'][k], self.ctx['hi'][k], self.lo0, self.hi0, ih, ic)
        if self.shield:
            lo, hi = float(self.ctx['lo'][k]), float(self.ctx['hi'][k])
            if self.shield == 2:
                k1 = min(k + 1, len(self.ctx['lo']) - 1)
                lo, hi = max(lo, float(self.ctx['lo'][k1])), min(hi, float(self.ctx['hi'][k1]))
            elif self.shield >= 4:
                # heating can recover ~1 K/h, cooling ~0.5 K/h (conservative); look 6 h ahead
                n_ = len(self.ctx['lo']) if hasattr(self, 'ctx') else n
                for j in range(1, 7):
                    kj = min(k + j, n_ - 1)
                    lo = max(lo, float(self.ctx['lo'][kj]) - SHIELD_RATE * (j - 1))
                    hi = min(hi, float(self.ctx['hi'][kj]) + 0.5 * SHIELD_RATE * (j - 1))
            elif self.shield >= 3:
                # anticipate a tightening band: be within RATE K per hour of every band edge in the next 4 h
                n = len(self.ctx['lo'])
                for j in range(1, 5):
                    kj = min(k + j, n - 1)
                    lo = max(lo, float(self.ctx['lo'][kj]) - SHIELD_RATE * (j - 1))
                    hi = min(hi, float(self.ctx['hi'][kj]) + SHIELD_RATE * (j - 1))
            h = max(h, lo)
            c = min(c, hi)
            if c < h + 0.5:
                c = h + 0.5
        return dict(heat_sp=h, cool_sp=c)


def action_sp(ctx, k, a):
    ih, ic = M.ACTIONS[a]
    return M.setpoints(ctx['lo'][k], ctx['hi'][k], float(ctx['pref_lo']), float(ctx['pref_hi']), ih, ic)


# ------------------------------------------------------------------------------------------- R8: card MPC + residual
from . import card_mpc as core   # noqa: E402  (numpy only; extends thermopilot_core, which stays v1.0)


class CoreAdapter:
    """Runs a tp.card_mpc controller (CardMPC or ResidualPilot) as an Episode strategy act(k, obs, info_prev)."""

    def __init__(self, ctrl, forecaster):
        self.ctrl, self.fc = ctrl, forecaster
        self.last = None

    def act(self, k, obs, info_prev):
        q = info_prev['q_hvac_kW'] * 1000.0 if info_prev is not None else 0.0
        e = info_prev['electricity_kW'] if info_prev is not None else 0.0
        To, G = self.fc.get(k)
        sp = self.ctrl.step(k, obs['T_zone'], q, e, To, G)
        self.last = self.ctrl.last
        return sp


def card_mpc(ep, pref_hi, shield=0):
    """The anchor alone on an Episode (nominal card, noisy forecast), optionally behind the v1.0 shield."""
    from .features import full_card
    card = full_card(ep.model, ep.ctx, pref_hi)
    m = core.CardMPC(card, ep.ctx, ep.lam, ep.model.base_W, ep.gas_price)
    if shield:
        inner = m.step
        def step(k, *a):
            sp = inner(k, *a)
            h, c = core.shield_sp(ep.ctx, k, sp['heat_sp'], sp['cool_sp'], shield)
            return dict(heat_sp=h, cool_sp=c)
        m.step = step
    return CoreAdapter(m, ep.fc_noisy)


def residual(ep, W, pref_hi):
    from .features import full_card
    card = full_card(ep.model, ep.ctx, pref_hi)
    return CoreAdapter(core.ResidualPilot(W, card, ep.ctx, ep.lam, ep.model.base_W, ep.gas_price), ep.fc_noisy)
