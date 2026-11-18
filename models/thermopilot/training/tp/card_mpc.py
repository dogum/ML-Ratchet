"""ThermoPilot rung R8 (17 Nov 2026, rejected): the residual pilot on the card MPC, numpy only.

The card MPC itself (CardMPC, shield_sp, price_scale) moved into thermopilot_core.py with v1.1 (rung R9, the card MPC
alone behind the shield); this module re-exports it and keeps R8's residual network wrapper.
"""
import math
import os
import sys

import numpy as np

_MODEL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _MODEL not in sys.path:
    sys.path.insert(0, _MODEL)
from thermopilot_core import (ACTIONS, DC, DH, GAS_RES, H, SHIELD_RATE, CardMPC, Featurizer,  # noqa: E402,F401
                              ThermoPilot, mlp_forward, part_power, price_scale, setpoints, shield_sp)




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
