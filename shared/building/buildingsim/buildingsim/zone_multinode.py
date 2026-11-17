"""Multi-node zone (zone_model='multinode'): an RC network with explicit wall, roof and floor layers.

Nodes: zone air (air + furniture capacity), an internal-mass node (partitions/furniture), and for each opaque
construction (walls lumped over orientations with their area-weighted sol-air temperature, roof, ground floor) a
ladder of nodes through its layers. Interior surfaces exchange with the air through a combined film
(h_i = 7.7 W/m2K walls, 9.3 roof... we use 1/R_si per ISO 6946: walls 0.13, roof 0.10, floor 0.17 m2K/W) and with
each other through the air node (no separate radiant star). Exterior surfaces see the sol-air / attic / ground
boundary through R_se = 0.04 m2K/W (floors: none, the ground boundary is the buffer-space temperature of zone.py).
Windows are massless (U A between outdoor and air). Solar through windows lands on the floor (60 %) and the internal
mass (40 %); internal gains are half convective (air), half radiant (interior surfaces by area). HVAC heat can be
injected at the air, the interior surfaces or the floor (radiant).

Layers: spec['geometry']['constructions'] = {'wall'|'roof'|'floor': [(thickness m, k W/mK, rho kg/m3, cp J/kgK),
... inside -> outside]}; each layer is split into nodes so that a node is <= ~3 cm of masonry or one node per light
layer. Without explicit layers, default stacks by mass class reproduce the spec U-value exactly (insulation thickness
is solved for) with an interior lining whose capacity follows the class.
Time scheme: implicit Euler (L-stable, 1st order); the air-exchange conductance H_ve enters only the air-node
diagonal, so A(H_ve) = A0 + H_ve e0 e0^T is solved by a precomputed inverse and Sherman-Morrison.
"""
import math

import numpy as np

R_SI = {'wall': 0.13, 'roof': 0.10, 'floor': 0.17}
R_SE = 0.04

# default interior linings (inside the insulation) by mass class: list of layers (t, k, rho, cp)
_LINING = {'very_light': [(0.0125, 0.16, 950, 840)],
           'light': [(0.0125, 0.16, 950, 840), (0.02, 0.14, 530, 900)],
           'medium': [(0.0125, 0.16, 950, 840), (0.10, 0.51, 1400, 1000)],
           'heavy': [(0.015, 0.7, 1400, 1000), (0.15, 1.13, 2000, 1000)],
           'very_heavy': [(0.015, 0.7, 1400, 1000), (0.25, 1.4, 2200, 1000)]}
_FLOOR = {'very_light': [(0.02, 0.14, 650, 1200)], 'light': [(0.025, 0.14, 650, 1200)],
          'medium': [(0.08, 1.13, 1400, 1000)], 'heavy': [(0.10, 1.4, 2000, 1000)],
          'very_heavy': [(0.15, 1.4, 2200, 1000)]}
_INT_MASS_KJ = {'very_light': 10.0, 'light': 20.0, 'medium': 40.0, 'heavy': 60.0, 'very_heavy': 80.0}


def default_stack(kind, U, mass):
    """Layers inside -> outside giving overall U (W/m2K) including R_si and R_se (none for floors)."""
    lin = list((_FLOOR if kind == 'floor' else _LINING)[mass])
    R_lin = sum(t / k for t, k, _, _ in lin)
    R_target = 1.0 / max(U, 1e-3) - R_SI[kind] - (R_SE if kind != 'floor' else 0.0)
    R_ins = max(R_target - R_lin - (0.01 / 0.14 if kind != 'floor' else 0.0), 0.02)
    stack = lin + [(R_ins * 0.04, 0.04, 30, 1400)]
    if kind != 'floor':
        stack.append((0.01, 0.14, 530, 900))       # sheathing / cladding
    return stack


def _discretise(stack):
    """Split layers into nodes: returns (C per node J/m2K, R between consecutive nodes m2K/W, R from the inner
    surface to node 0, R from the last node to the outer surface)."""
    Cs, Rs = [], []          # node capacities and the resistance of the half-cells
    for t, k, rho, cp in stack:
        R = t / k
        C = t * rho * cp
        n = 1 if C < 2e4 else max(1, min(6, int(math.ceil(t / 0.03))))
        for _ in range(n):
            Cs.append(C / n)
            Rs.append(R / n)
    # node i sits at the centre of cell i: R between i and i+1 = (Rs[i] + Rs[i+1]) / 2
    Rb = [(Rs[i] + Rs[i + 1]) / 2 for i in range(len(Rs) - 1)]
    return np.array(Cs), np.array(Rb), Rs[0] / 2, Rs[-1] / 2


class ZoneMultiNode:
    kind = 'multinode'

    def __init__(self, spec, geo, bnd, n, dt_s, theta0=20.0):
        g = spec['geometry']
        mass = spec['thermal_mass'] if isinstance(spec['thermal_mass'], str) else 'medium'
        cons = g.get('constructions') or {}
        A_f = geo['A_f']
        self.dt = dt_s
        self.bnd = bnd
        self.k = 0
        # areas and boundary conductance weights
        A_wall = sum(geo['opaque'].values())
        A_roof = geo['roof']
        self.H_tr_w = float(spec['window_U']) * sum(geo['win'].values())
        stacks = []
        if A_wall > 0:
            stacks.append(('wall', A_wall, cons.get('wall') or default_stack('wall', float(spec['wall_U']), mass),
                           R_SI['wall'], R_SE))
        if A_roof > 0:
            stacks.append(('roof', A_roof, cons.get('roof') or default_stack('roof', float(spec['roof_U']), mass),
                           R_SI['roof'], R_SE))
        H_g = bnd['H_ground']
        if H_g > 0 and geo['ground'] > 0:
            U_eq = H_g / geo['ground']
            stacks.append(('floor', geo['ground'], cons.get('floor') or default_stack('floor', U_eq, mass),
                           R_SI['floor'], 0.0))
        nodes = 2
        self.stacks = []
        for kind, A, st, rsi, rse in stacks:
            C, Rb, r_in, r_out = _discretise(st)
            idx = list(range(nodes, nodes + len(C)))
            nodes += len(C)
            self.stacks.append(dict(kind=kind, A=A, C=C * A, G=(1.0 / Rb) * A if len(Rb) else np.zeros(0),
                                    G_in=A / (rsi + r_in), G_out=A / (rse + r_out), idx=idx))
        self.n = nodes
        C = np.zeros(nodes)
        C[0] = 1.2 * 1006.0 * geo['volume'] * 5.0          # air + light furniture
        im = g.get('internal_mass_kJ_m2K')
        C[1] = (_INT_MASS_KJ[mass] if im is None else float(im)) * 1000.0 * A_f + 1.0   # partitions / furniture
        G = np.zeros((nodes, nodes))
        A_int = 2.0 * A_f
        h_int = 3.0 * A_int                               # internal mass <-> air
        G[0, 1] = G[1, 0] = h_int
        for s_ in self.stacks:
            i0 = s_['idx'][0]
            G[0, i0] += s_['G_in']
            G[i0, 0] += s_['G_in']
            for j, gg in enumerate(s_['G']):
                a, b = s_['idx'][j], s_['idx'][j + 1]
                G[a, b] += gg
                G[b, a] += gg
            C[s_['idx']] = s_['C']
        self.Cv = C
        self.Gm = G
        # boundary conductances: windows to outdoor at the air node, each stack's last node to its boundary
        Gb = np.zeros(nodes)
        Gb[0] = self.H_tr_w
        for s_ in self.stacks:
            Gb[s_['idx'][-1]] += s_['G_out']
        self.Gb = Gb
        A0 = np.diag(C / dt_s + G.sum(axis=1) + Gb) - G
        self.Ainv = np.linalg.inv(A0)
        self.Ainv_col0 = self.Ainv[:, 0].copy()
        self.A00 = self.Ainv[0, 0]
        # injection/distribution vectors
        surf = [s_['idx'][0] for s_ in self.stacks]
        areas = np.array([s_['A'] for s_ in self.stacks] + [A_int])
        self.v_rad = np.zeros(nodes)
        for i, a in zip(surf + [1], areas):
            self.v_rad[i] += a / areas.sum()
        self.v_sol = np.zeros(nodes)
        fl = [s_ for s_ in self.stacks if s_['kind'] == 'floor']
        sf = float(g.get('solar_floor_frac', 0.6))
        if fl:
            self.v_sol[fl[0]['idx'][0]] = sf
            self.v_sol[1] = 1.0 - sf
        else:
            self.v_sol[1] = 1.0
        self.v_floor = np.zeros(nodes)
        if fl:
            self.v_floor[fl[0]['idx'][0]] = 1.0
        else:
            self.v_floor[1] = 1.0
        self.surf_idx = np.array(surf + [1])
        self.surf_w = areas / areas.sum()
        # boundary temperature arrays per stack
        self.Tb = {'wall': bnd['T_wall'], 'roof': bnd['T_roof'], 'floor': bnd['T_ground']}
        self.theta_m = np.full(nodes, float(theta0))
        self.C_m = float(C.sum())
        self.UA = self.H_tr_w + bnd['H_wall'] + bnd['H_roof'] + bnd['H_ground']
        self.H_tr_is = 1e9      # for code that queries it (no ISO interior coupling here)
        self.H_tr_em = 0.0
        self.H_op = bnd['H_wall'] + bnd['H_roof'] + bnd['H_ground']
        self.w_wall = self.w_roof = self.w_gnd = 0.0
        self.A_t = 4.5 * A_f
        self.tau_h = self.C_m / max(self.UA, 1.0) / 3600.0
        self.w = 1.0
        self._bvec = np.zeros(nodes)

    def time_constant_h(self, H_ve):
        return self.C_m / (self.UA + H_ve) / 3600.0

    def _rhs(self, x_prev, th_e, th_sup, H_ve, phi_int, phi_sol, qa, qs, qm):
        k = self.k
        b = self.Cv / self.dt * x_prev
        b[0] += self.H_tr_w * th_e + H_ve * th_sup + 0.5 * phi_int + qa
        for s_ in self.stacks:
            b[s_['idx'][-1]] += s_['G_out'] * self.Tb[s_['kind']][k]
        b += self.v_rad * (0.5 * phi_int + qs) + self.v_sol * phi_sol + self.v_floor * qm
        return b

    def solve(self, x_prev, th_e, th_em, H_ve, th_sup, phi_int, phi_sol, qa, qs, qm):
        b = self._rhs(x_prev, th_e, th_sup, H_ve, phi_int, phi_sol, qa, qs, qm)
        y = self.Ainv @ b
        x = y - (H_ve * y[0] / (1.0 + H_ve * self.A00)) * self.Ainv_col0
        ts = float(x[self.surf_idx] @ self.surf_w)
        return x, x, ts, float(x[0])

    def ledger_flows(self, x_prev, x, k, th_e):
        """Heat flows into the zone over the step [W] (consistent with the implicit scheme)."""
        out = dict(win=self.H_tr_w * (th_e - x[0]), wall=0.0, roof=0.0, ground=0.0)
        for s_ in self.stacks:
            f = s_['G_out'] * (self.Tb[s_['kind']][k] - x[s_['idx'][-1]])
            out['ground' if s_['kind'] == 'floor' else s_['kind']] += f
        out['storage'] = float(self.Cv @ (x - x_prev)) / self.dt
        return out
