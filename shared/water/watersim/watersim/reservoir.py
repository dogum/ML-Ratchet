"""Reservoirs: storage-area-elevation curves, operating policies (hook), spill, evaporation and hydropower.

Mass balance per day (hm3):  S' = S + I dt - E A - R dt - Spill dt,   dt = 0.0864 hm3 per (m3/s day)
  R      release through outlets/turbines = clip(policy(obs), min_release, max_release), never below dead storage
  Spill  free overflow above capacity ('free'), or an ogee weir Q = Cw Lw h^1.5 over a surcharge pool ('weir')
  E      open-water evaporation (mm/day; Penman 1948 from the forcing, or a factor times PET) over the area A(S)
Hydropower: P = eta rho g H Q_turbine, H = mean pool level - tailwater, Q_turbine = min(R, turbine_max).
Geometry: tabulated curves, or a power law V = C (h/H)^b (b = 2.5 default; A = dV/dh), the usual shape of
storage-depth curves (e.g. Liebe et al. 2005 use V ~ A^1.4, i.e. b ~ 3.5 for small reservoirs; larger valleys 2-3).

Policies (policy=...): 'sop' (standard operating policy: meet demand, store the rest, spill when full), 'rule_curve'
(seasonal target storage, Loucks & van Beek 2017 style), 'hanasaki' (Hanasaki et al. 2006 generic rule), 'linear'
(R = a + b S), 'run_of_river' (R = I), or any callable obs -> release (m3/s). obs keys: t, date, doy, storage_hm3,
frac_full, level_m, area_km2, inflow_m3s (today's), inflow_mean_m3s, demand_m3s, capacity_hm3, dead_hm3,
min_release_m3s, max_release_m3s, history (dict of lists so far).
"""
import math

import numpy as np

from .constants import G, RHO_W

DT_HM3 = 0.0864  # hm3 per (m3/s x 1 day)


class Reservoir:
    def __init__(self, capacity_hm3, depth_m=None, area_km2=None, curve=None, shape_b=2.5, dead_frac=0.05,
                 init_frac=0.7, upstream_frac=1.0, policy='sop', demand_m3s=None, min_release_m3s=0.0,
                 max_release_m3s=None, spill='free', weir_length_m=None, weir_coef=2.1, turbine_max_m3s=None,
                 efficiency=0.88, base_elev_m=0.0, tailwater_m=None, evap='penman', evap_factor=1.1,
                 policy_params=None, name='res', consumptive_frac=1.0):
        self.name = name
        self.C = float(capacity_hm3)
        self.b = float(shape_b)
        if curve is not None:
            self.curve = {k: np.asarray(v, float) for k, v in curve.items()}
            self.H = float(self.curve['elev_m'][-1] - self.curve['elev_m'][0])
            base_elev_m = float(self.curve['elev_m'][0])
        else:
            if depth_m is None and area_km2 is not None:
                depth_m = self.b * self.C * 1e6 / (area_km2 * 1e6)
            if depth_m is None:
                depth_m = 10.0 * self.C ** 0.25 + 5.0
            self.H = float(depth_m)
            self.curve = None
        self.base = float(base_elev_m)
        self.tail = self.base if tailwater_m is None else float(tailwater_m)
        self.dead = dead_frac * self.C
        self.init_frac = init_frac
        self.upstream_frac = float(upstream_frac)
        self.policy = policy
        self.pp = dict(policy_params or {})
        self.demand = demand_m3s
        self.rmin = float(min_release_m3s)
        self.rmax = max_release_m3s
        self.spill_mode = spill
        self.weir_L = weir_length_m
        self.weir_c = weir_coef
        self.turb = turbine_max_m3s
        self.eta = efficiency
        self.evap_mode = evap
        self.evap_factor = evap_factor
        self.consumptive_frac = consumptive_frac  # share of withdrawals (demand not returned to the river)
        self.S = self.init_frac * self.C

    # ----------------------------------------------------------------- geometry
    def level(self, S):
        """Pool elevation (m) for storage S (hm3)."""
        if self.curve is not None:
            return float(np.interp(S, self.curve['storage_hm3'], self.curve['elev_m']))
        h = self.H * (max(S, 0.0) / self.C) ** (1.0 / self.b)
        return self.base + h

    def area(self, S):
        """Surface area (km2) for storage S (hm3)."""
        if self.curve is not None:
            return float(np.interp(S, self.curve['storage_hm3'], self.curve['area_km2']))
        h = self.H * (max(S, 0.0) / self.C) ** (1.0 / self.b)
        # A = dV/dh = b C h^(b-1) / H^b  (hm3/m -> km2: 1 hm3/m = 1 km2)
        return self.b * self.C * h ** (self.b - 1) / self.H ** self.b if h > 0 else 0.0

    def storage_from_level(self, z):
        if self.curve is not None:
            return float(np.interp(z, self.curve['elev_m'], self.curve['storage_hm3']))
        h = max(z - self.base, 0.0)
        return self.C * (h / self.H) ** self.b

    def table(self, n=21):
        S = np.linspace(0, self.C, n)
        return {'storage_hm3': S, 'elev_m': np.array([self.level(s) for s in S]),
                'area_km2': np.array([self.area(s) for s in S])}

    # ----------------------------------------------------------------- policies
    def _policy_release(self, obs):
        p = self.policy
        if callable(p):
            return float(p(obs))
        kind = p.get('type') if isinstance(p, dict) else p
        prm = {**self.pp, **(p if isinstance(p, dict) else {})}
        I = obs['inflow_m3s']
        D = obs['demand_m3s']
        S = obs['storage_hm3']
        if kind == 'run_of_river':
            return I
        if kind == 'sop':
            avail = (S - self.dead) / DT_HM3 + I
            return min(D, max(avail, 0.0))
        if kind == 'linear':
            return prm.get('a', 0.0) + prm.get('b', 0.01) * S / DT_HM3
        if kind == 'rule_curve':
            lo, hi = prm.get('s_lo', 0.5), prm.get('s_hi', 0.9)
            dfull = prm.get('doy_full', 150)        # day of year the target is highest
            tgt = lo + (hi - lo) * 0.5 * (1 + math.cos(2 * math.pi * (obs['doy'] - dfull) / 365.25))
            tadj = prm.get('t_adjust_days', 30.0)
            return max(D + (S - tgt * self.C) / (tadj * DT_HM3), 0.0)
        if kind == 'hanasaki':
            # Hanasaki et al. (2006): R' = mean inflow (adjusted for demand), k = S at start of year / (0.85 C)
            imean = obs['inflow_mean_m3s']
            dmean = prm.get('demand_mean_m3s', D)
            if dmean >= 0.5 * imean:
                rp = imean / 2 * (1 + D / dmean) if dmean > 0 else imean
            else:
                rp = imean + D - dmean
            k = obs['storage_year_start_hm3'] / (0.85 * self.C)
            c = self.C / max(imean * 365.25 * DT_HM3, 1e-9)
            if c >= 0.5:
                return k * rp
            w = (c / 0.5) ** 2
            return w * k * rp + (1 - w) * I
        raise ValueError(f'unknown policy {p!r}')

    # ----------------------------------------------------------------- stepping
    def reset(self, init_frac=None, storage_hm3=None):
        if storage_hm3 is not None:
            self.S = float(storage_hm3)
        else:
            self.S = (self.init_frac if init_frac is None else init_frac) * self.C
        return self.S

    def step(self, inflow_m3s, release_m3s, evap_mm):
        """Advance one day with a requested release. Returns a dict of the day's fluxes and the new state."""
        S0 = self.S
        A0 = self.area(S0)
        I = max(float(inflow_m3s), 0.0)
        e_hm3 = max(evap_mm, 0.0) * A0 * 1e-3
        avail = S0 + I * DT_HM3
        e_hm3 = min(e_hm3, max(avail, 0.0))
        avail -= e_hm3
        R = float(release_m3s)
        R = max(R, self.rmin)
        if self.rmax is not None:
            R = min(R, self.rmax)
        R = max(min(R, (avail - self.dead) / DT_HM3), 0.0) if avail > self.dead else 0.0
        S1 = avail - R * DT_HM3
        spill = 0.0
        if self.spill_mode == 'weir' and self.weir_L:
            # ogee weir over a surcharge pool (vertical walls at the full-pool area), net inflow spread over 24 substeps
            nsub = 24
            afull = max(self.area(self.C), 1e-6)
            net = S1 - S0
            S1 = S0
            for _ in range(nsub):
                S1 += net / nsub
                if S1 > self.C:
                    hz = (S1 - self.C) / afull            # hm3 / km2 = m
                    q = self.weir_c * self.weir_L * hz ** 1.5
                    dv = min(q * DT_HM3 / nsub, S1 - self.C)
                    spill += dv / DT_HM3
                    S1 -= dv
        elif S1 > self.C:
            spill = (S1 - self.C) / DT_HM3
            S1 = self.C
        self.S = S1
        head = 0.5 * (self.level(S0) + self.level(S1)) - self.tail
        qt = R if self.turb is None else min(R, self.turb)
        power = self.eta * RHO_W * G * max(head, 0.0) * qt / 1e6  # MW
        return {'storage_hm3': S1, 'release_m3s': R, 'spill_m3s': spill, 'evap_hm3': e_hm3,
                'evap_mm_res': evap_mm, 'area_km2': self.area(S1), 'level_m': self.level(S1),
                'head_m': head, 'power_MW': power, 'energy_MWh': power * 24.0, 'inflow_m3s': I}

    def run(self, inflow_m3s, dates, evap_mm, demand_m3s=None, policy=None):
        """Operate over a series. inflow (m3/s), dates (DatetimeIndex), evap_mm (open-water mm/day).
        Returns a dict of arrays plus 'withdrawal_m3s' (the part of the release that serves demand)."""
        if policy is not None:
            self.policy = policy
        I = np.asarray(inflow_m3s, float)
        n = len(I)
        imean = float(np.mean(I)) if n else 0.0
        D = demand_m3s if demand_m3s is not None else self.demand
        if D is None:
            D = 0.6 * imean
        Darr = np.broadcast_to(np.asarray(D, float), (n,)).copy()
        keys = ['storage_hm3', 'release_m3s', 'spill_m3s', 'evap_hm3', 'area_km2', 'level_m', 'head_m',
                'power_MW', 'energy_MWh', 'inflow_m3s']
        out = {k: np.zeros(n) for k in keys}
        hist = {'inflow_m3s': [], 'release_m3s': [], 'storage_hm3': []}
        S_start = self.S
        s_year = self.S
        for t in range(n):
            d = dates[t]
            if t == 0 or (d.month == self.pp.get('year_start_month', 10) and d.day == 1):
                s_year = self.S
            obs = {'t': t, 'date': d, 'doy': d.dayofyear, 'storage_hm3': self.S, 'frac_full': self.S / self.C,
                   'level_m': self.level(self.S), 'area_km2': self.area(self.S), 'inflow_m3s': I[t],
                   'inflow_mean_m3s': imean, 'demand_m3s': Darr[t], 'capacity_hm3': self.C, 'dead_hm3': self.dead,
                   'min_release_m3s': self.rmin, 'max_release_m3s': self.rmax, 'storage_year_start_hm3': s_year,
                   'history': hist}
            r = self._policy_release(obs)
            st = self.step(I[t], r, float(evap_mm[t]))
            for k in keys:
                out[k][t] = st[k]
            hist['inflow_m3s'].append(I[t])
            hist['release_m3s'].append(st['release_m3s'])
            hist['storage_hm3'].append(st['storage_hm3'])
        out['withdrawal_m3s'] = np.minimum(out['release_m3s'], Darr) * self.consumptive_frac \
            if self.pp.get('withdraw', False) else np.zeros(n)
        out['storage_start_hm3'] = S_start
        out['demand_m3s'] = Darr
        return out


def from_spec(d, mean_inflow_m3s=None):
    """Build a Reservoir from a dict (keys = Reservoir arguments)."""
    if isinstance(d, Reservoir):
        return d
    d = dict(d)
    if 'capacity_hm3' not in d and 'capacity_years' in d and mean_inflow_m3s is not None:
        d['capacity_hm3'] = d.pop('capacity_years') * mean_inflow_m3s * 365.25 * DT_HM3
    d.pop('capacity_years', None)
    d.pop('inflow_from', None)
    return Reservoir(**d)
