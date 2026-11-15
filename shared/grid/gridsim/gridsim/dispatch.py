"""Unit commitment and economic dispatch (rolling daily horizon) with HiGHS.

Model per horizon (T hours, clusters c of identical units, K heat-rate segments):

  min  sum_ct [ noload_c u_ct + sum_k mc_ckt seg_ckt + start_c su_ct ] + sum offers*use + VOLL*shed + ORDC blocks
  s.t. balance_t:   sum_c (pmin_c u_ct + sum_k seg_ckt) + use_t + hydro_t + dis_t - ch_t + shed_t - dump_t = D_t - fixed_t
       reserve_t:   sum_c r_ct + r_storage_t + r_hydro_t + short_t >= R_t
       seg_ckt <= w_ck u_ct,   pmin_c u_ct + sum_k seg_ckt + r_ct <= pmax_c u_ct,   r_ct <= 10 min ramp * u_ct
       u_ct - u_c,t-1 = su_ct - sd_ct;  min up: sum_{last UT h} su <= u;  min down: sum_{last DT h} sd <= N - u
       ramping (clustered, Morales-Espana style start/shut-down capability), storage SOC balance, hydro energy budget
       optional DC network: -F_l <= sum_b PTDF_lb inj_bt <= F_l (lazy constraints)

u is integer (0..units available) in 'milp', continuous in 'lp'.  Prices are the duals of the balance rows: from the
LP itself ('lp'), or from the LP with the MILP commitment fixed ('milp', the usual market practice).  'merit' is an
hourly merit-order stack with no inter-temporal constraints (fast baseline).
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .fleet import (prepare_fleet, merge_combined_cycles, cluster_fleet, fuel_curve, fuel_prices_daily, THERMAL,
                    EIA930_FUEL, STORAGE)

try:
    import highspy
    _HAVE_HIGHSPY = True
except Exception:  # noqa
    _HAVE_HIGHSPY = False

# planned-outage seasonal weights (mean 1): maintenance in spring and autumn, none in summer peak (ERCOT/US practice)
PLANNED_SEASON = np.array([1.2, 1.4, 2.0, 2.1, 1.3, 0.3, 0.0, 0.0, 0.4, 1.4, 1.6, 0.3])
PLANNED_SEASON = PLANNED_SEASON / PLANNED_SEASON.mean()


# ---------------------------------------------------------------------- compiled fleet
class CompiledFleet:
    """Arrays for the optimiser.  Thermal clusters, variable renewables, hydro, storage."""

    def __init__(self, units: pd.DataFrame, n_seg=3, n_clusters=None, cluster=True):
        self.units = units
        th = units[units.tech.isin(THERMAL)].copy()
        if cluster:
            th = cluster_fleet(th, n_clusters)
        else:
            th["cluster"] = np.arange(len(th))
        self.thermal_units = th
        cl = []
        for cid, g in th.groupby("cluster"):
            n = len(g)
            pmax = g.pmax_MW.mean()
            pmin = g.pmin_MW.mean()
            # cluster heat-rate curve = mean of member curves on a normalised load grid
            fr = np.linspace(pmin / pmax if pmax > 0 else 0, 1, 4)
            pts = []
            for f in fr:
                hr = []
                for c, pm in zip(g.heat_rate_curve, g.pmax_MW):
                    if c is None:
                        continue
                    P = np.array([a for a, _ in c]); H = np.array([b for _, b in c])
                    hr.append(np.interp(f * pm, P, H))
                pts.append([f * pmax, float(np.mean(hr)) if hr else 0.0])
            fc = fuel_curve(pts if any(p[1] > 0 for p in pts) else None, pmax, pmin, n_seg)
            cl.append(dict(cluster=cid, tech=g.tech.iloc[0], n=n, ids=list(g.id), pmax=pmax, pmin=pmin,
                           ru=60 * g.ramp_up_MW_per_min.mean(), rd=60 * g.ramp_down_MW_per_min.mean(),
                           r10=10 * g.ramp_up_MW_per_min.mean(),
                           ut=int(round(g.min_up_h.max())), dt=int(round(g.min_down_h.max())),
                           start=g["start_cost_$"].mean(), vom=g["vom_$_per_MWh"].mean(),
                           co2=float((g.co2_kg_per_MMBtu * g.pmax_MW).sum() / g.pmax_MW.sum()),
                           H=float((g.inertia_H_s * g.mva_base).sum() / g.mva_base.sum()), mva=g.mva_base.mean(),
                           efor=g.efor.mean(), planned=g.planned_frac.mean(),
                           online=g.online.values, retired=g.retired.values, fuel=g.fuel.iloc[0],
                           F0=fc["F0"], Fmin=fc["Fmin"], widths=fc["widths"], ihr=fc["ihr"],
                           members=g.index.values))
        self.clusters = cl
        self.C = len(cl)
        self.K = n_seg
        for key in ["n", "pmax", "pmin", "ru", "rd", "r10", "ut", "dt", "start", "vom", "co2", "H", "mva", "efor",
                    "planned", "F0", "Fmin"]:
            setattr(self, key, np.array([c[key] for c in cl], float))
        self.widths = np.array([c["widths"] for c in cl]).reshape(self.C, n_seg)
        self.ihr = np.array([c["ihr"] for c in cl]).reshape(self.C, n_seg)
        self.tech = np.array([c["tech"] for c in cl])
        self.must_run = self.tech == "nuclear"
        # renewables / hydro / storage aggregates (time-varying capacity from online dates)
        self.wind = units[units.tech == "wind"]
        self.solar = units[units.tech == "solar_pv"]
        self.hydro = units[units.tech.isin(["hydro_res", "hydro_ror"])]
        self.storage = units[units.tech.isin(STORAGE)]

    def online_count(self, times):
        """Units of each cluster online (by online/retired dates) at each time -> (T, C)."""
        tn = pd.DatetimeIndex(times).tz_localize(None).values
        out = np.zeros((len(tn), self.C))
        for j, c in enumerate(self.clusters):
            on = c["online"]; rt = c["retired"]
            ok = (pd.isna(on)[None, :] | (on[None, :] <= tn[:, None])) & \
                 (pd.isna(rt)[None, :] | (rt[None, :] > tn[:, None]))
            out[:, j] = ok.sum(1)
        return out

    def capacity_of(self, df, times):
        if len(df) == 0:
            return np.zeros(len(times))
        tn = pd.DatetimeIndex(times).tz_localize(None).values
        on = df.online.values; rt = df.retired.values
        ok = (pd.isna(on)[None, :] | (on[None, :] <= tn[:, None])) & (pd.isna(rt)[None, :] | (rt[None, :] > tn[:, None]))
        return ok @ df.pmax_MW.values


def availability(cf: CompiledFleet, times, mode="expected", seed=0, mttr_h=60.0):
    """Available units (T, C) and derate factor (T, C) for each cluster.

    expected: all units online, capacity derated by (1 - EFOR - planned share x seasonal weight).
    random: two-state Markov forced outages (EFOR, MTTR) + planned outages placed in low-season weeks, per unit.
    """
    times = pd.DatetimeIndex(times)
    n_on = cf.online_count(times)
    month = times.tz_localize(None).month.values - 1 if times.tz is not None else times.month.values - 1
    if mode == "expected":
        der = 1.0 - cf.efor[None, :] - cf.planned[None, :] * PLANNED_SEASON[month][:, None]
        return n_on, np.clip(der, 0.2, 1.0)
    rng = np.random.default_rng(seed)
    T = len(times)
    avail = np.zeros((T, cf.C))
    for j, c in enumerate(cf.clusters):
        n = c["n"]
        # forced outages: Markov chain per unit, hourly
        lam_rep = 1.0 / mttr_h
        lam_fail = lam_rep * c["efor"] / max(1 - c["efor"], 1e-6)
        up = rng.random(n) > c["efor"]
        state = np.zeros((T, n), bool)
        u = rng.random((T, n))
        for t in range(T):
            up = np.where(up, u[t] > lam_fail, u[t] < lam_rep)
            state[t] = up
        # planned: each unit gets one outage of planned_frac x 8760 h a year, placed with the seasonal weights
        dur = int(round(c["planned"] * 8760 * T / 8760))
        if dur > 0:
            w = PLANNED_SEASON[month] + 1e-3
            p = w / w.sum()
            for i in range(n):
                s0 = rng.choice(T, p=p)
                state[s0:s0 + dur, i] = False
        avail[:, j] = state.sum(1)
    return np.minimum(avail, n_on), np.ones((T, cf.C))


# ---------------------------------------------------------------------- LP/MILP plumbing
class _Model:
    def __init__(self):
        self.n = 0
        self.lb, self.ub, self.c, self.integ = [], [], [], []
        self.rows_i, self.rows_j, self.rows_v = [], [], []
        self.rlo, self.rhi = [], []
        self.nr = 0

    def var(self, shape, lb=0.0, ub=np.inf, cost=0.0, integ=False):
        size = int(np.prod(shape))
        idx = np.arange(self.n, self.n + size).reshape(shape)
        self.n += size
        self.lb.append(np.broadcast_to(np.asarray(lb, float), shape).ravel())
        self.ub.append(np.broadcast_to(np.asarray(ub, float), shape).ravel())
        self.c.append(np.broadcast_to(np.asarray(cost, float), shape).ravel())
        self.integ.append(np.broadcast_to(np.asarray(integ, int), shape).ravel())
        return idx

    def rows(self, nrows, lo, hi):
        idx = np.arange(self.nr, self.nr + nrows)
        self.nr += nrows
        self.rlo.append(np.broadcast_to(np.asarray(lo, float), (nrows,)).ravel())
        self.rhi.append(np.broadcast_to(np.asarray(hi, float), (nrows,)).ravel())
        return idx

    def coef(self, r, j, v):
        r, j, v = np.broadcast_arrays(np.asarray(r), np.asarray(j), np.asarray(v, float))
        self.rows_i.append(r.ravel()); self.rows_j.append(j.ravel()); self.rows_v.append(v.ravel())

    def build(self):
        A = sp.csc_matrix((np.concatenate(self.rows_v), (np.concatenate(self.rows_i), np.concatenate(self.rows_j))),
                          shape=(self.nr, self.n))
        return (A, np.concatenate(self.c), np.concatenate(self.lb), np.concatenate(self.ub),
                np.concatenate(self.integ), np.concatenate(self.rlo), np.concatenate(self.rhi))


def _solve(A, c, lb, ub, integ, rlo, rhi, mip=True, mip_gap=0.005, time_limit=60.0, duals=True, fix_idx=None,
           pricing="fixed"):
    """Solve with highspy (MIP, then LP with integers fixed for duals) or scipy fallback.  Returns x, row duals, info."""
    info = {}
    if _HAVE_HIGHSPY:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lb, ub
        lp.row_lower_, lp.row_upper_ = rlo, rhi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
        if mip and integ.any():
            lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integ]
            h.setOptionValue("mip_rel_gap", mip_gap)
            h.setOptionValue("time_limit", float(time_limit))
        h.passModel(lp)
        h.run()
        st = h.modelStatusToString(h.getModelStatus())
        sol = h.getSolution()
        x = np.array(sol.col_value)
        info["status"] = st
        info["obj"] = h.getInfo().objective_function_value
        if mip and integ.any():
            info["mip_gap"] = h.getInfo().mip_gap
        y = None
        if duals and len(x) == A.shape[1]:
            if mip and integ.any():
                # fix integer columns (and their transition partners) and re-solve as LP
                lb2, ub2 = lb.copy(), ub.copy()
                if pricing == "fixed":
                    fix = np.where(integ > 0)[0] if fix_idx is None else fix_idx
                    lb2[fix] = np.round(x[fix]); ub2[fix] = np.round(x[fix])
                h2 = highspy.Highs(); h2.setOptionValue("output_flag", False); h2.setOptionValue("threads", 1)
                lp.integrality_ = []
                lp.col_lower_, lp.col_upper_ = lb2, ub2
                h2.passModel(lp); h2.run()
                s2 = h2.getSolution()
                y = np.array(s2.row_dual)
                info["lp_fixed_status"] = h2.modelStatusToString(h2.getModelStatus())
            else:
                y = np.array(sol.row_dual)
        return x, y, info
    # scipy fallback
    from scipy.optimize import milp, linprog, LinearConstraint, Bounds
    if mip and integ.any():
        r = milp(c, constraints=LinearConstraint(A, rlo, rhi), integrality=integ, bounds=Bounds(lb, ub),
                 options=dict(mip_rel_gap=mip_gap, time_limit=time_limit))
        x = r.x
        info["status"] = r.message
        info["obj"] = r.fun
        lb2, ub2 = lb.copy(), ub.copy()
        fix = np.where(integ > 0)[0] if fix_idx is None else fix_idx
        lb2[fix] = np.round(x[fix]); ub2[fix] = np.round(x[fix])
    else:
        lb2, ub2 = lb, ub
        x = None
    if not duals and x is not None:
        return x, None, info
    eq = rlo == rhi
    A = A.tocsr()
    up = (~eq) & np.isfinite(rhi)
    lo_ = (~eq) & np.isfinite(rlo)
    A_ub = sp.vstack([A[up], -A[lo_]])
    b_ub = np.r_[rhi[up], -rlo[lo_]]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A[eq], b_eq=rlo[eq], bounds=list(zip(lb2, ub2)), method="highs")
    if x is None:
        x = r.x; info["status"] = r.message; info["obj"] = r.fun
    y = np.zeros(A.shape[0])
    y[eq] = r.eqlin.marginals
    yy = np.zeros(A.shape[0]); yy[np.where(up)[0]] = r.ineqlin.marginals[:up.sum()]
    yy[np.where(lo_)[0]] -= r.ineqlin.marginals[up.sum():]
    y = y + yy
    return x, y, info


# ---------------------------------------------------------------------- main entry
def dispatch(fleet, demand_MW, wind_cf=None, solar_cf=None, prices=None, network=None, reserves="auto",
             horizon="day", method="milp", imports_MW=None, wind_MW=None, solar_MW=None, fixed_MW=None,
             hydro_MW=None, outages="expected", seed=0, lookahead_h=12, offers=None, voll=9000.0,
             region=None, n_clusters=None, cluster=True, mip_gap=0.005, time_limit=60.0, storage_end="cyclic",
             verbose=False, ordc=None, start_state=None, must_run=("nuclear",), hr_source="eia923",
             offer_adders=None, pricing="relaxed", soft_history=False):
    """Hourly unit commitment / economic dispatch.

    fleet: DataFrame or list of contract generator specs (or a path); filtered to units online in the period.
    demand_MW: hourly pandas Series (UTC index).  wind_cf / solar_cf: capacity-factor Series (x online capacity) or
    wind_MW / solar_MW: available MW Series.  imports_MW: net imports Series (fixed injection, + = into the region).
    fixed_MW: dict tech -> Series of exogenous output (e.g. observed nuclear); those units are removed from the
    optimisation.  hydro_MW: Series of hydro energy available per hour (used as a daily energy budget).
    prices: fuel prices: None/'eia923' (plant/state monthly EIA-923 delivered cost + daily Henry Hub moves),
    'constant' (spec values), dict fuel-code -> $/MMBtu (scalar or daily Series), or a DataFrame (days x unit row).
    network: optional dict(net=Network, gen_bus={unit id: internal bus}, load_share=array per bus, optional
    wind_share/solar_share/hydro_share/storage_share arrays, rate_scale): DC line limits on every hour (PTDF, lazy
    constraints), unit-level commitment, nodal prices in res.lmp (hours x buses) and flows in res.line_flow_MW.
    `fleet_from_case(ppc)` builds a fleet + network dict from a MATPOWER case.
    reserves: 'auto' (max(largest online unit, 6 % of demand)), a number, or a Series [MW of 10-min up reserve].
    method: 'milp' | 'lp' | 'merit'.  horizon: 'day' (24 h, + lookahead_h) or hours.
    outages: 'expected' (seasonal derates), 'random' (Markov forced + planned outages, `seed`), or 'none'.
    offers: dict tech -> $/MWh offer for wind/solar/hydro (default wind -10, solar -5, hydro 0).
    soft_history: False (default, as in v1.0/v1.1). True: minimum up/down rows and the first-hour ramp rows from the
        previous day get a penalised slack (10 x VOLL x the cluster's unit size per unit-hour, 10 x VOLL per MW, so a
        slack is never cheaper than shedding load and is used only when the day is otherwise infeasible), so a day whose
        carried history or a
        random outage starting while a unit is on would be infeasible is solved with the smallest violation instead
        of being re-solved without history (HANDOFF P0.6; `res.history_slack` reports what was used).
    pricing: 'relaxed' (default; duals of the LP relaxation, an approximation of convex-hull pricing that carries
    commitment costs into prices) or 'fixed' (duals of the LP with the MILP commitment fixed).  On 4 dev weeks of
    ERCOT 2019, 'relaxed' gave price MAE $8.7 vs $12.7/MWh and rank correlation 0.88 vs 0.76.
    offer_adders: dict tech -> $/MWh added to that technology's marginal (and no-load) cost (dev-year calibration).
    must_run: technologies whose available units are always committed (at least at pmin); default nuclear only.
    Adding 'coal_steam' reproduces the self-commitment of US coal units (see LIMITS).
    Returns a DispatchResult.
    """
    t0 = time.time()
    demand_MW = pd.Series(demand_MW).astype(float)
    times = demand_MW.index
    N = len(times)
    if isinstance(fleet, pd.DataFrame) and "synthetic" in fleet and fleet["synthetic"].fillna(False).all():
        hr_source = "spec"                       # generated fleets: never look up real EIA plants
        if prices in (None, "eia923"):
            prices = "constant"
    units = prepare_fleet(fleet, region=region, start=times[0], end=times[-1], hr_source=hr_source)
    fixed_MW = dict(fixed_MW or {})
    if fixed_MW:
        units = units[~units.tech.isin(list(fixed_MW))]
    units = merge_combined_cycles(units)
    cf = CompiledFleet(units, n_clusters=n_clusters, cluster=cluster and network is None)
    cf.must_run = np.isin(cf.tech, list(must_run or ()))
    C, K = cf.C, cf.K

    def _s(x):
        if x is None:
            return np.zeros(N)
        if np.isscalar(x):
            return np.full(N, float(x))
        return pd.Series(x).reindex(times).interpolate(limit_direction="both").fillna(0).values.astype(float)

    wcap = cf.capacity_of(cf.wind, times)
    scap = cf.capacity_of(cf.solar, times)
    wind_av = _s(wind_MW) if wind_MW is not None else _s(wind_cf) * wcap
    solar_av = _s(solar_MW) if solar_MW is not None else _s(solar_cf) * scap
    imp = _s(imports_MW)
    fixed = sum((_s(v) for v in fixed_MW.values()), np.zeros(N))
    hcap = cf.capacity_of(cf.hydro, times)
    hyd = _s(hydro_MW) if hydro_MW is not None else 0.25 * hcap
    D = demand_MW.values
    offers_ = {"wind": -10.0, "solar_pv": -5.0, "hydro": 0.0}
    offers_.update(offers or {})

    # fuel prices per cluster per day
    days = pd.date_range(times[0].tz_localize(None).normalize() if times.tz else times[0].normalize(),
                         (times[-1].tz_localize(None) if times.tz else times[-1]).normalize(), freq="D")
    th = cf.thermal_units
    if isinstance(prices, pd.DataFrame):
        fp = prices.reindex(days).ffill().bfill()
    elif prices in (None, "eia923") and "plant_id" in th:
        fp = fuel_prices_daily(th, days[0], days[-1], mode="eia923")
    else:
        fp = fuel_prices_daily(th, days[0], days[-1], mode="constant")
        if isinstance(prices, dict):
            for fuel, v in prices.items():
                cols = [i for i in th.index if str(th.at[i, "fuel"]) == fuel]
                if not cols:
                    continue
                vv = (pd.Series(v).reindex(days).ffill().bfill().values if not np.isscalar(v) else float(v))
                fp.loc[:, cols] = np.broadcast_to(np.asarray(vv)[:, None] if not np.isscalar(vv) else vv,
                                                  (len(days), len(cols)))
    # cluster fuel price = capacity-weighted mean of members
    fpc = np.zeros((len(days), C))
    for j, c in enumerate(cf.clusters):
        w = th.loc[c["members"], "pmax_MW"].values
        fpc[:, j] = fp[c["members"]].values @ w / w.sum()
    day_of = ((times.tz_localize(None) if times.tz else times).normalize())
    day_idx = np.searchsorted(days.values, day_of.values)

    navail, derate = availability(cf, times, mode="expected" if outages == "none" else outages, seed=seed)
    if outages == "none":
        derate = np.ones_like(derate)
    pmax_eff = cf.pmax[None, :] * derate                         # (N, C)
    if reserves == "auto":
        big = np.where(navail > 0, cf.pmax[None, :], 0).max(1)
        R = np.maximum(big, 0.06 * D)
    else:
        R = _s(reserves)
    # storage aggregate
    st = cf.storage
    if len(st):
        Pst = cf.capacity_of(st, times)
        e = st.energy_MWh.fillna(st.pmax_MW * 2).values
        Est = cf.capacity_of(st.assign(pmax_MW=e), times)
        eta = float(np.nanmean(st.eta_rt.fillna(0.85))) if "eta_rt" in st else 0.85
    else:
        Pst = Est = np.zeros(N); eta = 0.85
    eta_c = eta_d = np.sqrt(eta)

    res = DispatchResult(times, cf, method)
    if method == "merit":
        _merit(res, cf, D - imp - fixed, wind_av, solar_av, hyd, navail, pmax_eff, fpc[day_idx], offers_)
        res.finish(D, imp, fixed, fixed_MW, wind_av, solar_av, R, fpc[day_idx], time.time() - t0)
        return res

    NW = None
    if network is not None:
        NW = _network_setup(network, cf, units)
        res.line_flow_MW = np.zeros((N, len(NW["lim"])))
        res.lmp = np.zeros((N, NW["net"].nb))
        res.monitored_lines = NW["lim"]
    H = 24 if horizon == "day" else int(horizon)
    L = int(lookahead_h)
    state = dict(u=None, p=None, su_hist=np.zeros((C, 168)), sd_hist=np.zeros((C, 168)), soc=0.5 * Est[0])
    if start_state:
        state.update(start_state)
    ords = ordc or [(0.3, 250.0), (0.4, 1500.0), (10.0, 5000.0)]   # (fraction of requirement, $/MWh) blocks
    stats = []
    n_relaxed = 0
    slack_used = dict(minupdown_unit_h=0.0, ramp_MW=0.0)
    for d0 in range(0, N, H):
        for attempt in (0, 1):
          relax = attempt == 1          # second attempt: drop carried history (min up/down, ramp from yesterday)
          hrs = np.arange(d0, min(d0 + H + L, N))
          keep = min(H, N - d0)
          T = len(hrs)
          m = _Model()
          integ = method == "milp"
          nav = navail[hrs].T                                         # (C, T)
          pmx = pmax_eff[hrs].T
          pmn = np.minimum(cf.pmin[:, None], pmx)
          wid = cf.widths[:, :, None] * ((pmx - pmn) / np.maximum(cf.pmax - cf.pmin, 1e-6)[:, None])[:, None, :]
          price_f = fpc[day_idx[hrs]].T                               # (C, T)
          adder = np.array([float((offer_adders or {}).get(t, 0.0)) for t in cf.tech])
          noload = cf.Fmin[:, None] * price_f + (cf.vom + adder)[:, None] * pmn
          mc = cf.ihr[:, :, None] * price_f[:, None, :] + (cf.vom + adder)[:, None, None]
          must = cf.must_run[:, None] & (nav > 0)
          u = m.var((C, T), lb=np.where(must, nav, 0), ub=nav, cost=noload, integ=integ & (cf.n[:, None] <= 50))
          su = m.var((C, T), 0, np.inf, cf.start[:, None])
          sd = m.var((C, T), 0, np.inf, 0.0)
          seg = m.var((C, K, T), 0, wid * nav[:, None, :], mc)
          r = m.var((C, T), 0, np.inf, 0.01)
          w = m.var((T,), 0, wind_av[hrs], offers_["wind"])
          s = m.var((T,), 0, solar_av[hrs], offers_["solar_pv"])
          hy = m.var((T,), 0, hcap[hrs], offers_["hydro"])
          hr_ = m.var((T,), 0, np.maximum(hcap[hrs] - 0, 0), 0.0)
          shed = m.var((T,), 0, np.inf, voll)
          dump = m.var((T,), 0, np.inf, 50.0)
          blocks = [m.var((T,), 0, f * R[hrs], pr) for f, pr in ords]
          has_st = Pst[hrs].max() > 0
          if has_st:
              ch = m.var((T,), 0, Pst[hrs], 0.5)
              dis = m.var((T,), 0, Pst[hrs], 0.5)
              soc = m.var((T,), 0.05 * Est[hrs], 0.95 * Est[hrs], 0.0)
              rs = m.var((T,), 0, Pst[hrs], 0.005)
          # balance
          rb = m.rows(T, D[hrs] - imp[hrs] - fixed[hrs], D[hrs] - imp[hrs] - fixed[hrs])
          tt = np.arange(T)
          m.coef(np.repeat(rb[None, :], C, 0), u, pmn)
          m.coef(np.broadcast_to(rb[None, None, :], (C, K, T)), seg, 1.0)
          for v in (w, s, hy):
              m.coef(rb, v, 1.0)
          m.coef(rb, shed, 1.0); m.coef(rb, dump, -1.0)
          if has_st:
              m.coef(rb, dis, 1.0); m.coef(rb, ch, -1.0)
          # reserve
          rr = m.rows(T, R[hrs], np.inf)
          m.coef(np.repeat(rr[None, :], C, 0), r, 1.0)
          m.coef(rr, hr_, 1.0)
          for b in blocks:
              m.coef(rr, b, 1.0)
          if has_st:
              m.coef(rr, rs, 1.0)
          # hydro: output + reserve <= capacity; energy budget over the window
          rh = m.rows(T, -np.inf, hcap[hrs])
          m.coef(rh, hy, 1.0); m.coef(rh, hr_, 1.0)
          re = m.rows(1, -np.inf, hyd[hrs].sum())
          m.coef(np.full(T, re[0]), hy, 1.0)
          # segment <= width * u
          rs_ = m.rows(C * K * T, -np.inf, 0.0).reshape(C, K, T)
          m.coef(rs_, seg, 1.0); m.coef(rs_, np.broadcast_to(u[:, None, :], (C, K, T)), -wid)
          # headroom: (pmin - pmax) u + sum seg + r <= 0
          rhd = m.rows(C * T, -np.inf, 0.0).reshape(C, T)
          m.coef(rhd, u, pmn - pmx)
          m.coef(np.broadcast_to(rhd[:, None, :], (C, K, T)), seg, 1.0)
          m.coef(rhd, r, 1.0)
          # reserve <= 10-min ramp * u
          slow = np.where(cf.r10 < cf.pmax - cf.pmin)[0]
          if len(slow):
              r10r = m.rows(len(slow) * T, -np.inf, 0.0).reshape(len(slow), T)
              m.coef(r10r, r[slow], 1.0); m.coef(r10r, u[slow], -cf.r10[slow, None])
          # transitions u_t - u_{t-1} - su_t + sd_t = u_prev (t=0) else 0
          u_prev = state["u"] if state["u"] is not None else nav[:, 0] * (cf.pmin < 0.5 * cf.pmax)  # warm start guess
          if state["u"] is None:
              u_prev = np.where(cf.must_run, nav[:, 0], np.round(0.5 * nav[:, 0]))
          u_prev = np.minimum(u_prev, nav[:, 0])
          rt = m.rows(C * T, 0, 0).reshape(C, T)
          rhs = np.zeros((C, T)); rhs[:, 0] = u_prev
          m.rlo[-1] = rhs.ravel(); m.rhi[-1] = rhs.ravel()
          m.coef(rt, u, 1.0); m.coef(rt[:, 1:], u[:, :-1], -1.0); m.coef(rt, su, -1.0); m.coef(rt, sd, 1.0)
          # min up / down (window capped at T); history from previous days on the RHS
          soft_vars = []
          for kind, Tm, hist, var in (("up", cf.ut, state["su_hist"], su), ("down", cf.dt, state["sd_hist"], sd)):
              cs = np.where((Tm >= 2) & ~cf.must_run)[0]
              for c_ in cs:
                  Wn = int(min(Tm[c_], 168))
                  # rows t: sum_{tau=max(0,t-W+1)}^{t} var[c,tau] (+-) u[c,t] <= rhs
                  ii, jj = [], []
                  for t_ in range(T):
                      lo_ = max(0, t_ - Wn + 1)
                      ii.extend([t_] * (t_ - lo_ + 1)); jj.extend(range(lo_, t_ + 1))
                  ii = np.array(ii); jj = np.array(jj)
                  histsum = np.array([hist[c_, max(0, 168 - (Wn - 1 - t_)):].sum() if Wn - 1 - t_ > 0 else 0.0
                                      for t_ in range(T)])
                  histsum = np.clip(histsum, 0, nav[c_]) * (0.0 if relax else 1.0)
                  if kind == "up":
                      rows_ = m.rows(T, -np.inf, -histsum)
                      m.coef(rows_[ii], var[c_, jj], 1.0); m.coef(rows_, u[c_], -1.0)
                  else:
                      rows_ = m.rows(T, -np.inf, nav[c_] - histsum)
                      m.coef(rows_[ii], var[c_, jj], 1.0); m.coef(rows_, u[c_], 1.0)
                  if soft_history:
                      sl = m.var((T,), 0, np.inf, 10.0 * voll * max(float(cf.pmax[c_]), 1.0))
                      m.coef(rows_, sl, -1.0); soft_vars.append(("minupdown_unit_h", sl, keep))
          # ramping for slow clusters: p_t - p_{t-1} <= RU (u_t - su_t) + SU su_t ; p_{t-1} - p_t <= RD u_t + SD sd_t
          rampc = np.where((cf.ru < cf.pmax - cf.pmin) & ~cf.must_run)[0]
          if len(rampc):
              p_prev = state["p"] if (state["p"] is not None and not relax) else None
              for kind in ("up", "down"):
                  nrc = len(rampc)
                  rows_ = m.rows(nrc * T, -np.inf, 0.0).reshape(nrc, T)
                  sgn = 1.0 if kind == "up" else -1.0
                  RU = cf.ru[rampc] if kind == "up" else cf.rd[rampc]
                  SU = np.maximum(cf.pmin[rampc], RU)
                  # + p_t
                  m.coef(rows_, u[rampc], sgn * pmn[rampc])
                  m.coef(np.broadcast_to(rows_[:, None, :], (nrc, K, T)), seg[rampc], sgn)
                  # - p_{t-1}
                  m.coef(rows_[:, 1:], u[rampc, :-1], -sgn * pmn[rampc, :-1])
                  m.coef(np.broadcast_to(rows_[:, None, 1:], (nrc, K, T - 1)), seg[rampc][:, :, :-1], -sgn)
                  if kind == "up":
                      m.coef(rows_, u[rampc], -RU[:, None]); m.coef(rows_, su[rampc], RU[:, None] - SU[:, None])
                  else:
                      m.coef(rows_, u[rampc], -RU[:, None]); m.coef(rows_, sd[rampc], -SU[:, None])
                  if p_prev is not None:
                      rhs = np.zeros((nrc, T)); rhs[:, 0] = sgn * p_prev[rampc]
                      m.rhi[-1] = rhs.ravel()
                      if soft_history:
                          sl = m.var((nrc,), 0, np.inf, 10.0 * voll)
                          m.coef(rows_[:, 0], sl, -1.0); soft_vars.append(("ramp_MW", sl, None))
                  else:
                      rhs = np.zeros((nrc, T)); rhs[:, 0] = 1e9
                      m.rhi[-1] = rhs.ravel()
          # storage
          if has_st:
              rsoc = m.rows(T, 0, 0)
              rhs = np.zeros(T); rhs[0] = state["soc"]
              m.rlo[-1] = rhs; m.rhi[-1] = rhs
              m.coef(rsoc, soc, 1.0); m.coef(rsoc[1:], soc[:-1], -1.0)
              m.coef(rsoc, ch, -eta_c); m.coef(rsoc, dis, 1.0 / eta_d)
              r1 = m.rows(T, -np.inf, Pst[hrs]); m.coef(r1, rs, 1.0); m.coef(r1, dis, 1.0); m.coef(r1, ch, -1.0)
              r2 = m.rows(T, -np.inf, -0.05 * Est[hrs] * eta_d); m.coef(r2, rs, 1.0); m.coef(r2, soc, -eta_d)
              if storage_end == "cyclic":
                  rend = m.rows(1, state["soc"], np.inf)
                  m.coef(rend, soc[keep - 1], 1.0)
          A, c, lb, ub, ig, rlo, rhi = m.build()
          fix_idx = np.r_[u.ravel(), su.ravel(), sd.ravel()]
          t_build = time.time()
          if NW is None:
              x, y, info = _solve(A, c, lb, ub, ig, rlo, rhi, mip=integ, mip_gap=mip_gap, time_limit=time_limit,
                                  fix_idx=fix_idx, pricing=pricing)
          else:
              # DC network: lazy line-limit rows  F_l >= |sum_b PTDF_lb inj_bt|  added until no overload remains
              vars_ = dict(u=u, seg=seg, w=w, s=s, hy=hy, shed=shed, dump=dump,
                           ch=ch if has_st else None, dis=dis if has_st else None)
              mon = np.array(sorted(NW["monitor"]), int)
              for it in range(8):
                  Af, flo, fhi = _flow_rows(NW, mon, vars_, pmn, T, A.shape[1], D[hrs] - imp[hrs] - fixed[hrs], K)
                  A2 = sp.vstack([A, Af]).tocsc() if len(mon) else A
                  x, y, info = _solve(A2, c, lb, ub, ig, np.r_[rlo, flo], np.r_[rhi, fhi], mip=integ,
                                      mip_gap=mip_gap, time_limit=time_limit, fix_idx=fix_idx,
                                      pricing=pricing)
                  if len(x) != A.shape[1]:
                      break
                  fl = _flows(NW, x, vars_, pmn, T, D[hrs] - imp[hrs] - fixed[hrs])       # (nlim, T)
                  over = NW["lim"][np.abs(fl).max(1) > NW["rate"][NW["lim"]] * 1.0005 + 1e-6]
                  new_ = np.setdiff1d(over, mon)
                  if len(new_) == 0:
                      break
                  mon = np.union1d(mon, new_)
              NW["monitor"] = set(mon.tolist())        # keep for the next day (warm list)
              info["n_monitored_lines"] = int(len(mon))
              info["flows"] = fl
              info["mu"] = y[A.shape[0]:].reshape(len(mon), T) if (y is not None and len(mon)) else None
              info["mon"] = mon
          info["solve_s"] = time.time() - t_build
          info["n_var"], info["n_row"] = A.shape[1], A.shape[0]
          info["relaxed_history"] = relax
          if soft_vars and x is not None and len(x):
              for key, sl, kp in soft_vars:
                  v = np.asarray(x)[np.asarray(sl)]
                  slack_used[key] += float(v[:kp].sum() if kp is not None and v.ndim == 1 else v.sum())
          if info.get("status") == "Optimal" or (integ and str(info.get("status")).startswith("Time limit")
                                                 and np.isfinite(info.get("mip_gap", np.inf))):
              break
          if relax:
              raise RuntimeError(f"dispatch failed at hour {d0}: {info}")
          n_relaxed += 1
          first_status = info.get("status")
        if attempt == 1:
            info["first_attempt_status"] = first_status
        stats.append(info)
        k = np.arange(keep)
        gi = hrs[:keep]
        res.u[gi] = x[u[:, k]].T
        res.su[gi] = x[su[:, k]].T
        res.sd[gi] = x[sd[:, k]].T
        res.seg[gi] = np.transpose(x[seg[:, :, k]], (2, 0, 1))
        res.p[gi] = res.u[gi] * pmn[:, k].T + res.seg[gi].sum(2)
        res.r[gi] = x[r[:, k]].T
        res.wind[gi] = x[w[k]]; res.solar[gi] = x[s[k]]; res.hydro[gi] = x[hy[k]]
        res.shed[gi] = x[shed[k]]; res.dump[gi] = x[dump[k]]
        res.reserve_short[gi] = sum(x[b[k]] for b in blocks)
        if has_st:
            res.ch[gi] = x[ch[k]]; res.dis[gi] = x[dis[k]]; res.soc[gi] = x[soc[k]]
        if y is not None:
            res.price[gi] = y[rb[k]]
            res.reserve_price[gi] = y[rr[k]]
        if NW is not None:
            res.line_flow_MW[gi] = info["flows"][:, k].T
            if info.get("mu") is not None and y is not None:
                res.lmp[gi] = res.price[gi][:, None] + (NW["H"][info["mon"]].T @ info["mu"][:, k]).T
            else:
                res.lmp[gi] = res.price[gi][:, None]
            for kk in ("flows", "mu", "mon"):
                info.pop(kk, None)
        # carry state
        state["u"] = np.round(res.u[gi[-1]]) if integ else res.u[gi[-1]]
        state["p"] = res.p[gi[-1]]
        state["su_hist"] = np.concatenate([state["su_hist"], res.su[gi].T], 1)[:, -168:]
        state["sd_hist"] = np.concatenate([state["sd_hist"], res.sd[gi].T], 1)[:, -168:]
        if has_st:
            state["soc"] = res.soc[gi[-1]]
        if verbose and (d0 // H) % 30 == 0:
            print(f"  day {d0 // H}: {info.get('status')} gap {info.get('mip_gap', 0):.4f} {time.time() - t0:.1f}s",
                  flush=True)
    res.solver_stats = stats
    res.n_days_relaxed = n_relaxed
    res.history_slack = slack_used
    res.final_state = state
    res.finish(D, imp, fixed, fixed_MW, wind_av, solar_av, R, fpc[day_idx], time.time() - t0, derate=derate,
               navail=navail)
    return res


def _merit(res, cf, net_load, wind, solar, hydro, navail, pmax_eff, fprice, offers):
    """Hourly merit order: renewables first (curtailed if net load < 0), then thermal by full-load marginal cost,
    nuclear must-run.  Fully vectorised."""
    N = len(net_load)
    C = cf.C
    cap = navail * pmax_eff                                        # (N, C)
    avg_ihr = (cf.ihr * cf.widths).sum(1) / np.maximum(cf.widths.sum(1), 1e-9)
    mc = avg_ihr[None, :] * fprice + cf.vom[None, :]
    mc = np.where(cf.must_run[None, :], -1e3, mc)
    rem = net_load - hydro
    ren = wind + solar
    use_ren = np.minimum(ren, np.maximum(rem, 0) + 0.0)
    rem2 = rem - use_ren
    order = np.argsort(mc, axis=1)
    capo = np.take_along_axis(cap, order, 1)
    cum = np.cumsum(capo, 1)
    disp_o = np.clip(rem2[:, None] - (cum - capo), 0, capo)
    p = np.zeros_like(cap)
    np.put_along_axis(p, order, disp_o, 1)
    marg = (disp_o > 1e-6) & (disp_o < capo - 1e-6)
    mco = np.take_along_axis(mc, order, 1)
    price = np.where(marg.any(1), (mco * marg).max(1), np.where(rem2 <= 0, offers["wind"], mco[:, -1]))
    res.p[:] = p
    res.u[:] = np.where(p > 0, np.ceil(p / np.maximum(pmax_eff, 1e-9)), 0)
    frac = np.where(ren > 0, use_ren / np.maximum(ren, 1e-9), 0)
    res.wind[:] = wind * frac
    res.solar[:] = solar * frac
    res.hydro[:] = np.minimum(hydro, np.maximum(net_load, 0))
    res.shed[:] = np.maximum(rem2 - cap.sum(1), 0)
    res.price[:] = price
    # fuel for emissions: assume the average incremental heat rate + no-load share
    res.seg[:, :, 0] = p
    res._merit_ihr = avg_ihr


class DispatchResult:
    """Container for dispatch outputs.  Arrays are hourly (N, ...); DataFrames via properties."""

    def __init__(self, times, cf, method):
        N, C, K = len(times), cf.C, cf.K
        self.times, self.cf, self.method = times, cf, method
        z = lambda *s: np.zeros(s)  # noqa
        self.u, self.su, self.sd, self.p, self.r = z(N, C), z(N, C), z(N, C), z(N, C), z(N, C)
        self.seg = z(N, C, K)
        self.wind, self.solar, self.hydro, self.shed, self.dump = z(N), z(N), z(N), z(N), z(N)
        self.ch, self.dis, self.soc, self.price, self.reserve_price, self.reserve_short = (z(N) for _ in range(6))
        self._merit_ihr = None
        self.line_flow_MW = None
        self.lmp = None

    def finish(self, D, imp, fixed, fixed_MW, wind_av, solar_av, R, fprice, wall, derate=None, navail=None):
        cf = self.cf
        self.demand, self.imports, self.fixed_total = D, imp, fixed
        self.fixed_MW = fixed_MW
        self.wind_avail, self.solar_avail, self.reserve_req = wind_av, solar_av, R
        self.navail, self.derate = navail, derate
        if self._merit_ihr is not None:
            fuel = self.p * self._merit_ihr[None, :] + (self.u * 0)
        else:
            fuel = self.u * cf.Fmin[None, :] + (self.seg * cf.ihr[None, :, :]).sum(2)
        start_heat = self.su * np.array([c["start"] for c in cf.clusters])[None, :] / np.maximum(fprice, 0.5)
        self.fuel_MMBtu = fuel + start_heat
        self.co2_t = self.fuel_MMBtu * cf.co2[None, :] / 1000.0
        self.fuel_cost = self.fuel_MMBtu * fprice
        self.wall_s = wall
        bal = (self.p.sum(1) + self.wind + self.solar + self.hydro + self.dis - self.ch + self.shed - self.dump
               + imp + fixed - D)
        self.balance_residual_MW = bal

    # ---- views
    @property
    def gen_by_tech(self):
        """MW by technology (DataFrame, hourly)."""
        cols = {}
        for t in np.unique(self.cf.tech):
            cols[t] = self.p[:, self.cf.tech == t].sum(1)
        cols["wind"] = self.wind
        cols["solar_pv"] = self.solar
        cols["hydro"] = self.hydro
        for k, v in (self.fixed_MW or {}).items():
            cols[k] = pd.Series(v).reindex(self.times).interpolate(limit_direction="both").fillna(0).values
        cols["storage_net"] = self.dis - self.ch
        return pd.DataFrame(cols, index=self.times)

    @property
    def gen_by_fuel(self):
        """MW by EIA-930 fuel category (COL, NG, NUC, OIL, WAT, SUN, WND, OTH)."""
        g = self.gen_by_tech
        out = {}
        for t in g:
            f = EIA930_FUEL.get(t, "WAT" if t == "hydro" else "OTH")
            out[f] = out.get(f, 0) + g[t].values
        return pd.DataFrame(out, index=self.times)

    @property
    def curtailment_MW(self):
        return pd.DataFrame({"wind": self.wind_avail - self.wind, "solar_pv": self.solar_avail - self.solar},
                            index=self.times)

    @property
    def summary(self):
        g = self.gen_by_fuel
        return {"energy_TWh": {k: float(v.sum() / 1e6) for k, v in g.items()},
                "co2_Mt": float(self.co2_t.sum() / 1e6),
                "mean_price": float(np.mean(self.price)), "shed_MWh": float(self.shed.sum()),
                "curtail_TWh": float(self.curtailment_MW.values.sum() / 1e6),
                "max_balance_residual_MW": float(np.abs(self.balance_residual_MW).max()),
                "starts": float(self.su.sum()), "wall_s": self.wall_s}

    @property
    def inertia_MWs(self):
        """Hourly kinetic energy of online synchronous machines [MW s]: committed units x H x MVA, plus fixed
        injections (e.g. observed nuclear, H 5 s on MW/0.9 MVA) and hydro (H 3 s)."""
        cf = self.cf
        E = (np.round(self.u) * (cf.H * cf.mva)[None, :]).sum(1)
        E = E + self.hydro / 0.9 * 3.0
        from .fleet import tech_default
        for k, v in (self.fixed_MW or {}).items():
            mw = pd.Series(v).reindex(self.times).interpolate(limit_direction="both").fillna(0).values
            E = E + mw / 0.9 * tech_default(k, "H")
        return E

    def frame(self):
        """Hourly DataFrame: demand, price, reserve, by-fuel MW, CO2, inertia."""
        df = self.gen_by_fuel.add_prefix("MW_")
        df.insert(0, "demand_MW", self.demand)
        df["imports_MW"] = self.imports
        df["price_$_per_MWh"] = self.price
        df["reserve_price"] = self.reserve_price
        df["co2_t"] = self.co2_t.sum(1)
        df["shed_MW"] = self.shed
        df["reserve_short_MW"] = self.reserve_short
        df["inertia_MWs"] = self.inertia_MWs
        df["wind_avail_MW"] = self.wind_avail
        df["solar_avail_MW"] = self.solar_avail
        return df

    def system_state(self, hour, region="ERCO", **kw):
        """Frequency-model system state at hour index `hour` (online synchronous units with H, MVA, headroom)."""
        from .frequency import make_system_state
        cf = self.cf
        units = []
        for j, c in enumerate(cf.clusters):
            n_on = self.u[hour, j]
            if n_on < 1e-6:
                continue
            pm = c["pmax"] * (self.derate[hour, j] if self.derate is not None else 1.0)
            from .frequency import TECH_DYN
            d = TECH_DYN.get(c["tech"], TECH_DYN["other"])
            head = min(pm * n_on - self.p[hour, j], d["head"] * pm * n_on)
            # units carrying scheduled spinning reserve hold that headroom for governor response
            up = max(head, min(self.r[hour, j], pm * n_on - self.p[hour, j])) if d["resp"] else 0.0
            units.append(dict(tech=c["tech"], MW=float(self.p[hour, j]), pmax_MW=float(pm * n_on),
                              pmin_MW=float(c["pmin"] * n_on), mva=float(c["mva"] * n_on), H_s=float(c["H"]),
                              n=float(n_on), governor=dict(up_MW=float(max(up, 0.0)))))
        for tech, mw in (("wind", self.wind[hour]), ("solar_pv", self.solar[hour])):
            if mw > 0:
                units.append(dict(tech=tech, MW=float(mw), pmax_MW=float(mw), H_s=0.0, mva=float(mw)))
        if self.hydro[hour] > 0:
            units.append(dict(tech="hydro_res", MW=float(self.hydro[hour]),
                              pmax_MW=float(max(self.hydro[hour], 1.0)), mva=float(self.hydro[hour] / 0.9)))
        for k, v in (self.fixed_MW or {}).items():
            mw = float(pd.Series(v).reindex(self.times).interpolate(limit_direction="both").fillna(0).values[hour])
            if mw > 0:
                units.append(dict(tech=k, MW=mw, pmax_MW=mw, mva=mw / 0.9))
        stP = self.cf.capacity_of(self.cf.storage, self.times[hour:hour + 1])[0] if len(self.cf.storage) else 0
        if stP > 0:
            units.append(dict(tech="battery", MW=float(self.dis[hour] - self.ch[hour]), pmax_MW=float(stP),
                              pmin_MW=-float(stP), mva=float(stP), H_s=0.0))
        return make_system_state(units, load_MW=float(self.demand[hour]), region=region, **kw)


# ---------------------------------------------------------------------- DC network helpers
def _network_setup(network, cf, units):
    """network: dict(net=Network, gen_bus={unit id: internal bus index}, load_share=array(nb), optional
    wind_share / solar_share / hydro_share / storage_share arrays, monitor=list of line indices, rate_scale)."""
    net = network["net"]
    nb = net.nb
    H = net.ptdf()
    ls = np.asarray(network.get("load_share"), float)
    ls = ls / ls.sum()
    shares = {k: (np.asarray(network[k + "_share"], float) / np.sum(network[k + "_share"])
                  if network.get(k + "_share") is not None else ls) for k in ("wind", "solar", "hydro", "storage")}
    gb = np.array([network["gen_bus"][c["ids"][0]] for c in cf.clusters], int)
    rate = net.rate_MW * network.get("rate_scale", 1.0)
    lim = np.where(np.isfinite(rate))[0]
    return dict(net=net, H=H, gb=gb, Hg=H[:, gb], ls=ls, Hl=H @ ls, Hw=H @ shares["wind"], Hs=H @ shares["solar"],
                Hh=H @ shares["hydro"], Hst=H @ shares["storage"], rate=rate, lim=lim,
                monitor=set(network.get("monitor", [])))


def _flow_rows(NW, mon, v, pmn, T, nvar, netload, K):
    """Sparse rows (len(mon)*T x nvar) and bounds for line flows of the monitored lines."""
    if len(mon) == 0:
        return sp.csr_matrix((0, nvar)), np.zeros(0), np.zeros(0)
    nl = len(mon)
    C = v["u"].shape[0]
    rows, cols, vals = [], [], []
    rid = np.arange(nl * T).reshape(nl, T)
    Hg = NW["Hg"][mon]                                    # (nl, C)
    for c_ in range(C):
        coef = Hg[:, c_]
        if not np.any(coef):
            continue
        rows.append(rid.ravel()); cols.append(np.tile(v["u"][c_], nl)); vals.append((coef[:, None] * pmn[c_][None, :]).ravel())
        for k_ in range(K):
            rows.append(rid.ravel()); cols.append(np.tile(v["seg"][c_, k_], nl)); vals.append(np.repeat(coef, T))
    for key, Hx, sg in (("w", "Hw", 1), ("s", "Hs", 1), ("hy", "Hh", 1), ("shed", "Hl", 1), ("dump", "Hl", -1),
                        ("dis", "Hst", 1), ("ch", "Hst", -1)):
        if v.get(key) is None:
            continue
        coef = NW[Hx][mon] * sg
        rows.append(rid.ravel()); cols.append(np.tile(v[key], nl)); vals.append(np.repeat(coef, T))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nl * T, nvar))
    cst = (NW["Hl"][mon][:, None] * netload[None, :])     # flow caused by the (distributed) net load
    F = NW["rate"][mon][:, None]
    return A, (-F + cst).ravel(), (F + cst).ravel()


def _flows(NW, x, v, pmn, T, netload):
    lim = NW["lim"]
    p = x[v["u"]] * pmn + x[v["seg"]].sum(1)               # (C, T)
    f = NW["Hg"][lim] @ p
    for key, Hx, sg in (("w", "Hw", 1), ("s", "Hs", 1), ("hy", "Hh", 1), ("shed", "Hl", 1), ("dump", "Hl", -1),
                        ("dis", "Hst", 1), ("ch", "Hst", -1)):
        if v.get(key) is not None:
            f = f + sg * NW[Hx][lim][:, None] * x[v[key]][None, :]
    return f - NW["Hl"][lim][:, None] * netload[None, :]


def fleet_from_case(ppc_or_net):
    """Contract fleet from a MATPOWER case (genfuel -> tech; linear gencost c1 -> VOM so offers = case costs) and the
    `network` dict for dispatch (gen_bus, load_share from bus Pd + Gs; pass demand = sum(Pd + Gs) to match the
    case)."""
    from .network import Network
    net = ppc_or_net if isinstance(ppc_or_net, Network) else Network.from_ppc(ppc_or_net)
    fuel_map = {"coal": "coal_steam", "ng": "gas_cc", "nuclear": "nuclear", "wind": "wind", "solar": "solar_pv",
                "hydro": "hydro_res", "oil": "oil", "dfo": "oil", "biomass": "biomass"}
    rows = []
    for g in range(net.gen.shape[0]):
        if net.gen[g, 7] <= 0:
            continue
        fuel = (net.genfuel[g] if net.genfuel else "ng").lower()
        tech = fuel_map.get(fuel, "gas_ct")
        gc = net.gencost[g]
        n = int(gc[3]); coef = gc[4:4 + n]
        c1 = float(coef[-2]) if n >= 2 else 0.0
        c2 = float(coef[-3]) if n >= 3 else 0.0
        pmax = float(net.gen[g, 8]); pmin = float(max(net.gen[g, 9], 0.0))
        rows.append(dict(id=f"g{g}", plant_id=g, tech=tech, pmax_MW=pmax, pmin_MW=pmin, heat_rate_curve=None,
                         **{"vom_$_per_MWh": c1 + c2 * pmax, "fuel_cost_$_per_MMBtu": 0.0, "start_cost_$": 0.0},
                         min_up_h=0.0, min_down_h=0.0, ramp_up_MW_per_min=max(pmax, 1.0), ramp_down_MW_per_min=max(pmax, 1.0),
                         co2_kg_per_MMBtu=0.0, inertia_H_s=None, bus=int(net.gen[g, 0])))
    fleet = pd.DataFrame(rows)
    th = fleet[~fleet.tech.isin(["wind", "solar_pv"])]
    pd_ = net.bus[:, 2] + net.bus[:, 4]          # signed load + shunt conductance (MW), as the DC-OPF sees it
    cap = lambda t: np.bincount(fleet[fleet.tech == t].bus, weights=fleet[fleet.tech == t].pmax_MW, minlength=net.nb)
    nwk = dict(net=net, gen_bus=dict(zip(fleet.id, fleet.bus)), load_share=pd_ / pd_.sum(),
               wind_share=cap("wind") if cap("wind").sum() > 0 else None,
               solar_share=cap("solar_pv") if cap("solar_pv").sum() > 0 else None)
    return fleet, nwk
