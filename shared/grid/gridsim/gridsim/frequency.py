"""System frequency after a power imbalance: aggregated swing equation(s) with governors, FFR, synthetic inertia,
load damping, UFLS and generator protection.  Vectorised over a batch of events.

Physics (per area a, all powers in MW, frequency deviation df in Hz):

    2 E_a / f0 * d(df_a)/dt = sum_i dPm_i + P_ffr + P_si + P_shed - P_lost(t) - D_a * P_load_a * df_a / f0 - P_tie_out_a
    dP_tie_ab / dt = 2 pi T_ab (df_a - df_b)            (T_ab synchronising coefficient, MW/rad)

E_a = sum_i H_i * MVA_i over online synchronous machines (+ any extra inertia), reduced when a lost unit carries
inertia.  Governor/turbine of unit (group) i:

    e_i   = -K_i * db(df) ,  K_i = P_rating_i / (R_i f0)       droop R, deadband db (no-step by default)
    x1_i  : lag T1 (valve / controller), rate limit, clipped to [-down_i, +up_i] (headroom)
    y_i   = LL_B( LL_A(x1_i) )                                 two lead-lags (num, den):
            steam reheat (1 + s Fhp Trh)/(1 + s Trh); hydro transient droop (1 + s Tr)/(1 + s Tr Rt/R) and
            penstock (1 - s Tw)/(1 + s Tw/2); CCGT steam bottoming (1 + s g Th)/(1 + s Th)
    dPm_i = y_i

Integration: fixed step `h` (default 0.01 s) exponential-Euler for lags (exact for a constant input over the step)
and explicit Euler for the swing equation; output every `dt` s.  Convergence is recorded in checks/frequency_analytic.json.
"""
from __future__ import annotations

import copy
import numpy as np

# Default dynamic parameters per technology (literature typical values; see README "Frequency" for sources).
# H on own MVA base [s]; R droop [pu]; db deadband [Hz] (None -> system default); T1 lag [s];
# LL_A, LL_B = (T_num, T_den) lead-lags; rate = max governor movement [pu of rating per s]; resp = governor active;
# head = max governor response as a fraction of rating (beyond spare capacity).
TECH_DYN = {
    "nuclear":    dict(H=5.0, R=0.05, T1=0.3, LL_A=(0.3 * 7.0, 7.0), LL_B=(0, 0), rate=0.05, resp=False, head=0.0),
    "coal_steam": dict(H=4.0, R=0.05, T1=0.3, LL_A=(0.3 * 7.0, 7.0), LL_B=(0, 0), rate=0.02, resp=True, head=0.06),
    "gas_steam":  dict(H=4.0, R=0.05, T1=0.3, LL_A=(0.3 * 7.0, 7.0), LL_B=(0, 0), rate=0.03, resp=True, head=0.06),
    "biomass":    dict(H=3.5, R=0.05, T1=0.3, LL_A=(0.3 * 7.0, 7.0), LL_B=(0, 0), rate=0.02, resp=True, head=0.05),
    "oil":        dict(H=4.0, R=0.05, T1=0.3, LL_A=(0.3 * 7.0, 7.0), LL_B=(0, 0), rate=0.03, resp=True, head=0.06),
    "gas_cc":     dict(H=5.0, R=0.05, T1=0.5, LL_A=(0.67 * 40.0, 40.0), LL_B=(0, 0), rate=0.10, resp=True, head=0.08),
    "gas_ct":     dict(H=4.0, R=0.05, T1=0.5, LL_A=(0, 0), LL_B=(0, 0), rate=0.15, resp=True, head=0.10),
    "gas_ice":    dict(H=1.5, R=0.05, T1=0.5, LL_A=(0, 0), LL_B=(0, 0), rate=0.20, resp=True, head=0.10),
    "hydro_res":  dict(H=3.0, R=0.05, T1=0.5, LL_A=(5.0, 5.0 * 0.38 / 0.05), LL_B=(-1.5, 0.75), rate=0.10, resp=True, head=0.15),
    "hydro_ror":  dict(H=2.5, R=0.05, T1=0.5, LL_A=(5.0, 5.0 * 0.38 / 0.05), LL_B=(-1.0, 0.5), rate=0.10, resp=True, head=0.05),
    "pumped":     dict(H=3.5, R=0.05, T1=0.5, LL_A=(5.0, 5.0 * 0.38 / 0.05), LL_B=(-1.5, 0.75), rate=0.10, resp=True, head=0.15),
    "geothermal": dict(H=3.0, R=0.05, T1=0.3, LL_A=(0, 0), LL_B=(0, 0), rate=0.01, resp=False, head=0.0),
    "other":      dict(H=3.0, R=0.05, T1=0.5, LL_A=(0, 0), LL_B=(0, 0), rate=0.05, resp=True, head=0.05),
    # inverter-based: no inertia; droop response only with headroom (curtailed or battery)
    "wind":       dict(H=0.0, R=0.05, T1=0.3, LL_A=(0, 0), LL_B=(0, 0), rate=1.0, resp=False, head=0.0),
    "solar_pv":   dict(H=0.0, R=0.05, T1=0.3, LL_A=(0, 0), LL_B=(0, 0), rate=1.0, resp=False, head=0.0),
    "battery":    dict(H=0.0, R=0.01, T1=0.2, LL_A=(0, 0), LL_B=(0, 0), rate=5.0, resp=True, head=1.0),
}
IBR_TECHS = {"wind", "solar_pv", "battery"}

# Interconnection defaults.  D = load damping in % load per % frequency.  GB: 0.75 (1.5 %/Hz) fitted on the first
# nadir of the 9 Aug 2019 dev event (checks/frequency_events.json).  ERCOT: 2.5 = an EFFECTIVE damping fitted on the
# 5 ERCOT dev events of splits.json with states from the calibrated dispatch (checks/frequency_dev_events.json; it was
# 3.75 with the earlier dispatch calibration, so it trades off against governor headroom); it stands for all fast
# response the model lacks (motor load, fast CT/IBR response), textbook load damping is 1-2.  US_EAST/US_WEST: 1.0
# (textbook, Kundur 1994; not calibrated).
# ERCOT's effective damping is fitted on the 5 ERCOT dev events with states from the calibrated dispatch, so it belongs
# to a physics version: read from that version's calibration file (checks/check_frequency_dev_events.py writes it).
from .fleet import calibration as _calibration
_ERCO_D = float(_calibration().get("frequency", {}).get("ERCO_D_pct_per_pct", 2.5))

SYSTEMS = {
    # ERCOT: 0.036 Hz governor deadband (BAL-001-TRE-1), UFLS 59.3/58.9/58.5 Hz for 5/10/10 % (ERCOT Nodal Op. Guide 2.6),
    # load resources tripped by UF relays at 59.7 Hz; FFR from batteries at 59.85 Hz.
    # deadbands by technology from BAL-001-TRE-1 (R6): +-0.034 Hz steam/hydro with mechanical governors, +-0.0166 Hz
    # all other generating units (gas turbines, combined cycles, inverter-based)
    "ERCO": dict(f0=60.0, db=0.0166, D=_ERCO_D,
                 db_by_tech={"coal_steam": 0.034, "gas_steam": 0.034, "biomass": 0.034, "oil": 0.034,
                             "nuclear": 0.034, "hydro_res": 0.034, "hydro_ror": 0.034, "pumped": 0.034},
                 ufls=[dict(f_Hz=59.3, frac=0.05, delay_s=0.3), dict(f_Hz=58.9, frac=0.10, delay_s=0.3),
                       dict(f_Hz=58.5, frac=0.10, delay_s=0.3)]),
    # GB: mandatory response deadband +-0.015 Hz (Grid Code BC3), LFDD 48.8 Hz first stage (Grid Code OC6.6, ~5 % per stage)
    "GB": dict(f0=50.0, db=0.015, D=0.75,
               ufls=[dict(f_Hz=48.8, frac=0.05, delay_s=0.2), dict(f_Hz=48.75, frac=0.05, delay_s=0.2),
                     dict(f_Hz=48.7, frac=0.10, delay_s=0.2), dict(f_Hz=48.6, frac=0.075, delay_s=0.2),
                     dict(f_Hz=48.5, frac=0.075, delay_s=0.2), dict(f_Hz=48.4, frac=0.075, delay_s=0.2),
                     dict(f_Hz=48.2, frac=0.075, delay_s=0.2), dict(f_Hz=48.0, frac=0.05, delay_s=0.2)]),
    "US_EAST": dict(f0=60.0, db=0.036, D=1.0,
                    ufls=[dict(f_Hz=59.5, frac=0.06, delay_s=0.3), dict(f_Hz=59.3, frac=0.06, delay_s=0.3),
                          dict(f_Hz=59.1, frac=0.06, delay_s=0.3), dict(f_Hz=58.9, frac=0.06, delay_s=0.3)]),
    "US_WEST": dict(f0=60.0, db=0.036, D=1.0,
                    ufls=[dict(f_Hz=59.1, frac=0.053, delay_s=0.3), dict(f_Hz=58.9, frac=0.059, delay_s=0.3),
                          dict(f_Hz=58.7, frac=0.065, delay_s=0.3), dict(f_Hz=58.5, frac=0.067, delay_s=0.3),
                          dict(f_Hz=58.3, frac=0.067, delay_s=0.3)]),
}
for _b in ("CISO", "BPAT", "PACE", "PACW", "NEVP", "AZPS", "SRP", "LDWP", "WALC"):
    SYSTEMS[_b] = SYSTEMS["US_WEST"]
for _b in ("PJM", "MISO", "SWPP", "NYIS", "ISNE", "SOCO", "TVA", "FPL", "DUK", "SC", "CPLE"):
    SYSTEMS[_b] = SYSTEMS["US_EAST"]


def make_system_state(units, load_MW, region="ERCO", **kw):
    """Build a system_state dict.

    units: list of dicts with `tech`, `MW` (current output), `pmax_MW` (rating, online), optional `pmin_MW`,
           `mva` (default pmax/0.9), `H_s`, `governor` {droop, deadband_Hz, T_s, headroom_frac, enabled},
           `area` (index), `n` (number of identical units in the group, information only).
    kw: f0, D (load damping, % load per % frequency), ufls (list), ffr (list of {MW, trigger_Hz, delay_s, ramp_s, area}),
        synthetic_inertia (list of {H_s, MVA, MW_max, area}), protection (list of {MW, kind 'rocof'|'uf', threshold,
        delay_s, window_s, area, H_MWs}), secondary {MW, T_s}, H_extra_MWs (per area), areas (n), ties
        (list of {a, b, T_MW_per_rad}), pload_area (fractions).
    """
    sysd = copy.deepcopy(SYSTEMS.get(region, SYSTEMS["ERCO"]))
    st = dict(region=region, f0=sysd["f0"], D=sysd["D"], db=sysd["db"], db_by_tech=sysd.get("db_by_tech", {}),
              ufls=sysd["ufls"], load_MW=float(load_MW),
              units=list(units), ffr=[], synthetic_inertia=[], protection=[], secondary=None, H_extra_MWs=0.0,
              areas=1, ties=[], pload_area=None)
    st.update(kw)
    return st


def inertia_MWs(system_state, area=None):
    """Kinetic energy stored in online synchronous machines [MW s] (= MJ)."""
    tot = 0.0
    for u in system_state["units"]:
        if area is not None and u.get("area", 0) != area:
            continue
        H, mva = _unit_H_mva(u)
        tot += H * mva
    extra = system_state.get("H_extra_MWs", 0.0)
    if np.ndim(extra):
        extra = extra[area] if area is not None else np.sum(extra)
    elif area not in (None, 0):
        extra = 0.0
    return tot + extra


def _unit_H_mva(u):
    d = TECH_DYN.get(u.get("tech", "other"), TECH_DYN["other"])
    H = u.get("H_s")
    H = d["H"] if H is None else H
    pmax = u.get("pmax_MW", u.get("MW", 0.0))
    mva = u.get("mva")
    mva = pmax / 0.9 if mva is None else mva
    if u.get("tech") in IBR_TECHS:
        H = u.get("H_s") or 0.0
    return H, mva


def _compile(state, f0):
    """Turn a system_state into flat parameter arrays (one row per unit group)."""
    units = state["units"]
    n = max(len(units), 1)
    P = {k: np.zeros(n) for k in ["K", "db", "T1", "rate", "up", "dn", "An", "Ad", "Bn", "Bd", "area"]}
    for i, u in enumerate(units):
        d = TECH_DYN.get(u.get("tech", "other"), TECH_DYN["other"])
        gov = dict(u.get("governor") or {})
        enabled = gov.get("enabled", d["resp"])
        pmax = float(u.get("pmax_MW", u.get("MW", 0.0)))
        p = float(u.get("MW", 0.0))
        pmin = float(u.get("pmin_MW", 0.0) or 0.0)
        R = gov.get("droop") or d["R"]
        db = gov.get("deadband_Hz")
        if db is None:
            db = (state.get("db_by_tech") or {}).get(u.get("tech"), state.get("db", 0.036))
        head = gov.get("headroom_frac")
        head = d["head"] if head is None else head
        P["K"][i] = (pmax / (R * f0)) if enabled else 0.0
        P["db"][i] = db
        P["T1"][i] = gov.get("T_s") or d["T1"]
        P["rate"][i] = d["rate"] * pmax
        P["up"][i] = max(0.0, min(pmax - p, head * pmax)) if enabled else 0.0
        P["dn"][i] = max(0.0, min(p - pmin, max(head, 0.1) * pmax)) if enabled else 0.0
        if enabled and gov.get("up_MW") is not None:      # explicit headroom (e.g. from aggregate_units)
            P["up"][i] = gov["up_MW"]
        if enabled and gov.get("dn_MW") is not None:
            P["dn"][i] = gov["dn_MW"]
        P["An"][i], P["Ad"][i] = gov.get("LL_A", d["LL_A"])
        P["Bn"][i], P["Bd"][i] = gov.get("LL_B", d["LL_B"])
        P["area"][i] = u.get("area", 0)
    P["area"] = P["area"].astype(int)
    return P


def _pad_stack(dicts, keys, n):
    out = {}
    for k in keys:
        arr = np.zeros((len(dicts), n))
        for b, d in enumerate(dicts):
            arr[b, :len(d[k])] = d[k]
        out[k] = arr
    return out


def _as_events(loss_MW, B):
    """loss_MW: scalar; a list of (t, MW[, area, H_MWs]) tuples; or a length-B list/array with one of those per state."""
    def is_event(e):     # one (t, MW[, area, H_MWs]) loss: a short sequence of numbers
        return isinstance(e, (list, tuple, np.ndarray)) and 2 <= len(e) <= 4 and all(np.isscalar(v) for v in e)

    def is_event_list(x):
        # A list of losses, not a list of per-state loss lists. Up to 14 Nov 2026 a per-state list of lists was taken
        # as ONE event list for every state (GridPulse's build passed an object array to avoid it; HANDOFF P0.6).
        return isinstance(x, (list, tuple)) and len(x) > 0 and all(is_event(e) for e in x)

    def one(x):
        if np.isscalar(x):
            return [(0.0, float(x), 0, 0.0)]
        return [(float(e[0]), float(e[1]), int(e[2]) if len(e) > 2 else 0, float(e[3]) if len(e) > 3 else 0.0)
                for e in x]
    if np.isscalar(loss_MW) or is_event_list(loss_MW):
        return [one(loss_MW)] * B
    if len(loss_MW) == B:
        return [one(x) for x in loss_MW]
    raise ValueError("loss_MW must be a scalar, a list of (t, MW) events, or one of those per state")


def frequency_event(system_state, loss_MW, t_end=60.0, dt=0.02, h=0.01, record=True, return_batch=False):
    """Simulate the frequency after losing `loss_MW` at t = 0 (or a list of (t_s, MW[, area, H_MWs]) losses).

    system_state: dict from `make_system_state` (or a list of them for a batch run; loss_MW can then be per state).
    Returns dict with `t` [s], `f` [Hz] (centre of inertia), `f_area` (n_area, nt), metrics (`rocof_Hz_s` over the
    first 0.5 s, `nadir_Hz`, `t_nadir_s`, `settling_Hz` (mean of last 10 s), `zenith_Hz`), `inertia_MWs`,
    `gov_MW` per unit group (nt, n), `ffr_MW`, `si_MW`, `ufls` (list of stage actions), `trips` (protection actions),
    `damping_MW`.  For a batch, every entry gets a leading batch axis and `ufls`/`trips` are lists per state.
    """
    states = system_state if isinstance(system_state, (list, tuple)) else [system_state]
    B = len(states)
    f0 = states[0]["f0"]
    A = max(int(s.get("areas", 1)) for s in states)
    comp = [_compile(s, f0) for s in states]
    n = max(len(c["K"]) for c in comp)
    keys = ["K", "db", "T1", "rate", "up", "dn", "An", "Ad", "Bn", "Bd", "area"]
    P = _pad_stack(comp, keys, n)
    area = P["area"].astype(int)
    onehot = np.zeros((B, n, A))
    np.put_along_axis(onehot, area[:, :, None], 1.0, axis=2)

    # per-area inertia, load, damping
    E = np.zeros((B, A))
    Pload = np.zeros((B, A))
    for b, s in enumerate(states):
        for a in range(A):
            E[b, a] = inertia_MWs(s, a) if A > 1 else inertia_MWs(s)
        frac = s.get("pload_area") or [1.0 / A] * A
        Pload[b] = np.asarray(frac) * s["load_MW"]
    E = np.maximum(E, 1.0)
    D = np.array([s["D"] for s in states])[:, None]

    events = _as_events(loss_MW, B)
    # FFR resources (pad)
    def lst(key):
        return [s.get(key) or [] for s in states]
    ffr = lst("ffr")
    nf = max(1, max(len(x) for x in ffr))
    F = {k: np.zeros((B, nf)) for k in ["MW", "trig", "delay", "ramp", "area"]}
    F["trig"][:] = -np.inf
    for b, xs in enumerate(ffr):
        for j, x in enumerate(xs):
            F["MW"][b, j] = x["MW"]
            F["trig"][b, j] = x.get("trigger_Hz", f0 - 0.15) - f0
            F["delay"][b, j] = x.get("delay_s", 0.25)
            F["ramp"][b, j] = max(x.get("ramp_s", 0.1), 1e-3)
            F["area"][b, j] = x.get("area", 0)
    F["area"] = F["area"].astype(int)
    si = lst("synthetic_inertia")
    ns = max(1, max(len(x) for x in si))
    S = {k: np.zeros((B, ns)) for k in ["K", "MW", "area"]}
    for b, xs in enumerate(si):
        for j, x in enumerate(xs):
            S["K"][b, j] = 2 * x.get("H_s", 4.0) * x.get("MVA", 0.0) / f0       # MW per Hz/s
            S["MW"][b, j] = x.get("MW_max", 0.1 * x.get("MVA", 0.0))
            S["area"][b, j] = x.get("area", 0)
    S["area"] = S["area"].astype(int)
    prot = lst("protection")
    npr = max(1, max(len(x) for x in prot))
    PR = {k: np.zeros((B, npr)) for k in ["MW", "kind", "thr", "delay", "win", "area", "H"]}
    PR["thr"][:] = np.inf
    for b, xs in enumerate(prot):
        for j, x in enumerate(xs):
            PR["MW"][b, j] = x["MW"]
            PR["kind"][b, j] = 0 if x["kind"] == "rocof" else 1
            PR["thr"][b, j] = x["threshold"] if x["kind"] == "rocof" else x["threshold"] - f0
            PR["delay"][b, j] = x.get("delay_s", 0.5)
            PR["win"][b, j] = x.get("window_s", 0.5 if x["kind"] == "rocof" else 0.0)
            PR["area"][b, j] = x.get("area", 0)
            PR["H"][b, j] = x.get("H_MWs", 0.0)
    PR["area"] = PR["area"].astype(int)
    uf = lst("ufls")
    nu = max(1, max(len(x) for x in uf))
    U = {k: np.zeros((B, nu)) for k in ["thr", "frac", "delay", "MW"]}
    U["thr"][:] = -np.inf
    for b, xs in enumerate(uf):
        for j, x in enumerate(xs):
            U["thr"][b, j] = x["f_Hz"] - f0
            U["frac"][b, j] = x.get("frac", 0.0)
            U["MW"][b, j] = x.get("MW", np.nan)
            U["delay"][b, j] = x.get("delay_s", 0.3)
    U_MW = np.where(np.isnan(U["MW"]), U["frac"] * np.array([s["load_MW"] for s in states])[:, None], U["MW"])
    sec = [s.get("secondary") for s in states]
    SEC_MW = np.array([(x or {}).get("MW", 0.0) for x in sec])
    SEC_T = np.array([(x or {}).get("T_s", 60.0) for x in sec])
    SEC_db = np.array([(x or {}).get("deadband_Hz", 0.0) for x in sec])
    ties = [s.get("ties") or [] for s in states]
    ntie = max(1, max(len(x) for x in ties))
    TI = {k: np.zeros((B, ntie)) for k in ["a", "b", "T"]}
    for b, xs in enumerate(ties):
        for j, x in enumerate(xs):
            TI["a"][b, j], TI["b"][b, j], TI["T"][b, j] = x["a"], x["b"], x["T_MW_per_rad"]
    TI["a"] = TI["a"].astype(int)
    TI["b"] = TI["b"].astype(int)

    # ---- precomputed incidence (B, item, area) and event tables
    def onehot_of(idx, m):
        oh = np.zeros((B, m, A))
        np.put_along_axis(oh, idx[:, :, None], 1.0, axis=2)
        return oh
    F_oh = onehot_of(F["area"], nf)
    TIa_oh = onehot_of(TI["a"], ntie)
    TIb_oh = onehot_of(TI["b"], ntie)
    S_oh = onehot_of(S["area"], ns)
    ksi = np.einsum("bj,bja->ba", S["K"], S_oh)
    msi = np.einsum("bj,bja->ba", S["MW"], S_oh)
    has_si = bool(ksi.any())
    has_ffr = bool(F["MW"].any())
    has_prot = bool(np.isfinite(PR["thr"]).any())
    has_sec = bool(SEC_MW.any())
    PR_oh = onehot_of(PR["area"], npr)
    ne = max(len(ev) for ev in events)
    EV_t = np.full((B, ne), np.inf); EV_MW = np.zeros((B, ne)); EV_H = np.zeros((B, ne)); EV_a = np.zeros((B, ne), int)
    for bb, ev in enumerate(events):
        for j, e in enumerate(ev):
            EV_t[bb, j], EV_MW[bb, j], EV_a[bb, j], EV_H[bb, j] = e
    EV_oh = onehot_of(EV_a, ne)
    ev_times = np.unique(EV_t[np.isfinite(EV_t)])
    U_max = U["thr"].max()
    lshare = Pload / np.maximum(Pload.sum(1, keepdims=True), 1e-9)

    # states
    df = np.zeros((B, A))
    x1 = np.zeros((B, n))
    xa = np.zeros((B, n))
    xb = np.zeros((B, n))
    ptie = np.zeros((B, ntie))
    ffr_t = np.full((B, nf), np.inf)       # time when FFR trigger condition started
    ffr_on = np.full((B, nf), np.inf)      # activation time
    ffr_p = np.zeros((B, nf))
    pr_t = np.full((B, npr), np.inf)
    pr_trip = np.full((B, npr), np.inf)
    uf_t = np.full((B, nu), np.inf)
    uf_trip = np.full((B, nu), np.inf)
    sec_x = np.zeros(B)
    si_flt = np.zeros((B, A))
    psi_a = np.zeros((B, A))
    pffr = np.zeros((B, A))
    tie_out = np.zeros((B, A))
    shed = np.zeros((B, A))
    lost_ev = np.zeros((B, A)); lostH_ev = np.zeros((B, A))
    lost_pr = np.zeros((B, A)); lostH_pr = np.zeros((B, A))

    nsteps = int(round(t_end / h))
    every = max(1, int(round(dt / h)))
    nout = nsteps // every + 1
    wsteps = np.maximum(1, np.round(PR["win"] / h).astype(int))
    hist_len = int(wsteps.max()) + 2
    hist = np.zeros((hist_len, B, A))   # ring buffer of df for RoCoF relays
    out_f = np.zeros((nout, B, A))
    rec_gov = np.zeros((nout, B, n)) if record else None
    out_ffr = np.zeros((nout, B))
    out_si = np.zeros((nout, B))
    out_damp = np.zeros((nout, B))
    out_shed = np.zeros((nout, B))
    out_E = np.zeros((nout, B))

    # lead-lag discretisation: y = (Tn/Td) u + (1 - Tn/Td) z,  dz/dt = (u - z)/Td  (bypass when Td == 0)
    def lag_coef(T):
        with np.errstate(divide="ignore", over="ignore"):
            return np.where(T > 0, 1.0 - np.exp(-h / np.where(T > 0, T, 1.0)), 1.0)
    a1 = lag_coef(P["T1"])
    aA = lag_coef(P["Ad"])
    aB = lag_coef(P["Bd"])
    rA = np.where(P["Ad"] > 0, P["An"] / np.where(P["Ad"] > 0, P["Ad"], 1.0), 1.0)
    rB = np.where(P["Bd"] > 0, P["Bn"] / np.where(P["Bd"] > 0, P["Bd"], 1.0), 1.0)
    useA = bool((P["Ad"] > 0).any()); useB = bool((P["Bd"] > 0).any())
    aS = 1.0 - np.exp(-h / 0.1)            # synthetic inertia derivative filter, 100 ms
    aSec = 1.0 - np.exp(-h / np.maximum(SEC_T, h))
    bidx = np.arange(B)
    rate_h = P["rate"] * h
    Kp, dbp, up, dn = P["K"], P["db"], P["up"], P["dn"]
    single = A == 1
    D_f0 = D / f0
    c_f = f0 / 2.0
    next_ev = 0
    Eeff = E.copy()

    k_out = 0
    for k in range(nsteps + 1):
        t = k * h
        # scheduled losses
        if next_ev < len(ev_times) and ev_times[next_ev] <= t + 1e-12:
            act = EV_t <= t + 1e-12
            lost_ev = np.einsum("bj,bja->ba", EV_MW * act, EV_oh)
            lostH_ev = np.einsum("bj,bja->ba", EV_H * act, EV_oh)
            Eeff = np.maximum(E - lostH_ev - lostH_pr, 1.0)
            while next_ev < len(ev_times) and ev_times[next_ev] <= t + 1e-12:
                next_ev += 1
        df_unit = df[:, :1] if single else np.take_along_axis(df, area, axis=1)          # (B, n)
        # governors
        dfd = np.sign(df_unit) * np.maximum(np.abs(df_unit) - dbp, 0.0)
        dx = np.clip((-Kp * dfd - x1) * a1, -rate_h, rate_h)
        x1 = np.clip(x1 + dx, -dn, up)
        gov = x1
        if useA:
            ya = rA * x1 + (1 - rA) * xa
            xa = xa + (x1 - xa) * aA
            gov = ya
        if useB:
            yb = rB * gov + (1 - rB) * xb
            xb = xb + (gov - xb) * aB
            gov = yb
        # FFR
        if has_ffr:
            dfr = df[:, :1] if single else np.take_along_axis(df, F["area"], axis=1)
            below = dfr <= F["trig"]
            ffr_t = np.where(below & ~np.isfinite(ffr_t), t, np.where(below | np.isfinite(ffr_on), ffr_t, np.inf))
            newon = np.isfinite(ffr_t) & ~np.isfinite(ffr_on) & (t - ffr_t >= F["delay"] - 1e-12)
            ffr_on = np.where(newon, t, ffr_on)
            ffr_p = np.where(np.isfinite(ffr_on), F["MW"] * np.clip((t - ffr_on) / F["ramp"], 0.0, 1.0), 0.0)
            pffr = ffr_p.sum(1, keepdims=True) if single else np.einsum("bj,bja->ba", ffr_p, F_oh)
        # protection (RoCoF relays use df change over a window; UF relays use df)
        if has_prot:
            hist[k % hist_len] = df
            kk = np.maximum(k - wsteps, 0) % hist_len
            prev = hist[kk, bidx[:, None], PR["area"]]
            dfp = df[bidx[:, None], PR["area"]]
            rocof = (dfp - prev) / (np.minimum(wsteps, max(k, 1)) * h)
            cond = np.where(PR["kind"] == 0, np.abs(rocof) >= PR["thr"], dfp <= PR["thr"]) & ~np.isfinite(pr_trip)
            pr_t = np.where(cond & ~np.isfinite(pr_t), t, np.where(cond, pr_t, np.inf))
            tr = np.isfinite(pr_t) & (t - pr_t >= PR["delay"] - 1e-12) & ~np.isfinite(pr_trip)
            if tr.any():
                pr_trip = np.where(tr, t, pr_trip)
                lost_pr = lost_pr + np.einsum("bj,bja->ba", PR["MW"] * tr, PR_oh)
                lostH_pr = lostH_pr + np.einsum("bj,bja->ba", PR["H"] * tr, PR_oh)
                Eeff = np.maximum(E - lostH_ev - lostH_pr, 1.0)
        fcoi = df[:, 0] if single else (df * Eeff).sum(1) / Eeff.sum(1)
        # UFLS on the centre-of-inertia frequency
        if fcoi.min() <= U_max or np.isfinite(uf_t).any():
            cond = (fcoi[:, None] <= U["thr"]) & ~np.isfinite(uf_trip)
            uf_t = np.where(cond & ~np.isfinite(uf_t), t, np.where(cond, uf_t, np.inf))
            tr = np.isfinite(uf_t) & (t - uf_t >= U["delay"] - 1e-12) & ~np.isfinite(uf_trip)
            if tr.any():
                uf_trip = np.where(tr, t, uf_trip)
                shed = shed + (U_MW * tr).sum(1)[:, None] * lshare
        # secondary control: first-order approach to its full MW while frequency is outside its deadband
        if has_sec:
            act = np.abs(fcoi) > SEC_db
            sec_x = np.where(act, sec_x + aSec * (-np.sign(fcoi) * SEC_MW - sec_x), sec_x)
        # power balance per area
        gen = gov.sum(1, keepdims=True) if single else np.einsum("bn,bna->ba", gov, onehot)
        damp = -D_f0 * np.maximum(Pload - shed, 0.0) * df            # >0 when frequency is low (load falls)
        if not single:
            tie_out = np.einsum("bj,bja->ba", ptie, TIa_oh) - np.einsum("bj,bja->ba", ptie, TIb_oh)
        if has_si:
            psi_a = np.clip(-ksi * si_flt, -msi, msi)
        bal = gen + pffr + psi_a + shed + damp - lost_ev - lost_pr - tie_out
        if has_sec:
            bal = bal + sec_x[:, None] * lshare
        dfdt = c_f / Eeff * bal
        if k % every == 0 and k_out < nout:
            out_f[k_out] = df
            if record:
                rec_gov[k_out] = gov
            out_ffr[k_out] = ffr_p.sum(1)
            out_si[k_out] = psi_a.sum(1)
            out_damp[k_out] = damp.sum(1)
            out_shed[k_out] = shed.sum(1)
            out_E[k_out] = Eeff.sum(1)
            k_out += 1
        if has_si:
            si_flt = si_flt + (dfdt - si_flt) * aS
        if not single:
            ptie = ptie + h * 2 * np.pi * TI["T"] * (df[bidx[:, None], TI["a"]] - df[bidx[:, None], TI["b"]])
        df = df + h * dfdt
    lost = lost_ev + lost_pr
    lostH = lostH_ev + lostH_pr

    tt = np.arange(nout) * h * every
    Etot = out_E
    fcoi = (out_f * np.repeat(Etot[:, :, None] / A, A, axis=2)).sum(2) / np.maximum(Etot, 1e-9) if A > 1 else out_f[:, :, 0]
    if A > 1:
        # E per area is constant except for lost inertia; recompute weights
        w = np.maximum(E - lostH, 1.0)
        fcoi = (out_f * w[None]).sum(2) / w.sum(1)[None]
    f = f0 + fcoi.T                       # (B, nt)
    i05 = int(round(0.5 / (h * every)))
    res = []
    for b in range(B):
        fb = f[b]
        i_n = int(np.argmin(fb))
        last = tt >= tt[-1] - 10.0
        r = {"t": tt, "f": fb, "f_area": f0 + out_f[:, b, :].T, "f0": f0,
             "rocof_Hz_s": float((fb[i05] - fb[0]) / tt[i05]) if i05 > 0 else float("nan"),
             "rocof_init_Hz_s": float(-sum(e[1] for e in events[b] if e[0] <= 1e-9) * f0 / (2 * E[b].sum())),
             "nadir_Hz": float(fb[i_n]), "t_nadir_s": float(tt[i_n]), "settling_Hz": float(fb[last].mean()),
             "zenith_Hz": float(fb.max()), "inertia_MWs": float(E[b].sum()),
             "ffr_MW": out_ffr[:, b], "si_MW": out_si[:, b], "damping_MW": out_damp[:, b], "shed_MW": out_shed[:, b],
             "ufls": [dict(stage=int(j), t_s=float(uf_trip[b, j]), MW=float(U_MW[b, j]))
                      for j in range(nu) if np.isfinite(uf_trip[b, j])],
             "trips": [dict(index=int(j), t_s=float(pr_trip[b, j]), MW=float(PR["MW"][b, j]))
                       for j in range(npr) if np.isfinite(pr_trip[b, j])],
             "lost_MW": float(lost[b].sum())}
        if record:
            r["gov_MW"] = rec_gov[:, b, :len(states[b]["units"])]
            r["gov_total_MW"] = rec_gov[:, b, :].sum(1)
            r["unit_tech"] = [u.get("tech") for u in states[b]["units"]]
        res.append(r)
    if B == 1 and not return_batch and not isinstance(system_state, (list, tuple)):
        return res[0]
    return res


def event_metrics(t, f, f0=None):
    """Standard metrics from a measured or simulated trace (t s from event start, f Hz)."""
    t = np.asarray(t, float); f = np.asarray(f, float)
    f0 = f0 or (50.0 if abs(f[0] - 50) < 5 else 60.0)
    pre = f[t < 0].mean() if np.any(t < 0) else f[0]
    post = t >= 0
    i_n = np.argmin(np.where(post, f, np.inf))
    m = (t >= 0) & (t <= 0.5)
    rocof = np.polyfit(t[m], f[m], 1)[0] if m.sum() >= 2 else np.nan
    return {"f_pre_Hz": float(pre), "nadir_Hz": float(f[i_n]), "t_nadir_s": float(t[i_n]),
            "rocof_Hz_s": float(rocof), "delta_nadir_Hz": float(f[i_n] - pre)}


def aggregate_units(units, by=("tech",), db=0.036, f0=60.0):
    """Merge unit dicts into one group per key (sums MW / rating / MVA, MVA-weighted H, summed per-unit headroom)
    to speed up event simulation.  Units in a group should share droop and dynamics (they do when grouped by tech)."""
    groups = {}
    for u in units:
        key = tuple(u.get(k) for k in by) + (u.get("area", 0),)
        H, mva = _unit_H_mva(u)
        P = _compile({"units": [u], "db": db}, f0)
        g = groups.setdefault(key, dict(tech=u.get("tech"), MW=0.0, pmax_MW=0.0, pmin_MW=0.0, mva=0.0, HS=0.0,
                                        area=u.get("area", 0), n=0, governor=dict(u.get("governor") or {}),
                                        _up=0.0, _dn=0.0, _K=0.0))
        g["MW"] += u.get("MW", 0.0)
        g["pmax_MW"] += u.get("pmax_MW", u.get("MW", 0.0))
        g["pmin_MW"] += u.get("pmin_MW", 0.0) or 0.0
        g["mva"] += mva
        g["HS"] += H * mva
        g["n"] += u.get("n", 1)
        g["_up"] += P["up"][0]
        g["_dn"] += P["dn"][0]
        g["_K"] += P["K"][0]
    out = []
    for g in groups.values():
        g["H_s"] = g.pop("HS") / max(g["mva"], 1e-9)
        up, dn, K = g.pop("_up"), g.pop("_dn"), g.pop("_K")
        g["governor"].update(up_MW=up, dn_MW=dn, enabled=K > 0)
        if K > 0:
            g["governor"]["droop"] = g["pmax_MW"] / (K * f0)
        out.append(g)
    return out
