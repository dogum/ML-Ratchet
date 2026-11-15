"""GridPulse state sampler: a wrapper around gridsim.generate_states that rescales each random state to a realistic
GB (50 Hz) or ERCOT (60 Hz) operating point and adds each system's own frequency services.

Every knob a calibration rung may move lives in PRIORS[version]; the bank records the version.
Real inputs used: GB half-hourly (demand, outturn inertia, IBR share) drawn jointly from the TRAIN-INPUT years
2018/2020/2022 only (NESO inertia + generation mix); ERCOT load from EIA-930 train-input years. No dev/exam labels.
"""
from common import *
import copy
import gridsim as gs
from gridsim.frequency import _compile, inertia_MWs

SYNC = {"nuclear", "coal_steam", "gas_cc", "gas_ct", "gas_steam", "hydro_res", "gas_ice", "oil", "biomass",
        "hydro_ror", "pumped", "geothermal", "other"}

PRIORS = {
    "p0": dict(
        GB=dict(loss=(150, 2000), D_med=0.75, D_sig=0.35, D_clip=(0.3, 2.0), gov_head=(300, 2500),
                dc=(0, 1800), dc_R=0.01, dc_T=(0.1, 0.5), ffr_p=0.5, ffr=(0, 600), ffr_trig=49.7,
                sec_frac=(0.0, 1.0), sec_T=(20, 60), db=0.015, E_mix=0.7, E_lu=(90e3, 420e3)),
        ERCO=dict(loss=(250, 2800), D_med=2.5, D_sig=0.3, D_clip=(1.0, 5.0), gov_head=(1000, 5000),
                  dc=(0, 2000), dc_R=(0.01, 0.05), dc_T=(0.1, 0.5), ffr_p=0.7, ffr=(0, 900), ffr_trig=59.85,
                  lr_p=0.9, lr=(500, 1500), lr_trig=59.7, sec_frac=(0.0, 0.0), sec_T=(20, 60), db=None,
                  h_med=5.0, h_sig=0.22, E_clip=(100e3, 420e3)),
    ),
}
PRIORS["p0"]["GB"]["loss_ramp_p"] = 0.0
PRIORS["p0"]["ERCO"]["loss_ramp_p"] = 0.0


def derive(base, name, GB=None, ERCO=None):
    """New prior version = base with some GB / ERCOT knobs changed."""
    PRIORS[name] = {"GB": dict(PRIORS[base]["GB"], **(GB or {})), "ERCO": dict(PRIORS[base]["ERCO"], **(ERCO or {}))}
    return name


# p1: GB response calibrated on the dev twins (2019 events: governor headroom 300-1500 MW, no Dynamic Containment)
# plus a DC-era component (2020+) built from NESO's service specs (not calibratable: no GB dev event after 2019).
derive("p0", "p1", GB=dict(gov_head=(300, 1500), D_med=0.75, ffr=(0, 300), ffr_p=0.7, dc_p=0.6, dc=(400, 1800),
                           dc_knee=True, dc_T=(0.2, 0.8), dm=(0, 500), dr=(0, 600), sec_frac=(0.0, 0.3), sec_T=(30, 90)),
       # ERCOT: dev twins (5 events) were 8-29 % too shallow with the effective damping 2.5 %/% (Phase 1 fit on
       # dispatch-derived states); with these random states damping 1.0-1.5 %/% brings them to ~0.8-1.0. Battery
       # primary response (5 % droop, BAL-001-TRE) and FFR are an exam-era component (2023+), half of the states.
       ERCO=dict(D_med=1.0, D_sig=0.35, D_clip=(0.4, 4.0), gov_head=(1000, 3500), dc_p=0.5, dc=(500, 4000),
                 dc_R=(0.03, 0.05), ffr_p=0.5, ffr=(0, 750)))
# p2: p1 + realistic loss shapes: 30 % of losses ramp out over 0.5-3 s (unit runbacks, staged trips; the 11 Jul 2019
# dev event falls more slowly than its label implies), label = total MW lost
derive("p1", "p2", GB=dict(loss_ramp_p=0.3), ERCO=dict(loss_ramp_p=0.3))

_GBOP = None
_ERLOAD = None


def gb_operating_points():
    """Joint (demand MW, inertia MW s, IBR share) per half hour of the train-input years."""
    global _GBOP
    if _GBOP is None:
        import griddata as gd
        yrs = gd.splits()['frequency_noise']['years']
        I = gd.gb_inertia(); I = I[I.time_utc.dt.year.isin(yrs)].set_index('time_utc')
        M = gd.gb_genmix(); M = M[M.time_utc.dt.year.isin(yrs)].set_index('time_utc')
        j = I.join(M, how='inner').dropna(subset=['inertia_outturn_GVAs', 'GENERATION'])
        ibr = (j.WIND + j.SOLAR + j.IMPORTS + j.STORAGE.clip(lower=0)) / j.GENERATION
        _GBOP = np.c_[j.GENERATION.values, j.inertia_outturn_GVAs.values * 1e3, ibr.clip(0, 0.95).values]
        _GBOP = _GBOP[(_GBOP[:, 0] > 10000) & (_GBOP[:, 1] > 60e3)]
    return _GBOP


def ercot_loads():
    global _ERLOAD
    if _ERLOAD is None:
        import griddata as gd
        yrs = gd.splits()['frequency_noise']['years']
        d = gd.eia930('ERCO', split='train_inputs', columns=['d'])['d'].dropna()
        _ERLOAD = d.values[d.values > 20000]
    return _ERLOAD


def _scale_state(st, s):
    st = copy.deepcopy(st)
    for u in st["units"]:
        for k in ("MW", "pmax_MW", "pmin_MW", "mva"):
            if u.get(k) is not None:
                u[k] = u[k] * s
        g = u.get("governor") or {}
        for k in ("up_MW", "dn_MW"):
            if g.get(k) is not None:
                g[k] = g[k] * s
    st["load_MW"] *= s
    st["H_extra_MWs"] = st.get("H_extra_MWs", 0.0) * s
    return st


def sample(n, seed, system="GB", version="p0", fix=None, reuse=4):
    """n states of one system. Returns (states, loss_events, labels DataFrame).
    fix: optional dict of demand_MW / inertia_MWs / loss_MW / ibr_share held fixed (twin tests)."""
    fix = fix or {}
    pr = PRIORS[version][system]
    rng = np.random.default_rng(seed)
    # each generate_states structure (unit mix, governors) is reused for `reuse` different operating points
    nb = int(np.ceil(n / reuse))
    base, _, blab = gs.generate_states(nb, seed=int(seed), systems=(system,))
    base = [base[i % len(base)] for i in range(n)]
    f0 = 50.0 if system == "GB" else 60.0
    states, events, labels = [], [], []
    for i, st0 in enumerate(base):
        # --- operating point
        if system == "GB":
            if rng.random() < pr["E_mix"]:
                op = gb_operating_points()[rng.integers(len(gb_operating_points()))]
                dem, E_t, ibr = op[0] * rng.uniform(0.95, 1.05), op[1] * rng.uniform(0.9, 1.1), op[2]
            else:
                dem = rng.uniform(16000, 46000); E_t = np.exp(rng.uniform(*np.log(pr["E_lu"])))
                ibr = rng.uniform(0.05, 0.8)
        else:
            dem = ercot_loads()[rng.integers(len(ercot_loads()))] * rng.uniform(0.95, 1.05)
            E_t = float(np.clip(dem * np.exp(np.log(pr["h_med"]) + pr["h_sig"] * rng.normal()), *pr["E_clip"]))
            ibr = rng.uniform(0.05, 0.75)
        dem = fix.get("demand_MW", dem); E_t = fix.get("inertia_MWs", E_t); ibr = fix.get("ibr_share", ibr)
        st = _scale_state(st0, dem / st0["load_MW"])
        # drop generate_states' battery droop units and FFR; system services are added below
        st["units"] = [u for u in st["units"] if u["tech"] != "battery"]
        st["ffr"] = []; st["synthetic_inertia"] = []
        E_now = inertia_MWs(st)
        k = E_t / max(E_now, 1.0)
        for u in st["units"]:
            if u["tech"] in SYNC:
                u["H_s"] = (u.get("H_s") or 0.0) * k
        st["H_extra_MWs"] = st.get("H_extra_MWs", 0.0) * k
        # --- governors: deadband and total headroom
        for u in st["units"]:
            g = u.setdefault("governor", {})
            if pr["db"] is not None:
                g["deadband_Hz"] = pr["db"]
            else:
                g["deadband_Hz"] = None          # system per-tech default (ERCOT BAL-001-TRE-1)
        # aggregate units per technology (GridSim's aggregate_units: sums gains and headroom), so a batch is not
        # padded to the largest fleet (generate_states fleets have up to ~1,300 units)
        dbt = gs.frequency.SYSTEMS[system].get("db_by_tech", {})
        for u in st["units"]:
            if u["governor"].get("deadband_Hz") is None:
                u["governor"]["deadband_Hz"] = dbt.get(u["tech"], gs.frequency.SYSTEMS[system]["db"])
        st["units"] = gs.frequency.aggregate_units(st["units"], by=("tech",), f0=f0)
        P = _compile(st, f0)
        head_t = rng.uniform(*pr["gov_head"])
        up = P["up"]; tot = up.sum()
        for j, u in enumerate(st["units"]):
            if up[j] > 0:
                u["governor"]["up_MW"] = float(up[j] * head_t / max(tot, 1e-9))
        # --- inverter services
        dc = rng.uniform(*pr["dc"]) if rng.random() < pr.get("dc_p", 1.0) else 0.0
        dcR = pr["dc_R"] if np.isscalar(pr["dc_R"]) else rng.uniform(*pr["dc_R"])
        db_dc = pr["db"] if pr["db"] is not None else 0.0166
        dm = dr = 0.0
        if dc > 1 and pr.get("dc_knee"):
            # NESO Dynamic Containment (low): +-0.015 Hz deadband, 5 % of contracted MW by 0.2 Hz, 100 % at 0.5 Hz,
            # full delivery within 1 s; Dynamic Moderation: 100 % at 0.2 Hz within 1 s; Dynamic Regulation: 100 % at
            # 0.2 Hz, slower (10 s). Piecewise-linear droop = two droop units with different deadbands.
            T = float(rng.uniform(*pr["dc_T"]))
            def bat(MW, db, full_at, T):
                R = (full_at - db) / f0
                return dict(tech="battery", MW=0.0, pmax_MW=MW, pmin_MW=-MW, H_s=0.0, mva=MW,
                            governor=dict(enabled=True, droop=R, deadband_Hz=db, up_MW=MW, dn_MW=MW, T_s=T))
            st["units"].append(bat(0.05 * dc, 0.015, 0.2, T))
            st["units"].append(bat(0.95 * dc, 0.2, 0.5, T))
            dm = rng.uniform(*pr["dm"]); dr = rng.uniform(*pr["dr"])
            if dm > 1:
                st["units"].append(bat(dm, 0.015, 0.2, T))
            if dr > 1:
                st["units"].append(bat(dr, 0.015, 0.2, float(rng.uniform(2.0, 10.0))))
        elif dc > 1:
            st["units"].append(dict(tech="battery", MW=0.0, pmax_MW=dc, pmin_MW=-dc, H_s=0.0, mva=dc,
                                    governor=dict(enabled=True, droop=dcR, deadband_Hz=db_dc, up_MW=dc, dn_MW=dc,
                                                  T_s=float(rng.uniform(*pr["dc_T"])))))
        ffr = 0.0
        if rng.random() < pr["ffr_p"]:
            ffr = rng.uniform(*pr["ffr"])
            st["ffr"].append(dict(MW=ffr, trigger_Hz=pr["ffr_trig"] + rng.uniform(-0.03, 0.03),
                                  delay_s=float(rng.uniform(0.1, 0.5)), ramp_s=float(rng.uniform(0.05, 0.5))))
        lr = 0.0
        if system == "ERCO" and rng.random() < pr["lr_p"]:
            lr = rng.uniform(*pr["lr"])
            st["ffr"].append(dict(MW=lr, trigger_Hz=pr["lr_trig"], delay_s=float(rng.uniform(0.3, 0.5)), ramp_s=0.05))
        # --- damping, secondary, loss
        D = float(np.clip(pr["D_med"] * np.exp(pr["D_sig"] * rng.normal()), *pr["D_clip"]))
        st["D"] = D
        L = float(fix.get("loss_MW", np.exp(rng.uniform(*np.log(pr["loss"])))))
        sec = rng.uniform(*pr["sec_frac"]) * L
        st["secondary"] = dict(MW=sec, T_s=float(rng.uniform(*pr["sec_T"])), deadband_Hz=0.015) if sec > 1 else None
        if rng.random() < pr.get("loss_ramp_p", 0.0):
            Tr = rng.uniform(0.5, 3.0); m = 6
            ev = [(Tr * q / (m - 1), L / m) for q in range(m)]
        else:
            ev = [(0.0, L)]
        st["region"] = system
        states.append(st); events.append(ev)
        P = _compile(st, f0)
        labels.append(dict(system=system, f0=f0, demand_MW=float(st["load_MW"]), ibr_share=float(ibr),
                           inertia_MWs=float(inertia_MWs(st)), loss_MW=L, gov_headroom_MW=float(head_t),
                           dc_MW=float(dc), dm_MW=float(dm), dr_MW=float(dr), headroom_MW=float(P["up"].sum()), ffr_MW=float(ffr + lr),
                           D=D, secondary_MW=float(sec), loss_ramp=int(len(ev) > 1), version=version))
    return states, events, pd.DataFrame(labels)


def run(states, events, t_end=68.0, dt=0.1, h=0.02):
    """frequency_event on a batch with per-state loss lists. (gridsim's _as_events reads a plain list of per-state
    lists as ONE event list; an object array avoids that - upstream candidate.)"""
    ea = np.empty(len(events), dtype=object)
    for i, e in enumerate(events):
        ea[i] = list(e)
    return gs.frequency_event(states, ea, t_end=t_end, dt=dt, h=h, record=False)
