"""Frequency dynamics vs analytical / independent linear-system solutions.  Writes grid/checks/frequency_analytic.json

1. RoCoF = dP f0 / (2 E) for pure inertia (several H, dP)
2. damping only: df_ss = -dP / (D P_load / f0), time constant 2E / (D P_load)
3. each governor/turbine model (gas turbine lag, steam reheat, hydro with transient droop + water hammer, CCGT) in its
   linear range vs scipy.signal.lsim of the same transfer functions (independent integrator): trace RMS, nadir error
4. closed-form nadir of the 2nd-order system (lag governor + damping)
5. deadband + headroom limits: quasi-steady-state frequency vs the analytic piecewise solution
6. step-size convergence of the nadir (h = 0.02 ... 0.001 s)
7. two areas: common steady-state frequency deviation and the inter-area oscillation frequency
"""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, sys, time
sys.path.insert(0, f"{GRID_CODE}/gridsim")
import numpy as np
from scipy import signal
from gridsim.frequency import frequency_event, make_system_state, TECH_DYN

out = {}
f0 = 50.0

# 1 RoCoF
rows = []
for H, mva, dP in [(2, 20000, 500), (4, 30000, 1000), (6, 50000, 1800), (3, 15000, 1320)]:
    st = make_system_state([dict(tech="coal_steam", MW=0, pmax_MW=mva * 0.9, H_s=H, mva=mva, governor=dict(enabled=False))],
                           load_MW=mva, region="GB", D=0, ufls=[])
    r = frequency_event(st, dP, t_end=2.0)
    ana = -dP * f0 / (2 * H * mva)
    rows.append(dict(H_s=H, MVA=mva, dP_MW=dP, rocof_sim=r["rocof_Hz_s"], rocof_analytic=ana,
                     rel_err=abs(r["rocof_Hz_s"] - ana) / abs(ana)))
out["rocof_pure_inertia"] = {"cases": rows, "max_rel_err": max(x["rel_err"] for x in rows)}

# 2 damping only
E, PL, D, dP = 5 * 30000, 30000, 2.0, 1000
st = make_system_state([dict(tech="coal_steam", MW=0, pmax_MW=27000, H_s=5, mva=30000, governor=dict(enabled=False))],
                       load_MW=PL, region="GB", D=D, ufls=[])
r = frequency_event(st, dP, t_end=60)
Dp = D * PL / f0
tau = 2 * E / f0 / Dp
i = np.searchsorted(r["t"], tau)
out["damping_only"] = {"df_ss_sim": float(r["f"][-1] - f0), "df_ss_analytic": -dP / Dp,
                       "df_at_tau_sim": float(r["f"][i] - f0), "df_at_tau_analytic": -dP / Dp * (1 - np.exp(-r["t"][i] / tau))}

# 3 governor models vs scipy lsim.  System: M s df = -dP - Dp df + G(s)(-K df)  ->  df/(-dP) = 1/(M s + Dp + K G(s))
def lti_ref(tech, K, Dp, M, dP, t):
    d = TECH_DYN[tech]
    T1 = d["T1"]
    num, den = np.array([1.0]), np.array([T1, 1.0])          # 1/(1 + s T1)
    for (Tn, Td) in (d["LL_A"], d["LL_B"]):
        if Td > 0:
            num = np.polymul(num, [Tn, 1.0]); den = np.polymul(den, [Td, 1.0])
    # closed loop df = -dP * den / (M s den + Dp den + K num)
    cl_den = np.polyadd(np.polymul([M, 0.0], den), np.polyadd(Dp * den, K * num))
    sys_ = signal.lti(den, cl_den)
    _, y, _ = signal.lsim(sys_, -dP * np.ones_like(t), t)
    return y

gov_rows = []
for tech in ["gas_ct", "coal_steam", "hydro_res", "gas_cc", "battery"]:
    H, mva, pmax, PL, D, dP = 4.0, 40000, 36000, 35000, 1.0, 1500
    M = 2 * H * mva / f0
    unit = dict(tech=tech, MW=pmax * 0.5, pmax_MW=pmax, H_s=H, mva=mva,
                governor=dict(enabled=True, deadband_Hz=0.0, headroom_frac=1.0, up_MW=1e9, dn_MW=1e9))
    if tech == "battery":
        unit["H_s"] = 0.0
        sync = dict(tech="coal_steam", MW=0, pmax_MW=36000, H_s=H, mva=mva, governor=dict(enabled=False))
        units = [sync, dict(unit, pmax_MW=2000, MW=0)]
        K = 2000 / (TECH_DYN["battery"]["R"] * f0)
    else:
        units = [unit]
        K = pmax / (TECH_DYN[tech]["R"] * f0)
    st = make_system_state(units, load_MW=PL, region="GB", D=D, ufls=[], db=0.0)
    # remove rate limits for the linear comparison
    import gridsim.frequency as F
    keep = F.TECH_DYN[tech]["rate"]; F.TECH_DYN[tech]["rate"] = 1e6
    r = frequency_event(st, dP, t_end=60, dt=0.01, h=0.002)
    F.TECH_DYN[tech]["rate"] = keep
    y = lti_ref(tech, K, D * PL / f0, M, dP, r["t"])
    sim = r["f"] - f0
    gov_rows.append(dict(tech=tech, rms_err_Hz=float(np.sqrt(np.mean((sim - y) ** 2))),
                         nadir_sim_Hz=float(sim.min()), nadir_ref_Hz=float(y.min()),
                         t_nadir_sim=float(r["t"][sim.argmin()]), t_nadir_ref=float(r["t"][y.argmin()]),
                         ss_sim=float(sim[-1]), ss_ref=float(y[-1])))
out["governor_vs_lsim"] = gov_rows

# 4 closed-form nadir: M s df = -dP - Dp df + Pm,  T s Pm = -K df - Pm
M, T, K, Dp, dP = 2 * 4 * 40000 / f0, 0.5, 36000 / (0.05 * f0), 1.0 * 35000 / f0, 1500
a2, a1, a0 = M * T, M + Dp * T, Dp + K
wn = np.sqrt(a0 / a2); zeta = a1 / (2 * np.sqrt(a0 * a2))
# df(t) = -dP/a0 * [1 - e^{-zeta wn t}(cos wd t + ((zeta wn - 1/T)/wd) sin wd t)]  (step of (1+sT)/(a2 s^2 + a1 s + a0))
wd = wn * np.sqrt(1 - zeta ** 2)
# zero of derivative: d/dt ... solve numerically on a fine grid of the closed form (no ODE solver involved)
tt = np.linspace(0, 20, 200001)
c = (zeta * wn - T * wn ** 2) / wd
dfc = -dP / a0 * (1 - np.exp(-zeta * wn * tt) * (np.cos(wd * tt) + c * np.sin(wd * tt)))
st = make_system_state([dict(tech="gas_ct", MW=0, pmax_MW=36000, H_s=4, mva=40000,
                             governor=dict(enabled=True, deadband_Hz=0.0, T_s=T, up_MW=1e9, dn_MW=1e9))],
                       load_MW=35000, region="GB", D=1.0, ufls=[], db=0.0)
import gridsim.frequency as F
keep = F.TECH_DYN["gas_ct"]["rate"]; F.TECH_DYN["gas_ct"]["rate"] = 1e6
rr = {}
for h in [0.02, 0.01, 0.005, 0.002, 0.001]:
    r = frequency_event(st, dP, t_end=20, dt=0.001 if h <= 0.001 else h, h=h)
    rr[h] = (float(r["f"].min() - f0), float(r["t"][r["f"].argmin()]))
F.TECH_DYN["gas_ct"]["rate"] = keep
out["closed_form_nadir"] = {"zeta": zeta, "wn_rad_s": wn, "nadir_closed_form_Hz": float(dfc.min()),
                            "t_nadir_closed_form_s": float(tt[dfc.argmin()]),
                            "nadir_sim_by_step": {str(k): v for k, v in rr.items()},
                            "err_at_default_h_0.01_mHz": abs(rr[0.01][0] - dfc.min()) * 1e3,
                            "err_at_h_0.02_mHz": abs(rr[0.02][0] - dfc.min()) * 1e3}

# 5 deadband + headroom: two groups, db 0.015 Hz, headroom 300 MW and 200 MW, loss 1000 MW, damping D=2 on 30 GW
PL, D, dP = 30000, 2.0, 1000
units = [dict(tech="gas_ct", MW=5000, pmax_MW=10000, H_s=4, mva=11000, governor=dict(deadband_Hz=0.015, up_MW=300)),
         dict(tech="gas_ct", MW=5000, pmax_MW=10000, H_s=4, mva=11000, governor=dict(deadband_Hz=0.015, up_MW=200)),
         dict(tech="coal_steam", MW=8000, pmax_MW=9000, H_s=4, mva=10000, governor=dict(enabled=False))]
st = make_system_state(units, load_MW=PL, region="GB", D=D, ufls=[])
r = frequency_event(st, dP, t_end=120)
# analytic: both groups saturate if K (|df| - db) > up; check: K = 10000/(0.05*50) = 4000 MW/Hz
K = 10000 / (0.05 * 50)
df_ss = -(dP - 500) / (D * PL / f0)
sat = K * (abs(df_ss) - 0.015) > 300
out["deadband_headroom"] = {"df_ss_sim": float(r["f"][-1] - f0), "df_ss_analytic": df_ss, "both_saturated": bool(sat),
                            "gov_total_end_MW": float(r["gov_total_MW"][-1])}

# 6 two areas: identical areas, loss in area 0; T = 2000 MW/rad
H, mva, PL, D = 4.0, 20000, 18000, 1.0
Tt = 2000.0
def two_area(gov_on, D):
    units = [dict(tech="gas_ct", MW=8000, pmax_MW=18000, H_s=H, mva=mva, area=a,
                  governor=dict(enabled=gov_on, deadband_Hz=0.0, up_MW=1e9, dn_MW=1e9)) for a in (0, 1)]
    st = make_system_state(units, load_MW=2 * PL, region="GB", D=D, ufls=[], db=0.0, areas=2,
                           ties=[dict(a=0, b=1, T_MW_per_rad=Tt)], pload_area=[0.5, 0.5])
    return frequency_event(st, [(0.0, 1000.0, 0)], t_end=60, dt=0.002, h=0.001)
r = two_area(True, D)
beta = 18000 / (0.05 * f0) + D * PL / f0
fa = r["f_area"]
r0 = two_area(False, 0.0)
d = r0["f_area"][0] - r0["f_area"][1]
# inter-area mode without governors or damping: d2(df1-df2)/dt2 = -2 pi T (1/M1 + 1/M2) (df1-df2)
M1 = 2 * H * mva / f0
w_ana = np.sqrt(2 * np.pi * Tt * (2 / M1))
zc = np.where(np.diff(np.sign(d[(r0["t"] > 0.01) & (r0["t"] < 10)])) != 0)[0]
w_sim = np.pi / np.mean(np.diff(r0["t"][(r0["t"] > 0.01) & (r0["t"] < 10)][zc])) if len(zc) > 2 else float("nan")
out["two_area"] = {"df_ss_area0_sim": float(fa[0][-1] - f0), "df_ss_area1_sim": float(fa[1][-1] - f0),
                   "df_ss_analytic": -1000 / (2 * beta), "interarea_mode_rad_s_sim_no_gov": float(w_sim),
                   "interarea_mode_rad_s_analytic": float(w_ana),
                   "coi_rocof_no_gov_sim": float(r0["rocof_Hz_s"]), "coi_rocof_analytic": -1000 * f0 / (2 * 2 * H * mva)}
units = [dict(tech="gas_ct", MW=8000, pmax_MW=18000, H_s=H, mva=mva)]

# speed
st = make_system_state(units[:1], load_MW=PL, region="GB", D=D)
t0 = time.perf_counter(); frequency_event(st, 1000, t_end=60); t1 = time.perf_counter() - t0
out["speed_single_60s_event_s"] = t1
sts = [make_system_state([dict(units[0], H_s=2 + 4 * np.random.rand())], load_MW=PL, region="GB", D=D) for _ in range(1000)]
t0 = time.perf_counter(); frequency_event(sts, list(500 + 1000 * np.random.rand(1000)), t_end=60, record=False)
out["speed_batch_1000_events_60s_s"] = time.perf_counter() - t0
json.dump(out, open(f"{GRID_CODE}/checks/frequency_analytic.json", "w"), indent=1)
print(json.dumps(out, indent=1))
