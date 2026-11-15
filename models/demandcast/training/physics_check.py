"""Independent checks of the DemandCast physics -> checks/physics.json"""
import json, os, numpy as np
from scipy.integrate import solve_ivp
from common import *
import buildings as B, renew as R
from gridsim.renewables import farm_curve
from gridsim.demand import specific_humidity

out = {}
# 1. envelope RC response: hourly exact-discretised filter vs an ODE solver (zero-order-hold input)
tau = 7.0; H = 240
rng = np.random.default_rng(0)
Tout = 15 + 8 * np.sin(np.arange(H) * 2 * np.pi / 24) + rng.normal(0, 1, H)
Tf = B.rc_filter(Tout, tau, x0=Tout[0])
sol = solve_ivp(lambda t, x: (Tout[min(int(t), H - 1)] - x) / tau, (0, H), [Tout[0]], t_eval=np.arange(1, H + 1),
                max_step=0.01, rtol=1e-10, atol=1e-10)
out['rc_filter_vs_solve_ivp_max_abs_K'] = float(np.max(np.abs(Tf - sol.y[0])))
# step response vs closed form
st = B.rc_filter(np.full(48, 10.0), tau, x0=0.0)
out['rc_step_vs_closed_form_max_abs_K'] = float(np.max(np.abs(st - 10 * (1 - np.exp(-np.arange(1, 49) / tau)))))

# 2. stock-averaged hinge E[max(Tset - T, 0)], Tset ~ N(mu, s): closed form vs Monte Carlo (1e6 draws)
mu, s = 19.0, 2.5
Ts = np.linspace(0, 30, 13)
mc = np.array([np.mean(np.maximum(rng.normal(mu, s, 1_000_000) - t, 0)) for t in Ts])
out['ghinge_vs_monte_carlo_max_abs_K'] = float(np.max(np.abs(B.ghinge(mu - Ts, s) - mc)))

# 3. degree-day limit: no thermal mass, no spread, constant gains -> heating energy = UA x HDD(balance point)
p = dict(B.PARAMS, tau1=0.05, tau2=0.05, sd_h=1e-3, sd_c=1e-3, a_sol=0, g_win=0, g_int=0, setback=0, setup=0,
         e_res=1.0, e_hp=0.0, ac=0.0, light=0, wh=0, P0=1000.0, ua=0.1)
Tday = np.repeat(np.linspace(-10, 30, 41), 24)
import pandas as pd
cal = B.calendar(pd.date_range('2019-01-07', periods=len(Tday), freq='h', tz='UTC'), 'UTC')
d = B.demand(dict(T=Tday, q=np.full(len(Tday), 5.0), ghi=np.zeros(len(Tday)), btm_cf=np.zeros(len(Tday))), cal, p)
hdd = np.maximum(p['mu_h'] - Tday, 0).sum() / 24
out['degree_day_heating_MWh'] = float(d['heat'].sum())
out['degree_day_formula_MWh'] = float(p['ua'] * p['P0'] * hdd * 24)

# 4. psychrometrics: specific humidity vs PsychroLib (ASHRAE Handbook formulas)
import psychrolib as pl
pl.SetUnitSystem(pl.SI)
errs = []
for T in (0, 10, 20, 30, 40):   # psychrolib switches to saturation over ice below 0 C; RH in MERRA-2/ASOS is w.r.t. water
    for rh in (0.2, 0.5, 0.9):
        w = pl.GetHumRatioFromRelHum(T, rh, 101325.0)
        q_ref = w / (1 + w)
        errs.append(abs(specific_humidity(T, rh, 1013.25) - q_ref) / q_ref)
out['specific_humidity_vs_psychrolib_max_rel'] = float(max(errs))
# latent factor: h_fg / c_p x ventilation share 0.3 / 1000 (K per g/kg)
out['latent_factor_K_per_gkg'] = float(2501.0 / 1.006 * 0.3 / 1000)

# 5. COP curves vs typical rating points (AHRI 210/240: heat pump COP ~3.4 at 8.3 C, ~2.0-2.3 at -8.3 C;
#    central AC EER ~11-12 Btu/Wh = COP 3.2-3.5 at 35 C)
out['cop_hp_8.3C'] = float(B.cop_hp(8.3)); out['cop_hp_-8.3C'] = float(B.cop_hp(-8.3)); out['cop_ac_35C'] = float(B.cop_ac(35.0))

# 6. wind farm curve lookup table vs GridSim's farm_curve evaluated directly
v = rng.uniform(0, 30, 20000); sp = rng.uniform(160, 460, 20000)
direct = np.array([farm_curve(np.array([vv]), s)[0] for vv, s in zip(v[:3000], sp[:3000])])
e = np.abs(R.farm_cf(v[:3000], sp[:3000]) - direct); away = np.abs(v[:3000] - 25) > 0.1
out['wind_table_vs_gridsim_mean_abs_cf'] = float(e.mean()); out['wind_table_vs_gridsim_max_abs_cf_away_from_cutout'] = float(e[away].max()); out['wind_table_vs_gridsim_max_abs_cf_at_25ms_cutout'] = float(e.max())

# 7. GridSim's own renewables and demand checks this builds on
out['gridsim_renewables_check'] = json.load(open(os.path.join(GRID, 'checks', 'renewables.json')))
os.makedirs(f'{ROOT}/checks', exist_ok=True)
jdump(out, f'{ROOT}/checks/physics.json')
print(json.dumps({k: v for k, v in out.items() if k != 'gridsim_renewables_check'}, indent=1))
