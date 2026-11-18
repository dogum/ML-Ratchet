"""Analytic and hand-calculation checks: linear-reservoir recession, unit hydrographs (S-curve, Nash cascade ODE),
Muskingum (X = 0 linear reservoir), Muskingum-Cunge vs the analytic diffusion wave (Hayami), kinematic wave vs the
characteristic solution, degree-day snow by hand, energy-balance snow terms by hand, FAO-56 worked examples (PET),
reservoir mass balance / geometry / hydropower."""
import math

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import gammainc

from _common import write
import watersim as ws
from watersim import routing as rt, snow as sn, pet as pt, reservoir as rv
from watersim import signatures as sg

res = {'what': 'analytic and hand-calculation checks of WaterSim components'}


def flat_forcing(n, P=0.0, T=10.0, PET=0.0, start='2000-01-01'):
    idx = pd.date_range(start, periods=n, freq='D')
    return pd.DataFrame({'P': np.broadcast_to(P, n).astype(float), 'T_mean': np.broadcast_to(T, n).astype(float),
                         'PET': np.broadcast_to(PET, n).astype(float)}, index=idx)


spec = {'id': 'analytic', 'area_km2': 100.0, 'lat': 45.0, 'elev_mean_m': 500.0}

# ------------------------------------------------------------------ 1. linear reservoir recession
n = 400
K2 = 0.05
o = ws.simulate(spec, flat_forcing(n), model='hbv', snow='none', routing='none', pet='forcing',
                params={'K2': K2, 'MAXBAS': 1.0, 'FC': 200.0}, init={'model': {'SM': 0.0, 'SUZ': 0.0, 'SLZ': 100.0}})
q = o['ts']['Q'].values
t = np.arange(1, n + 1)
exact_disc = 100.0 * K2 * (1 - K2) ** (t - 1)
k_cont = -math.log(1 - K2)
storage_cont = 100.0 * np.exp(-k_cont * t)        # continuous linear reservoir S(t) = S0 exp(-k t)
rec = sg.recession(q, min_len=5, drop=2)
o2 = ws.simulate(spec, flat_forcing(n), model='sacsma', snow='none', routing='none', pet='forcing',
                 params={'LZPK': 0.01, 'PCTIM': 0.0, 'ADIMP': 0.0, 'SIDE': 0.0, 'UH_SHAPE': 1.0, 'UH_SCALE': 0.01,
                         'UZTWM': 50.0, 'UZFWM': 40.0, 'LZTWM': 130.0, 'LZFSM': 25.0, 'LZFPM': 300.0, 'RSERV': 0.3},
                 init={'model': {'UZTWC': 50.0, 'UZFWC': 0.0, 'LZTWC': 130.0, 'LZFSC': 0.0, 'LZFPC': 200.0,
                                 'ADIMC': 180.0}})
q2 = o2['ts']['Q'].values
exact2 = 200.0 * 0.01 * 0.99 ** (t - 1)
res['linear_reservoir'] = {
    'hbv_lower_box': {'K2_per_day': K2, 'max_abs_err_vs_discrete_mm': float(np.max(np.abs(q - exact_disc))),
                      'max_rel_err_storage_vs_continuous_exp(-kt)': float(np.max(np.abs(
                          o['ts']['groundwater'].values - storage_cont) / storage_cont)),
                      'k_cont_per_day': k_cont,
                      'recession_fit_k_daily': rec['k_daily'], 'expected_k_daily': 1 - K2,
                      'brutsaert_nieber_b': rec['bn_b'], 'expected_b': 1.0, 'ledger_ok': o['ledger']['ok']},
    'sacsma_primary_baseflow': {'LZPK': 0.01, 'max_abs_err_mm': float(np.max(np.abs(q2 - exact2))),
                                'note': 'UH with shape 1, scale 0.01 d (= no delay); tension stores full so no resupply',
                                'ledger_ok': o2['ledger']['ok']},
}

# ------------------------------------------------------------------ 2. unit hydrographs
a, th = 2.5, 1.3
ordg = rt.gamma_uh(a, th, tol=1e-12, max_len=200)
step = rt.convolve_uh(np.ones(60), ordg)
scurve = gammainc(a, np.arange(1, 61) / th)
tri3 = rt.triangle(3.0)
# Nash cascade: n linear reservoirs (ODE) after an instantaneous unit input at t = 0; outflow volume per day
nres, kk = 3, 1.5


def nash(t, y):
    d = np.empty(nres)
    d[0] = -y[0] / kk
    for i in range(1, nres):
        d[i] = (y[i - 1] - y[i]) / kk
    return d


sol = solve_ivp(nash, (0, 40), np.r_[1.0, np.zeros(nres - 1)], dense_output=True, rtol=1e-11, atol=1e-13)
cum_out = 1 - sol.sol(np.arange(0, 41)).sum(0)            # volume released by day boundaries
ode_ord = np.diff(cum_out)
nash_ord = rt.nash_uh(nres, kk, tol=1e-14, max_len=40)
x = np.random.default_rng(0).gamma(0.6, 5, 300)
a1, b1 = rt.gr4j_ordinates(2.208)
res['unit_hydrograph'] = {
    'gamma_step_response_vs_S_curve_max_abs': float(np.max(np.abs(step - scurve))),
    'convolve_uh_vs_numpy_convolve_max_abs': float(np.max(np.abs(rt.convolve_uh(x, ordg) - np.convolve(x, ordg)[:300]))),
    'nash_cascade_ode_vs_gamma_ordinates_max_abs': float(np.max(np.abs(ode_ord[:len(nash_ord)] - nash_ord))),
    'hbv_triangle_maxbas3': tri3.tolist(), 'hbv_triangle_hand': [2 / 9, 5 / 9, 2 / 9],
    'hbv_triangle_max_abs': float(np.max(np.abs(tri3 - np.array([2 / 9, 5 / 9, 2 / 9])))),
    'gr4j_uh_sums': [float(a1.sum()), float(b1.sum())],
}

# ------------------------------------------------------------------ 3. Muskingum X = 0 = linear reservoir
# Daily values are end-of-day samples with linear inflow in between (textbook Muskingum convention).
rng = np.random.default_rng(1)
I = 10 + 50 * rng.gamma(0.5, 1, 365)
out = {}
for K in (2.0, 0.5, 5.0):
    O, S = rt.muskingum(I, K, 0.0, O0=I[0])
    # exact solution of dO/dt = (I(t) - O)/K with I linear over each day from I[d-1] to I[d]
    Oe = np.empty_like(I)
    o_ = I[0]
    ip = I[0]
    for d in range(len(I)):
        a_, b_ = ip, I[d] - ip          # I(s) = a + b s on s in [0, 1]
        e = math.exp(-1.0 / K)
        o_ = a_ + b_ * (1 - K) + (o_ - a_ + b_ * K) * e
        Oe[d] = o_
        ip = I[d]
    vol_err = (I.sum() - O.sum() - S[-1]) / I.sum()
    out[f'K={K}'] = {'max_rel_err_vs_exact_ode': float(np.max(np.abs(O - Oe) / Oe)),
                     'nse_vs_exact': float(1 - np.sum((O - Oe) ** 2) / np.sum((Oe - Oe.mean()) ** 2)),
                     'mass_residual_rel': float(vol_err)}
res['muskingum_X0_linear_reservoir'] = out
c0, c1, c2 = rt.muskingum_coefficients(2.0, 0.2, 1.0)
res['muskingum_coefficients_K2_X0.2_dt1'] = {'C0': c0, 'C1': c1, 'C2': c2, 'sum': c0 + c1 + c2,
                                              'hand': {'C0': (1 - 0.8) / (3.2 + 1), 'C1': (1 + 0.8) / 4.2,
                                                       'C2': (3.2 - 1) / 4.2}}

# ------------------------------------------------------------------ 4. Muskingum-Cunge vs analytic diffusion wave
B, S0, nm, Qr, L = 50.0, 0.0005, 0.035, 200.0, 300.0
y = (Qr * nm / (B * math.sqrt(S0))) ** 0.6
c = 5 / 3 * Qr / (B * y)
D = Qr / (2 * B * S0)
days = np.arange(40)
Iq = 50 + 400 * np.exp(-0.5 * ((days - 8) / 1.5) ** 2)
Omc, Smc = rt.muskingum_cunge(Iq, L, S0, B, nm, qref=Qr)
Ovp, _ = rt.muskingum_cunge(Iq, L, S0, B, nm)        # variable-parameter (nonlinear) for information
# analytic: convolve the piecewise-linear inflow (minus base) with the Hayami kernel at 1-minute resolution
dtm = 60.0
tt = np.arange(dtm, 41 * 86400, dtm)
Lm = L * 1000
h = Lm / np.sqrt(4 * np.pi * D * tt ** 3) * np.exp(-(Lm - c * tt) ** 2 / (4 * D * tt))
h = h / (h.sum() * dtm)
tf = np.arange(0, 41 * 86400, dtm)
Iin = np.interp(tf, days * 86400.0, Iq - 50)
Oan = 50 + np.convolve(Iin, h)[: len(tf)] * dtm
Oan_d = Oan[(days * 86400 / dtm).astype(int)]
res['muskingum_cunge_vs_diffusion_wave'] = {
    'case': f'wide channel B={B} m, S0={S0}, n={nm}, L={L} km, reference Q={Qr} m3/s -> c={c:.3f} m/s, '
            f'D={D:.0f} m2/s; Gaussian inflow 50 + 400 exp(-(t-8)^2/(2 1.5^2)) m3/s sampled daily; analytic = '
            f'Hayami kernel convolution of the same (linearly interpolated) inflow',
    'peak_analytic': float(Oan_d.max()), 'peak_mc': float(Omc.max()),
    'peak_rel_err': float(Omc.max() / Oan_d.max() - 1),
    'peak_day_analytic': int(Oan_d.argmax()), 'peak_day_mc': int(Omc.argmax()),
    'nse_mc_vs_analytic': float(1 - np.sum((Omc - Oan_d) ** 2) / np.sum((Oan_d - Oan_d.mean()) ** 2)),
    'mass_residual_rel': float((Iq.sum() * 86400 - Omc.sum() * 86400 - Smc[-1] * 86400) / (Iq.sum() * 86400)),
    'variable_parameter_peak': float(Ovp.max()),
}

# ------------------------------------------------------------------ 5. kinematic wave rising limb
L2, B2, S2, n2 = 20000.0, 20.0, 0.001, 0.035
qe = 50.0
ql = qe / L2
alpha = (n2 * B2 ** (2 / 3) / math.sqrt(S2)) ** 0.6
te = alpha * qe ** 0.6 / ql
nt = int(2 * te / 60)
outk, stork = rt._kw_core(np.full(nt, ql), np.zeros(nt), L2, S2, B2, n2, 60.0, 400, 1, np.zeros(400))
tk = (np.arange(nt) + 0.5) * 60
an = np.where(tk < te, (ql * tk / alpha) ** (1 / 0.6), qe)
m_ = tk < te
res['kinematic_wave_rising_limb'] = {
    'case': f'L={L2/1000} km, B={B2} m, S0={S2}, n={n2}, uniform lateral inflow to Q_eq={qe} m3/s from a dry '
            f'channel; analytic Q(L,t) = (q t / a)^(1/0.6) for t < t_e = {te/3600:.2f} h',
    'nse_vs_analytic': float(1 - np.sum((outk - an) ** 2) / np.sum((an - an.mean()) ** 2)),
    'rel_err_at_half_te': float(outk[np.argmin(np.abs(tk - te / 2))] / an[np.argmin(np.abs(tk - te / 2))] - 1),
    'equilibrium_rel_err': float(outk[-1] / qe - 1),
    'mass_residual_rel': float((ql * L2 * 60 * nt - outk.sum() * 60 - stork[-1]) / (ql * L2 * 60 * nt)),
}

# ------------------------------------------------------------------ 6. degree-day snow by hand
fd = pd.DataFrame({'P': [10, 0, 4, 0, 0], 'T_mean': [-5, 2, 1, -2, 5], 'PET': 0.0},
                  index=pd.date_range('2001-01-01', periods=5))
pdd = {'TT': 0.0, 'TTI': 0.0, 'CFMAX': 3.0, 'SFCF': 1.1, 'CFR': 0.05, 'CWH': 0.1, 'PGRAD': 0.0, 'TLAPSE': -6.5}
od = ws.simulate(spec, fd, model='none', snow='degree_day', routing='none', pet='forcing', params=pdd, init={})
hand_swe = [11.0, 5.5, 2.2, 2.2, 0.0]
hand_out = [0.0, 5.5, 7.3, 0.0, 2.2]
res['degree_day_by_hand'] = {
    'steps': 'TT=0, TTI=0, CFMAX=3, SFCF=1.1, CFR=0.05, CWH=0.1. d1 P10 T-5: snow 11. d2 T+2: melt 6, hold 0.5, '
             'out 5.5. d3 P4 T+1: melt 3, +rain 4, hold 0.2, out 7.3. d4 T-2: refreeze min(0.3, 0.2)=0.2. '
             'd5 T+5: melt all 2.2, out 2.2',
    'hand_SWE': hand_swe, 'sim_SWE': od['ts']['SWE'].round(12).tolist(),
    'hand_outflow': hand_out, 'sim_outflow': od['ts']['liquid_input'].round(12).tolist(),
    'max_abs_err': float(max(np.max(np.abs(od['ts']['SWE'].values - hand_swe)),
                             np.max(np.abs(od['ts']['liquid_input'].values - hand_out)))),
    'ledger_ok': od['ledger']['ok'],
}

# ------------------------------------------------------------------ 7. energy-balance snow terms by hand
prm = np.array([p[1] for p in sn.EB_PARAMS], float)
ice0, Ta, Rs, ea, u10, Rso, Pa = 100.0, 0.0, 20.0, 0.5, 2.0, 25.0, 90.0
r = sn.eb_step(ice0, 0.0, 0.0, 0.0, 0.0, Ta, Rs, ea, u10, Rso, Pa, prm, 1.0)
# hand: fresh snow albedo 0.85 blended with ground 0.2 over 15 mm, T = 0 -> Ts = 0, no sensible heat
alb = 0.2 + (0.85 - 0.2) * (1 - math.exp(-100 / 15))
SW = (1 - alb) * 20.0
epsc = min(1.24 * (5.0 / 273.15) ** (1 / 7), 1.0)
cf = 1 - 20 / 25
eps = (1 - 0.84 * cf) * epsc + 0.84 * cf
LWin = eps * 4.903e-9 * 273.15 ** 4
LWout = 0.98 * 4.903e-9 * 273.15 ** 4
z0 = 0.002
u2 = 2.0 * math.log(2 / z0) / math.log(10 / z0)
ra = math.log(2 / z0) ** 2 / (0.41 ** 2 * u2)
rho = 90000 / (287.05 * 273.15)
es0 = 0.61115
E = rho * 0.622 * (0.5 - es0) / 90.0 / ra * 86400
LE = 2.835 * E
Qhand = SW + LWin - LWout + 0.0 + LE + 0.173
res['energy_balance_by_hand'] = {
    'case': 'ice 100 mm, T=0 C, Rs=20 MJ/m2/d, ea=0.5 kPa, u10=2 m/s, Rso=25, P=90 kPa, default EB parameters',
    'terms_hand_MJ': {'SW': SW, 'LWin': LWin, 'LWout': LWout, 'H': 0.0, 'LE': LE, 'G': 0.173, 'Q': Qhand},
    'Q_sim_MJ': r[10], 'Q_rel_err': float(r[10] / Qhand - 1),
    'melt_sim_mm': r[6], 'melt_hand_mm': max(Qhand, 0) / 0.334,
    'sublimation_sim_mm': r[9], 'sublimation_hand_mm': -E,
    'mass_closure_mm': float(ice0 - (r[0] + r[1]) - r[8] - r[9]),
}

# ------------------------------------------------------------------ 8. PET: FAO-56 worked examples
ra8 = float(pt.extraterrestrial_radiation(-20.0, 246))
rs = 22.07 / 0.0864
et0 = float(pt.penman_monteith(np.array([16.9]), np.array([21.5]), np.array([12.3]), np.array([rs]),
                               np.array([1409.0]), np.array([2.078]), 50.8, np.array([187]), 100.0, wind_height=2.0)[0])
rn = float(pt.net_radiation(np.array([21.5]), np.array([12.3]), np.array([rs]), np.array([1409.0]), 50.8,
                            np.array([187]), 100.0)[0])
ra18 = float(pt.extraterrestrial_radiation(50.8, 187))
res['pet_fao56'] = {
    'example8_Ra_20S_3Sep': {'published': 32.2, 'sim': ra8, 'abs_err': abs(ra8 - 32.2)},
    'example18_Brussels_6Jul': {'published_ET0_mm': 3.9, 'sim_ET0_mm': et0, 'published_Ra': 41.09, 'sim_Ra': ra18,
                                'published_Rn': 13.28, 'sim_Rn': rn,
                                'inputs': 'lat 50.8N, z 100 m, Tmax 21.5, Tmin 12.3, ea 1.409 kPa, u2 2.078 m/s, '
                                          'Rs 22.07 MJ/m2/d (from n = 9.25 h)'},
    'hargreaves_vs_pm_same_day': float(pt.hargreaves(np.array([16.9]), np.array([21.5]), np.array([12.3]), 50.8,
                                                     np.array([187]))[0]),
    'priestley_taylor_same_day': float(pt.priestley_taylor(np.array([16.9]), np.array([21.5]), np.array([12.3]),
                                                           np.array([rs]), np.array([1409.0]), 50.8,
                                                           np.array([187]), 100.0)[0]),
}

# ------------------------------------------------------------------ 9. reservoir
R = rv.Reservoir(500.0, depth_m=60.0, shape_b=2.5, init_frac=0.5, dead_frac=0.05, efficiency=0.9, turbine_max_m3s=None)
Sg = np.linspace(10, 500, 50)
dh = 1e-4
num_area = np.array([(R.storage_from_level(R.level(s) + dh) - R.storage_from_level(R.level(s) - dh)) / (2 * dh)
                     for s in Sg])
an_area = np.array([R.area(s) for s in Sg])
idx = pd.date_range('2000-01-01', periods=730)
inflow = 20 + 30 * np.clip(np.sin(np.arange(730) / 365.25 * 2 * np.pi), 0, None) + rng.gamma(0.5, 10, 730)
ev = np.full(730, 4.0)
s0 = R.S
ro = R.run(inflow, idx, ev, demand_m3s=25.0, policy='sop')
bal = s0 + (inflow.sum() - ro['release_m3s'].sum() - ro['spill_m3s'].sum()) * rv.DT_HM3 - ro['evap_hm3'].sum() - ro['storage_hm3'][-1]
st = rv.Reservoir(100.0, depth_m=40.0).step(100.0, 100.0, 0.0)
R2 = rv.Reservoir(100.0, depth_m=50.0, init_frac=1.0, spill='weir', weir_length_m=30.0)
for _ in range(60):
    w = R2.step(80.0, 0.0, 0.0)
res['reservoir'] = {
    'area_vs_dV_dh_max_rel_err': float(np.max(np.abs(num_area - an_area) / an_area)),
    'mass_balance_residual_hm3': float(bal), 'storage_range_hm3': [float(ro['storage_hm3'].min()), float(ro['storage_hm3'].max())],
    'sop_demand_met_frac': float(np.mean(ro['release_m3s'] >= 25.0 - 1e-9)),
    'hydropower_hand': {'case': '100 hm3, depth 40 m, release 100 m3/s at constant level', 'head_m': st['head_m'],
                        'sim_MW': st['power_MW'], 'hand_MW': 0.88 * 1000 * 9.80665 * st['head_m'] * 100 / 1e6},
    'weir_steady_state': {'inflow_m3s': 80.0, 'spill_after_60d_m3s': w['spill_m3s'],
                          'surcharge_m': (R2.S - R2.C) / R2.area(R2.C),
                          'hand_head_m': (80.0 / (2.1 * 30.0)) ** (2 / 3),
                          'note': 'end-of-day level sits one hourly substep of inflow (80 m3/s x 1 h / 5 km2 = 0.058 m) '
                                  'below the head that passes the inflow (explicit hourly weir substeps)'},
}

# ------------------------------------------------------------------ verdicts
v = {
    'linear_reservoir': res['linear_reservoir']['hbv_lower_box']['max_abs_err_vs_discrete_mm'] < 1e-10 and
    res['linear_reservoir']['sacsma_primary_baseflow']['max_abs_err_mm'] < 1e-8,
    'unit_hydrograph': res['unit_hydrograph']['gamma_step_response_vs_S_curve_max_abs'] < 1e-9 and
    res['unit_hydrograph']['nash_cascade_ode_vs_gamma_ordinates_max_abs'] < 1e-8,
    'muskingum_X0': all(x['max_rel_err_vs_exact_ode'] < 0.01 and abs(x['mass_residual_rel']) < 1e-12 for x in out.values()),
    'muskingum_cunge': res['muskingum_cunge_vs_diffusion_wave']['nse_mc_vs_analytic'] > 0.99,
    'kinematic': res['kinematic_wave_rising_limb']['nse_vs_analytic'] > 0.99,
    'degree_day': res['degree_day_by_hand']['max_abs_err'] < 1e-12,
    'energy_balance': abs(res['energy_balance_by_hand']['Q_rel_err']) < 1e-3,
    'pet': abs(et0 - 3.9) < 0.05 and abs(ra8 - 32.2) < 0.05,
    'reservoir': abs(bal) < 1e-8 and res['reservoir']['area_vs_dV_dh_max_rel_err'] < 1e-4 and
    abs(res['reservoir']['weir_steady_state']['spill_after_60d_m3s'] / 80.0 - 1) < 1e-6 and
    abs(res['reservoir']['weir_steady_state']['surcharge_m'] / res['reservoir']['weir_steady_state']['hand_head_m'] - 1) < 0.06,
}
res['pass'] = v
write('analytic.json', res)
for k, val in v.items():
    print(k, val)
print(res['muskingum_X0_linear_reservoir'])
print(res['muskingum_cunge_vs_diffusion_wave'])
print(res['kinematic_wave_rising_limb'])
print(res['energy_balance_by_hand'])
print(res['pet_fao56'])
print(res['reservoir'])
print(res['linear_reservoir'])
