"""Print the README verification and speed tables from the check JSONs (so the docs quote the files exactly)."""
import os as _os
GRID_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/grid
GRID_ROOT = _os.environ.get('ML_RATCHET_GRID') or GRID_CODE   # data/, gridsim/refdata/, checks/out/ live here
import json, os
C = f"{GRID_CODE}/checks/"
J = lambda f: json.load(open(C + f)) if os.path.exists(C + f) else None  # noqa
rows = []
pf = J("powerflow.json")
if pf:
    mx = lambda k1, k2: max(v[k1][k2] for v in pf["cases"].values())  # noqa
    o = pf["dc_opf_congested"]
    rows.append(("`powerflow.json`", "AC Newton-Raphson and DC power flow vs PYPOWER 5.1 on case9/14/30/57/118/300, ACTIVSg200, ACTIVSg2000 (2,000 buses); pandapower on case9/30/118; DC-OPF with 8 tightened lines vs PYPOWER rundcopf",
                 f"AC max |ΔVa| {mx('ac','max_dVa_deg'):.0e}°, |ΔVm| {mx('ac','max_dVm_pu'):.0e} pu, |ΔPf| {mx('ac','max_dPf_MW'):.0e} MW; DC |ΔVa| {mx('dc','max_dVa_deg'):.0e}°; "
                 f"DC-OPF cost rel. diff {o['case_ACTIVSg2000']['cost_rel_diff']:.0e}, LMP max diff ${o['case_ACTIVSg2000']['max_dLMP_per_MWh']:.3f}/MWh (ACTIVSg2000), ${o['case_ACTIVSg200']['max_dLMP_per_MWh']:.4f} (ACTIVSg200); "
                 f"DC vs AC flows: median error {min(v['dc_vs_ac']['median_abs_flow_err_pct'] for v in pf['cases'].values()):.1f}-{max(v['dc_vs_ac']['median_abs_flow_err_pct'] for v in pf['cases'].values()):.1f} %"))
nd = J("network_dispatch.json")
if nd:
    rows.append(("`network_dispatch.json`", "dispatch(..., network=) with PTDF line limits vs the DC-OPF, same congestion, one hour; then 24 h unit commitment on ACTIVSg2000",
                 f"cost and LMPs identical (max ΔLMP {nd['case_ACTIVSg2000']['max_dLMP']:.0e} $/MWh ACTIVSg2000, {nd['case_ACTIVSg200']['max_dLMP']:.0e} ACTIVSg200); 24 h MILP: {nd['ACTIVSg2000_24h_milp']['lines_monitored']} lines monitored lazily, max overload {nd['ACTIVSg2000_24h_milp']['max_overload_pct']:.0e} %, {nd['ACTIVSg2000_24h_milp']['wall_s']:.0f} s"))
fa = J("frequency_analytic.json")
if fa:
    g = {x["tech"]: x for x in fa["governor_vs_lsim"]}
    rows.append(("`frequency_analytic.json`", "RoCoF = ΔP f0/2E; damping-only steady state; each governor model vs scipy.signal.lsim; closed-form nadir of the 2nd-order system; deadband + headroom steady state; two-area steady state and inter-area mode",
                 f"RoCoF rel. error {fa['rocof_pure_inertia']['max_rel_err']:.0e}; nadir vs closed form {fa['closed_form_nadir']['err_at_default_h_0.01_mHz']:.3f} mHz (h = 10 ms); governor traces RMS {min(x['rms_err_Hz'] for x in g.values())*1e3:.3f}-{max(x['rms_err_Hz'] for x in g.values())*1e3:.2f} mHz (hydro largest, non-minimum phase); "
                 f"deadband/headroom Δf_ss {fa['deadband_headroom']['df_ss_sim']:.4f} vs {fa['deadband_headroom']['df_ss_analytic']:.4f} Hz; inter-area mode {fa['two_area']['interarea_mode_rad_s_sim_no_gov']:.3f} vs {fa['two_area']['interarea_mode_rad_s_analytic']:.3f} rad/s"))
an = J("frequency_andes.json")
if an:
    rows.append(("`frequency_andes.json`", "aggregate model vs ANDES 2.0 full 4-machine Kundur two-area (GENROU, EXDC2, TGOV1), 200 MW load step",
                 f"nadir {an['gridsim']['nadir_Hz']:.3f} vs {an['andes']['nadir_Hz']:.3f} Hz at {an['gridsim']['t_nadir_s']:.2f} vs {an['andes']['t_nadir_s']:.2f} s; 30 s {an['gridsim']['f_30s_Hz']:.3f} vs {an['andes']['f_30s_Hz']:.3f} Hz; RMS {an['rms_diff_Hz']*1e3:.0f} mHz"))
fe = J("frequency_events.json")
if fe:
    s = fe["gb_2019_08_09"]["simulated"]
    rows.append(("`frequency_events.json`", "GB 9 Aug 2019 from the ESO technical report (losses, response held/delivered, RoCoF and 49 Hz relays, LFDD); one parameter fitted (load damping, on nadir 1)",
                 f"D = {fe['gb_2019_08_09']['damping_fit']['D_pct_per_Hz']} %/Hz; nadir 1 {s['nadir1_Hz']:.2f} Hz at {s['t_nadir1_s']:.1f} s (reported 49.1 at 24.5 s); peak before GT1A trip {s['max_Hz_between_nadir1_and_GT1A']:.2f} Hz (49.2); 350 MW RoCoF trip at {s['trips'][0]['t_s']:.1f} s; LFDD at {s['t_lfdd_s']:.1f} s (75.9)"))
de = J("frequency_dev_events.json")
if de:
    gb = {x["event"]: x for x in de["GB"]}
    rows.append(("`frequency_dev_events.json`", "the 4 GB and 5 ERCOT DEV events of splits.json (NESO 1 s; ERCOT operations reports); ERCOT state from that day's dispatch, inertia set to ERCOT's value; ERCOT effective damping fitted on these 5",
                 f"GB nadir error " + ", ".join(f"{k[3:11]} {v['nadir_dev_err_pct']:+.0f} %" + (" (B's loss 1,788 MW looks wrong: the 0.29 Hz drop at 333 GVA s implies ~850 MW)" if k == "GB_20191203T200030" else "") for k, v in gb.items()) +
                 f"; 9 Aug RMS vs the 1 s trace {gb['GB_20190809T155233']['rms_0_75s_Hz']*1e3:.0f} mHz over 0-75 s ({gb['GB_20190809T155233']['rms_Hz']*1e3:.0f} mHz to 100 s, after LFDD the simulation recovers too fast); ERCOT nadir errors " + ", ".join(f"{x['nadir_dev_err_pct']:+.0f} %" for x in de["ERCO"]) +
                 f" after fitting D = {de['ERCO_damping_fit']['D_pct_per_pct']} %/% (before: 41-70 % too deep); dispatch-derived inertia of those hours within " +
                 f"{min(x['sim_inertia_before_rescale_GWs']/x['inertia_GWs']-1 for x in de['ERCO'])*100:+.0f}/{max(x['sim_inertia_before_rescale_GWs']/x['inertia_GWs']-1 for x in de['ERCO'])*100:+.0f} % of reported"))
for y in (2019, 2022):
    d = J(f"ercot_dispatch_{y}.json")
    if not d:
        continue
    for name in ("economic", "calibrated", "merit"):
        r = d["configs"].get(name)
        if not r:
            continue
        b = r["by_fuel"]
        pr = r.get("price_vs_ercot_dam_hubavg", {})
        co2 = r.get("co2", {})
        rows.append((f"`ercot_dispatch_{y}.json` {name}", f"ERCOT {y} ({'dev' if y == 2019 else 'train-input year, not used for calibration'}), 8,760 h, real fleet + EIA-930 demand/wind/solar/interchange/nuclear vs EIA-930 by fuel, CAMPD CO2, ERCOT DAM hub price",
                     f"annual: gas {b['NG']['annual_err_pct']:+.1f} %, coal {b['COL']['annual_err_pct']:+.1f} %; hourly nMAE gas {b['NG']['hourly_nMAE_pct_of_mean']:.0f} % (r {b['NG']['hourly_corr']:.2f}), coal {b['COL']['hourly_nMAE_pct_of_mean']:.0f} % (r {b['COL']['hourly_corr']:.2f}); "
                     f"CO2 {co2.get('err_pct', float('nan')):+.1f} % vs CAMPD ({co2.get('campd_Mt_ercot_plants', float('nan')):.0f} Mt), hourly r {co2.get('hourly_corr', float('nan')):.2f}; "
                     f"price median {pr.get('sim_median', float('nan')):.1f} vs {pr.get('real_median', float('nan')):.1f} $/MWh, median abs. error {pr.get('median_abs_err', float('nan')):.1f}, rank r {pr.get('spearman', float('nan')):.2f}, hours ≤ $0 {pr.get('share_hours_sim_le_0', 0)*100:.0f} % (real {pr.get('share_hours_real_le_0', 0)*100:.0f} %); wall {r['wall_s']/60:.1f} min"))
ei = J("ercot_inertia.json")
if ei:
    for k, v in ei.items():
        rows.append((f"`ercot_inertia.json` {k}", "hourly inertia of the simulated commitment vs Texas RE's reported minimum and hours below 150 GW s",
                     f"min {v['sim']['min_GWs']:.0f} GW s vs {v['reported']['min_GWs']} reported ({v['min_err_pct']:+.0f} %); hours < 150 GW s: {v['sim']['hours_below_150']} vs {v['reported']['hours_below_150']}; median {v['sim']['p5_p50_p95_GWs'][1]:.0f} GW s"))
rn = J("renewables.json")
if rn:
    for y, v in rn["years"].items():
        rows.append((f"`renewables.json` {y}", "ERCOT wind and solar fleets from NASA POWER weather vs EIA-930",
                     f"wind bias {v['wind']['raw']['annual_bias_pct']:+.1f} % (dev-scaled {v['wind']['dev_scaled']['annual_bias_pct']:+.1f} %), hourly r {v['wind']['raw']['hourly_corr']:.2f}, nMAE {v['wind']['raw']['hourly_nMAE_pct']:.0f} %; solar bias {v['solar_pv']['raw']['annual_bias_pct']:+.1f} %, hourly r {v['solar_pv']['raw']['hourly_corr']:.2f}, nMAE {v['solar_pv']['raw']['hourly_nMAE_pct']:.0f} %"))
dm = J("demand.json")
if dm:
    t = dm["test"]
    rows.append(("`demand.json`", "weather-driven ERCOT demand, fit 2018-19, test 2021 (Uri week excluded) and 2022",
                 f"MAPE {t['test_2021']['MAPE_pct']:.1f} % / {t['test_2022']['MAPE_pct']:.1f} % (bias {t['test_2022']['bias_pct']:+.1f} % in 2022); ERCOT's own day-ahead forecast {t['test_2021']['ercot_day_ahead_forecast_MAPE_pct']:.1f} / {t['test_2022']['ercot_day_ahead_forecast_MAPE_pct']:.1f} %"))
ge = J("generator.json")
if ge:
    rows.append(("`generator.json`", "2,000 generated states and 40 generated fleets vs real ranges",
                 f"inertia per MW demand p5/p50/p95 {'/'.join(f'{x:.1f}' for x in ge['ERCO']['inertia_s_per_demand_p5_p50_p95'])} s (ERCOT 2019 min 4.6 s, GB 9 Aug 7.2 s); RoCoF at t=0 = formula to {max(ge['ERCO']['rocof_t0_vs_formula_max_rel_err'], ge['GB']['rocof_t0_vs_formula_max_rel_err']):.0e}; UFLS in {ge['ERCO']['share_with_ufls']*100:.0f} % (60 Hz) / {ge['GB']['share_with_ufls']*100:.0f} % (50 Hz) of random events; heat rates inside the ERCOT EIA-923 spread"))
print("| check (file) | what | result |\n|---|---|---|")
for a, b, c in rows:
    print(f"| {a} | {b} | {c} |")
sp = J("speed.json")
if sp:
    print("\n| task | time |\n|---|---|")
    p = sp["powerflow_ACTIVSg2000"]
    print(f"| AC power flow, ACTIVSg2000 (2,000 buses) | {p['ac_nr_s']*1e3:.0f} ms |")
    print(f"| DC power flow / PTDF build / DC-OPF, ACTIVSg2000 | {p['dc_s']*1e3:.0f} ms / {p['ptdf_build_s']:.2f} s / {p['dc_opf_s']:.2f} s |")
    print(f"| one 60 s frequency event (50 Hz output, 10 ms step) | {sp['frequency']['single_60s_event_dt0.02_h0.01_s']*1e3:.0f} ms |")
    print(f"| batch of {sp['frequency']['batch_events']} events with full generated states, 60 s each | {sp['frequency']['batch_60s_s_per_event']*1e3:.0f} ms per event |")
    print(f"| generate one random system state | {sp['generator_states_s_per_state']*1e3:.0f} ms |")
    print(f"| ERCOT week: merit / LP / MILP | {sp['dispatch_ercot_week_merit_s']:.1f} / {sp['dispatch_ercot_week_lp_s']:.1f} / {sp['dispatch_ercot_week_milp_s']:.1f} s |")
    print(f"| ERCOT solver time per day (median): LP / MILP | {sp['dispatch_ercot_lp_solve_s_per_day_median']:.2f} / {sp['dispatch_ercot_milp_solve_s_per_day_median']:.2f} s |")
    for y in (2019, 2022):
        d = J(f"ercot_dispatch_{y}.json")
        if d:
            print(f"| ERCOT {y} full year (8,760 h), wall incl. setup, shared CPU | " + ", ".join(f"{k} {v['wall_s']/60:.1f} min" for k, v in d["configs"].items()) + " |")
