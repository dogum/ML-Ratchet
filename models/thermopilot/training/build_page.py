"""Assemble web/thermopilot.html from page_src.html + tp.js + data.
usage: build_page.py TAG WEIGHTS.npz LAB.json"""
import os, sys, json, base64
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE   # repo layout: tp/, results/, ratchet.json are in training/
sys.path.insert(0, ROOT)
from tp import env
from _paths import WORK

TARIFF_NAMES = {'pge_e_tou_c': 'PG&E E-TOU-C (CA)', 'aps_r3': 'APS R-3 demand (AZ)', 'srp_e26': 'SRP E-26 (AZ)',
                'austin_res_tou': 'Austin Energy TOU (TX)', 'rtp_ercot': 'ERCOT real-time (TX)', 'duke_fl_rt': 'Duke RT (FL)',
                'rmp_ut_sched1_tou': 'Rocky Mtn Power TOU (UT)', 'xcel_co_re_tou': 'Xcel RE-TOU (CO)',
                'xcel_mn_a04_tod': 'Xcel A04 TOD (MN)', 'nyseg_sc12_tou': 'NYSEG SC-12 (NY)', 'coned_sc1_rate3': 'Con Ed SC-1 R III (NY)',
                'pseg_rs_tou_3p': 'PSE&G RS-TOU (NJ)', 'pge_b19_sec': 'PG&E B-19', 'ouc_gsd_tou': 'OUC GSD TOU',
                'xcel_mn_a15_tod': 'Xcel A15 TOD', 'pepco_gtlv': 'Pepco GT LV'}


def weights_js(path):
    z = np.load(path)
    layers = []
    i = 0
    while f'W{i}' in z.files:
        Wm = z[f'W{i}'].astype(np.float32)
        layers.append(dict(W=base64.b64encode(Wm.tobytes()).decode(), b=[float(v) for v in z[f'b{i}']], nin=Wm.shape[0], nout=Wm.shape[1]))
        i += 1
    return dict(mu=[float(v) for v in z['mu']], sd=[float(v) for v in z['sd']], layers=layers,
                shield=int(z['shield']) if 'shield' in z.files else 0)


def main():
    tag, wpath, labp = sys.argv[1:4]
    lab = json.load(open(labp)); lab.pop('check', None)
    exam = json.load(open(os.path.join(ROOT, 'results', f'exam_{tag}.json')))
    for s in exam.values():
        s.pop('by_month', None); s.pop('by_kind', None)
        if 'by_tariff' in s:
            s['by_tariff'] = {k: dict(n=v['n'], t={c: dict(bill_saving_vs_fixed=v['t'][c]['bill_saving_vs_fixed']) for c in v['t']})
                              for k, v in s['by_tariff'].items()}
    R = json.load(open(os.path.join(ROOT, 'ratchet.json')))
    rat = [dict(tag=r['tag'], change=r['change'], why=r['why'], kept=r['kept']) for r in R['rungs']]
    sig = json.load(open(os.path.join(ROOT, 'results', 'wpuq', 'signature.json')))
    ok = [h for h in sig['homes'] if h.get('ok')]
    sm = sig['summary']
    prof = {k: list(np.mean([h[k + '_jan']['profile'] for h in ok], 0)) for k in ('real', 'sim', 'calib')}
    rank_f = os.path.join(ROOT, 'results', 'wpuq', 'rank.json')
    rk = json.load(open(rank_f))['summary'] if os.path.exists(rank_f) else None
    rtxt = ''
    if rk:
        g, c = rk['mean_J_saving_vs_fixed']['thermopilot']['generic'], rk['mean_J_saving_vs_fixed']['thermopilot']['calibrated']
        gm, cm = rk['mean_J_saving_vs_fixed']['mpc_c']['generic'], rk['mean_J_saving_vs_fixed']['mpc_c']['calibrated']
        rtxt = (f"Does the ranking survive? On the {rk['n']} WPuQ homes we re-ran every controller on both twins (January 2019). "
                f"The order of the seven controllers agrees with a median Kendall tau of {rk['kendall_tau_median']:.2f}. "
                f"ThermoPilot's saving on cost + comfort vs the fixed thermostat is {100*g:.1f} % on the generic twins and {100*c:.1f} % "
                f"on the scaled twins (MPC: {100*gm:.1f} % and {100*cm:.1f} %); ThermoPilot vs setback flips order in {rk['tp_vs_setback_order_flips']} of {rk['n']} homes. "
                "So the ranking of controllers is more robust than the absolute numbers, but the real homes run their heat pumps far more evenly through the day than any twin, "
                "which means savings from shifting load in time are the least trustworthy part of the simulated exam.")
    W = dict(homes=[dict(real=h['real']['slope'], sim=h['sim']['slope'], calib=h['calib']['slope']) for h in ok], profiles=prof,
             text1=f"Generic twins need {sm['slope_ratio_sim_real_median']:.1f}× the real homes' extra heat-pump energy per kelvin (median; IQR {sm['slope_ratio_iqr'][0]:.1f}–{sm['slope_ratio_iqr'][1]:.1f}×): these German homes are much tighter than a typical US house. Scaling each twin's envelope (median ×{sm['f_env_median']:.2f}) matches the slope.",
             text2=f"Real heat pumps run almost flat through a January day; the twins follow the night-time cold and the midday sun (median hourly-profile correlation r = {sm['profile_r_median']:.2f}). Real homes likely buffer with hot-water tanks and run weather-compensated flow temperatures, which a single-zone air model does not have.",
             text3=rtxt)
    ex = exam.get('exam_recs', {}).get('all', {})
    tpr = ex.get('thermopilot', {}); mpc = ex.get('mpc_c', {}); orc = ex.get('oracle', {}); sb = ex.get('setback', {})
    keys = dict(items=[
        (f"{100*tpr.get('bill_saving_vs_fixed', 0):.1f} %", 'lower bill than a fixed thermostat (RECS exam twins)'),
        (f"{-100*tpr.get('bill_saving_vs_setback', 0):+.1f} %", "bill change vs the occupant's own setback schedule (about the same)"),
        (f"{tpr.get('DH', 0):.1f} K h", f"outside the comfort band per home-month (setback {sb.get('DH', 0):.1f})"),
        (f"{tpr.get('skill', 0):.2f}", f"share of the perfect-foresight saving captured (MPC {mpc.get('skill', 0):.2f})")],
        note=f"Exam: 96 home-months of RECS respondent twins, January and July 2017, under the real weather and the real tariff of their state. Perfect foresight saves {100*orc.get('bill_saving_vs_fixed', 0):.1f} % of the bill there.")
    limits = [
        'The exam is simulated. There is no real thermostat log in this project (ecobee is gated), so every number on this page is BuildingSim with real inputs, not measured savings.',
        'The WPuQ check shows the simulator puts heat-pump load at the wrong times of day for those German homes and overstates their heat loss 2–3× with generic twins. Savings that come from moving load in time are the least certain part.',
        'Thermostat physics are idealised: one zone, the setpoint is met within each hour, no short-cycling or minimum run times, no humidity control. "On/off per day" counts hours that change state, not compressor starts.',
        'Twins come from survey characteristics matched to a ResStock dwelling, not from audits. HUE and WPuQ use stand-in US tariffs because their own utilities are not in the tariff set.',
        'The perfect-foresight bound is a dynamic program on a 0.25 K grid with a 24 h horizon and a menu of 36 setpoint pairs; with demand charges it also tries monthly peak targets. It is not a proven optimum.',
        'The comfort weight (λ = $3 per kelvin-hour per 150 m²) is a choice. With it, every controller is pushed to stay inside the band; a smaller λ would trade comfort for money.',
        'Gas prices are the 2025 US averages from EIA; tariffs are 2024–26 URDB rates applied to 2017 weather.']
    credits = [
        'Building Data Genome 2 (Miller et al. 2020), CC BY-SA 4.0: site weather and building metadata. Derived data stay share-alike.',
        'EIA RECS 2020 microdata (public domain): home characteristics and thermostat settings for twins and comfort bands.',
        'NREL ResStock 2025 R1 / ComStock 2025 R3 (OEDI, CC BY 4.0): generator priors and twin matching.',
        'HUE (Makonin 2019, Harvard Dataverse, CC BY 4.0) and WPuQ (Schlemminger et al. 2022, Zenodo, CC BY 4.0).',
        'URDB tariffs (OpenEI), ERCOT real-time prices via GridData, TMYx weather (climate.onebuilding.org), NASA POWER irradiance, EIA natural gas prices.',
        'BuildingSim (ISO 13790 5R1C, EnergyPlus-derived equipment curves, pvlib) and BuildingData, ML Ratchet phase 1.']
    usage = ("import numpy as np, thermopilot_core as tp\n"
             "W = tp.load('thermopilot_v1.npz')\n"
             "card = dict(UA=180.0, C=2.4e7, area=160.0, kind='hp', gas_heat=False, fan_cont=False, res=True,\n"
             "            P_fan=300.0, backup=0.0, Tg=..., Qmax=..., Qh=..., Ph=..., eff=..., Qs=..., Pc=...)  # tables vs T_out\n"
             "ctx = tp.make_ctx(local_hours, price, lo, hi, pref_lo=20.5, pref_hi=24.0, state=state)\n"
             "pilot = tp.ThermoPilot(W, card, ctx, lam=3.0, base_W=np.zeros(len(price)))\n"
             "sp = pilot.step(k, T_zone, q_hvac_prev_W, elec_prev_kW, forecast_T[24], forecast_GHI[24])\n"
             "# -> {'heat_sp': 21.0, 'cool_sp': 100.0}   (python3 demo.py replays three exam episodes)")
    data = dict(lab=lab, exam=exam, ratchet=rat, wpuq=W, weights=weights_js(wpath), keys=keys, limits=limits,
                credits=credits, usage=usage, tariffNames=TARIFF_NAMES, lab_gas=env.GAS_RES)
    src = open(os.path.join(HERE, 'page_src.html')).read()
    js = open(os.path.join(HERE, 'tp.js')).read()
    out = src.replace('__DATA__', json.dumps(data, separators=(',', ':')).replace('</', '<\\/')).replace('__TPJS__', js)
    os.makedirs(WORK, exist_ok=True)
    p = os.path.join(WORK, 'thermopilot.html')
    open(p, 'w').write(out)
    print(p, os.path.getsize(p) / 1e6, 'MB')


if __name__ == '__main__':
    main()
