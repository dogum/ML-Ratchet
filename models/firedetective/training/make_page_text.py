"""Page prose + headline numbers (out/page_text.json) from out/exam.json, out/page_tables.json and ratchet.json.
Run after make_text.py."""
from common import *

E = json.load(open(latest('out/exam.json')))
T = json.load(open(f'{ROOT}/out/page_tables.json'))
R = json.load(open(latest('ratchet.json')))
A = E['all']
sev, sevall, wind, lf = A['forest'], A['all_fuels'], A['wind'], A['lfmc']
f2 = lambda x: f'{x:.2f}'

headline = [
    dict(k='Burn severity, forest', v=f"ρ {f2(sev['rho']['network']['rho'])}",
         d=f"Spearman with the MTBS high-severity share on {sev['n']} exam forest fires. Best classical index: "
           f"FWI {f2(sev['rho']['fwi_max']['rho'])}, ERC {f2(sev['rho']['erc_max']['rho'])}."),
    dict(k='Wind, read blind', v=f"ρ {f2(wind['rho']['network']['rho'])}",
         d=f"On {wind['n']} exam fires with the wind inputs removed. Rothermel back-calculation: "
           f"{f2(wind['rho']['rothermel_bc']['rho'])}. It ranks windy fires but misses the level (MAE worse than the median)."),
    dict(k='Live fuel moisture', v=f"ρ {f2(lf['rho']['network']['rho'])}",
         d=f"A failure. On {lf['n']} fires with Globe-LFMC samples, the dev-site climatology does better "
           f"(ρ {f2(lf['rho']['clim']['rho'])}, MAE {lf['mae']['clim']['mae']:.0f} vs {lf['mae']['network']['mae']:.0f} %)."),
]
exam_intro = (f"The exam was run once, after the ratchet closed, on {E['n_records']} real exam fires from 2017-2023 "
              "(every timber fire with at least one mapped growth day, plus a seeded draw of 250 others). "
              "Severity and the MTBS map are compared only after a two-number linear map fitted on the 40 dev fires; "
              "every classical index gets the same kind of map. Brackets are 95 % bootstrap intervals over fires. "
              f"The 80 % error bars cover {100 * sev['cov80_network']:.0f} % of forest severity truths (too narrow), "
              f"{100 * wind['cov80_network']:.0f} % of wind and {100 * lf['cov80_network']:.0f} % of live moisture truths.")
flow = [
    ['FireSim', 'Rothermel surface spread, Van Wagner / Cruz crown fire, Albini spotting, Nelson dead-fuel moisture, '
                'minimum-travel-time spread on 180 m grids (Phase 1, checked against 436 published BehavePlus cases and pyretechnics).'],
    ['Real places and weather', 'Coarsened LANDFIRE windows of dev fires plus parametric landscapes; real HRRR hourly weather '
                                'from dev fire dates; the matching gridMET days are the network\'s weather input.'],
    ['Randomised drivers', 'Rate-of-spread factor, live and dead moisture, crowning, spotting, initial attack, '
                           'anchor-and-flank lines, night recovery, wind adjustment factor.'],
    ['Simulated satellite', 'FireSim\'s VIIRS emulator (detection vs FRP, scan angle, clouds) and a FEDS-like half-day '
                            'perimeter emulator tuned on dev fires (median area ratio 1.03).'],
    ['Network', '1-D CNN over up to 12 half-day steps plus fuel and terrain summaries; 3 seeds, 189k weights in all; '
                'Gaussian heads give a mean and a sigma for every driver.'],
    ['Graded on reality', 'MTBS burn severity, gridMET wind (with the wind inputs blanked) and Globe-LFMC samples.'],
]
limits = [
    f"Live fuel moisture fails on the exam (ρ {f2(lf['rho']['network']['rho'])} vs climatology {f2(lf['rho']['clim']['rho'])}). "
    "Even in simulation the network reads it poorly (R² 0.14), and the dev score (ρ 0.55 on only 9 fires) did not carry over.",
    f"Wind is ranked, not measured: ρ {f2(wind['rho']['network']['rho'])}, but MAE {wind['mae']['network']['mae']:.2f} m/s is worse "
    f"than guessing the dev median ({wind['mae']['dumb']['mae']:.2f}). On the recent 2022-23 fires ρ drops to "
    f"{f2(E['recent']['wind']['rho']['network']['rho'])}.",
    f"Across all fuels the simple Van Wagner crown-fire potential ranks severity better than the network "
    f"(ρ {f2(sevall['rho']['cfp']['rho'])} vs {f2(sevall['rho']['network']['rho'])}); the network only wins inside forests.",
    f"Severity error bars are too narrow on the exam: the 80 % band covers {100 * sev['cov80_network']:.0f} % of forest fires, "
    "even after a dev widening factor of 2.5.",
    "Crown fraction in FireSim and MTBS 'high severity' are different things (a burnt-out surface fire can be high severity). "
    "The map between them is fitted on 30 dev fires.",
    "Simulated fires still grow on 91 % of half-days against 50 % in reality, and are rounder (elongation 0.40 vs 0.70). "
    "Some drivers are not identifiable from these inputs even in simulation: rate-of-spread factor R² 0.14, fuel-model mismatch 0.01, spotting AUC 0.47. They are shown as simulator readings only.",
    "Only 15 recent (2022-23) forest fires have severity truth, because LANDFIRE codes MTBS severity with a lag.",
]
credits = ("Data: MTBS burn severity via LANDFIRE Annual Disturbance (USGS/USFS), LANDFIRE fuels and topography (LFPS), "
           "FEDS fire event data (Chen et al. 2022), VIIRS 375 m active fire (NASA FIRMS / LANCE), gridMET (Abatzoglou 2013), "
           "HRRR (NOAA, AWS Open Data), Globe-LFMC (Yebra et al. 2019). All public; see the model card for licences. "
           "Simulator: FireSim (ML Ratchet Wildfire Phase 1). Part of ML Ratchet: trained on physics, graded on reality.")
out = dict(headline=headline, exam=dict(rung=E['rung'], n=E['n_records']), exam_intro=exam_intro, flow=flow,
           limits=limits, credits=credits, exam_tables=T['exam_tables'])
json.dump(out, open(f'{ROOT}/out/page_text.json', 'w'), indent=1)
print('ok')
