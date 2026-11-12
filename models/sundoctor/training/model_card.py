import _paths as _P  # portable paths (see _paths.py)
import json
R = json.load(open(_P.WORK + '/ratchet.json')); D = json.load(open(_P.LAB + '/data.json')); ph = json.load(open(_P.WORK + '/physics_check.json'))
E = D['exam']; I = E['inject']; P = E['episodes']; X = E['real']; V = D['release']['val']
rows = []
for i, r in enumerate(R):
    v = r['val']; s = lambda k: f"{v[k]['skill']:.2f}"
    rows.append(f"| {i + 1} | {r['note'].split(':')[0]} | {'kept' if r['accepted'] else 'rejected'} | {r['objective']:.3f} | {s('pf')} | {s('frac')} | {s('soil')} | {s('shade')} | {s('clip')} | {s('outage')} | {v['step_auc']:.3f} | {100 * v['count_acc']:.1f}% | {100 * r['real']['count_acc']:.1f}% / {100 * r['real']['count_acc_when_dead']:.0f}% |")
conf = X['confusion_true_rows_0_1_2_3plus']
card = f"""# SunDoctor model card (v1.0, rung 3)

**What it is.** A 2-D convolutional network ({590603:,} weights) that reads 14 days x 96 fifteen-minute steps (local *solar* time)
of one PV system: AC power per kW of DC nameplate, plane-of-array irradiance, air temperature, and physics-derived channels
(clear-sky POA, a PVWatts expectation from the measured POA, sun angles, a data mask, and the system's own healthy level).
For every day it returns, each with a 68% error bar:

- `pf`: performance factor of the clean, unshaded, unclipped array, relative to nameplate physics
- `frac`: fraction of the array working (dead strings: `round(n_strings x (1 - frac))`)
- `soil`, `shade`, `clip`, `outage`: soiling loss, and the share of energy lost to shading, inverter clipping and outages
- `step_p`: probability that part of the array failed or was repaired that day

Plus per window: `eta0` (baseline efficiency) and `ac_limit` (inverter limit per kW DC).

**Files.** `sundoctor.py` (numpy, plus pandas for CSV input), `sundoctor_weights.npz` (BatchNorm fused), `examples/`
(six months of real data from PVDAQ system 2107, inverter 11), `training/`.

## The simulator (all training data)

Vectorized physics in solar time, each block checked against pvlib 0.16:

| Block | Model | Max / median difference vs pvlib |
|---|---|---|
| Sun position | Spencer declination + equation of time | elevation {ph['sun_elevation_maxerr_deg']:.2f} deg, azimuth {ph['sun_azimuth_maxerr_deg']:.2f} deg |
| Clear sky | Ineichen-Perez | GHI {100 * ph['clearsky_ghi_rel_err_noon']:.2f}%, DNI {100 * ph['clearsky_dni_rel_err']:.2f}% |
| Decomposition | Erbs | DHI {100 * ph['erbs_dhi_rel_err']:.2f}% |
| Transposition | Hay-Davies | POA {100 * ph['poa_rel_err']:.4f}% |
| Cell temperature | Faiman | {ph['faiman_maxerr_C']:.3f} C |
| DC / inverter | PVWatts (AC clipping) | exact |

Systems: latitude 20-52 N, tilt 0-40 deg, azimuth 90-270, DC/AC 0.95-1.5, temperature coefficient -0.28 to -0.48 %/C,
baseline efficiency 0.80-1.02. Weather: clear, hazy, broken-cloud (Markov) and overcast days. Faults: 1-12 strings per
inverter with failures or repairs inside the window; soiling at 0-0.6 %/day reset by rain or cleaning; row-to-row shading
(profile-angle model) or horizon obstacles applied to the beam; outages, trips and curtailment. Measurements: power
noise, irradiance gain error and noise (20% of systems with satellite-like 10% noise), timestamp shifts, missing data.
The system's healthy reference level is supplied in 65% of windows. Total: {D['release']['samples']:,} simulated fortnights.

## The ratchet

Objective on a frozen set of 2,000 simulated fortnights (seed 777): mean skill (1 - MAE / MAE of predicting the median)
over pf, frac, soil, shade, clip and outage, + 0.5 x step-detection AUC + 0.5 x dead-string counting accuracy. A candidate
is kept only if the objective rises by more than 0.01. The real farm was never used to choose. Rung 2 introduced beta-NLL
(Seitzer et al. 2022, beta = 0.5): with plain Gaussian NLL, rung 1 had learned to wave rare large losses away with huge
error bars (outage skill 0.00).

| Rung | Change | Decision | Objective | pf | frac | soil | shade | clip | outage | step AUC | count (sim) | real: count / when dead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Rungs 1-2 were re-scored on the real farm with the final answer key and reference rule (decisions unaffected).
Rung 5 did slightly better on the real farm than rung 3, but the ratchet does not look at the farm, so rung 3 shipped.

## Final exam: PVDAQ system 2107 (Arbuckle, California)

24 identical ABB TRIO-27.6 string inverters, each with 6 strings of 20 Hyundai 310 W modules (37.2 kW DC, 30 kW AC),
fixed 25 deg tilt facing south, POA and air temperature on site, Nov 2017 - Nov 2023 at 5-15 min. Timestamps were shifted
by -3.5 min so that clear-day POA is symmetric about solar noon (power and irradiance are mutually aligned).

**1. Sibling answer key.** For each inverter-day, the ratio to the fleet's healthy level (80th percentile of the 24)
on unclipped daylight; days whose morning and afternoon disagree by more than 0.06 are ambiguous and dropped.
SunDoctor saw only the one inverter, with a reference level computed from its own predictions (97.5th percentile of daily
pf over the previous year). {X['n_inverter_days']:,} clean inverter-days:

| Siblings say \\ SunDoctor says | 0 dead | 1 | 2 | 3+ |
|---|---|---|---|---|
| 0 dead | {conf[0][0]:,} | {conf[0][1]:,} | {conf[0][2]:,} | {conf[0][3]:,} |
| 1 | {conf[1][0]:,} | {conf[1][1]:,} | {conf[1][2]:,} | {conf[1][3]:,} |
| 2 | {conf[2][0]:,} | {conf[2][1]:,} | {conf[2][2]:,} | {conf[2][3]:,} |
| 3+ | {conf[3][0]:,} | {conf[3][1]:,} | {conf[3][2]:,} | {conf[3][3]:,} |

Overall {100 * X['count_acc']:.1f}% right; healthy days called healthy {100 * X['healthy_called_healthy']:.1f}%; days with dead strings counted
right {100 * X['count_acc_when_dead']:.1f}% (most misses are faults that had already lasted long enough to pull the reference down).
Discrete events (ten clean days before): string failures flagged within a day {P['onsets_found_pm1day']}/{P['string_failures']}, counted as exactly one dead
string {P['counted_one_dead_after']}/{P['string_failures']}; repairs flagged within a day {P['repairs_found']}/{P['repairs']}; inverters going offline {P['offline_found']}/{P['offline_events']}.

**2. Injections into real healthy fortnights** ({I['n_windows']} windows the siblings call healthy):
dead string(s) on day 7 counted exactly afterwards {100 * I['dead_count_right_after']:.1f}%, onset within a day {100 * I['dead_onset_found_pm1']:.0f}%, healthy before
{100 * I['dead_healthy_before']:.1f}%; dust ramps of 2-8%: performance-drop error {100 * I['soil_pf_drop_mae']:.1f}%, mistaken for dead strings {100 * I['soil_mistaken_for_dead_string']:.1f}%;
tree shade (true mean loss {100 * I['shade_true_mean']:.1f}%): error {100 * I['shade_mae']:.2f}%, correlation {I['shade_corr']:.2f}; inverter limit 0.70 per kW DC:
clipping-loss error {100 * I['clip_mae']:.1f}% (true mean {100 * I['clip_true_mean']:.1f}%), AC-limit error {I['ac_limit_mae']:.3f}; 3-hour trip size error
{100 * I['outage_trip_mae']:.1f}%, dead day called an outage {100 * I['outage_dead_day_pred']:.1f}%.

**3. RdTools** on inverter 11's healthy days: degradation {E['rdtools']['rdtools_yoy_degradation_pct_per_yr']:.2f} %/yr (YoY, 68% CI
{E['rdtools']['rdtools_ci68'][0]:.2f} to {E['rdtools']['rdtools_ci68'][1]:.2f}) vs SunDoctor {E['rdtools']['sundoctor_degradation_pct_per_yr']:.2f} %/yr. Soiling (SRR): RdTools {100 * E['soiling']['mean_loss_rdtools']:.1f}%
mean loss, SunDoctor {100 * E['soiling']['mean_loss_sundoctor']:.1f}%. SunDoctor attributes the winter shortfall to row-to-row shading (up to
~8% in December) and spring losses to clipping; RdTools, which has no shading model, counts the winter shortfall as soiling.
SunDoctor likely under-reports gradual summer soiling.

## Limits

- Needs plane-of-array irradiance. Satellite irradiance was simulated (10% noise) but not tested on real satellite data.
- Separates dead strings from dust partly by memory (its own best level over the past year). A string dead from the
  first day of the record looks like a weaker system.
- Soiling is its weakest output.
- One real farm tested (fixed tilt, California). No trackers, snow, bifacial modules or microinverters in training.
- The on-site POA sensor reads relatively low in the morning compared with the array (orientation mismatch); training
  assumed sensors aligned with the array.
- Morning-only string disconnects (e.g. insulation faults until dew dries) are not modelled.
"""
open(_P.WORK + '/release/MODEL_CARD.md', 'w').write(card); print('card ok', len(card))
