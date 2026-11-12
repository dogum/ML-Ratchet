"""Write release/MODEL_CARD.md and release/README.md from the result files (final.json, ratchet.json, physics_check.json,
calib_v2_record.json, twin_net_r1.json, logs/tables.md from webtext.py). Run after final_exam.py and webtext.py."""
import os, json, numpy as np
import evaluate as E
ROOT = E.ROOT; REL = os.path.join(ROOT, 'release')
fin = json.load(open(os.path.join(ROOT, 'final.json'))); rat = json.load(open(os.path.join(ROOT, 'ratchet.json')))
phys = json.load(open(os.path.join(ROOT, 'physics_check.json'))); cal = json.load(open(os.path.join(ROOT, 'calib_v2_record.json')))
tw = json.load(open(os.path.join(ROOT, 'twin_net_r1.json')))['summary']; tw4 = json.load(open(os.path.join(ROOT, 'twin_net_r4.json')))['summary']; tables = open(os.path.join(ROOT, 'logs/tables.md')).read()
base = json.load(open(os.path.join(ROOT, 'baseline.json'))); dv = base['dumb_values']
H = fin['exam_confirmed']; N = H['network']; sf = fin['sigma_factor']; ridge = phys['ridge']
inc = [r for r in rat if r['accepted']][-1]

card = f"""# SwellCast v1.0 model card

## What it does
SwellCast reads ten days of wave spectra from one ocean buoy and answers: is swell from a distant storm arriving near
the middle of this window, how far away was the storm, when did it peak, and which direction is it in? It is Walter
Munk's 1963 idea (the swell frequency at the buoy rises linearly in time, df/dt = g / (4 pi D)) done by a small
neural network trained only on simulated swell laid on top of real buoy records.

## Inputs
A 240-hour window of NOAA NDBC spectral data: energy density (m^2/Hz) on 34 frequency bins 0.0325-0.1975 Hz every
3 hours (80 steps), plus the first directional moments r1 cos(alpha1), r1 sin(alpha1) (NaN where the buoy has no
directional data). `pipeline.py` builds this from the public NDBC files (swden `w`, swdir `d`, swr1 `j`; historical
or realtime). The window should be centred near the expected arrival of 0.06 Hz (16.7 s) swell; training used a
random offset of up to +-36 h.

## Outputs
`SwellCast().read(window)` returns: `D_km` and `logD_sig` (68% error bar in log10 km), `t0` (storm peak time, hours
from the window centre) and `t0_sig`, `dir` (degrees, direction the swell comes FROM) and `kappa` (von Mises
concentration; 1-sigma ~ 57.3/sqrt(kappa) deg), `hs` (storm wave height, m; weak skill), `p_swell` and `detected`
(score above the threshold that gives 10% false alarms on real quiet 2024 windows). Error bars are already widened by
the dev-fitted factors (distance x{sf['logD']:.2f}, time x{sf['t0']:.2f}, direction x{sf['dir']:.2f}).

## Physics and simulator (training/swellsim.py)
- Deep-water group speed c_g = g/(4 pi f); a component emitted at t_s from distance D arrives at t_s + D/c_g.
  Check (physics_check.json): group speed exact (max rel. err {phys['group_speed']['max_rel_err']:.0e}); the simulated
  ridge slope matches g/(4 pi D) within {100 * max(abs(r['rel_err']) for r in ridge):.1f}% at 1,500-9,000 km and the
  intercept gives t0 within {max(abs(r['t0_from_intercept_h']) for r in ridge):.1f} h; energy received / emitted = {phys['energy_conservation']['received_over_emitted']:.4f}.
- Storm: a moving, finite patch (7 sub-areas, radius 80-600 km) emitting a JONSWAP sea that grows and decays
  (Hasselmann et al. 1973), partly isotropic and partly beamed along the storm heading.
- Spherical spreading over the great circle (Munk 1963) and swell dissipation by air friction (Ardhuin et al. 2009,
  WAVEWATCH III ST4 form), with the swell's own decaying orbital velocity.
- Directional moments from each sub-area's bearing, intrinsic r1, mixed with the real background by energy.
- chi-square (24 dof) spectral estimation noise, NDBC 0.01 m^2/Hz quantisation, optional local wind-sea burst.
- v2 (dev-calibrated): weaker, longer-lived storms, shorter periods, broader directions, crossing swells from other
  storms, the NDBC low-frequency cutoff. v3: 65% of storms are real IBTrACS 1985-2007 hurricane tracks.

## Real noise
Every training window is a real 240-h NDBC record from 27 buoys (2008-2025), at least 70% complete, not overlapping any
candidate hurricane window at that buoy, and with long-period energy below the buoy's 60th percentile. The simulated
swell is added to it. 25% of each batch is a real null window with nothing added (detection negatives).

## Network
SwellNet: 2-D CNN over frequency x time, 5 input channels (log energy, anomaly against the window's per-frequency
median, two directional moments, data mask), 4 residual stages (24-64 channels), 296k parameters. Heads: mean and
log-sigma for log10 D, t0, log10 Hs (beta-NLL), a 2-vector for direction (von Mises NLL), a detection logit.
`swellcast.py` is numpy-only with BatchNorm folded into the convolutions; max |numpy - torch| = 1.8e-5 on dev windows.

## Truth, dev and exam (training/events.py)
Truth = IBTrACS v04r01 best tracks: storms with peak 1-min wind >= 64 kt, the point of peak wind (D, bearing, time).
Every (storm, buoy) pair 1,000-8,000 km apart over open water, 2008-2025, >= 70% data, not crowded by another
hurricane within 72 h. 'Visible': the energy along the ridge predicted by the true storm is >= 2x what it was 48 h
earlier. 'Confirmed' (headline): visible AND the buoy's measured direction along that ridge is within 45 deg of the
storm bearing. Dev = 18 storms by seed 2026 (29 confirmed arrivals); exam = all other storms ({N['n']} confirmed,
{fin['exam_True']['network']['n']} visible, {fin['exam_None']['network']['n']} pairs in all). Nothing in the exam was used to choose anything.

## Ratchet
Objective = mean skill (distance, time, direction, storm Hs, detection) on a frozen simulated set; kept only if it
rises by > 0.01 AND the dev errors are no more than 5% worse than the incumbent. When the physics changed, a new frozen
set was built and every rung re-scored in order (current frozen set: {inc['val_set']}). Incumbent: {inc['ckpt']}.

{tables.split('## ratchet')[1].strip()}

Twin test (dev only, training/twin_net.py): each dev storm re-simulated from its real track and injected into quiet
records of the same buoy. r1's errors on the twins correlate with its real errors (r = {tw['corr_logD_err']:.2f} for log D,
{tw['corr_t0_err']:.2f} for time); twin log-D MAE {tw['twin_logD_mae']:.3f} vs real {tw['real_logD_mae']:.3f}. Shipped rung r4: twin log-D MAE {tw4['twin_logD_mae']:.3f} vs real {tw4['real_logD_mae']:.3f}, twin t0 MAE {tw4['twin_t0_mae']:.0f} h vs real {tw4['real_t0_mae']:.0f} h. Calibration record:
calib_v2_record.json (real ridge r1 {cal['dev_real']['r1_on_ridge']:.2f} vs v1 sims {cal['sim_v1']['r1_on_ridge']:.2f}).

## Final exam (real, never used for any choice)
Dumb baseline = always the dev median ({10 ** dv['logD']:,.0f} km, t0 {dv['t0']:.0f} h, direction {dv['dir']:.0f} deg).
Classical = pick the swell ridge and fit f = a (t - t0), D = g/(4 pi a), direction = energy-weighted buoy direction on the ridge.
Detection thresholds for both methods give 10% false alarms on real quiet 2024 windows; false alarms reported on
held-out quiet 2025 windows.

{tables.split('## ratchet')[0].strip()}

Coverage of the 68% error bars on the confirmed exam (ideal 68%): distance {100 * N['logD_in_1sig']:.0f}%, time
{100 * N['t0_in_1sig']:.0f}%, direction {100 * N['dir_in_1sig']:.0f}%.

By basin (confirmed exam, network): """ + '; '.join(f"{k} n={v['n']}: D {v['D_med_pct']:.0f}%, t0 {v['t0_med_h']:.0f} h, dir {v.get('dir_med_deg', np.nan):.0f} deg" for k, v in H['by_basin'].items()) + """.
By distance: """ + '; '.join(f"{k} n={v['n']}: D {v['D_med_pct']:.0f}%, t0 {v['t0_med_h']:.0f} h" for k, v in H['by_distance'].items()) + f""".

## Limits (read these)
- On distance and time the network does NOT beat the dumb median guess on this exam (median distance error
  {N['D_med_pct']:.0f}% vs {H['dumb']['D_med_pct']:.0f}%, time {N['t0_med_h']:.0f} h vs {H['dumb']['t0_med_h']:.0f} h). Most exam pairs sit at similar
  distances and the window centring gives every method the arrival time to within +-36 h. It does beat the classical
  ridge fit on distance, time and detection, matches it on direction, and its error bars are calibrated.
- Fails most on far storms (5,000-8,000 km) and West Pacific typhoons.
- Truth only for tropical cyclones; winter extratropical storms (most long-period swell at Hawaii/California) are not graded.
- One-point truth (peak wind) for a storm that makes swell for days along a track.
- The quiet false-alarm windows can still hold real swell from unlisted storms, so false-alarm rates are upper bounds.
- Simulator versions 2 and 3 matched the dev data better but networks trained on them read real dev arrivals worse; the
  reason is not yet understood.

## Data credits and licences
NOAA National Data Buoy Center spectral wave data (public domain). IBTrACS v04r01: Knapp et al. 2010, BAMS 91, 363-376;
NOAA NCEI (public domain). Coastlines: Natural Earth (public domain). Physics: Munk et al. 1963; Snodgrass et al. 1966;
Hasselmann et al. 1973; Ardhuin et al. 2009. Open-Meteo archive/marine API was tried but its daily limit was exceeded
from this shared IP, so it was not used.
"""
open(os.path.join(REL, 'MODEL_CARD.md'), 'w').write(card)

readme = f"""# SwellCast v1.0

Finds swell from distant storms in one buoy's wave spectra and says how far away the storm was, when it peaked and
which direction it is in. Trained only on physics simulations laid on real NOAA buoy records; tested on {N['n']} real
hurricane and typhoon swells it never saw. See MODEL_CARD.md for the physics, the exam and the limits.

## Run
    python3 demo.py        # numpy + scipy only; raw NDBC files -> window -> answer, then 5 real exam windows

    import pipeline as PL, swellcast as SC
    rec = PL.read_ndbc('51004w2018.txt.gz', '51004d2018.txt.gz', '51004j2018.txt.gz')
    win, tc = PL.window(rec, PL.hour('2018-09-17T12'))     # 10 days centred on this hour
    print(SC.SwellCast().read(win))

Files: `swellcast.py` (numpy network), `pipeline.py` (NDBC reader and windowing), `swellcast_weights.npz`,
`examples/` (raw NDBC files for buoy 51004 in 2018 and five pre-cut exam windows, a fixed random draw),
`training/` (every script, result json files: ratchet.json, final.json, physics_check.json, ...).

## Headline (confirmed exam, n={N['n']})
| | distance error (median) | storm time error (median) | direction error (median) | detected |
|---|---|---|---|---|
| SwellCast | {N['D_med_pct']:.0f}% | {N['t0_med_h']:.0f} h | {N['dir_med_deg']:.0f} deg | {100 * N['detect_rate']:.0f}% |
| Classical ridge fit | {H['classical']['D_med_pct']:.0f}% | {H['classical']['t0_med_h']:.0f} h | {H['classical']['dir_med_deg']:.0f} deg | {100 * H['classical_detect_rate']:.0f}% |
| Dev median guess | {H['dumb']['D_med_pct']:.0f}% | {H['dumb']['t0_med_h']:.0f} h | {H['dumb']['dir_med_deg']:.0f} deg | - |

Data: NOAA NDBC and IBTrACS (public domain). Part of the ML Ratchet physics + ML collection.
"""
open(os.path.join(REL, 'README.md'), 'w').write(readme); print('wrote MODEL_CARD.md, README.md')
