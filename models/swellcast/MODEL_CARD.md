# SwellCast v1.0 model card

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
the dev-fitted factors (distance x1.19, time x1.45, direction x1.33).

## Physics and simulator (training/swellsim.py)
- Deep-water group speed c_g = g/(4 pi f); a component emitted at t_s from distance D arrives at t_s + D/c_g.
  Check (physics_check.json): group speed exact (max rel. err 3e-11); the simulated
  ridge slope matches g/(4 pi D) within 1.4% at 1,500-9,000 km and the
  intercept gives t0 within 0.5 h; energy received / emitted = 1.0003.
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
storm bearing. Dev = 18 storms by seed 2026 (29 confirmed arrivals); exam = all other storms (288 confirmed,
632 visible, 2081 pairs in all). Nothing in the exam was used to choose anything.

## Ratchet
Objective = mean skill (distance, time, direction, storm Hs, detection) on a frozen simulated set; kept only if it
rises by > 0.01 AND the dev errors are no more than 5% worse than the incumbent. When the physics changed, a new frozen
set was built and every rung re-scored in order (current frozen set: val3). Incumbent: r4.pt.

| Rung | What changed | Frozen set | Objective | skills D/t0/dir/Hs/det | Dev D med / t0 med / dir med | Exam (not used) | Decision |
|---|---|---|---|---|---|---|---|
| r1 | r1: SwellNet 296k params, train1 (30k v1 sims in real NDBC backgrounds) + null_train1, 4000 steps | val3 | 0.386 | 0.30/0.25/0.62/0.00/0.75 | 40% / 35 h / 16 deg | 34% / 32 h / 21 deg | KEPT: first rung |
| r2 | r2: fine-tune r1 on train2 (12k v2 sims: dev-calibrated priors, crossing swells, NDBC low-f cutoff) + null_train2, 2500 steps | val3 | 0.442 | 0.37/0.34/0.68/0.00/0.82 | 58% / 50 h / 31 deg | 45% / 41 h / 24 deg | rejected: score 0.442 vs 0.386; dev errors +30% vs incumbent (gate fails) |
| r3 | r3: v3 physics - 65% of storms are REAL IBTrACS 1985-2007 hurricane tracks (moving, growing, decaying; Hs = a*Vmax) at their real place relative to the buoy; fine-tune r1 on train3 (12k) + null_train2, 2200 steps | val3 | 0.582 | 0.51/0.46/0.77/0.30/0.87 | 46% / 56 h / 24 deg | 43% / 39 h / 23 deg | rejected: score 0.582 vs 0.386; dev errors +21% vs incumbent (gate fails) |
| r4 | r4: domain randomisation over all simulator versions - fine-tune r1 on train1+train2+train3 (54k: v1, v2, v3 with real tracks) + null_train2, 3000 steps | val3 | 0.536 | 0.45/0.42/0.74/0.19/0.88 | 37% / 37 h / 22 deg | 38% / 33 h / 19 deg | KEPT: score 0.536 vs 0.386; dev errors +2% vs incumbent |
| r5 | r5: continue r4 for 3000 more steps on the same union (train1+2+3) at half the learning rate | val3 | 0.545 | 0.46/0.44/0.74/0.21/0.87 | 43% / 36 h / 19 deg | 41% / 34 h / 19 deg | rejected: score 0.545 vs 0.536 (needs +0.01); dev errors -5% vs incumbent |

Twin test (dev only, training/twin_net.py): each dev storm re-simulated from its real track and injected into quiet
records of the same buoy. r1's errors on the twins correlate with its real errors (r = 0.60 for log D,
0.65 for time); twin log-D MAE 0.135 vs real 0.200. Shipped rung r4: twin log-D MAE 0.124 vs real 0.207, twin t0 MAE 30 h vs real 45 h. Calibration record:
calib_v2_record.json (real ridge r1 0.47 vs v1 sims 0.79).

## Final exam (real, never used for any choice)
Dumb baseline = always the dev median (2,364 km, t0 -48 h, direction 171 deg).
Classical = pick the swell ridge and fit f = a (t - t0), D = g/(4 pi a), direction = energy-weighted buoy direction on the ridge.
Detection thresholds for both methods give 10% false alarms on real quiet 2024 windows; false alarms reported on
held-out quiet 2025 windows.

## exam confirmed
| Method | D error median | D within 20% | log10 D MAE | t0 error median | t0 MAE | dir error median | dir mean | detected | false alarms (quiet 2025) |
|---|---|---|---|---|---|---|---|---|---|
| SwellCast network | 38% | 30% | 0.200 | 33 h | 42 h | 19 deg | 32 deg | 63% | 11% |
| Classical ridge fit (Munk) | 59% | 15% | 0.326 | 48 h | 59 h | 20 deg | 28 deg | 43% | 10% |
| Dumb: dev median, always | 28% | 36% | 0.175 | 24 h | 33 h | 56 deg | 57 deg | always | 100% |

## exam visible
| Method | D error median | D within 20% | log10 D MAE | t0 error median | t0 MAE | dir error median | dir mean | detected | false alarms (quiet 2025) |
|---|---|---|---|---|---|---|---|---|---|
| SwellCast network | 42% | 24% | 0.225 | 39 h | 45 h | 36 deg | 52 deg | 58% | 11% |
| Classical ridge fit (Munk) | 66% | 12% | 0.356 | 61 h | 67 h | 45 deg | 59 deg | 46% | 10% |
| Dumb: dev median, always | 38% | 29% | 0.212 | 30 h | 40 h | 54 deg | 58 deg | always | 100% |

## exam all
| Method | D error median | D within 20% | log10 D MAE | t0 error median | t0 MAE | dir error median | dir mean | detected | false alarms (quiet 2025) |
|---|---|---|---|---|---|---|---|---|---|
| SwellCast network | 44% | 23% | 0.229 | 41 h | 47 h | 43 deg | 58 deg | 34% | 11% |
| Classical ridge fit (Munk) | 71% | 10% | 0.389 | 66 h | 79 h | 59 deg | 68 deg | 27% | 10% |
| Dumb: dev median, always | 40% | 28% | 0.224 | 33 h | 43 h | 62 deg | 64 deg | always | 100% |

## dev confirmed
| Method | D error median | D within 20% | log10 D MAE | t0 error median | t0 MAE | dir error median | dir mean | detected | false alarms (quiet 2025) |
|---|---|---|---|---|---|---|---|---|---|
| SwellCast network | 37% | 10% | 0.207 | 37 h | 45 h | 22 deg | 29 deg | 55% | 11% |
| Classical ridge fit (Munk) | 43% | 24% | 0.278 | 26 h | 43 h | 18 deg | 34 deg | 24% | 10% |
| Dumb: dev median, always | 17% | 52% | 0.131 | 24 h | 29 h | 17 deg | 29 deg | always | 100% |

Coverage of the 68% error bars on the confirmed exam (ideal 68%): distance 67%, time
74%, direction 70%.

By basin (confirmed exam, network): EP n=138: D 32%, t0 30 h, dir 18 deg; NA n=119: D 36%, t0 30 h, dir 19 deg; WP n=29: D 52%, t0 73 h, dir 45 deg.
By distance: 1000-2500 km n=133: D 49%, t0 30 h; 2500-5000 km n=114: D 24%, t0 29 h; 5000-8000 km n=41: D 52%, t0 59 h.

## Limits (read these)
- On distance and time the network does NOT beat the dumb median guess on this exam (median distance error
  38% vs 28%, time 33 h vs 24 h). Most exam pairs sit at similar
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
