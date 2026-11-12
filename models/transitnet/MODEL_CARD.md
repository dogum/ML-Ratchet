# TransitNet model card (v1.0 · rung 3)

**What it is.** A 1-D convolutional network (four input branches, about 320,000 weights) that reads a TESS 2-minute light curve of a star with a candidate periodic dip
(period, epoch, duration) and returns what the dip is (planet, eclipsing binary on the star, background eclipsing binary
blended into its light, or nothing real) and, for a planet, the radius ratio Rp/R*, impact parameter b, a/R* and the
duration, each with a 68% interval. Trained on **zero real planets**.

## The physics

Quadratic limb darkening; the light a disk of radius p blocks at separation z, integrated exactly over the covered annuli
(48-point Gauss-Legendre in a variable that removes the square-root endpoints), valid for any p, so the same function
eclipses a star by a bigger one. Keplerian orbits (eccentric beyond 4 days for planets), 2-minute exposures averaged.
Eclipsing binaries: both eclipses, surface-brightness ratio, ellipsoidal variation, third light. Background binaries: the
same, diluted 3 to 3,000 times, moving the flux-weighted centroid by (1 - m)/m x (1 - phi) x separation (simulated from
rung 2 on; the network that read centroids, rung 4, was not kept).

Against batman (Kreidberg 2015, exact Mandel & Agol quadratic limb darkening), instantaneous flux:

| k | b | e | depth (ppm) | max error (ppm) |
|---|---|---|---|---|
| 0.1 | 0.3 | 0 | 11877 | 0.006 |
| 0.02 | 0.0 | 0 | 471 | 0.943 |
| 0.15 | 0.95 | 0 | 12420 | 0.004 |
| 0.25 | 1.1 | 0 | 11321 | 0.005 |
| 0.08 | 0.5 | 0.3 | 7554 | 0.006 |
| 0.05 | 0.7 | 0.6 | 2541 | 0.001 |
| 0.01 | 0.4 | 0 | 115 | 0.014 |

A star eclipsed by a bigger one, against pixel-by-pixel integration (4001 x 4001):

| z | p | ours | brute force |
|---|---|---|---|
| 0.0 | 3.0 | 1.00000 | 1.00000 |
| 2.5 | 3.0 | 0.79711 | 0.79767 |
| 3.4 | 3.0 | 0.21414 | 0.21413 |
| 1.2 | 1.5 | 0.63216 | 0.63235 |
| 0.5 | 1.0 | 0.71373 | 0.71446 |

## The data

Noise: 2,200 ordinary TESS 2-minute targets (no TOI, no known planet), 220 from each of sectors 3, 9, 15, 22, 28, 36, 44, 50, 57
and 64; each simulated reading stitches 1 to 3 of them of similar brightness. Signals: planets (45%: P 0.4-25 d log-uniform,
Rp/R* 0.01-0.3 log-uniform, b mostly 0-1, a/R* from the TIC density by Kepler's law, limb darkening drawn in Kipping's
q1, q2), eclipsing binaries (20%), background binaries (20%), nothing (15%). Catalogue-like candidate ephemerides: period errors
of 1e-7 to 3e-5 of the period (5x the stated error 15% of the time), epoch within 0.15 durations, duration within 15%, binaries
at half their period 15% of the time. 60,000 training readings; 5,000 frozen for the ratchet.

## The ratchet

Every candidate network was scored on a frozen set of 5,000 simulated readings: half the score for telling planets from everything else (2 AUC - 1), half for the four planet parameters (skill against guessing the median, floored at 0). It was kept only if that rose by more than 0.01 and it did no worse on 40 real dev stars that no exam uses: 20 confirmed planets and 20 false positives (log-loss within 0.05, radius-ratio error within 1 point). When the dev planets showed the simulated limb darkening was wrong, the frozen set was regenerated and every rung re-scored. The exam is reported, never used to choose.

| Rung | Change | Decision | Sim score | Sim AUC | Dev log-loss | Dev Rp/R* error | Exam AUC | Exam Rp/R* error | Exam b error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline | kept | 0.838 | 0.938 | 1.224 | 5.5% | 0.781 | 5.4% | 0.091 |
| 2 | Limb darkening from real planets | rejected | 0.735 | 0.922 | 0.856 | 34.0% | 0.760 | 36.6% | 0.104 |
| 3 | Corrections, not guesses | kept | 0.859 | 0.948 | 0.935 | 3.0% | 0.835 | 3.4% | 0.085 |
| 4 | Where the light dims | rejected | 0.861 | 0.951 | 0.981 | 2.9% | 0.832 | 3.6% | 0.086 |

## Final exam

Every confirmed transiting planet with an orbit under 25 days that TESS watched at 2-minute cadence and saw at signal-to-noise 7 or more, and every TESS Object of Interest that follow-up proved a false positive. The 40 dev stars are left out. Published answers come from the NASA Exoplanet Archive's composite table; they were fitted with more data than one TESS light curve.

Planet or not (probability above 0.5 called a planet), AUC 0.835:

| Real signals | Count | Called a planet |
|---|---|---|
| Confirmed planets | 1196 | 98% |
| False positives, all | 832 | 66% |
| flagged by a centroid offset or a nearby star | 307 | 64% |
| eclipsing or spectroscopic binaries | 262 | 62% |
| other reasons | 263 | 71% |
| False alarms (no real signal) | 33 | 85% |

Parameters by signal-to-noise (median absolute error; coverage: published value inside TransitNet's 68% interval with the
published error added):

| SNR | Planets | Rp/R* error | sqrt(depth) | b error | a/R* error | Rp/R* coverage |
|---|---|---|---|---|---|---|
| 7 to 20 | 386 | 6.7% | 6.8% | 0.096 | 5.4% | 76% |
| 20 to 100 | 452 | 3.5% | 5.7% | 0.085 | 4.3% | 68% |
| 100+ | 337 | 1.6% | 4.4% | 0.070 | 4.2% | 66% |

Against a least-squares transit fit on the same folded data (0.8 s per star, fixed limb darkening, no error bars, no vetting):

| SNR and method | Planets | Rp/R* | b | a/R* |
|---|---|---|---|---|
| 7 to 20: TransitNet | 398 | 6.7% | 0.10 | 5.4% |
| 7 to 20: Least-squares fit | 398 | 10.7% | 0.31 | 28.9% |
| 7 to 20: square root of the depth | 398 | 6.8% | — | — |
| 20 to 100: TransitNet | 460 | 3.5% | 0.09 | 4.3% |
| 20 to 100: Least-squares fit | 460 | 4.7% | 0.23 | 12.6% |
| 20 to 100: square root of the depth | 460 | 5.7% | — | — |
| 100 and up: TransitNet | 338 | 1.6% | 0.07 | 4.2% |
| 100 and up: Least-squares fit | 338 | 2.3% | 0.09 | 3.7% |
| 100 and up: square root of the depth | 338 | 4.4% | — | — |
| all: TransitNet | 1196 | 3.4% | 0.09 | 4.6% |
| all: Least-squares fit | 1196 | 4.7% | 0.20 | 10.7% |
| all: square root of the depth | 1196 | 5.5% | — | — |

## Limits

- It cannot rule out most false positives from TESS photometry alone: it calls 66% of the exam's false positives planets. Many need ground-based images, spectra or radial velocities; TransitNet's job is to rank them, and its score is best used that way.
- It knows TESS 2-minute light curves of single stars, orbits under 25 days, and dips the pipeline can fold. Single transits, very long periods, 20-second or 30-minute data, and circumbinary planets are outside what it learned.
- Limb darkening is a broad prior, set from the dev planets' own fits, not from stellar atmosphere tables (the Claret tables could not be downloaded here). Its error bars include that; very hot or very cool stars may be biased.
- Published answers are not ground truth either: they come from different papers, bandpasses and dilution corrections. Part of the scatter in the exam is theirs.
- Impact parameter is the hardest number: 0.09 typical error, worse near grazing. a/R* assumes the orbit's shape is unknown to it; eccentric orbits shift it.
- Stellar radius comes from the TIC catalogue. Planet sizes in Earth radii carry its error, which TransitNet does not see.

Data: NASA TESS (SPOC light curves via MAST), NASA Exoplanet Archive (pscomppars), ExoFOP TOI list.
