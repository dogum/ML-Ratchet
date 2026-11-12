# StarQuake v1.0 — model card

## What it does
Reads a Kepler long-cadence light curve of a red giant (27 days to 4 years) and returns:
whether solar-like oscillations are present (numax 5-275 uHz), the frequency of maximum oscillation power **numax** and the
large frequency separation **Delta nu**, each with a 1-sigma error bar, and (given Teff) mass and radius from the asteroseismic
scaling relations M = (numax/3090)^3 (dnu/135.1)^-4 (Teff/5777)^1.5, R = (numax/3090) (dnu/135.1)^-2 (Teff/5777)^0.5.
The solar references (3090 uHz, 135.1 uHz, 5777 K) are those of Yu et al. 2018, so results compare directly with that catalogue.
(With the IAU 5772 K, solar numax would scale by sqrt(5777/5772), a 0.04% change.)

Released rung: **r5**. numpy/scipy inference (`starquake.py`), BatchNorm folded into convolutions; numpy equals torch to 7e-7; the in-browser JS port equals Python to 1e-6.

## Inputs and preprocessing (`pipeline.py`)
PDCSAP flux, quality-flagged cadences removed, per-quarter median normalisation to ppm, gaps of <= 3 cadences interpolated,
10-day running-mean high-pass, 5-sigma clip, power density (ppm^2/uHz, Parseval-normalised for the duty cycle).
Features: log power on 1024 log-spaced bins 3-283 uHz; a 48 x 110 comb map (autocorrelation of the spectrum within +-3.5 expected
separations of 48 trial centres, lags 0.56-1.65 expected separations); log observing length and duty cycle.

## Physics in the simulator (`training/starsim.py`)
Granulation: two Harvey-like super-Lorentzians, amplitudes and timescales scaling with numax and mass (Kallinger et al. 2014, Table 2) with scatter.
Envelope: Gaussian, FWHM 0.66 numax^0.88 (Mosser et al. 2012), dev-calibrated narrowing at high numax (v2). Radial-mode amplitude
from Yu et al. 2018 eq. 8 (RGB/clump coefficients), divided by sqrt(2) for the Kjeldsen et al. 2008 convention (v2), 12% of stars diluted.
Modes: universal red-giant pattern (Mosser et al. 2011): eps(dnu), d01, d02, d03, curvature alpha = 0.008; visibilities 1, 1.54, 0.63, 0.071.
Dipole mixed modes (70% of stars) from tan(theta_p) = q tan(theta_g) with DPi1 and q drawn for RGB or clump; power and width scaled by (1 - zeta).
Linewidths 0.05-0.3 uHz (log-normal, Teff^3). Stochastic excitation: random complex Gaussian amplitude in every frequency bin (chi^2 2 dof).
Apodisation sinc^2(pi nu / 2 nu_Nyq). dnu from M, Teff and numax via the scaling relations + 2% scatter; clump stars 2% lower.
Synthesis by inverse FFT on a grid twice the observed span; added to a real quiet-dwarf Kepler light curve (real noise and gaps).

Verification (`training/physics_check.json`): chi^2 statistics mean 1.004, variance 1.001 (expected 1, 1);
Parseval variance ratio 0.989-0.988; classical recovery on 24 simulated stars: numax 1.40%, dnu 0.58% median error;
mixed modes 23 found vs 22.2 expected (numax 60, DPi1 75 s);
scaling relations reproduce Yu et al.'s 16,094 masses and radii (median ratio 1.000, scatter 0.26% / 0.04% from rounding);
simulated dnu-numax slope 0.764 (Yu catalogue 0.739, Mosser 2012 0.751); the Sun gives 3090 uHz / 135.1 uHz (by construction of the references).
Twin test on the 18 dev stars (`twin_test.json`): v1 humps -0.24 dex vs real, v2 -0.03 dex.
Network twin test (`twin_net.json`, r1): median |error| real 1.6% / 0.9%, twins 1.2% / 1.1% (numax / dnu).

## Data
- Truth: Yu et al. 2018 (ApJS 236, 42), 16,094 Kepler red giants, journal machine-readable tables. Exam: 420 stars (70 per numax stratum 8-15-25-40-70-130-250 uHz, seed 2026), >= 1,200 d of data. Dev: 18 other stars (3 per stratum).
- Noise: 360 + 60 quiet Kepler dwarfs (DR25: logg > 4.25, no KOI/TCE), screened for variability in-band; 232 train / 42 validation kept.
- Exam negatives: 120 more dwarfs, NOT screened (seed 2027).
- Gaia DR3 FLAME radii for 222 exam stars (ESA Gaia TAP).
- Light curves: Kepler long cadence from MAST (stpubdata public S3 bucket).

## Network
Two CNN branches (1-D on the log spectrum, 2-D on the comb map) + MLP head, 412k parameters, beta-NLL (beta 0.5) on ln numax and ln dnu,
BCE on detection, EMA weights, AdamW, cosine schedule. Training data: 40,000 simulations (sim1 v1 physics 15k, sim2 v2 15k, sim3 v2 10k with 30% non-oscillating).

## Ratchet (`training/ratchet.json`)
Objective on the frozen simulated set (val2: 5,000 v2 giants in 42 held-out noise stars; valneg: 2,000 non-oscillating):
mean of numax-within-5% rate, dnu-within-2% rate, detection balanced accuracy. Keep if objective > incumbent + 0.01 AND the mean ratio of four
dev median errors (whole and 90 d; numax and dnu) vs the incumbent is <= 1.10. When the physics changed (v1 -> v2) all rungs were re-scored on val2 and decisions re-taken.

| rung | what changed | objective | numax hit / dnu hit / det | dev med err (whole numax, dnu; 90 d numax, dnu) | decision |
|---|---|---|---|---|---|
| r1 | v1 physics (published relations), sim1 15k, 3500 steps | 0.593 | 0.759 / 0.519 / 0.500 | 1.60% / 0.89% / 1.35% / 2.15% | kept: first rung |
| r2 | v2 physics (dev-calibrated: Kjeldsen amplitude convention, narrower humps), sim2 15k, 3500 steps | 0.584 | 0.738 / 0.514 / 0.500 | 1.66% / 1.20% / 2.49% / 2.00% | rejected: score 0.584 vs 0.593 (needs +0.01); dev errors +29% vs the incumbent (gate: at most +10%) |
| r3 | adds a detection output; trained on sim2 + sim3 (30% non-oscillating: quiet dwarfs alone, or giants with numax below the band), 25k, 3500 steps | 0.738 | 0.734 / 0.499 / 0.980 | 1.65% / 1.32% / 2.61% / 1.80% | rejected: score 0.738 vs 0.593; dev errors +32% vs the incumbent (gate: at most +10%) |
| r4 | warm start from r1 + detection output; trained on all banks (v1 + v2 physics: the amplitude convention spans both; 30% of sim3 non-oscillating), 40k, 4000 steps | 0.737 | 0.747 / 0.482 / 0.982 | 1.28% / 1.16% / 1.71% / 2.05% | kept: score 0.737 vs 0.593; dev errors +8% vs the incumbent |
| r5 | r4 trained further: 4000 more steps, batch 96, lr 7e-4, same 40k banks | 0.757 | 0.762 / 0.530 / 0.979 | 1.01% / 0.78% / 2.34% / 1.57% | kept: score 0.757 vs 0.737; dev errors -10% vs the incumbent |

Note: r4's objective gain came from the detection output (its frozen-set dnu hit rate was lower than r1's, 0.482 vs 0.519); r5, the same network trained longer, recovered it (0.530) and cut the dev errors by 10%.

## Exam (420 Kepler red giants, never used for any decision)
Median |ln error| / fraction within 5% (numax) or 2% (dnu). Coverage: fraction within 1 and 2 sigma, sigma = sqrt(sigma_net^2 + sigma_published^2), widen factors numax x1.00, dnu x1.43 fitted on dev.

| length | n | StarQuake numax | classical numax | dumb | StarQuake dnu | classical dnu | dumb | cov 1-sigma (numax/dnu) | cov 2-sigma |
|---|---|---|---|---|---|---|---|---|---|
| whole (~4 yr) | 420 | 1.49% / 89% | 0.89% / 91% | 77% | 0.89% / 78% | 0.64% / 84% | 56% | 72% / 78% | 94% / 97% |
| 365 d | 418 | 1.68% / 89% | 1.67% / 79% | 77% | 1.02% / 74% | 1.05% / 69% | 58% | 71% / 78% | 94% / 97% |
| 90 d | 420 | 2.17% / 82% | 3.89% / 57% | 77% | 2.07% / 48% | 3.06% / 42% | 56% | 66% / 65% | 93% / 92% |
| 54 d | 420 | 2.76% / 77% | 5.97% / 46% | 77% | 2.47% / 41% | 7.63% / 28% | 56% | 63% / 66% | 92% / 90% |
| 27 d | 420 | 3.61% / 62% | 16.05% / 34% | 77% | 4.60% / 23% | 26.80% / 13% | 56% | 54% / 58% | 90% / 88% |

Classical = SYD-style pipeline (version chosen on dev: multi-start background fit for pieces < 300 d).

Detection (420 giants should be yes; 120 unscreened quiet dwarfs should be no). StarQuake: p > 0.5 fixed. Classical: comb strength threshold chosen on dev.

| length | StarQuake TPR | StarQuake FPR | classical TPR | classical FPR |
|---|---|---|---|---|
| 0 | 99.8% | 11.7% | 96.9% | 4.2% |
| 365 | 99.8% | 12.5% | 91.4% | 30.8% |
| 90 | 99.8% | 15.0% | 62.4% | 53.3% |
| 54 | 100.0% | 10.8% | 69.0% | 56.7% |
| 27 | 100.0% | 14.2% | 39.5% | 38.3% |

Mass and radius (whole light curve, Yu Teff): median |difference| vs Yu et al. 4.5% (mass), 1.9% (radius); bias 3.0% / 1.1%;
median error bars 9.0% / 3.8%. Against Gaia DR3 FLAME radii (222 stars): R/R_Gaia median 1.076 (scatter 7.1%);
Yu et al.'s radii 1.063 (8.1%); classical 1.055 (8.7%).

## Limits
- On whole 4-year light curves the classical pipeline is better (0.9% vs 1.5% median νmax error, 0.6% vs 0.9% for Δν). The published answers were made with a similar pipeline on the same data, which favours it. StarQuake's advantage is on short light curves.
- Δν from short light curves is hard: from 27 days only 23% of stars are within 2% and 46% are off by more than 5% (often by one comb tooth). The error bars do not fully catch these jumps.
- False alarms: 12%–15% of unscreened quiet dwarfs are called oscillating (spots, eclipses and instrument trouble look like a hump). The classical comb test has fewer false alarms on full light curves (4%) but far more on short pieces.
- Masses carry a +3% median offset against Yu et al. (a small νmax bias, cubed). Radii from uncorrected scaling relations sit 8% above Gaia's (Yu et al.'s own: 6%); this is the known scaling-relation offset, not corrected here.
- Error bars are too narrow on short pieces: within 1 sigma 72% / 78% of stars on whole light curves but 54% / 58% at 27 days (numax / dnu). The widening factor was fitted on dev pooled over lengths; a per-length factor is the obvious fix.
- The most luminous giants (numax 8-15 uHz) read about 4% high in numax even from 4 years of data; the simulated twins of dev stars show the same bias, so it is the network, not the simulator.
- Trained for Kepler long cadence (29.4 min) and red giants with νmax 5–275 μHz. TESS light curves need a retrain on TESS noise; main-sequence stars (νmax > 1000 μHz) are out of reach at this cadence.
- The simulator has no red-clump vs RGB mixed-mode fine structure tuned on real stars, no rotation splitting, no stellar activity and no magnetic suppression beyond a simple dipole-suppression draw.

## Credits and licences
Data: Kepler mission light curves (NASA; Mikulski Archive for Space Telescopes, public stpubdata bucket).
Truth: Yu, Huber, Bedding et al. 2018, ApJS 236, 42 (machine-readable tables from the journal). Gaia DR3 FLAME radii
(ESA/Gaia/DPAC, via the ESA Gaia TAP service). Kepler DR25 stellar properties (NASA Exoplanet Archive) for the quiet dwarfs.
Physics: Kallinger et al. 2014; Mosser et al. 2011, 2012, 2015; Kjeldsen & Bedding 1995; Kjeldsen et al. 2008; Vrard et al. 2018;
Huber et al. 2009 (SYD). Built with numpy, scipy, PyTorch, astropy, astroquery. Code MIT; data under their providers' terms.
