# StarQuake v1.1.1 — model card

## What it does
Reads a Kepler long-cadence light curve of a red giant (27 days to 4 years) and returns:
whether solar-like oscillations are present (numax 5-275 uHz), the frequency of maximum oscillation power **numax** and the
large frequency separation **Delta nu**, each with a 1-sigma error bar, and (given Teff) mass and radius from the asteroseismic
scaling relations M = (numax/3090)^3 (dnu/135.1)^-4 (Teff/5777)^1.5, R = (numax/3090) (dnu/135.1)^-2 (Teff/5777)^0.5.
The solar references (3090 uHz, 135.1 uHz, 5777 K) are those of Yu et al. 2018, so results compare directly with that catalogue.
(With the IAU 5772 K, solar numax would scale by sqrt(5777/5772), a 0.04% change.)

Released: **v1.1** = rung r5's network + a classical backbone (section v1.1). numpy/scipy inference (`starquake.py`), BatchNorm folded into convolutions; numpy equals torch to 7e-7; the in-browser JS port equals Python to 1e-6.

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
Parseval variance ratio 0.988–0.990; classical recovery on 24 simulated stars: numax 1.40%, dnu 0.58% median error;
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

## v1.1: classical backbone (15 Nov 2026)
The released network did not change. `starquake.py` now also runs the classical SYD-style pipeline (`classical.py`) on
every reading and moves the network's ln νmax and ln Δν towards the classical values by a weight w(T) that depends on
the observing length T, wherever the two agree within 5 network error bars (else the classical reading is taken to have
failed). Weights (`training/hybrid.py`, fitted on 3,000 simulated giants in real noise, never on real stars):

| length bin | < 40 d | 40–70 d | 70–150 d | 150–600 d | ≥ 600 d |
|---|---|---|---|---|---|
| w νmax | 0.30 | 0.20 | 0.05 | 0.25 | 0.40 |
| w Δν | 0.65 | 0.10 | 0.45 | 0.45 | 0.70 |

The error bar is the network's, scaled per bin by the ratio of blended to network median errors on the same
simulations, then widened on dev (νmax ×1.00, Δν ×1.71). `read_ppm(..., hybrid=False)` gives the v1.0 reading.
Ratchet (new frozen set val3, since val2 could not be rebuilt): objective 0.745 (r5) → 0.788; dev errors −10%
(whole νmax 1.01 → 0.78%, Δν 0.78 → 0.51%; 90-day νmax 2.34 → 2.24%, Δν 1.57 → 1.91%). Refitting the weights on two
other simulated sets gives 0.788 and 0.786.

### Exam, v1.1 (run once, after the release choice)
Median |ln error| / fraction within 5% (νmax) or 2% (Δν). v1.0 re-scored on the same readings; classical from the same
spectra. 1σ coverage with the dev-fitted factors and the published error in quadrature.

| length | n | v1.1 νmax | v1.0 νmax | classical νmax | v1.1 Δν | v1.0 Δν | classical Δν | v1.1 cov 1σ (νmax / Δν) |
|---|---|---|---|---|---|---|---|---|
| whole (~4 yr) | 420 | 1.01% / 95% | 1.49% / 89% | 0.89% / 91% | 0.65% / 90% | 0.89% / 78% | 0.64% / 84% | 81% / 84% |
| 365 d | 418 | 1.41% / 92% | 1.67% / 89% | 1.67% / 79% | 0.81% / 79% | 1.02% / 74% | 1.05% / 69% | 78% / 82% |
| 90 d | 420 | 2.22% / 82% | 2.17% / 82% | 3.89% / 57% | 1.77% / 54% | 2.07% / 48% | 3.06% / 42% | 66% / 67% |
| 54 d | 420 | 2.69% / 77% | 2.76% / 77% | 5.97% / 46% | 2.61% / 41% | 2.47% / 41% | 7.63% / 28% | 63% / 73% |
| 27 d | 420 | 3.44% / 64% | 3.61% / 62% | 16.05% / 34% | 4.29% / 24% | 4.60% / 23% | 26.80% / 13% | 58% / 63% |

Dumb baseline (dev median): 77% νmax, 56% Δν. Detection is unchanged (the network's own output).
Mass and radius, whole light curves: median |difference| vs Yu et al. 4.0% (mass), 1.6% (radius)
(v1.0 4.5% / 1.9%). Against Gaia DR3 FLAME radii (the same 222 stars as v1.0; the ESA archive
could not be reached on 15 Nov 2026, so the radii stored in `lab/index.html` were used): R/R_Gaia 1.063, scatter
8.0% (v1.0 1.076, 7.1%; Yu 1.063, 8.1%; classical 1.055, 8.7%).

What got worse: νmax at 90 days (2.17 → 2.22%), Δν at 54 days (2.47 → 2.61%), the radius scatter against Gaia
(7.1 → 8.0%, now level with Yu's radii instead of ahead), and 1σ coverage on whole light curves (now 81% / 84%, too wide).

## Exam, v1.0 (420 Kepler red giants, never used for any decision)
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

## 27 Nov 2026: trimmed record ends (v1.1.1) and a hard-negative rung (not promoted)

**Trimmed ends, dated re-score.** Flagged or empty cadences at the start or end of a record add only gaps, yet they
lower the duty cycle and change the FFT grid (audit). 22 of the 420 exam stars (36 of the 858 archived light curves)
start or end on a gap. The released v1.1 was re-read once on the exam with the ends trimmed (`training/hybrid.py
real exam exam_trim` with `SQ_TRIM=1`, scored by `hybrid.py exam_trim`; record `training/exam_trim_result.json`).
92 of 2,098 readings change, all from those 22 stars; the other readings are identical.

| Length | νmax v1.1 → trimmed | Δν v1.1 → trimmed | classical νmax / Δν (trimmed) |
|---|---|---|---|
| 4 yr | 1.01 → 0.99% | 0.65 → 0.62% | 0.89 / 0.63% |
| 1 yr | 1.41 → 1.41% | 0.81 → 0.81% | 1.67 / 1.05% |
| 90 d | 2.22 → 2.22% | 1.77 → 1.75% | 3.89 / 3.03% |
| 54 d | 2.69 → 2.68% | 2.61 → 2.63% | 5.97 / 7.63% |
| 27 d | 3.44 → 3.50% | 4.29 → 4.37% | 16.51 / 26.80% |

Radius scatter against Gaia: 8.04 → 8.12% (Yu et al. 8.06%). Shorter pieces move only because the random piece positions
depend on the record length. No verdict changes. From v1.1.1 `pipeline.grid_lc`, `read_npz` and `load_kepler` trim by
default (`trim=False` gives the v1.1 grid). The demo's three stars have no edge gaps; its output is unchanged.

**High-RMS non-oscillators.** The noise screen had dropped every dwarf with strong low-frequency variability (spots,
rotation, eclipses), so training never saw them. On 999 pieces of the 18 dropped held-out (noise_val) dwarfs, v1.1 calls
**16.9%** oscillating (quiet dwarfs: 6.1%). Rung r6 (`training/run_r6.sh`, `eval_r6.py`) retrained the network from
scratch on r5's bank recipe plus 5,000 examples on the 129 dropped noise_train dwarfs, refitted the hybrid, and was judged
as h1 was:

| | val3 objective | dev errors (whole νmax, Δν; 90 d νmax, Δν) | false alarms, high-RMS / quiet |
|---|---|---|---|
| h1 (released) | 0.788 | 0.78 / 0.51 / 2.24 / 1.91% | 16.9% / 6.1% |
| r6 | 0.795 | 1.10 / 0.85 / 1.91 / 1.78% | 11.1% / 5.6% |

r6 is **rejected**: +0.007 on val3 (needs +0.01) and dev errors +22% (gate: at most +10%); whole-light-curve dev errors
got worse. It does cut false alarms on high-RMS dwarfs by a third. v1.1's network stays; the exam was not opened.

## Limits
- On whole 4-year light curves the classical pipeline is better on the median νmax error (0.89% vs 0.99% for v1.1.1, 1.01% for v1.1, 1.49% for v1.0); for Δν v1.1.1 matches it (0.62% vs 0.63%). The published answers were made with a similar pipeline on the same data, which favours it. StarQuake's advantage is on short light curves.
- Δν from short light curves is hard: from 27 days only 23% of stars are within 2% and 46% are off by more than 5% (often by one comb tooth). The error bars do not fully catch these jumps.
- False alarms: 11–15% of unscreened quiet dwarfs are called oscillating (spots, eclipses and instrument trouble look like a hump); 16.9% of the high-RMS dwarfs the screen dropped (27 Nov 2026). The classical comb test has fewer false alarms on full light curves (4%) but far more on short pieces.
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
