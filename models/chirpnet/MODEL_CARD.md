# ChirpNet model card (v2.0)

ChirpNet v2 is four small networks, all trained only on simulations and tested on real LIGO data:

| File | What it does | Input |
|---|---|---|
| `chirpnet_weights.npz` | black-hole **finder** (rung 4): detection statistic, chirp mass ± 68%, mass ratio, merger time | 2 s, 2 × 4096 samples at 2048 Hz |
| `chirpnet_coinc_weights.npz` | **coincidence checker** (rung 8, two-stream): confirms both detectors see the same chirp | same |
| `chirpnet_pe_weights.npz` | **posterior** network (DINGO-style normalizing flow): samples of chirp mass, mass ratio, χ_eff, distance, inclination | same + both noise spectra |
| `chirpnet_bns_weights.npz` | **neutron-star** detector and chirp-mass estimator (time-frequency 2-D CNN) | 16 s, 2 × 16384 samples at 1024 Hz |

Finder plus checker together are the released detector, rung 9 (882,399 weights). `chirpnet.py` (numpy and scipy only)
runs all four. `demo.py` reproduces the numbers below. `training/` holds every script that produced the weights (needs PyTorch and LALSuite).

## 1. Black-hole detector

**Pipeline.** Raw 4096 Hz strain is whitened with its own Welch PSD (4 s Hann segments, median average), band-limited to
20-900 Hz, resampled to 2048 Hz and normalized to unit-variance noise. Excursions beyond 25σ are gated out, as in LIGO's
own searches. The finder slides over the data in 1/16 s steps. Wherever it scores above 0 the checker also runs, and if
the checker scores below -1.0 the window's statistic drops by 30. That only ever lowers a score, so every value above 0 is exact.

**Training data (all simulated).** 8,000 aligned-spin IMRPhenomD binary black holes. Chirp mass is log-uniform in
5.5-45 M☉ (detector frame), q in 0.2-1, spins in [-0.85, 0.85]. Each is projected onto H1/L1 for a random sky position
with the real light-travel delay, whitened with O1/O2 and design noise spectra, and dropped into fresh band-limited
Gaussian noise. Loud sine-Gaussian glitches go into one detector in 30-35% of windows. From rung 6 on, some negatives
hold a chirp in one detector only. Across all rungs: 2,496,000 training windows.

### The ratchet

Every candidate was scored on a frozen validation set (20,000 noise windows to set a false-alarm probability of 1e-3,
7,200 held-out signals at SNR 5-20, a glitchy version of both, and one-detector chirps). Real LIGO events never chose anything.

- **Rule 1** (rungs 1-3): Gaussian sensitivity must rise. It wrongly rejected rung 3, which had gained 55 points of glitch robustness.
- **Rule 2** (rungs 3-5): one objective, Gaussian sensitivity + glitchy sensitivity − 2 × median chirp-mass error, must rise by 0.01.
- **Rule 3** (rung 6): a stress test on time-slid real data showed that a loud chirp in one detector alone fools rung 4.
  A fix must halve one-detector false alarms and may not drop the objective by more than 0.01.
- **Rule 4** (rungs 7-9): as rule 3, but counting only loud (SNR ≥ 12) one-detector chirps, since quiet ones are ambiguous.
  Rungs 6-8 each cut the fakes to a few percent but lost too much sensitivity. Rung 9 splits the job: the rung 4 finder
  proposes and the rung 8 checker confirms. Its veto level (-1.0) was chosen on simulations with a fresh random seed:
  the most sensitive level whose loud-fake rate was at most a quarter of rung 4's.

| Rung | Change | Decision | Found at SNR 8 | Found at SNR 10 | Chirp-mass error (median, SNR ≥ 10) | 68% coverage | Found with glitches | Loud one-detector fakes flagged | Objective |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline | kept | 30% | 71% | 11.6% | 0.76 | 0% | not measured | 0.199 |
| 2 | SNR curriculum | kept | 54% | 89% | 5.8% | 0.72 | 0% | not measured | 0.477 |
| 3 | Glitch training | rejected | 52% | 88% | 5.3% | 0.70 | 55% | not measured | 1.029 |
| 3 (re-judged) | Rung 3 re-judged under rule 2 (single objective that also counts glitch robustness) | kept | 52% | 88% | 5.3% | 0.70 | 55% | not measured | 1.029 |
| 4 | Polish | kept | 58% | 91% | 4.9% | 0.72 | 58% | 100% | 1.105 |
| 5 | Test-time averaging | rejected | 57% | 91% | 4.8% | 0.73 | 59% | not measured | 1.109 |
| 6 | Coincidence training | rejected | 36% | 61% | 5.5% | 0.74 | 41% | 2% | 0.721 |
| 7 | Loud singles | rejected | 38% | 59% | 5.4% | 0.74 | 42% | 1% | 0.726 |
| 8 | Two-stream | rejected | 48% | 80% | 5.3% | 0.78 | 52% | 1% | 0.953 |
| 9 | Cascade | kept | 58% | 90% | 4.9% | 0.72 | 60% | 17% | 1.116 |

### Final exam: real LIGO data

The four events use 32 s GWOSC recordings. For the background, L1 is slid against H1 in 0.25 s steps from 3 to 25 s,
and windows near the event in either detector are removed. That gives 57,760 windows and 2.0 h of livetime;
the loudest statistic was 6.20. The release threshold is 7.0. The "alone" scores are the full
pipeline's statistic with the partner detector slid by 5, 9 or 13 s (the maximum of the three).

| Event | Statistic | Coincidence (checker) | H1 alone / L1 alone | FAR per hour (upper limit) | ChirpNet chirp mass, 68% | LIGO GWTC-1, detector frame, 90% | Merger time vs LIGO |
|---|---|---|---|---|---|---|---|
| GW150914 | 11.5 | 7.9 | -18.0 / -21.5 | < 0.50 | 32.1 [28.9, 35.7] | 31.2 [29.5, 33.0] | -14 ms |
| GW151012 | 9.6 | 7.6 | 5.5 / 1.3 | < 0.50 | 18.6 [16.2, 21.4] | 18.4 [16.9, 20.9] | -24 ms |
| GW151226 | 10.1 | 6.5 | 7.9 / 1.8 | < 0.50 | 10.2 [9.5, 10.9] | 9.7 [9.4, 10.0] | -2 ms |
| GW170104 | 9.8 | 7.1 | 5.7 / 8.9 | < 0.50 | 23.2 [20.9, 25.8] | 25.7 [23.5, 28.3] | -1 ms |

### Final exam, part 2: O3 (12 Nov 2026)

The released model, unchanged, on eight O3 events named in advance (32 s, 4 kHz GWOSC files; `training/o3_exam.py`,
results in `training/logs/o3_exam.json`). Background: L1 slid against H1 by 3–25 s within each file, event windows
removed: 29,879 windows, 1.04 h, loudest 7.24 (one window above the 7.0 threshold).

| Event | Statistic | Chirp mass (detector frame), ChirpNet | GWTC | Posterior medians inside 90% |
|---|---|---|---|---|
| GW190408_181802 | 11.07 | 24.5 [22.3, 26.9] | 23.9 | 5 / 5 |
| GW190412 | 12.41 | 15.4 [14.7, 16.1] | 15.3 | 5 / 5 |
| GW190521 | −0.63 (missed) | — | 98.7 | — |
| GW190828_063405 | 14.11 | 33.5 [30.6, 36.6] | 33.9 | 5 / 5 |
| GW191204_171526 | 12.60 | 10.0 [9.6, 10.5] | 9.7 | 5 / 5 |
| GW200129_065458 | 12.35 | 31.4 [28.4, 34.7] | 32.1 | 5 / 5 |
| GW200224_222234 | 9.89 | 35.0 [30.9, 39.7] | 41.1 | 5 / 5 |
| GW200311_115853 | 11.85 | 32.8 [29.3, 36.7] | 32.7 | 5 / 5 |

GW190521's detector-frame chirp mass (98.7) is more than twice the training range's upper edge (45); it was kept in
the exam because it was named in advance. Merger times are 24–100 ms from the catalogue's geocentric GPS times.

## 2. Posteriors (DINGO-style neural posterior estimation)

A ChirpNet trunk (started from rung 4) embeds the 2 s window. Each detector's noise spectrum is summarized as its ASD at
10 frequencies from 25 to 560 Hz. Together they condition an 8-layer RealNVP flow over (log ℳ, q, χ_eff, log₁₀ d_L,
cos ι, t_c). Training draws use physical amplitudes: waveforms are scaled from 400 Mpc, distances follow a uniform-in-volume
prior out to each waveform's horizon, PSDs are jittered by 0.6-1.6×, and only draws with network SNR ≥ 6 are kept
(4,000 steps × 128). Source-frame masses use the sampled distance and the Planck 2015 cosmology. For the real events
the window is placed using ChirpNet's own merger time.

**Calibration** (P-P test, 400 held-out injections): largest gap between the rank CDF and the diagonal is
log ℳ 0.04, q 0.057, χ_eff 0.025, d_L 0.028, cos ι 0.03. Chance alone allows about 0.07.
The merger-time posterior (0.09) is too wide to use; it is not reported.

| Event | m₁ (90%) | LIGO | m₂ (90%) | LIGO | ℳ source (90%) | LIGO | χ_eff (90%) | LIGO | d_L Mpc (90%) | LIGO |
|---|---|---|---|---|---|---|---|---|---|---|
| GW150914 | 44.5 [33.8, 71.9] | 35.6 | 26.5 [16.3, 35.2] | 30.6 | 29.6 [25.3, 34.1] | 28.6 | 0.15 [-0.19, 0.46] | -0.01 | 419 [204, 856] | 440 |
| GW151012 | 23.4 [17.6, 37.2] | 23.2 | 14.0 [8.7, 19.0] | 13.6 | 15.6 [12.9, 18.8] | 15.2 | -0.04 [-0.55, 0.46] | +0.05 | 650 [304, 1289] | 1080 |
| GW151226 | 13.4 [10.5, 21.9] | 13.7 | 8.4 [5.4, 10.7] | 7.7 | 9.2 [8.2, 10.2] | 8.9 | 0.21 [-0.33, 0.69] | +0.18 | 338 [168, 633] | 450 |
| GW170104 | 30.6 [23.7, 47.8] | 30.8 | 18.6 [11.6, 24.9] | 20.0 | 20.6 [17.6, 24.0] | 21.4 | -0.14 [-0.54, 0.26] | -0.04 | 748 [355, 1462] | 990 |

Every LIGO median falls inside ChirpNet's 90% interval.

## 3. Neutron stars

**Pipeline.** Minutes of 4096 Hz strain are whitened with an 8 s-segment Welch PSD, band-limited to 25-500 Hz,
resampled to 1024 Hz, and gated at 25σ. The model reads 16 s windows. A fixed front end computes power in 0.375 s Hann
frames every 1/16 s (2.7 Hz resolution, long enough to integrate a slowly sweeping chirp coherently). It averages the power
into 64 log-spaced bands from 24 to 450 Hz and takes log(1 + power). A 2-D residual CNN (6 stages, 16→96 channels) does the rest.

**Training data.** 3,000 IMRPhenomD_NRTidalv2 binary neutron stars: detector-frame chirp mass 0.85-2.0 M☉, q 0.6-1,
component masses 0.95-2.6 M☉, tidal deformabilities 0-1500, aligned spins up to 0.05. The merger is placed 8-15.8 s
into the window. Glitches go into 20% of windows. Validation uses 600 held-out waveforms.

**Ratchet objective:** Gaussian sensitivity (SNR 6-12) − 2 × median chirp-mass error; glitch false alarms ≤ 5%.

| Rung | Change | Decision | Found at SNR 10 | 12 | 15 | 20 | Chirp-mass error | Objective | GW170817 statistic vs loudest background | GW170817 ℳ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Plain CNN | kept | 1% | 6% | 17% | 77% | 14.9% | -0.286 | 10.8 vs 12.9 | 1.4603 |
| 2 | Time-frequency front end | kept | 10% | 45% | 94% | 100% | 2.6% | 0.054 | 11.1 vs 5.5 | 1.1823 |
| 3 | Polish | kept | 14% | 53% | 97% | 100% | 2.2% | 0.093 | 14.4 vs 5.0 | 1.1365 |

**Final exam: GW170817.** The 2048 s cleaned H1 and L1 data products (LIGO-P1700349) are used. Background comes from
L1 slid circularly against H1 by 12 shifts of 29-503 s, with windows holding the event in either detector removed:
11,086 windows over 6.2 h, loudest 4.98. The released model scores GW170817 at **14.4**
with chirp mass **1.1365 M☉** (LIGO: 1.1975 +0.0001/−0.0001, detector frame) and merger time
+963 ms from LIGO's. The raw Livingston data (LIGO-P1700337) still carries the famous glitch
1.1 s before merger. With automatic gating it scores 12.0 (ℳ 1.1455); with the glitch left in it scores 9.8. (The first release of `chirpnet.py` had its gate tapers reversed and gave 12.2 here; fixed 12 Nov 2026 to match the training code.)

## Limits

- **Coincidence is not perfect.** The checker lets about 1 in 6 loud one-detector chirps through in simulation, and
  GW151226's Hanford signal alone still scores above the threshold. The neutron-star model has no checker at all
  (H1 alone 8.2, L1 alone 11.2, both above its background).
- Aligned spins only; no precession or higher-order modes. The black-hole model covers chirp masses of about 6-45 M☉ (detector frame).
- False-alarm rates are bounded by hours of time slides (2 h for black holes, 6 h for neutron stars).
  They are upper limits, nowhere near the significance of LIGO's production searches.
- The neutron-star chirp mass is good to about 2% on simulations, versus LIGO's 0.01%, because 16 s windows give up the early
  inspiral. On GW170817 the released rung is 5% low and its 68% interval misses LIGO's value; rung 2 had been within 1.3%.
- Real events tested come from O1–O3 (2015–2020); untested on O4. Mergers heavier than the training range (GW190521) are missed.
