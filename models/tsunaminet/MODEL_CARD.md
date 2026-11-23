# TsunamiNet v1.0 — model card

**What it does.** After a Pacific subduction earthquake, TsunamiNet reads the first DART bottom-pressure records
(up to three buoys, a 90-minute window around each buoy's predicted arrival) and returns a posterior over the
tsunami source, then forecasts the largest wave at other DART buoys. It was trained only on simulated tsunamis laid
into real quiet DART noise, and graded once on 28 real DART-recorded tsunamis. Research only: not a warning system.

## Inputs
- 1-minute DART water-column height from NDBC historical files, processed by `pipeline.py` (causal):
  tide fit on the 3 days before the origin (8 constituents + quadratic trend), glitch removal, a 1-minute grid from
  12 h before to 24 h after the origin, gaps linearly filled for filtering, then a causal 2nd-order Butterworth
  high-pass with a 3-hour cutoff. Minutes without a real sample stay masked.
- The USGS epicentre and origin time (to pick the 9 x 2 unit sources around the epicentre and predict arrivals).
- Nothing else from seismology: no magnitude, no mechanism.

## Outputs
- Samples of θ = (Mw, along-trench centroid in units of 100 km from the epicentre's unit-source column, deep-row
  share of moment, log10 rupture length). The release reports median, 68 % and 90 % ranges, widened on dev
  (Mw ×3.18, centroid ×4.40; widen-only, never narrowed).
- For any other DART with Green's functions: the maximum |η| in [arrival − 10, arrival + 180] min, as a median and
  68 % range (posterior samples pushed through the Green's functions with the p1 delay and gain, spread ×2.08).

## Physics
- **Source.** Okada (1985) vertical displacement for 1 m of thrust (rake 90°) on each of 1,317 Pacific SIFT unit
  sources (100 × 50 km; NOAA's `info_sz.dat` geometry, μ = 4 × 10¹⁰ Pa). Checks: vs GeoClaw's independent code,
  largest relative difference 4 × 10⁻¹³; vs NOAA MOST's own deformation grid for unit source `acsza1`, r = 0.995,
  peak uplift 0.352 vs 0.349 m.
- **Propagation.** Linear shallow-water equations on a spherical C-grid (no Coriolis, no friction), ETOPO 2022
  averaged to 4' then 8', land below 20 m depth is a wall, sponge at the open edges, dt = 20 s, 24 h. Checks
  (`checks/swe_check.json`): 1-D speed 198.5 vs 198.1 m/s; Green's law ×1.416 vs 1.414 at 1,000 m and ×1.868 vs
  1.943 at 281 m; 2-D front at 607 km in both, peak −2.1 % vs the exact Hankel-transform solution.
- **Green's functions by reciprocity.** The discrete operator is symmetric under the cell-area inner product, so
  one run started from a unit hump in a DART's cell gives that DART's response to every unit source:
  η_DART(t) = (1/A_DART) Σ_j A_j η0_j G_j(t). Forward vs reciprocal on the real grid agree to 2.4 × 10⁻⁶.
  68 DART sites × 1,317 unit sources × 1,441 minutes.
- **Resolution.** NOAA's propagation database uses 4'. A 4' run took 577 s per site on the shared 2-core machine
  (16 h for the bank), so the bank is 8'. At DART 46409 against 4': 3-hour peak ratio median 0.99 (≤ 1,500 km) and
  0.97 (> 3,000 km); input-window correlation 0.98 near and 0.73–0.81 far (`checks/gf_resolution_check.json`).

## Simulator and noise
- Random interface earthquakes Mw 7.3–9.3 on any Pacific SIFT column; length from Strasser et al. (2010) with
  scatter; heterogeneous slip along strike; a random share on the deeper row; the epicentre inside the rupture.
- The DART network of a random date 2006–2025 (NDBC deployment history). The three earliest-arriving buoys whose
  window has ≥ 50 % real minutes are the input, exactly as for real events.
- Real noise: 955 24-hour windows that start at a DART event-mode trigger more than 2 days from every NGDC tsunami
  (712 train, 243 for the frozen sets, split by station-year), with their real gaps.
- **Physics p1** (from the dev twin test, `twin_p0.json`): arrivals 1.65 % of the travel time later than the linear
  model (IQR 1.1–1.8 %, as Watada et al. 2014 found from elastic loading and compressibility), real/twin amplitude
  ×1.60 with 0.17 dex scatter per DART, rupture spreading at 2.5 km/s.

## Network
Per-DART 1-D CNN (4 conv layers, 48–128 channels) over 20 channels × 90 minutes (record, mask, the buoy's own
response to the 18 unit sources around the epicentre), mean/max pooling over the buoys, plus 23 features of a
SIFT-style non-negative least-squares fit on the same windows (the physics anchor), conditioning an 8-layer RealNVP
flow (ChirpNet-PE pattern) over θ. 520k weights. numpy == torch to 2.2 × 10⁻⁶.

## Ratchet
Rule: keep a rung if the frozen-set score (mean skill of the posterior-median Mw and centroid vs the set's median)
beats the incumbent by > 0.01 AND the dev error (Mw MAE/0.3 + centroid MAE/100 km + amplitude MAE/0.2 dex on 5 real
dev events) is ≤ 5 % worse. Physics changed once (p0 → p1); every rung was scored on val_p1.

| Rung | Change | Frozen val_p1 | Sim Mw MAE | Sim centroid (km) | Dev Mw MAE | Dev centroid (km) | Dev amp (dex) | Decision |
|---|---|---|---|---|---|---|---|---|
| r1 | p0 textbook physics | 0.327 (0.438 on val_p0) | 0.155 | 82 | 0.21 | 91 | 0.40 | kept (first) |
| r2 | p1 physics from the twin test | 0.663 | 0.092 | 39 | 0.21 | 79 | 0.24 | kept |
| r4 | mixed p0 + p1 bank | 0.641 | 0.104 | 40 | 0.28 | 54 | 0.28 | rejected (−0.022) |
| r3 | + NNLS physics anchor | 0.718 | 0.091 | 30 | 0.23 | 57 | 0.26 | **released** |
| r5 | r3 polished at lower learning rate | 0.724 | 0.090 | 29 | 0.22 | 59 | 0.25 | rejected (+0.006) |

## Exam (run once, 28 events, 2007–2025)
Truth: USGS W-phase Mw and the moment-weighted centroid of the USGS finite-fault model. Amplitudes: what 119
held-out DARTs recorded (15 events). 95 % intervals are bootstrap over events.

| Metric | TsunamiNet | SIFT-style NNLS | NNLS, p1 GFs | Seismic W-phase | Dumb | Verdict |
|---|---|---|---|---|---|---|
| Mw mean abs error | **0.23** [0.18, 0.27] | 0.25 | 0.28 | (is the truth) | 0.30 (Mw 7.9) | win |
| Centroid distance, median (km) | **45** | 82 | 83 | – | 56 (epicentre column) | win |
| Centroid along the trench, mean (km) | 42 [29, 58] | 98 | 102 | – | **39** [26, 54] | loss (tie) |
| Amplitude mean abs log10, 119 forecasts | **0.20** [0.16, 0.27] | 0.23 | 0.21 | 0.28 | 0.34 (2.8 cm) | win |
| Amplitude within factor 2 | **79 %** | 71 % | 74 % | 56 % | 52 % | win |
| Amplitude, each event once (15) | 0.34 | 0.32 | 0.30 | 0.35 | **0.27** | loss |

Seismic W-phase rows use a standard rupture (Strasser length, both rows, equal split) centred on the epicentre's
column, the pre-DART SIFT forecast; with the p1 delay and gain it scores 0.31 pooled. The USGS finite-fault twin
(hindsight, p1) scores 0.30.

**Coverage.** Mw 68 % range: 0.68 (interface 0.79, other 0.44); Mw 90 %: 0.96. Centroid 68 %: 0.89 (too wide).
Amplitude 68 %: 0.78.

**By mechanism.** Interface thrusts (19 events, 106 forecasts): Mw 0.23, amplitude 0.19 dex (NNLS 0.22, seismic
0.28, dumb 0.36). Other mechanisms (9 events, 13 forecasts): Mw 0.23, amplitude 0.29 dex, where seismic scaling
with p1 (0.13) and the dumb guess (0.22) win.

## Where it fails
- Magnitudes read low (bias −0.22 Mw). Tohoku 2011: 8.67 [8.44, 8.92] against 9.1. Large events whose slip runs
  past the 9 basis columns and the 2 shallow rows are under-read.
- The along-trench centroid only ties the epicentre's own column.
- Small events with one or two held-out buoys: a constant 2.8 cm is closer, event by event.
- Outer-rise, strike-slip and intraslab events are read as interface thrusts.

## Limits
8' linear propagation; thrust on two rows only; 68 DART sites (no tide gauges); dev of 5 events sets the widening;
detiding uses the 3 days before the event (hindcast data from NDBC real-time files).

## Data and credits
NDBC DART historical data and station metadata; NCEI/WDS Global Historical Tsunami Database; USGS ComCat W-phase
moment tensors and finite-fault models; ETOPO 2022 (NCEI); NOAA NCTR SIFT unit sources and the MOST deformation grid
for `acsza1` (ComMIT compressed server). GeoClaw (BSD-3) only as an independent Okada check. All NOAA/USGS data are
US Government works.
