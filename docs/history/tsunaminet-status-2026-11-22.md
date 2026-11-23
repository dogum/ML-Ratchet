# TsunamiNet — status (v1.0, 22 Nov 2026)

> Kept as written on 22 Nov 2026 in the cloud project, with names of people and build roles made neutral and the claude.ai page links replaced by each model's `lab/` in this repository. Paths and status refer to that project.

From the first DART bottom-pressure records after a Pacific subduction earthquake, a normalizing flow gives the
tsunami source as a posterior (Mw, centroid along the trench, deep-row share, rupture length) on NOAA's SIFT unit
sources, and pushes it through linear shallow-water Green's functions to forecast the largest wave at other DARTs.
Trained only on simulated tsunamis in real DART noise; graded once on 28 real ones. ML Ratchet B batch, project 2;
it fills the "normalizing flow" family.

- Page (the network runs live in the browser on all 28 exam events; lab with sliders; exam; ratchet; checks): `models/tsunaminet/lab/`
- Release: `tsunaminet-v1.0.zip`, containing:
  - numpy + scipy inference (`tsunaminet.py`, `pipeline.py`), weights with the dev calibration, compact 8' Green's functions (68 DART sites, 25 MB)
  - `demo.py` on four exam events (one per stratum, seeded; about 2 s)
  - model card, `model.yaml`, `ratchet.yaml`, `splits.json`, all scripts, logs and checks, the lab page
  - `for_repo/models/tsunaminet/` in the repo layout

## Honest headline

- **Exam, 28 real tsunamis (2007–2025), run once.**
  - Mw error 0.23 [0.18, 0.27]. SIFT-style NNLS on the same windows: 0.25. Dev-median guess: 0.30. Bias −0.22: it reads low. Tohoku 8.67 [8.44, 8.92] against 9.1.
  - Centroid: median miss 45 km from the USGS finite-fault centroid (NNLS 82, epicentre column 56). Along the trench it ties the epicentre: 42 vs 39 km.
  - Held-out DART amplitude (119 forecasts, 15 events): mean |log10 error| 0.20 [0.16, 0.27]; 79 % within a factor 2.
    - NNLS: 0.23 and 71 %.
    - NNLS with dev-calibrated Green's functions: 0.21 and 74 %.
    - Seismic W-phase scaling: 0.28 and 56 %.
    - Constant 2.8 cm: 0.34 and 52 %.
    - USGS finite-fault twin (hindsight): 0.30.
- **Where it loses:** weighted event by event, the constant 2.8 cm wins (0.27 against 0.34). Most events are small, with one or two held-out DARTs.
- **Coverage of the dev-widened 68 % ranges:** Mw 0.68, centroid 0.89 (too wide), amplitude 0.78. Interface thrusts (19 events): Mw coverage 0.79; other mechanisms (9): 0.44.

## Recipe as built

- **Physics.**
  - Okada (1985) for 1 m of thrust on each of 1,317 Pacific SIFT unit sources (NOAA geometry).
  - Linear shallow-water equations on a spherical C-grid over ETOPO 2022: 8', dt 20 s, 24 h, numba.
  - Green's functions by reciprocity: one receiver-started run per DART site gives its response to every unit source. 68 sites.
- **Checks** (`training/checks/`):
  - Okada vs GeoClaw: 4e-13. Okada vs NOAA MOST's own deformation for acsza1: r 0.995, peak 0.352 vs 0.349 m.
  - 1-D speed: 198.5 vs 198.1 m/s.
  - Green's law: 1.416 vs 1.414 at 1,000 m; 1.868 vs 1.943 at 281 m.
  - 2-D Hankel solution: front at 607 km in both, peak −2.1 %.
  - Reciprocity: 2.4e-6.
  - 8' vs 4': peak ratio 0.97–0.99; window correlation 0.98 near, 0.73–0.81 far.
  - numpy vs torch: 2.2e-6. JS vs numpy: 4.2e-7.
- **Noise:** 955 real quiet DART windows from event-mode triggers more than 2 days from any NGDC tsunami, with their real gaps (712 train, 243 frozen).
- **Processing (records and Green's functions alike):** causal tide fit on the 3 days before the origin, glitch removal, causal 3-h Butterworth high-pass.
- **Twin test** (dev, USGS finite faults) → physics p1:
  - arrivals 1.65 % of travel time late (IQR 1.1–1.8 %);
  - amplitudes ×1.60, sd 0.17 dex;
  - rupture at 2.5 km/s.
- **Network:** per-DART CNN over [record, mask, 18 unit-source responses] × 90 min. Mean/max pooling over ≤ 3 DARTs, plus 23 NNLS-anchor features. 8-layer RealNVP. 520k weights.
- **Dev:** 5 seeded events: Kuril 2006 and 2007, Solomon Sea 2016, Shumagin Oct 2020, Kamchatka Sep 2025. Three of them are not interface thrusts.
- **Calibration (dev, widen-only):** Mw ×3.18, centroid ×4.40, amplitude ×2.08.

## Rungs

Rule: keep if the frozen-set score is more than 0.01 above the incumbent's **and** the dev error is no more than 5 % worse. The frozen set is val_p1, 3,000 sims; r1 was re-scored on it.

| Rung | Change | Frozen | Dev Mw | Dev centroid | Dev amp | Kept |
|---|---|---|---|---|---|---|
| r1 | p0 textbook physics | 0.327 (0.438 on val_p0) | 0.21 | 91 km | 0.40 | base |
| r2 | p1 physics from the twin test | 0.663 | 0.21 | 79 km | 0.24 | yes |
| r4 | mixed p0 + p1 bank | 0.641 | 0.28 | 54 km | 0.28 | no (−0.022) |
| r3 | + NNLS physics anchor | 0.718 | 0.23 | 57 km | 0.26 | **yes, released** |
| r5 | r3 polished | 0.724 | 0.22 | 59 km | 0.25 | no (+0.006) |

- r1's flow never learned the centroid (skill 0.006). NNLS on the same windows recovered it (correlation 0.6–0.8), which motivated the anchor.

## Limits

- 8' propagation, not NOAA's 4' (a 4' site took 577 s on the shared machine). Linear: no dispersion, Coriolis or Kajiura filter.
- Thrust on the two shallow rows only, 9 columns read: long ruptures (Tohoku) and non-thrust events are under-read.
- Green's functions exist for 68 DART sites only; there are no coastal tide gauges.
- The 5 dev events set the widening.

## Next rungs worth trying

- 4' Green's functions for the 68 sites (about 10 min each on one core). Then re-run the twin test and look at the −0.22 Mw bias.
- A wider basis (more columns and the deeper rows) for M9 ruptures.
- A mechanism prior from the W-phase (outer-rise and strike-slip unit sources).
- Tide gauges through nested grids or a dev-fitted amplification per gauge.

## Upstream candidates

- `dart.py`: causal NDBC DART reader, detider and QC; it is reusable by any DART model.
- `swe.py` + `gf_bank.py`: reciprocal Green's functions on any grid; a shared ocean-wave platform could start from them.

## help_wanted

- NOAA's compressed propagation database answers, but its amplitude encoding is undocumented. The documented THREDDS/OPeNDAP access (sift.pmel.noaa.gov/thredds) returns 404. MOST waveforms were therefore compared only through the deformation grid.

## Sandbox gotchas

- A 4' Green's-function run took 577 s per site and an 8' run 70–120 s, at load 6–9 from four builds.
- The first twin test showed several-cm tide-fit errors that looked like waves, and a 30 m DART glitch. Fix: causal high-pass and QC (> 5 % outlier samples or > 3 cm rms in the pre-event fit, 9-sample despike).
- `tsim.GFBank` once defaulted to the 4' folder, which held one site. A bank and a twin test silently used one DART. They were caught and regenerated.
