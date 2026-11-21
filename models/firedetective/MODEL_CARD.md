# FireDetective v1.0: model card

ML Ratchet, Wildfire Phase 2, project 3. Trained on physics, graded on reality.

## Imported into ML Ratchet (21 Nov 2026)

This card is the release card, kept as written except for the file paths. Three notes from the import (details in
NOTEBOOK.md and `model.yaml`):

- **Rule note.** The exam input build (`exam.py build`), including the truth values stored in the same records, ran
  before the ratchet closed, while R4 was the incumbent. Nothing was predicted, scored, compared or opened before the
  close; R6 was decided on the frozen sim set and dev only, and `exam.py score` ran once (see "Exam process
  disclosure" below).
- **Dev in the gate.** The 40 dev fires tuned the FEDS emulator, gave the twin findings behind physics v2 and v3,
  fit the severity map and the widening factors, and are also the gate of every rung. The gate is partly in-sample.
- **Verdicts with intervals** (paired bootstrap over the shipped per-fire exam rows, or Zou's interval from the
  shipped 95 % intervals; `training/import_check.py`): forest severity rank is a tie with FWI and ERC
  (+0.04 [-0.06, +0.15]); its error is a narrow win (vs ERC -0.004 [-0.008, -0.001]). Across all fuels the crown-fire
  potential ranks higher but not decisively (-0.067 [-0.146, +0.013]): a tie, not a loss. Blind wind rank wins (vs
  back-calculation +0.16 [+0.07, +0.25]); its error loses to the dev median (+0.12 [+0.06, +0.18]). Live moisture
  loses on error (+6.4 [+2.0, +10.6] % vs climatology); its rank difference is not decisive. The recent-fire wind
  rho below is 0.1349 in `training/results/exam.json` (0.13, not 0.14). Globe-LFMC 2.0, which FireData used, is CC0.

## What it does

FireDetective reads a wildfire's satellite record and says what drove it, with error bars. The input is the fire's first 6 days (up to 12 half-day steps):
- the burned-area mask at each VIIRS overpass (FEDS-style, 180 m grid),
- the VIIRS 375 m fire pixels and their fire radiative power (FRP),
- daily gridMET weather on the fire days,
- LANDFIRE fuels, canopy and terrain.

It outputs, per fire:

| Output | Meaning | Graded on real fires? |
|---|---|---|
| `hs`, `hs_sd` | share of the burned area MTBS maps as high severity (from the simulated crown-fire share, through a 2-number map fitted on dev) | yes, MTBS severity |
| `wind`, `wind_lo`, `wind_hi` | daily-mean 10 m wind on the fire days, inferred with the wind inputs removed | yes, gridMET wind |
| `lfmc`, `lfmc_sd` | live fuel moisture, % dry mass | yes, Globe-LFMC samples (it fails) |
| `crown` | simulated crown-fire share (passive + active) | simulator label only |
| `ros_adj`, `m1`, `wind_regime`, `p_spot`, `p_ia` | spread-rate factor, afternoon 1-h dead moisture, wind- vs slope-driven share, spotting, initial attack | simulator labels only; several are not identifiable (see Limits) |

## Why these real targets

Most fire "drivers" have no field truth. We picked targets that are measured independently of the inputs:
1. **MTBS burn severity.** High severity is the closest real proxy for crown fire. It comes from Landsat dNBR, not from VIIRS or FEDS. Fetched per fire from LANDFIRE Annual Disturbance (LFPS; MTBS codes 11–14 and 111–114), because the per-fire MTBS mosaics need an e-mail queue.
2. **gridMET wind on the fire days.** The network gets the wind columns blanked ("wind-blind mode") and must read the wind from the fire's growth, shape and FRP. Rothermel back-calculation is the classical rival.
3. **Globe-LFMC** live fuel moisture within 25 km and ±10 days.

Rejected: suppression truth (FEDS growth stopping is confounded with weather), GOFER (4 dev / 24 exam fires, too few).

Classical diagnostics compared: Van Wagner crown-fire potential (from LANDFIRE canopy and gridMET moisture), max ERC, BI and FWI over the fire days, Rothermel wind back-calculation from the observed head spread, LFMC climatology (fuel type × month from dev sites), and a dumb dev median.

## Physics and simulator

- **FireSim** (Phase 1, shared, not edited): Rothermel 1972 + Albini 1976 surface spread, Scott & Burgan 40 fuel models, Van Wagner 1977 initiation, Cruz 2005 / Rothermel 1991 crown spread, Albini spotting, Nelson dead-fuel moisture, minimum-travel-time spread. Checks in `fire/checks/`: 436 of 436 published BehavePlus/Rothermel cases within tolerance (`published_examples.json`); WAF vs pyretechnics max difference 3e-8 and crown fire-type agreement 1.0 (`pyretechnics_crosscheck.json`); two-fuel-strip arrival time vs hand solution max relative error 3e-9 (`analytic.json`); Van Wagner I0 example 2476 vs quoted 2500 kW/m (`crown.json`).
- **Observations:** FireSim's VIIRS emulator (detection probability vs FRP, scan angle, cloud omission, commission), then a FEDS-like perimeter emulator (pixel footprints, closing, hole filling). The emulator was tuned on REAL dev VIIRS pixels against real FEDS perimeters: median log10 area ratio +0.012, MAD 0.020, 1,646 dev half-day steps (`out/calib_feds.json`).
- **Real noise and real places:** simulations run on coarsened (180 m) LANDFIRE windows of the 40 dev fires and on parametric landscapes, under real HRRR hourly weather from dev fire dates; the network sees the matching real gridMET days as its weather input, so its weather has real analysis error.
- **Randomised drivers:** spread-rate factor, live and dead moisture factors, crown fire on/off and Cruz vs Rothermel, spotting, initial attack, and (physics v2/v3, from dev twin tests) night fine-fuel recovery, anchor-and-flank control lines, and a fixed wind adjustment factor U(0.25, 0.6) in half the runs.
- **Physics versions:** v1 (FireSim as shipped), v2 (+ night recovery, control lines), v3 (+ wind-sensitive WAF). Frozen validation sets `val_v1`, `val_v2`, `val_v3` use the same seeds. The changes live in `training/simbank.py` as wrappers.

## Network

- 1-D CNN over up to 12 half-day steps (25 features per step: area growth, VIIRS counts and FRP, day/night, weather, wind relative to the growth direction) plus 17 static fuel/terrain/canopy summaries. 63k weights per member; 3 seeds (189k weights in all).
- Gaussian heads (mean and log-sigma) for each regression target, logistic heads for spotting and initial attack. In training, half the samples have the wind columns blanked (wind-hidden mode).
- Training: about 4,100 accepted simulations (fires ≥ 2 km²), 60 epochs, Adam, one CPU thread, about 4 minutes per seed.
- Release: numpy only. numpy vs torch max difference 6e-7 (means), 9e-6 (log-sigma) on 40 dev fires (`weights/export_check.json`). The browser port (JS) matches numpy to 2e-14.

## Dev set and calibration (dev only)

- 40 dev fires (Phase 1 frozen split, CONUS 2017–21): 30 with MTBS severity (16 forest), 40 with wind, 9 with LFMC samples.
- Severity map: `hs = 0.017 + 0.271 × crown` (fitted on 30 dev fires); the same 2-number map is fitted for every classical index.
- Widen-only sigma factors fitted on dev so the 80 % band covers ≥ 80 % of dev truths: severity × 2.53, wind × 1.98, LFMC × 1.09.

## Ratchet

Rule: keep a rung if it beats the incumbent on the frozen simulated set by more than +0.01 on the primary score (mean clipped R² of crown share, blind wind and LFMC), and the dev composite (mean of Spearman ρ for severity and wind on the 40 dev fires) drops by no more than 0.05 (stated threshold; the dev set is small). When the physics changed, the incumbent was re-scored on the new frozen set first.

| Rung | Change | Frozen | Sim | Inc. sim | Dev ρ sev. all / forest | Dev ρ wind | Dev ρ LFMC (n=9) | Dev comp. | Decision |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 1-D CNN, bank v1 (1,340 sims) | val_v1 | 0.240 | | 0.18 / 0.49 | 0.02 | −0.03 | 0.103 | keep (base) |
| R1 | drop the shape-elongation input (sims rounder than real fires: 0.40 vs 0.70) | val_v1 | 0.254 | 0.240 | 0.22 / 0.43 | −0.03 | 0.13 | 0.097 | keep |
| R2 | v3 physics: night recovery, control lines, wind-sensitive WAF; 2,121 sims | val_v3 | 0.261 | 0.225 | 0.35 / 0.53 | 0.16 | 0.35 | 0.256 | keep |
| R3 | more v3 sims (2,548) | val_v3 | 0.307 | 0.261 | 0.44 / 0.69 | 0.13 | 0.62 | 0.282 | keep |
| R4 | 3-seed ensemble of R3 | val_v3 | 0.338 | 0.307 | 0.50 / 0.71 | 0.20 | 0.67 | 0.348 | keep |
| R5 | training sims re-weighted toward dev replay ROS factors and fire-season LFMC | val_v3 | 0.274 | 0.338 | 0.48 / 0.71 | 0.06 | 0.67 | 0.269 | reject (sim −0.064, dev −0.080) |
| **R6** | **R4 recipe on the full v3 bank (about 4,100 sims)** | val_v3 | **0.376** | 0.338 | 0.58 / 0.75 | 0.11 | 0.55 | 0.344 | **keep, released** |

R6 on the frozen sim set: crown share R² 0.68 (0.51 wind-blind), blind wind R² 0.31, LFMC 0.14, 1-h moisture 0.72 (0.22 blind), spread-rate factor 0.14 (0.01 blind), fuel-model mismatch 0.01, spotting AUC 0.47, initial attack AUC 0.81.

### Twin test (dev, `training/twin.py`, `out/twin_*.json`)

Each dev fire was re-simulated from its first mapped perimeter with its own fuels and weather and seen through the same emulators.

| Input feature | Sim twins (v1) | Real dev | Correlation |
|---|---|---|---|
| log10 burned area at the last step (km²) | 1.44 | 1.13 | 0.61 |
| share of half-days with growth | 0.91 | 0.50 | 0.25 |
| log10(1 + VIIRS pixels per step) | 1.38 | 0.85 | 0.64 |
| log10(1 + FRP per pixel), night | 0.86 | 0.50 | 0.19 |
| shape elongation | 0.40 | 0.70 | 0.43 |

Twin findings that became rungs: night slow-down and control lines (v2), and a wind-sensitive WAF (v3: sim-vs-real per-step growth ρ 0.36 → 0.44). Not fixed: growth on too many half-days, rounder shapes, too much night FRP.

## Exam (run once, after the ratchet closed)

**Exam set (fixed rule, seed 20261119):** exam fires 2017–2023 with at least one FEDS growth day; every timber fire (TU/TL) plus a seeded draw of 250 others: 944 fires (718 main 2017–21, 226 recent 2022–23). 2024 is excluded because LANDFIRE has no disturbance layer for it yet. Brackets are 95 % bootstrap intervals over fires.

**Burn severity (MTBS high-severity share), forest fires (forest share ≥ 0.5), n = 357**

| Method | Spearman ρ | MAE (fraction) |
|---|---|---|
| **FireDetective** | **0.58** [0.51, 0.64] | **0.088** [0.079, 0.098] |
| FWI max | 0.54 [0.45, 0.61] | |
| ERC max | 0.53 [0.44, 0.62] | 0.092 [0.084, 0.102] |
| Van Wagner crown-fire potential | 0.47 [0.37, 0.55] | 0.097 [0.088, 0.107] |
| BI max | 0.43 [0.33, 0.51] | |
| Dev median (dumb) | | 0.107 [0.096, 0.118] |

80 % band coverage: 69 % (too narrow). Truth median 0.073.

All fuels (n = 531): FireDetective ρ 0.48 [0.42, 0.55], **Van Wagner crown potential 0.55** [0.48, 0.61], ERC 0.39. MAE 0.080 vs dumb 0.090, crown potential 0.085. Coverage 77 %.

**Wind on the fire days, read blind, n = 944**

| Method | Spearman ρ | MAE (m/s) |
|---|---|---|
| **FireDetective** (wind inputs removed) | **0.30** [0.24, 0.35] | 1.08 [1.01, 1.16] |
| Rothermel back-calculation (dev-mapped) | 0.13 [0.06, 0.19] | 1.05 [1.00, 1.11] |
| Dev median (dumb) | | **0.96** [0.91, 1.02] |

Coverage 83 %. It ranks windy fires better than the classical method but its level is worse than the dumb median.

**Live fuel moisture, n = 175 (fails)**

| Method | Spearman ρ | MAE (% dry mass) |
|---|---|---|
| FireDetective | 0.17 [0.02, 0.31] | 27.7 [24.7, 30.7] |
| **Climatology: fuel type × month (dev sites)** | **0.34** [0.18, 0.47] | **21.3** [18.4, 24.5] |
| Dev-site median (dumb) | | 22.5 [19.7, 25.7] |

Coverage 90 %.

**By period:** main forest severity ρ 0.58 (ERC 0.52), n = 342; recent ρ 0.84 (ERC 0.79), n = 15 only. Wind main ρ 0.35 (back-calc 0.11); **recent ρ 0.14 (back-calc 0.21)**, n = 226. LFMC main 0.13 (clim 0.40); recent 0.27 (clim −0.07), n = 26.

### Exam process disclosure

The exam script has two parts: `exam.py build` (fetch LANDFIRE and build input records) and `exam.py score`. In the first session, `build` was started at 10:15 local time, while R4 was the incumbent and before the ratchet was closed. That session was then interrupted. On resume we checked: the build had finished (944/944 records, 0 failures). It fetched LANDFIRE, built the input records (the truth fields sit in the same files) and computed the classical index values. It made no prediction, score or comparison, and the records were not opened. After the resume, one more rung (R6: the same recipe with more simulations) was trained and judged on the frozen sim set and the dev set only. The ratchet was then closed, R6 was calibrated on dev, and `exam.py score` was run once. The records are rung-independent, so building them early did not change any decision; we record it because the rule says the exam comes after the ratchet closes.

## Limits

1. **Live fuel moisture fails** on the exam. Even in simulation it is read poorly (R² 0.14); the dev score (9 fires) did not carry over.
2. **Wind is ranked, not measured.** MAE is worse than the dumb median, and on recent fires the classical back-calculation ranks better.
3. **Outside forests, the Van Wagner crown-fire potential beats the network** for severity. Crown share in FireSim is not the same thing as MTBS high severity; the map between them is fitted on 30 dev fires.
4. **Severity error bars are too narrow** (69 % coverage for an 80 % band, after a dev widening of 2.5).
5. **Simulator gaps** (twin test): sims grow on 91 % of half-days vs 50 % real, are rounder, and have too much night FRP.
6. **Not identifiable from these inputs:** spread-rate factor, fuel-model mismatch, spotting. Their outputs are simulator readings, not measurements.
7. Severity truth lags: only 15 recent forest fires have MTBS severity in LANDFIRE.

## Files

- `firedetective.py` numpy model; `feats.py` features; `pipeline.py` raw observations → features → predictions; `demo.py` 4 real exam fires in `examples/` (a seeded draw within strata: 2 forest main, 1 non-forest main, 1 recent; runs in about 0.1 s).
- `weights/firedetective_v1.npz`, `weights/calib.json`, `weights/export_check.json`; `training/ratchet.json`; `training/results/` (exam, calibration, twin, dev baselines; the `out/` files named above).
- `training/`: every script (simbank, train, ratchet, calibrate, twin, baselines, exam, export, page builder, JS port), with `_paths.py` for the repository paths, and `import_check.py` (21 Nov 2026).
- `lab/index.html`: the released page (the network runs in the browser on all 944 exam fires).

## Data credits

MTBS burn severity (USGS/USFS) via LANDFIRE Annual Disturbance; LANDFIRE fuels, canopy and topography (LFPS; public domain); FEDS fire event data set (Chen et al. 2022, Scientific Data; CC BY); VIIRS 375 m active fire (NASA LANCE/FIRMS); gridMET (Abatzoglou 2013; public); HRRR (NOAA, AWS Open Data); Globe-LFMC (Yebra et al. 2019, Scientific Data; CC BY). Fire lists and splits: ML Ratchet Wildfire Phase 1 (FireData). Simulator: FireSim (Phase 1).
