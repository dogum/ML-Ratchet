# FireDetective lab notebook

Wildfire Phase 2, project 3. Built on 18-19 Nov 2026 in a cloud sandbox (workspace `fire/firedetective/`, next to the
Wildfire Phase 1 tree in `fire/`), released on 19 Nov and imported into ML Ratchet on 21 Nov 2026. Historical entries
are kept as written; add new dated entries below, do not rewrite these.

## 19 Nov 2026 — build notes (firedetective-status.md, as written)

From a wildfire's satellite record (half-day FEDS-style burned area, VIIRS fire pixels and FRP, gridMET weather, LANDFIRE fuels), a small 1-D CNN infers what drove the fire, with error bars: the share burned at high severity (crown fire), the wind on the fire days (read with the wind inputs removed) and live fuel moisture. It was trained only on FireSim simulations and graded once on 944 real exam fires. Wildfire Phase 2, project 3.

- Release: `firedetective-v1.0.zip` (1.0 MB), containing:
  - numpy inference (`firedetective.py`), features (`feats.py`), `pipeline.py` from raw observations
  - weights (3-seed ensemble, 189k weights), dev calibration
  - `demo.py` on 4 real exam fires (seeded draw within strata; runs in 0.1 s)
  - `MODEL_CARD.md`, `ratchet.json`, result JSONs, all training scripts and the JS port

### Honest headline

- **Burn severity, forest fires (n = 357):** Spearman **0.58** [0.51, 0.64] against the MTBS high-severity share. Best classical: FWI 0.54, ERC 0.53, Van Wagner crown potential 0.47. MAE 0.088 vs dumb 0.107 and ERC 0.092.
- **All fuels (n = 531):** the simple Van Wagner crown potential wins (0.55 vs 0.48).
- **Wind, read blind (n = 944):** Spearman **0.30** vs Rothermel back-calculation 0.13. But MAE 1.08 m/s is worse than guessing the dev median (0.96). On recent 2022–23 fires it drops to 0.14 (back-calc 0.21).
- **Live fuel moisture (n = 175): fails.** 0.17 vs fuel-type × month climatology 0.34; MAE 27.7 vs 21.3 %.
- **Error bars:** the 80 % band covers 69 % of forest severity truths (too narrow), 83 % of wind, 90 % of LFMC.

### Recipe as built

- **Physics:** FireSim (Phase 1, not edited): Rothermel/Albini surface spread, Van Wagner / Cruz crown fire, Albini spotting, Nelson moisture, MTT spread. Checks in `fire/checks/`: 436/436 published cases; pyretechnics WAF 3e-8; two-fuel strip vs hand 3e-9.
- **Physics knobs from twin tests** (wrappers in `src/simbank.py`):
  - v2: night fine-fuel recovery, anchor-and-flank control lines
  - v3: fixed WAF U(0.25, 0.6) in half the runs
- **Bank:** about 4,100 accepted sims (v1 1,340, v2 108, v3 2,652); frozen sets val_v1/v2/v3 (about 650 each, same seeds). Each sim:
  - a coarsened 180 m LANDFIRE window of a dev fire, or a parametric landscape
  - real HRRR hourly weather from dev fire dates; the network sees the real gridMET days, so its weather carries real analysis error
- **Observations:** FireSim's VIIRS emulator, then a FEDS-like perimeter emulator tuned on real dev VIIRS pixels vs real FEDS perimeters (median log10 area ratio +0.012, 1,646 steps).
- **Network:** 1-D CNN over 12 half-day steps × 25 features plus 17 statics. Gaussian heads; half the training samples are wind-blind. About 4 min per seed on one thread. numpy == torch to 9e-6; JS == numpy to 2e-14.
- **Real targets** (independent of the inputs):
  - MTBS severity via LANDFIRE Annual Disturbance (the per-fire mosaics need an e-mail queue)
  - gridMET wind
  - Globe-LFMC within 25 km and ±10 days
- **Dev:** 40 dev fires (30 with severity, 9 with LFMC). Dev calibration: severity map 0.017 + 0.271 × crown; widen factors 2.53 / 1.98 / 1.09.

### Rungs

Rule: more than +0.01 on the frozen-sim primary (mean clipped R² of crown share, blind wind, LFMC) **and** a dev composite (mean ρ of severity and wind on 40 dev fires) no more than 0.05 lower.

| Rung | Change | Sim (inc.) | Dev ρ sev / wind | Kept |
|---|---|---|---|---|
| R0 | 1-D CNN, bank v1 | 0.240 | 0.18 / 0.02 | base |
| R1 | drop shape-elongation input (sims too round) | 0.254 (0.240) | 0.22 / −0.03 | yes |
| R2 | v3 physics (night recovery, control lines, WAF); new frozen set | 0.261 (0.225) | 0.35 / 0.16 | yes |
| R3 | more v3 sims (2,548) | 0.307 (0.261) | 0.44 / 0.13 | yes |
| R4 | 3-seed ensemble | 0.338 (0.307) | 0.50 / 0.20 | yes |
| R5 | re-weight sims toward dev ROS factors and LFMC | 0.274 (0.338) | 0.48 / 0.06 | no |
| **R6** | **R4 recipe on the full bank (about 4,100 sims)** | **0.376** (0.338) | 0.58 / 0.11 | **yes, released** |

- **Twin test** (dev fires re-simulated from their first perimeter):
  - Sims grow on 91 % of half-days vs 50 % real.
  - Sims have 3× the VIIRS pixels, too much night FRP and rounder shapes (elongation 0.40 vs 0.70).
  - The WAF knob raised sim-vs-real per-step growth ρ from 0.36 to 0.44.
- **Classical diagnostics on dev (forest, n = 16):** ERC ρ 0.71, FWI 0.70, crown potential 0.41. Rothermel wind back-calc vs gridMET −0.12 (n = 40).

### Exam process note

`exam.py build` (input records only) ran before the ratchet closed. It was started during R4, before an interruption. On resume we checked it:
- 944/944 records, no predictions, no scores
- the records were not inspected
- they are rung-independent

R6 was then decided on sim and dev only, the ratchet was closed, and `exam.py score` was run once. This is disclosed in the model card and `ratchet.json`.

### Limits

1. LFMC fails; it is weak even in simulation (R² 0.14).
2. Wind is ranked but its level is worse than the median.
3. Outside forests, crown-fire potential beats the network.
4. Severity bands are too narrow.
5. Simulator gaps remain (growth too continuous, shapes too round, night FRP too high).
6. Spread-rate factor, fuel-model mismatch and spotting are not identifiable from these inputs (sim R² 0.14 / 0.01, AUC 0.47).
7. Only 15 recent forest fires have severity truth.

### Next ideas

- **Fire persistence physics.** Add a FireSim knob for day-to-day burnout and stalls (sims grow on 91 % of half-days vs 50 % real). This is also FireCast's over-prediction problem.
- **Severity-aware labels.** Simulate a dNBR-like severity (heat per unit area + crown) instead of crown share, so the dev map is not needed.
- **Conformal or heteroscedastic severity bands** fitted on more dev fires.
- **Live moisture.** Drop it or replace it with a phenology input. Use per-fire MTBS mosaics if the e-mail queue can be used.

### Upstream candidates for FireSim

- Night fine-fuel recovery multiplier and anchor-and-flank containment lines (v2)
- Canopy WAF sensitivity (v3)
- A FEDS perimeter emulator (closing radius 3 cells at 180 m), calibrated on dev

### Sandbox gotchas

- LFPS LANDFIRE export works per fire at 180 m with 4 threads (network-bound).
- MTBS severity mosaics need an e-mail queue: not used.
- `setsid nohup` returns at once; record the child PID with `ps`.
- Training is cheap (about 4 min per seed); bank shards take 1–2 min each per 100 attempts (about 45 % accepted).

## 19 Nov 2026 — from the Wildfire handoff (sections 5 and 6, FireDetective parts, as written)

**Exam: 944 fires, 2017–23**

| Target | FireDetective | Best classical | Notes |
|---|---|---|---|
| Burn severity, forest fires (Spearman ρ) | **0.58** | FWI 0.54, ERC 0.53 | Across all fuels the Van Wagner crown-fire potential wins: 0.55 vs 0.48 |
| Wind on fire days, read without wind inputs (ρ) | **0.30** | Rothermel back-calculation 0.13 | Level wrong; recent fires 0.14 vs 0.21 |
| Live fuel moisture (ρ) | 0.17 | Climatology **0.34** | **Fails** |

- **Error bars:** severity bands too narrow (80 % band covers 69 %).
- **Rule note:** the exam *input* build ran before the ratchet closed. Nothing was scored or opened; this is disclosed
  in the card and in `ratchet.json`.
- **Next:** 1. The persistence knob (upstream item 2). 2. Simulate a dNBR-like severity label. 3. Conformal severity
  bands; drop or rework live moisture. 4. Validate against GOFER hourly perimeters.

Rule notes (section 6): FireDetective built its exam inputs (including truth values) before the ratchet closed. They
were never opened or scored before close. All four Phase 2 agents were interrupted once (an interface stall) and
resumed from their STATUS notes; each checked that no exam had started before resuming. Dev twins and dev replays
fed physics calibration *and* the gates in several models, so the gates are partly in-sample. The exams were
untouched.

## 2026-11-21 — imported into ML Ratchet

Imported from `firedetective-v1.0.zip` (19 Nov 2026). Inference code, weights, calibration, examples, result JSONs
and every shipped number are as shipped; nothing was retrained, re-scored on new exam data or re-run.

### Layout
- Kept the release layout: `firedetective.py`, `feats.py`, `pipeline.py` and `weights/` (`firedetective_v1.npz`,
  `calib.json`, `export_check.json`) at the top, unchanged.
- `demo_data/` → `examples/` (the contract's name); `demo.py` reads `examples/` (one path changed). Output unchanged.
- `ratchet.json` → `training/ratchet.json`; `results/*.json` (the sandbox `out/`) → `training/results/`.
- The zip's `README.md` is replaced by the generated one (its private page link is gone). The status doc's page link
  is left out of the entry above.
- `training/` holds every script as run. Path edits only, each marked `# import:`: `_paths.py` is new; `common.py`
  takes `ROOT = WORK` and the `NUMBA_CACHE_DIR` from it instead of `/home/claude/fire/firedetective`; `baselines.py`
  and `twin.py` read `shared/fire/checks/replay_calibration.json`; `build_page.py` reads `template.html` and
  `fdnet.js` from `training/` and the weights from a re-run of `export.py` if there is one, else `weights/`;
  `build_page.py`, `calibrate.py`, `exam.py`, `make_text.py` and `make_page_text.py` read a shipped result through
  `latest()` (a re-run's copy under WORK first); `export.py` copies `feats.py` from `training/` and `firedetective.py`
  from the model folder; the `run_*.sh` scripts `cd` to `training/` and log under WORK.
- New: `training/import_check.py` (below) and its output `training/import_check.json`, `web/wrap_lab.py`,
  `web/export_instrument.py`, `examples/README.md`, `lab/README.md`, `model.yaml`, `ratchet.yaml`.

### The lab page
`lab/index.html` (3.63 MB) is the published page. `training/build_page.py` cannot rebuild it offline: it needs the
944 exam input records (`data/real/exam/*.pkl` from `exam.py build`, which needs the fire data rebuild and LANDFIRE
downloads) and `out/page_text.json` (from `make_text.py` and `make_page_text.py`, not shipped). The published copy was
checked instead: it is byte for byte `template.html` with the data and `fdnet.js` filled in, inside a host wrapper.
`web/wrap_lab.py` removed the wrapper and the host-only meta tag and gave the page its own doctype, head and body;
nothing inside the page changed. The page embeds every exam fire's input features, predictions and truths, and the 3
members' weights rounded to 5 significant figures.

### Checks run on import
- `python demo.py`: 4 fires in 0.05 s, identical to the zip's own demo run; recorded as `tests/golden/firedetective.txt`.
- The four example fires' raw observations through `pipeline.py` and the released weights reproduce their rows in
  `results/exam.json` exactly to the shipped digits (hs, hs_sd, crown, wind, lfmc, ros, m1, spot), and their truth and
  classical values match too. So the released pipeline from raw observations is the one that was graded.
- `training/import_check.py`: the 944 per-fire rows in `results/exam.json` re-derive the headline numbers (forest
  severity rho 0.5813 vs 0.5796, wind rho 0.2983 vs 0.2977, LFMC MAE 27.750 vs 27.749; every difference ≤ 0.003, from
  the rows' rounding).
- Intervals for the verdicts, from shipped numbers only: a paired bootstrap over the per-fire rows (10,000 resamples,
  seed 20261119) where the rows carry the baseline, else Zou's interval from exam.json's own 95 % intervals. The
  verdicts in `model.yaml` follow from them (loss if a baseline is better with an interval excluding 0, win only if
  every baseline is beaten that way):
  - forest severity rho: **tie** (vs FWI +0.044 [-0.058, +0.153] Zou; vs ERC +0.046 [-0.041, +0.134] paired);
  - forest severity MAE: **win** (vs ERC -0.004 [-0.008, -0.001]; vs crown potential and the dev median too);
  - all-fuel severity rho: **tie** (vs crown potential -0.067 [-0.146, +0.013] paired);
  - all-fuel severity MAE: **win** (vs crown potential -0.006 [-0.009, -0.002]);
  - blind wind rho: **win** (vs back-calculation +0.162 [+0.071, +0.251]);
  - wind MAE: **loss** (vs dev median +0.116 [+0.058, +0.175]);
  - LFMC rho: **tie** (vs climatology -0.171 [-0.365, +0.044], Zou); LFMC MAE: **loss** (vs climatology +6.4
    [+2.0, +10.6]).
- Severity ±1 sigma coverage from the rows: 60.6 % in forests (358 rows), 69.1 % across fuels. The rows carry no
  wind or LFMC sigma, so their 68 % coverage cannot be computed; their 80 % coverages are 83 % and 90 % (exam.json).

### Found on import (recorded, not fixed)
1. **All-fuel severity is a tie, not a loss.** The card, status doc and handoff say the crown-fire potential "wins"
   across fuels (0.55 vs 0.48). The paired interval of the difference includes 0 (-0.067 [-0.146, +0.013]).
2. **Forest severity "best" is not decisive.** 0.58 vs FWI 0.54 and ERC 0.53 is a tie by interval. The narrow
   decisive gain is in the error (MAE), not the rank.
3. **LFMC rho alone is not decisive** over the whole exam (Zou [-0.365, +0.044]); the failure is in the error, where
   the network is worse than a constant.
4. **Recent wind rho is 0.135, not 0.14.** exam.json gives 0.1349; the card and status doc round it to 0.14 (the page
   says 0.13). The handoff's "0.14 vs 0.21" comes from the card.
5. **The Rothermel back-calculation changes sign between dev and exam.** On dev it correlated -0.12 with gridMET wind,
   so its dev-fitted map has a negative slope (-0.049 per m/s): the dev-mapped baseline used for the wind MAE ranks
   winds backwards. The exam rho of 0.13 uses the raw back-calculation (positive). 36 of the 923 back-calculations
   hit their 30 m/s cap and 21 fires have none. The demo prints the raw back-calculated wind (27.3 m/s for Cooks
   Peak), not the mapped one.
6. **Forest count in the rows:** the rows round the forest share to 2 decimals, so one fire just below 0.5 counts as
   forest in the rows (358 vs 357).
7. **Bank counts:** R3's description says the bank has 131 v2 sims; the status doc says 108 v2 sims in the final bank
   of about 4,100. The banks are not shipped, so this cannot be checked.
8. **val_v2** was built (`results/compare_val_v2.json`) but no rung was scored on it; R2 went straight to v3 and
   val_v3.
9. **Dev in the gate:** the 40 dev fires tuned the FEDS emulator, gave the twin findings behind physics v2/v3, gate
   every rung, fit the severity map and the widening, and set R5's rejected re-weighting. R6's dev wind rho (0.11) and
   wind MAE (1.21 m/s) are worse than R4's (0.20, 1.10); R6 passed on the composite.
10. **Licence:** the card credits Globe-LFMC as the 2019 paper under CC BY; FireData used Globe-LFMC 2.0 (2024),
    which is CC0 (`shared/fire/data/DATA.md`).
11. The card's back-calculation MAE interval [1.00, 1.11] is [0.995, 1.113] in exam.json (the page shows [0.99,
    1.11]). The card's claim that the browser port matches numpy to 2e-14 has no shipped log; the site instrument's
    own TypeScript port matches to 8e-16 (below).
12. The page's "Use it" text still says `unzip firedetective-v1.0.zip` and "about 2 s"; kept as published.

### Live instrument
`web/export_instrument.py` writes `site/public/instruments/firedetective/` (1.41 MB: weights 0.38 MB float16, the
four example fires' raw observations 0.86 MB, reference 0.17 MB). `site/src/instruments/firedetective/features.ts`
ports `feats.py` (numpy's pairwise float32 means, linear percentile and half-to-even rounding included), `net.ts`
ports `FireDetective.raw` and `.predict`, and `whatif.ts` the exporter's what-ifs: lose VIIRS pixels (seeded), FRP ×,
drier or wetter dead fuels, wind ×, canopy base height ×, fewer half-day passes. Parity: 40 cases (4 fires × 10
what-ifs), worst relative difference 8.3e-16 over the 13 outputs and all 365 features; float16 weights move the
outputs by ≤ 3.1e-4 relative.
