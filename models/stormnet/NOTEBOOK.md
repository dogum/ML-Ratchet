# StormNet build notebook

## 17 November 2026 · protocol recorded before opening any index series

At the end of a UTC hour, estimate the minimum of the **next three complete hourly Dst values** in nT. Inputs are 36 completed hours of solar wind (GSM Bz, speed and dynamic pressure) and the last completed Dst. The last value is available to every competitor. OMNI has already shifted upstream measurements to Earth. This is a retrospective experiment conditional on an available index, with no claim to operational lead time.

**Freeze the data roles.** Use only wind from 1998–2005 as the driving record. Keep 2005 exclusively for the two frozen synthetic validation banks, disjoint from even the training wind. Discard real Dst from this archive when building either bank. Dev uses whole storm and quiet blocks in 2008, 2011 and 2014. Select up to six storms and six 72-hour quiet blocks per year, seed 1729, stratified by storm depth. Commit the actual dates. Reserve every eligible issue hour in 2016–2020 for one primary exam. Leave 2022–2025 sealed for a future release. Unassigned years separate the roles. An issue needs 36 past wind hours, current Dst and three future verification values. Allow wind gaps of at most two hours to be carried forward from the past. Record exclusions.

**Physics and opponents.** Start with the pressure-corrected O’Brien–McPherron injection/decay model. Verify its step response, free recovery and varying forcing, including a fast and slow reservoir, against independent integration and limiting cases. Fit the single-reservoir model and the complete two-reservoir model on exactly the dev blocks that choose network rungs. Include published coefficients and last-value persistence. The integrations project future wind using a causal persistence/relaxation rule. Fit that wind law from the unlabelled training era. Every forecast method stops at the same cutoff. Compare how the model reads simulated and observed twins of the dev cases.

**Rungs, fixed before running.** R1 fits a small instantaneous nonlinear readout on single-reservoir simulations. R2 adds causal wind history and physical state estimates to the same teacher. R3 trains the history model on fast/slow physics calibrated on dev. R4 mixes the two simulation families. The supervised target always comes from simulated Dst. Use independent, seeded train and validation banks and freeze their fingerprints. Judge R1 and R2 on v1, then replay them in order on v2 before judging R3 and R4. Skill equals one minus the mean of two MAE ratios to persistence, for all issues and for storm-strength targets (≤ −50 nT). Keep only a gain greater than 0.01 with no dev year’s corresponding error ratio more than 5% worse. Fit a single widen-only factor on the released rung’s dev residuals for nominal 68% intervals.

**Exam and reporting.** Commit the release choice, calibration, input cutoffs, split and scoring program before reading exam years. Publish MAE for all issues, targets ≤ −50, ≤ −100 and ≤ −150 nT, plus a false-alarm rate on targets above −50. Report sample and independent block counts, nominal 68% coverage and paired 95% intervals on error differences to the best competitor. Use 2,000 resamples of whole storms or quiet blocks. Mark rows with fewer than five blocks inconclusive. A loss stays on the card. Select replay examples by fixed seed within strata, with different blocks. Disclose any departure in a new entry before running the affected experiment.

**Release scope.** Standalone NumPy module, reproducible bank/fit/exam commands, recorded physics and twin checks, model card and ladder, browser replay with an actual network forward pass, data credits and parity checks. No shared simulator changes.

### Release choice and pre-exam checks

The 34 frozen dev blocks contain 2,408 eligible issues. Pressure coupling hit the 2 nT/√nPa lower search bound in both dev fits. R2 moved from rejected on val1 to kept on val2. R3 cleared R2 by 0.0143 on val2 and passed all three years. The mixed R4 lost 0.0049. Select R3 and its 1.16749 widening factor before opening 2016–2020. A diagnostic at initialisation seeds 104 and 105 will quantify readout seed variation without changing this predeclared release. As an independent clock/column check, compare March 2011 in OMNI with the final hourly series published directly by Kyoto.

Use a paired block bootstrap stratified by storm/quiet designation. In each draw, compare the model with the strongest baseline in that draw. Use a tie for an interval spanning zero and an informational verdict for fewer than five blocks. The quiet row is the false-alarm rate among issues whose next-three-hour minimum is above −50. For replay select one distinct block uniformly in each nonempty severity band (most intense first), then one issue uniformly in that block. Bundle up to six preceding consecutive eligible issues for a time scrub. The observed future is shown only on the original-input replay.

### 17 November 2026 · primary exam and release reproduction

The first plan is in commit `3c8f02b`. Commit `6fc38b3` freezes the R3 weights, comparators, exam code and replay sampling before the primary download. Scoring all eligible 2016–2020 issues left 43,439 of 43,810 possible issues across 629 blocks. Missing wind histories accounted for 371 exclusions.

| Target | Issues / blocks | StormNet | Best comparator | Paired difference and 95% interval |
|---|---:|---:|---|---|
| All hours, MAE | 43,439 / 629 | 3.466 nT | Published O’Brien–McPherron, 3.447 nT | +0.019 [−0.011, +0.048] nT |
| ≤ −50 nT, MAE | 811 / 51 | 9.636 nT | Tuned two-state, 9.487 nT | +0.149 [−0.221, +0.519] nT |
| ≤ −100 nT, MAE | 57 / 5 | 17.763 nT | Tuned O’Brien–McPherron, 17.899 nT | −0.136 [−2.077, +2.045] nT |
| ≤ −150 nT, MAE | 7 / 1 | 26.078 nT | Tuned O’Brien–McPherron, 27.379 nT | −1.301 nT, one independent storm |
| > −50 nT, false alarms | 42,628 / 629 | 0.296% | Published O’Brien–McPherron, 0.169% | +0.127 [+0.078, +0.180] percentage points |

The aggregate 68% band covered 71.7%. Coverage fell to 49.0% on storm-strength and 40.4% on intense targets. Three of the seven extreme targets fell inside the band. One storm cannot measure between-storm variation. The original raw bootstrap returned the same 3/7 coverage on every draw for that row. The release report will encode its coverage interval as missing, the same convention already used for its paired difference. This reporting correction does not change any prediction, score or decision.

Dev twins missed the real absolute Dst level by 8–14 nT even where the error magnitudes matched. The R3 minus R2 gains at seeds 103, 104 and 105 were 0.0143, 0.0048 and 0.0128 on val2. The 0.01 keep margin depends on the seed. The four bundled replays come from the frozen stratified draw. They are interactive examples rather than an additional scorecard.

For the release check, reproduce the archived years from the pinned provider files in a fresh work directory. Compare the learned weights, all four bank fingerprints, the entire exam and the replay arrays. Exercise the same NumPy and browser network on every selected issue at three wind settings. Keep 2022–2025 sealed.

The backlog also asks about storm onsets. Add a descriptive **dev-only** check to distinguish warnings of a new crossing from forecasts issued in storms already underway. Use the unchanged R3 and four opponents. For each already-selected dev storm block, find its first observed hourly Dst ≤ −50 nT following a completed hour above −50. Require all three eligible forecast issues one, two and three hours before the crossing hour ends. Count one hit per storm if any of those minima crosses −50. Record the earliest such lead in hours to the completed Dst observation, a 95% Wilson interval across storms, and any excluded blocks. Keep this retrospective check outside the primary exam and all model decisions.

All 16 selected dev storms had usable inputs at all three issue times. R3 warned before 12 first crossings (75%, Wilson 95% interval 51–90%). Published physics warned on nine, both tuned opponents on 15, and persistence on zero. Median lead on hits was one hour to the completed observation for all four forecasting methods that warned. The dev result is descriptive because the same blocks calibrated the physics and chose the rung.

A build in a new work directory reproduced the source hashes, four bank fingerprints, dev reports, scientific figure, ladder, release weights and replay files byte for byte. Every scored value and paired difference matched the original primary run. Python and TypeScript agreed on 84 recording/issue/field combinations. The release weight SHA-256 is `4ca58aacb745d53108a643b54e33e069cd457db2f640ca62d6b7734894e68e49`.

## 17 November 2026 · review and the v1.1 plan (written before any v1.1 experiment)

A review for the owner (Claude) rebuilt v1.0 on Windows from the pinned provider files. The calibration
reproduces to about 1e-8 relative (`single` gain 6.2123453 against 6.2123453), but the bank fingerprints differ at
bit level, presumably from platform maths libraries; no score that matters moves. The release stays the Linux build.

**Dev diagnostic** (`training/dev_diagnostic.py`, dev only). The dev-tuned two-state forecast misses by 4.36 nT on
the 2,408 dev issues. Re-running the same physics with the actual next three hours of wind, which no forecast has,
only reaches 4.21 nT; in main phases it is worse (6.65 → 7.34 nT). So the error is the physics, not the unknown
wind. Truth sits deeper than the forecast: −1.26 nT overall, −3.63 nT on storm targets, −3.14 nT in recovery.
The forecast recovers too fast.

One structural reason: the forecast splits the observed current into fast and slow reservoirs using a ratio
reconstructed from 36 hours of wind alone, while the observed Dst of those same 36 hours is available at issue time
and unused. Long storms keep more current in the slow reservoir than 36 hours of wind can show.

**Plan, fixed now.** Same split, data roles, protocol, rule (gain > 0.01 on the frozen set and no dev year more than
5 % worse) and sealed reserve.

1. *Physics ring-3, two candidates, each fitted on exactly the dev blocks:*
   - a. state assimilation: estimate the fast and slow currents at issue time by least squares against the
     observed Dst of the previous 36 hours, with the same physics as the forecast, instead of the wind-only ratio;
   - b. pressure-dependent injection (Wang, Chao & Lin 2003): injection × (P / 3 nPa)^γ, γ fitted on dev
     (γ = 0 is the current model).
   Keep a candidate only if it lowers dev MAE on all issues and on storm targets in every dev year. Any kept
   physics becomes the new **fair classical baseline** as well as the anchor.
2. *Rungs, judged on new frozen sets built from the kept physics (all earlier rungs re-scored there):*
   - r5: r3's readout with the past 36 hours of Dst as inputs (trend, assimilated states) and 256 fixed random
     units instead of 48, trained on banks whose simulated past Dst carries the simulator's own unresolved noise.
   - Seeds 103 / 104 / 105, judged on the seed mean (R3 showed the 0.01 margin is seed-sensitive).
3. *Error bars:* widen-only factors per issue-time electric-field bin (< 1, 1-3, ≥ 3 mV/m), fitted on the
   released rung's dev residuals.
4. *Exam 2, the sealed reserve 2022-2025 (solar maximum):* opened once, after the v1.1 release decision is
   committed, for v1.0, v1.1 and every baseline, with the same rows, block bootstrap and rules as the primary exam.
   2022-2025 Dst in OMNI is the provisional or quicklook index; the year files and their index versions are
   recorded.
5. The lab and instrument are rebuilt around whole-storm replays once exam 2 is scored.

## 17 November 2026 · v1.1 decisions (dev and simulation only; reserve still sealed)

- **Physics ring-3: neither candidate kept** (`training/calibration_v3.json`). Rule fixed above: lower dev MAE on all
  issues and on storm targets in every dev year.
  - Assimilated split: storm MAE 11.77 → 11.69 (2008), 8.63 → 8.62 (2011), 6.61 → 6.61 nT (2014); all-issue MAE
    slightly worse in 2011 and 2014.
  - Pressure-dependent injection (γ = 0.106): storm MAE 11.77 → 11.33 and 8.63 → 8.57 nT, but 2014 worse (6.61 → 6.79).
  The physics is near what three dev years can resolve. The baselines and simulator stay ring-2.
- **r5 rejected** (`training/v11_results.json`). With the window's observed Dst and 256 units, seeds 103 / 104 / 105
  score 0.1283 / 0.1410 / 0.1283 on val2 (mean 0.1325) against r3's 0.1340, and every dev year is worse. One seed
  alone would have passed; the seed mean decides, as planned. The rebuilt r3 reproduces the released one exactly on
  the rebuilt val2 (0.1340) and dev.
- **v1.0.1 error bars** (`training/calibration_sigma.json`). The plan named electric-field bins, but on dev those bins
  were already covered (69 %, 64 %, 68 % with the single factor). Issues where the last Dst is already ≤ −50 nT were
  covered 55 %. The bins therefore follow the last observed Dst: factors 1.575 (≤ −50), 1.232 (−50 to −30) and
  1.092 (> −30 nT), replacing the single 1.167. This change was decided on dev, before the reserve was opened, and
  moves only the bars; every point forecast is v1.0's.
- **Release:** v1.0.1 = r3 with the binned bars. Next, exactly as planned: open the sealed 2022-2025 reserve once
  and score v1.0, v1.0.1 and every baseline on the primary exam's rows, bootstrap and rules.

## 17 November 2026 · exam 2 (the sealed reserve) and the storm replay

The reserve was opened once, after `a36824b`, by `training/exam_reserve.py`: 2022-2025, 35,026 possible issues,
1,958 excluded for wind gaps, **33,068 scored in 515 blocks, 85 of them storms**; deepest block 9-19 May 2024 (−406 nT).

| Row (v1.0.1) | StormNet | Best competitor | Difference [95 %] | Verdict |
|---|---:|---|---|---|
| All hours | 4.356 nT | published O’Brien–McPherron 4.343 | +0.013 [−0.029, +0.055] | tie |
| ≤ −50 nT | 10.690 | dev-tuned O’Brien–McPherron 10.616 | +0.074 [−0.206, +0.341] | tie |
| ≤ −100 nT | 23.650 | dev-tuned O’Brien–McPherron 23.086 | +0.564 [−0.151, +1.238] | tie |
| ≤ −150 nT | 36.158 | dev-tuned O’Brien–McPherron 34.991 | +1.167 [+0.070, +2.161] | loss |
| Quiet false alarms | 0.615 % | published 0.376 % | +0.238 [+0.178, +0.304] pp | loss |

Persistence: 5.03, 14.20, 30.21 and 44.07 nT on the four MAE rows. Coverage of the 68 % bars, v1.0 → v1.0.1:
65.3 → 63.2 % (all), 51.5 → 59.9 % (storm), 41.9 → 52.9 % (intense), 41.7 → 50.9 % (extreme).

Reading: with three times as many storm blocks as the primary exam, the verdict is the same. StormNet is far
better than persistence and better than published physics, and level with physics tuned on the same dev storms. The
network adds nothing measurable over its anchor on real storms; the same finding as WakeNet's on 17 Nov 2026.

The instrument and lab now replay six whole storms chosen by rule (`training/storm_gallery.py`, by observed minimum
only): the deepest storm of 2022, 2023, 2024 and 2025, the deepest primary-exam storm (Aug 2018) and one quiet block
drawn with the example seed. The browser reruns every hourly forecast with all three physics competitors and the
anchor's fast and slow reservoirs; 120 Python reference cases, including what-if winds, match to 4.5e-15.
