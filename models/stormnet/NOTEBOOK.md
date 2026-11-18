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
