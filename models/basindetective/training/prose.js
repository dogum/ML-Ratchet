$('examLede').innerHTML = `Each exam basin's network input was its 1999–2008 forcing and streamflow plus the 1989–1999 forcing.
The truth is how the six signatures of its real 1989–1999 gauge record differ from 1999–2008. Spearman ρ ranks basins:
1 means the network orders all 619 basins exactly by how much they changed. The classical answers are B's per-basin
calibrated GR4J model (two ways to read a change from it) and the Budyko-Fu curve with its shape fitted to each basin's own
gauged decade. <b>Result: a tie.</b> On rank correlation BasinDetective is within 0.05 of the best classical answer on every
target and ahead on low flow; on median absolute error it is best or level on four of six (runoff ratio, baseflow index, low and high flow). It does this without fitting any
basin, and with error bars that hold (if anything they are too wide: 73–83 % of truths fall inside the 68 % bars).`;
$('regLede').innerHTML = `This is where the network fails. On simulated basins it separates regulated from natural flow with
AUC 0.91. On real exam basins, against HydroATLAS reservoir storage, it scores AUC 0.49: chance, like the two classical
scores. Only against USGS peak-flow codes 5 and 6 ("regulation", "diversion"; 11 basins) does it reach 0.71. CAMELS was chosen
to be minimally impacted, and the simulated reservoirs (generic standard-operating, rule-curve and Hanasaki policies) do
not look like the few real ones.`;
$('hiddenLede').innerHTML = `Nobody has measured a real basin's groundwater time constant or soil capacity, so these outputs are
checked on <b>twins</b>: each of the 52 dev basins re-simulated with its five dev-calibrated model setups, real forcing and
gauge noise, where the parameters are known. On the exam there is only a plausibility check: the predicted precipitation
multiplier rises with the gauged runoff ratio (ρ 0.65) and is 1.16 for the 42 basins whose rivers carry more water than
ERA5-Land says fell on them; the groundwater share tracks the filter BFI (ρ 0.80).`;
$('ratchetLede').innerHTML = `A rung is kept only if it beats the incumbent on the frozen simulated set of the current physics by
more than 0.01 (mean of R² over 11 outputs and 2·AUC−1 for regulation) and the real dev composite (52 dev basins, gauged decade
to the non-exam 1980s) drops by no more than 0.02. Each physics change built a new frozen set, and all earlier rungs were
re-scored on it: R1 scores 0.546 on its own set but 0.410 on the v4 set. R4 is rejected although its dev error bars were
better: the rule is the rule. R5 (v3 and v4 banks together) is the release.`;
$('steps').innerHTML = [
 `<b>Physics.</b> WaterSim, the shared Phase 1 simulator: HBV, GR4J, SAC-SMA and VIC-lite runoff models, degree-day and
  energy-balance snow on elevation bands, unit-hydrograph routing, reservoirs with operating policies. GR4J matches airGR to
  1e-7 mm, HBV matches TUWmodel to 1e-12 mm, and every run closes its water balance to 1e-15.`,
 `<b>Simulated basins in real weather.</b> 10,000 basins per physics version: 40 % copy a dev basin's attributes and use its own
  real ERA5-Land forcing from 1951–1988, 35 % are generated basins on a random CAMELS basin's forcing from those years, 25 % use
  a stochastic weather generator. Each run is laid out like the exam: one spin-up year, a ten-year decade the network sees only
  as forcing, then nine gauged years. A random model structure and snow routine per basin, so the network cannot learn one
  model's habits.`,
 `<b>The forcing is not the truth.</b> The simulator is driven by "true" precipitation; the network reads the forcing product.
  Between them: a basin multiplier, daily error, extra snowfall, a temperature offset, a year-to-year error and a slow drift
  between decades. Streamflow goes through a gauge model (slow rating errors, revisions, 3 significant figures in ft³/s, gaps).`,
 `<b>Features and network.</b> About 150 numbers per basin: climate of both decades and their differences, and the gauged
  decade's signatures, flow-duration curve, monthly regime, recession, baseflow filters and rain-flow correlations. The
  signature code is an exact port of the classical code that defines the real truth (max difference 1.3e-6). Five MLPs (2 × 256
  GELU, about 100,000 weights each) give a mean and a sigma per output and a probability of regulation.`,
 `<b>Dev twins found the physics.</b> Twins of the 52 dev basins (their calibrated models driven by their own forcing) were read
  by the network next to the real records. Three gaps became rungs: real records have smooth recessions and many flat days
  (v2: gauge noise fitted to them); ERA5-Land's precipitation level drifts between decades, because twins said the 1980s
  carried 21 % more flow than 1999–2008 while the gauges said 6 % (v3: a drift of 2.9 ± 3.8 % per decade, scaled from the
  dev measurement); annual flows track ERA5-Land precipitation less closely than twins do (v4: 10 % year-to-year error).`,
 `<b>Error bars.</b> One factor per output, widen-only, fitted on dev (the six changes on the 52 real dev basins, hidden
  properties on the 260 twins): mean flow ×1.15, BFI ×1.49, half-flow date ×1.73, the rest 1.0–1.12.`,
 `<b>Exam.</b> Run once on the 619 exam basins with the released model. Nothing was chosen on it.`
].map(s => `<li>${s}</li>`).join('');
$('limitsList').innerHTML = [
 `It ties per-basin calibrated GR4J on the decade-to-decade changes; it does not beat it on rank correlation. Its advantages are
  calibrated error bars, no fitting per basin (milliseconds per basin), and a better low-flow answer.`,
 `Regulation detection does not transfer to real basins (AUC 0.49). Simulated reservoirs are generic; the HydroATLAS truth may
  itself be wrong where a pour point snaps to a bigger river.`,
 `Error bars are too wide on the exam (73–83 % inside the 68 % bars) because they were fitted on the 1980s dev window, where
  the forcing drift was larger. The widen-only rule cannot narrow them.`,
 `The drift rung was measured on the 1980s and scaled linearly to the 1990s. If ERA5-Land drifted differently, it is wrong.`,
 `Where ERA5-Land badly undercatches precipitation (gauged runoff ratio above 1; 42 exam basins) skill collapses for every
  method (mean-flow ρ 0.22).`,
 `Hidden properties have no real truth. The "slow store" means a different store in each model structure.`,
 `It needs about nine years of daily streamflow (gauged mode). Baseflow-index changes are barely predictable by anyone (ρ 0.20–0.25).`
].map(s => `<li style="margin-bottom:8px">${s}</li>`).join('');
$('credits').innerHTML = `<p><b>Data.</b> Caravan v1.6 (Kratzert et al. 2023, CC BY 4.0) with CAMELS-US streamflow (Newman et al. 2015,
Addor et al. 2017) and ERA5-Land forcing (Muñoz-Sabater et al. 2021, Copernicus); HydroATLAS attributes (Linke et al. 2019,
CC BY 4.0) including GRanD reservoir volumes; USGS NWIS annual peaks (public domain). Classical baselines: WaterData's GR4J
calibrations and signatures (Phase 1). Simulator: WaterSim (Phase 1).</p>
<p>BasinDetective v1.0 · ML Ratchet, Watershed Phase 2 · trained only on simulations, graded on reality.</p>`;
