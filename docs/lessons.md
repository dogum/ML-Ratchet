# Lessons from twenty-six models

What building ChirpNet, SunDoctor, CellSense, TransitNet, QuakeNet, BearingNet, SwellCast, BeatNet and StarQuake,
then the vehicle platform with EnergyNet, TripDetective and HybridPilot, the grid platform with GridPulse,
BatteryPilot, PlantDetective and DemandCast, WakeNet, the building platform with LoadCast, BuildingDetective,
FaultFinder and ThermoPilot, StormNet, and the water platform with FlowCast, SnowSense, BasinDetective and
ReservoirPilot, taught, in rough order of how much it mattered.

## The simulator is the model

- **Plot real dev data next to simulations of the same case, early.** QuakeNet's two biggest gains — surface waves
  6.5% slower than ak135, and a scattering coda — came from doing this. Frozen-set score 0.556 → 0.808; exam distance
  error 11° → 5.7°.
- **The twin test finds what is missing.** Re-simulate each dev unit with the full simulator and compare. BearingNet's
  rung 1 read 100% of simulated twins right and 22% of real dev files: real faults were 10–25 dB louder than the
  textbook prior and rang longer. Fixing it took dev balanced accuracy from 0.31 to 0.68.
- **Priors and nuisance physics matter as much as the main physics.** Aging rates (CellSense), limb darkening
  (TransitNet), the world's mix of earthquake depths and sizes (QuakeNet), storm behaviour (SwellCast). Fit them on
  dev units, or match them to real data that predates the exam.
- **Inject into real instrument noise** whenever the instrument has a public archive. It is the cheapest realism there is.

## The ratchet keeps you honest

- **It regularly rejects "calibrated" simulators** that match dev statistics better but read real dev data worse
  (SwellCast v2/v3, StarQuake v2, BeatNet r4). Mixing old and calibrated banks is the usual fix. The cause is not
  always understood — and the ratchet doesn't need it to be.
- **Version the frozen set with the physics, and re-take every decision.** Decisions can flip both ways
  (QuakeNet rungs 5 and 6).
- **Change the rule in the open when it's wrong.** ChirpNet's first rule rejected a rung with 55 points more glitch
  robustness; the rule was replaced and the log shows both.
- **A small dev set can be too kind.** BearingNet's Paderborn dev scored 0.78; the exam 0.45. Draw dev units from
  several machines, conditions or strata.

## Report like you mean it

- **Always include the dumb baseline.** SwellCast's median guess wins on distance and time because the exam
  geography is narrow. QuakeNet's shallow depth loses to always guessing 31 km. Both pages say so.
- **Predict corrections to a model-free estimate**, not absolute values. TransitNet's radius head failed to learn
  absolute k (34% error); predicting the correction to √depth fixed it and removed a 3–5% bias.
- **Check every catalogue input.** TransitNet: catalogue periods drift; per-sector alignment was needed.
  SunDoctor: the Colorado plant's clock was in Mountain time with daylight saving, and its main irradiance sensor
  read half the true sunlight for three and a half years (May 2019 to October 2022).
- **Error bars need their coverage reported.** QuakeNet's depth bars cover 51% of a nominal 68%. StarQuake's cover
  54–58% on 27-day light curves. Those numbers are the next rungs.
- **Compare with the physics tuned on your own dev data.** WakeNet and StormNet both beat textbook physics and both
  tie it once its few free settings are fitted on the same dev data their simulators were (17 Nov 2026). WakeNet lost
  17 of 18 re-scored rows to dev-tuned FLORIS; StormNet ties dev-tuned O'Brien–McPherron on 85 storm blocks at solar
  maximum. A network trained on its own anchor's physics can only add what the simulator knows and the anchor does
  not. If the simulator is the anchor plus scatter, expect a tie, and give the simulator something more.
- **Before blaming the unknown future, test with it.** StormNet's dev forecasts improved only from 4.36 to 4.21 nT
  when the physics was given the actual next three hours of wind, so a better wind forecast was not the lever.

## Working practice

- Long jobs need resumable checkpoints; cloud containers restart (2–3 times per model here).
- Four builds in parallel on a shared 2-core machine worked: each took 1–2 hours, kept to one thread and small models.
- Never kill processes by name on a shared machine — another build's `simbank.py` is not yours.
- If a data host is blocked, don't route around it. Record it and ask. ChirpNet's O3 exam waited on exactly that: GWOSC
  could not be reached from the shared build machine, and the exam ran from another connection on 12 Nov 2026 (7 of 7
  in-range events found). CellSense's other chemistries are still waiting.

## Big systems (the vehicle platform)

- **Build a shared platform in two phases.** Phase 1: one simulator and one real-data/splits package, built in
  parallel against a written contract. Phase 2: the models in parallel on top. Freeze the shared layers during
  Phase 2, let each model patch in its own wrappers, and upstream the fixes afterwards with a new physics version.
- **Builds checking each other's data catch real bugs.** EnergyNet found swapped EPA cycle labels and a driver-and-load
  mass counted twice (136 kg) in RoadData.
- **A controller can only be examined in the simulator.** HybridPilot's exam is real routes driven by simulated twins.
  It needs a separate simulator-versus-real check: RoadSim burns about 20% less fuel than real hybrids, a gap far larger
  than the 1–3% between strategies, so the ranking holds only inside its physics.
- **Fleet priors can matter more than physics.** TripDetective's dev mass bias fell from +106 kg to −9 kg by matching
  the simulated fleet's mass distribution (median 2,200 kg) to the real one (1,720 kg).
- **Grade at matched end states.** Correcting hybrid fuel with a fixed battery-energy factor credited ECMS with 105% of
  the optimum; re-solving DP to each strategy's own final charge gave 99.6%.

## Big systems (the grid platform)

- **The two-phase recipe worked a second time.** The two Phase 1 builds cross-checked each other's data: GridSim's
  twin of a GB dev event showed its 1,788 MW loss label was about twice what the frequency drop implies (metered output:
  854 MW). The four Phase 2 builds filed concrete upstream fixes, each with a measured effect.
- **Watch "defaults" for exam-period leaks.** GridSim's outage rates cited Texas reliability aggregates published in
  2024-2025, inside the exam years. Nothing chose from them, but a default is data too: date-check every one.
- **Physics wins where the physics is; reporting quirks decide the rest.** DemandCast beats the grid operators' own
  day-ahead demand forecasts in 6 of 8 regions but loses on solar, because EIA-930 solar timestamps moved by an hour
  between years. PlantDetective reads coal and gas-steam heat rates at 4.4-4.7% against EIA-923's 10-22%, but loses on
  combined cycles, where CAMPD reports only the gas turbine's output.
- **A skill can come from the wrong input.** GridPulse's inertia beats a constant guess (16.4% vs 20.4%) but not a
  one-line rule from demand fitted on four dev events (15.1%); from the trace alone it is 20.3%. One-second data blur
  the first-second slope that carries inertia. A dumb baseline built from the context inputs is the one to beat.
- **A tie is a result.** BatteryPilot captures 0.667 of perfect hindsight in ERCOT against 0.685 for the classical
  day-ahead optimiser: it wins with 4-hour batteries and loses with 1-hour ones and in CAISO. Its dev twins capture
  91-94% while real 2019 gives 80%, so the simulated price spikes are still too predictable.
- **Plan archive downloads away from shared IPs.** The forecast-weather archive rate-limited the shared build
  machine's address after a few requests, which left DemandCast's realistic forecast-weather exam at one year.

## Big systems (the building platform)

- **A twin from metadata alone can be right on one meter and far off on the others.** BuildingSim's twins of the
  52 BDG2 dev buildings got electricity right on the median (0.98×) but read campus chilled water, steam and hot water
  about 3× low. The models that leaned on the thermal meters paid for it: LoadCast with no history forecasts them about
  2× low and loses to a median guess.
- **Real logs record commands, not positions.** FaultFinder's twin test found a lab valve stuck at 100 % logged as 0 %,
  and a damper at 100 % mixing only 81-90 % outdoor air. Adding the installed damper curve and larger sensor offsets
  (rung r4) raised real dev AUC from 0.71 to 0.73, and dev false alarms fell from 45 % (r2) to 36 %; on exam days it
  still raises 33 %.
- **A signature the simulator gets wrong does worse than no skill.** BuildingDetective tells heat pumps from
  resistance heat at AUC 0.94 on simulated homes and 0.40 on real RECS homes: worse than chance, because simulated heat
  pumps run at night and real ones run flat.
- **When the exam is simulated, check the order on real data.** ThermoPilot is graded on twins of real homes. On 37 real
  heat-pump homes its order against a setback schedule flipped in 18 between generic and calibrated twins, so that
  comparison is not settled by the simulator.
- **Share-alike data travel with the files.** BDG2 is CC BY-SA 4.0. Its excerpts live in folders marked with their own
  licence notice, apart from the MIT code, and the instruments and lab pages carry the same notice.

## Big systems (the water platform)

- **Simulation training wins where the physics is local and the cost is exact, and ties where real behaviour must
  be borrowed.** SnowSense (snow at one point, from that point's gauge) beats every weather-only physics baseline run,
  and ReservoirPilot (one dam's mass balance, a fixed cost) halves a tuned rule's cost. FlowCast loses to plain
  regionalisation, which borrows real flow from similar gauged basins, and BasinDetective cannot see human regulation.
- **The weather input can cap every method at once.** All four water models and all their baselines run on
  ERA5-Land, which under-catches mountain precipitation (50 of 671 basins carry more water out than falls in) and
  drifts about 6 % between decades. Calibrated GR4J reaches a median NSE of 0.47 on it, against about 0.6-0.7 with
  the CAMELS forcing in the literature. Compare only with baselines on the same weather, and say so on every table.
- **The loss can be the sim-to-real gap.** FlowCast's first rung scaled each basin's error by the spread of its own
  flow, a standard choice for real data. In simulation that spread depends on the hidden parameters, so wet, flashy
  draws were down-weighted: the network ran low (dev high-flow bias −51 %) on real basins and on their twins alike.
  Scaling by an input-derived runoff ratio instead (r0b) raised dev NSE from 0.390 to 0.436.
- **A metric that ignores a cost gets exploited.** ReservoirPilot halves the cost J of a classical rule tuned in the
  same simulator, partly by keeping reservoirs low: J charges nothing for storage. It holds a median 38 % of capacity
  against SOP's 59 % and ends at least 20 % of capacity lower on a third of the dams. Report the storage beside J.
- **What the simulator draws generically does not transfer.** BasinDetective finds generated reservoirs at AUC 0.91
  in simulation and real regulated basins at 0.49, chance. The bank's reservoirs were random sizes and policies, not
  the hydropower peaking and withdrawals real dams show.
- **Points are not areas.** At SNOTEL points the calibrated snow threshold sits about +1.3 °C above the default; in
  whole basins it shifts the other way. Default energy-balance snow put point peaks about 20 % low. SnowSense's
  dev-fitted parameter prior (rung R3) took its dev NSE from 0.805 to 0.880.
- **A Gaussian mean drags a zero-bounded series late.** SnowSense's Gaussian head melted out 10 days after its own
  twin, because the mean of many zero-SWE days pulls the curve up; quantile heads fixed it (R4).
- **Rung decisions near the margin are noise.** FlowCast's r5 gained +0.02 on a 300-basin frozen subset and lost
  0.006 on the full 800. Size the frozen set for the margin.
