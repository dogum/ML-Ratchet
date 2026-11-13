# Lessons from twelve models

What building ChirpNet, SunDoctor, CellSense, TransitNet, QuakeNet, BearingNet, SwellCast, BeatNet and StarQuake,
and then the vehicle platform with EnergyNet, TripDetective and HybridPilot, taught, in rough order of how much it
mattered.

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
  read half the true sunlight for three years.
- **Error bars need their coverage reported.** QuakeNet's depth bars cover 51% of a nominal 68%. StarQuake's cover
  54–58% on 27-day light curves. Those numbers are the next rungs.

## Working practice

- Long jobs need resumable checkpoints; cloud containers restart (2–3 times per model here).
- Four agents in parallel on a 2-core sandbox worked: each took 1–2 hours, kept to one thread and small models.
- Never kill processes by name on a shared machine — another agent's `simbank.py` is not yours.
- If a data host is blocked, don't route around it. Record it and ask (ChirpNet's O3 exam and CellSense's other
  chemistries are waiting on exactly that).

## Big systems (the vehicle platform)

- **Build a shared platform in two phases.** Phase 1: one simulator and one real-data/splits package, built in
  parallel against a written contract. Phase 2: the models in parallel on top. Freeze the shared layers during
  Phase 2, let each model patch in its own wrappers, and upstream the fixes afterwards with a new physics version.
- **Agents checking each other's data catch real bugs.** EnergyNet found swapped EPA cycle labels and a driver-and-load
  mass counted twice (136 kg) in RoadData.
- **A controller can only be examined in the simulator.** HybridPilot's exam is real routes driven by simulated twins.
  It needs a separate simulator-versus-real check: RoadSim burns about 20% less fuel than real hybrids, a gap far larger
  than the 1–3% between strategies, so the ranking holds only inside its physics.
- **Fleet priors can matter more than physics.** TripDetective's dev mass bias fell from +106 kg to −9 kg by matching
  the simulated fleet's mass distribution (median 2,200 kg) to the real one (1,720 kg).
- **Grade at matched end states.** Correcting hybrid fuel with a fixed battery-energy factor credited ECMS with 105% of
  the optimum; re-solving DP to each strategy's own final charge gave 99.6%.
