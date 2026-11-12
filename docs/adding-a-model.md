# Adding a model

A practical walk-through. The [method](method.md) says why; this says how. If you are an AI research agent, read
[agent-brief.md](agent-brief.md) too.

## 0. Pick a problem worth it

A good candidate has all three:

1. **Physics you can simulate** from governing equations, and an independent reference to check the simulator against.
2. **Public real data** with a documented download and a licence you can work with — enough for a noise pool, a
   handful of dev units, and an exam.
3. **A classical method to beat**, plus an obvious dumb baseline.

[backlog.md](backlog.md) lists candidates with their data sources already checked. Open an issue with the
*New model idea* template before starting something big.

## 1. Scaffold

```bash
pip install -e ".[dev]"
mlratchet new gridpulse --name GridPulse
mlratchet validate gridpulse        # tells you what is still TODO
```

This creates `models/gridpulse/` from the template with every file in [the contract](model-contract.md), a
`training/` folder with stubs for each step, and a checklist README.

## 2. Physics, then check it

Write `training/simulator.py`. Keep the physics, the prior over the world, and the noise separate, and version the
physics (`PHYSICS_VERSION`). Write `training/physics_check.py`, run it, and commit `physics_check.json`.

## 3. Real noise, and the split

- `training/noise.py`: fetch real background recordings from a documented public endpoint and screen them (quiet
  periods, healthy machines, stars that don't oscillate in your band).
- `training/split.py`: split real labelled units into **dev** (a handful — 3 to 40 — chosen by a fixed seed, ideally
  spread over machines/conditions/strata) and **exam** (everything else). Split by unit, never by window. Commit
  `split.json`.

## 4. Banks and the frozen set

```bash
cd models/gridpulse/training
python simbank.py train1 50000 1
python simbank.py val1 5000 99       # frozen: never regenerate it for one rung's benefit
```

Banks are float16 and resumable. When the physics version changes, build `val2` and re-score every rung.

## 5. Train, judge, repeat

```bash
python train.py --name r1 --bank train1 --steps 5000
python ratchet.py r1 "textbook physics"
```

`ratchet.py` uses `mlratchet.Ratchet`: a rung is kept only if it beats the incumbent on the frozen set by more than
the margin AND passes the dev gate. It writes `../ratchet.yaml` either way.

Between rungs, look at dev:

- Plot real dev units next to simulations of the same case. Measure the mismatch. Build it into the simulator.
  That is your next rung.
- Run a **twin test**: re-simulate each dev unit with the full simulator and compare the network's readings of twin
  and real. Where they differ, the simulator is missing something.
- Expect calibrated simulators to sometimes read real dev data *worse*. Mixing old and new banks is the usual fix
  (SwellCast, StarQuake). Let the dev gate decide.

## 6. Error bars

Fit one widen-only factor per output on dev residuals (`mlratchet.calibrate.widen_factor`) and store the factors in
the weights' `meta`. Never shrink bars.

## 7. The exam — once

`training/exam.py`: model vs a classical method vs a dumb baseline fitted on dev, with exam coverage of the 68% bars
and a breakdown by stratum. Run it after the ratchet has chosen the release. Copy the headline rows into
`model.yaml` with honest verdicts; put the full tables in `MODEL_CARD.md`.

## 8. Ship

- `training/export.py`: fold BatchNorm, write `<id>_weights.npz`, check numpy == framework to ~1e-5.
- `<id>.py`: numpy-only inference. `demo.py` on 3–15 real exam examples drawn by a fixed seed within strata.
- Fill in `MODEL_CARD.md` and `NOTEBOOK.md`; set `status: released` and mark the released rung.
- `mlratchet readme --write gridpulse`, then record the golden demo output:
  `python tests/test_demos.py --regen` and check the diff is only your model.
- Optional but loved: an interactive `lab/index.html` with the network running in the browser.
- Add your data sources to `DATA_LICENSES.md`.

## Compute

Every model so far was built on a shared 2-core, 7 GB cloud sandbox in 1–2 hours of wall time per model. Small is
enough: one thread, ≤ ~500k parameters (QuakeNet's 1.44 M is the outlier), banks ≤ ~60k examples, ≤ ~30 minutes per
rung, a resumable checkpoint every ~500 steps.
