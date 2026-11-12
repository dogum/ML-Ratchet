# __NAME__ (in progress)

This folder was scaffolded by `mlratchet new __ID__`. Once `model.yaml` and `ratchet.yaml` are filled in,
`mlratchet readme --write __ID__` replaces this file with the standard README.

Work through the recipe in order (full guide: [docs/adding-a-model.md](../../docs/adding-a-model.md)):

- [ ] **Pick the problem.** Real physics you can simulate, public real data to grade on, a classical method to beat.
- [ ] **Physics.** `training/simulator.py`. Verify it against an independent reference in `training/physics_check.py`
      and record the number in `training/physics_check.json`.
- [ ] **Real noise.** Lay simulated signals into real instrument recordings (`training/noise.py`).
- [ ] **Split by unit, by seed.** A handful of real dev units; everything else is the exam. Write the seed down.
- [ ] **Frozen simulated set.** Build it once per physics version (`training/simbank.py val1 ...`).
- [ ] **Train** on simulations only (`training/train.py`). No real labelled example is ever a training sample.
- [ ] **Ratchet.** `training/ratchet.py` judges each rung: frozen-set gain > margin AND dev gate. Log every rung.
- [ ] **Calibrate the simulator on dev.** Plot real dev data next to simulations of the same case; build the
      mismatch into the simulator as the next rung. Try a twin test.
- [ ] **Error bars.** One widen-only factor per output, fitted on dev (`mlratchet.calibrate.widen_factor`).
- [ ] **Final exam, once.** Model vs a classical method vs a dumb baseline, with coverage and where it fails.
- [ ] **Ship.** numpy-only `__ID__.py`, weights `.npz`, `demo.py` on a fixed random draw of real exam examples,
      `MODEL_CARD.md`, and optionally an interactive page in `lab/`.
