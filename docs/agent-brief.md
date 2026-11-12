# Brief for an AI research agent building a model in ML Ratchet

Read this fully before starting. It is the recipe as a checklist, adapted from the brief the last four models
(BearingNet, SwellCast, BeatNet, StarQuake) were built from by parallel agents. The original is in
[history/agent-brief-2026-11.md](history/agent-brief-2026-11.md).

You are building one new model for an open-source collection of physics-first ML models. The maintainer enjoys
learning outside their field and prefers visual output and plain writing. Follow the recipe exactly.

## Where you work

- Scaffold with `mlratchet new <id> --name <Name>`. Work only inside `models/<id>/`.
- Put downloads, banks, checkpoints and builds in `models/<id>/work/` (or `$ML_RATCHET_WORK/<id>/`) via
  `training/_paths.py`. Never commit them.
- Keep `models/<id>/NOTEBOOK.md` current: what's done, what's next, numbers, gotchas — enough to resume after a
  restart.

## The recipe (non-negotiable)

1. **Real physics.** Write the simulator from the governing physics. Verify it against an independent reference
   (a published formula, a library, a known case) and record the check (number + file).
2. **Simulated data in real noise.** Where the instrument has public recordings, inject simulated signals into REAL
   background recordings (quiet periods, healthy machines, noise databases).
3. **Train a network only on simulations.** No real labelled example is ever a training sample.
4. **Ratchet.** Keep a frozen simulated validation set, versioned with the physics (physics changes → new frozen set
   → re-score every rung → re-take every decision in order). A rung is kept only if it beats the incumbent on the
   frozen set by more than a stated margin (0.01 on a skill score in [0, 1] is usual) AND passes a dev gate on a
   SMALL real dev set (a handful of units, chosen by a fixed seed, never in the exam). Log every rung in
   `ratchet.yaml` with the reason. Use `mlratchet.Ratchet`.
5. **Calibrate the simulator on dev only.** Plot real dev data against simulations of the same case, measure the
   mismatch, build it into the simulator as the next rung. Run a twin test.
6. **Final exam on real data never used for choosing anything.** Report it with a classical method and a dumb
   baseline; error bars with their coverage (one widen-only factor per output, fitted on dev); and where it fails.
   Never cherry-pick examples (fixed random draw within strata).
7. **Ship** per [the model contract](model-contract.md): numpy-only inference (BatchNorm folded, numpy == torch to
   ~1e-5), weights `.npz`, `demo.py` on real exam examples that runs in seconds, `MODEL_CARD.md`, `model.yaml`,
   `ratchet.yaml`, `training/`. Run `mlratchet validate <id>`, `mlratchet readme --write <id>`, and
   `python tests/test_demos.py --regen`; check that only your model's golden file changed.
8. **Optional showpiece:** `lab/index.html` — a hero that states the thesis, a real-data explorer, the network
   running live in the browser in JS (verify JS == Python on a few examples), a small lab with sliders, the exam vs
   baselines, the ratchet ladder, limits, credits. Self-contained, under ~8 MB, light and dark themes, phone width.

## Be a good neighbour on shared machines

- One thread (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`); `nice` heavy jobs.
- Small networks (≤ ~500k parameters), banks ≤ ~60k examples in float16, ≤ ~30 minutes per rung,
  ≤ ~1.5 GB RAM per process.
- Never `pkill -f` / `killall`, never kill processes you didn't start. Name your scripts distinctively.
- Long jobs: detach them, write resumable checkpoints every ~500 steps, and poll — containers restart.

## Network and data rules

- Never disable TLS verification or route around a proxy.
- If a host refuses or is blocked, do not try to get around it (no mirrors of blocked content, no probing
  undocumented APIs). Find another documented public source, or record it as blocked in `NOTEBOOK.md` and in the
  card's `help_wanted`.
- Download only from documented public endpoints; respect licences; cite every dataset in `model.yaml` and
  `DATA_LICENSES.md`.

## Writing style

Plain, short, direct sentences. Real units and numbers. No hype. Say what failed.

## Your final report

Page/lab path, the recipe as built (physics check numbers, noise source, dev set), the rungs table (kept/rejected
and why), the exam table vs baselines with coverage, the top 3 limits, anything blocked, and the 2–3 best next rungs.
Numbers over adjectives.
