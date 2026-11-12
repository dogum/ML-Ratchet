# The method

Physics first, data second. Every model in ML Ratchet is built the same way, in six steps. The steps are simple.
The discipline is in never skipping one.

```
physics ─► simulator ─► simulated data in REAL noise ─► network
               ▲                                           │
               │ the measured mismatch                     ▼
               │ becomes the next rung          ratchet: frozen simulated set
               └──── dev set (a few real units) ◄──── + dev gate
                                                           │
                                                           ▼
                           final exam on real data that never chose anything
```

## 1. Real physics, checked

Write the simulator from the governing physics: Green's functions for earthquakes, general relativity for mergers,
deep-water dispersion for swell, an eikonal voxel heart for ECGs. Then check it against something independent — a
published formula, a reference library, a known case — and record the number.

| Model | Check | Result |
|---|---|---|
| QuakeNet | Local synthesis vs EarthScope Syngine full synthetics | r = 0.9999 |
| TransitNet | Limb-darkened transit vs batman | ≤ 0.94 ppm |
| SunDoctor | Row shading vs pvlib | 1e-15 |
| CellSense | Single-particle model vs PyBaMM | capacity within 0.2% |
| SwellCast | Ridge slope vs Munk's g/(4πD) | within 1.4% at 1,500 km |
| BearingNet | Fault frequencies vs CWRU's published multiples | 1.6e-4 |
| StarQuake | Scaling relations reproduce Yu et al.'s 16,094 masses | ratio 1.000 |

## 2. Simulated signals in real noise

Where the instrument has public recordings, lay the simulated signal into **real** background: quiet-time
seismometer windows, healthy-machine vibration, quiet dwarf stars, buoy records with no storm, the MIT-BIH noise
stress test. The network never sees idealised noise, so it never learns to rely on it.

## 3. Train only on simulations

No real labelled example is ever a training sample. This is the rule that makes the exam mean something: the
network cannot have memorised the answer key, because it never saw one.

## 4. The ratchet

A new version of the model — a **rung** — is kept only if both of these hold:

1. **It beats the incumbent on a frozen simulated validation set by more than a stated margin.** The usual margin is
   0.01 on a skill score in [0, 1]. The set is frozen: built once, never regenerated for a rung's benefit.
2. **It passes a dev gate on a small real dev set.** A handful of real units (records, stars, cells, machines) chosen
   by a fixed seed, never in the exam. Typical gates: dev errors no more than 5% worse; mean dev-error ratio ≤ 1.10;
   dev S/V F1 drops by no more than 0.02.

Every rung is logged, kept or not, with the reason (`models/*/ratchet.yaml`). Rejected rungs are as informative as
kept ones: across the collection, 24 of the 61 rung decisions were rejections.

**When the physics changes, the frozen set changes.** Build a new frozen set from the new physics, score every
rung on it, and re-take every decision in order. QuakeNet's rung 5 was rejected on val5 by 0.002 and kept when every
decision was re-taken on val7; rung 6 passed on val7 and fell short on val8. `mlratchet.Ratchet.rescore` does this.

**The rule can change, in the open.** ChirpNet's first rule (Gaussian sensitivity must rise) wrongly rejected a rung
that had gained 55 points of glitch robustness. The rule was replaced, the rung re-judged, and both decisions are in
the log.

## 5. Calibrate the simulator on dev only

The biggest wins came from plotting real dev data next to simulations of the same case, measuring the mismatch, and
building it into the simulator as the next rung:

- **QuakeNet:** real surface waves arrive about 6.5% slower than ak135 predicts, and real records have a scattering
  coda. Building both in took the frozen-set score from 0.556 to 0.808 and the exam distance error from 11° to 5.7°.
- **BearingNet:** real faults were 10–25 dB louder than the textbook prior allowed and rang longer. Fixing that took
  dev balanced accuracy from 0.31 to 0.68.
- **TransitNet:** the limb-darkening prior came from free fits to the dev planets.

**The twin test** is the sharpest tool: simulate each dev unit's twin with the full simulator — same event, same
station, same star — and check whether the network reads twin and real alike. If it reads twins right and reals
wrong, the gap is in the simulator, and the twin tells you where. QuakeNet's twins showed the shallow-depth problem
lives in surface waves after S; BearingNet's showed the fault-level prior was wrong.

## 6. The final exam, once

Real data that was never used to choose anything. Report it honestly:

- **Beside a classical method and a dumb baseline** (always guess the dev median, or the majority class). SwellCast
  loses to the median guess on distance and time, and the page says so.
- **With error bars and their coverage.** Fit one widen-only factor per output on dev (`mlratchet.calibrate`), then
  report what fraction of exam truths fall inside the 68% bars — 51% for QuakeNet depth is a finding, not a typo.
- **Where it fails.** Never cherry-pick examples: demo examples are a fixed random draw within strata.

## Why bother

Because the alternative is a model you can't trust. A network trained on real labels can memorise quirks of one
dataset. A network trained only on physics, kept only when it improves on a frozen test and a real dev set, and
graded once on real data it never saw, tells you exactly how far physics plus learning gets you — and where it
doesn't.

See also: [the model contract](model-contract.md) · [adding a model](adding-a-model.md) · [lessons](lessons.md).
