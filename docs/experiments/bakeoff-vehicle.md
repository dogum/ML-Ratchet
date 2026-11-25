# Vehicle Bake-off: one problem, every learner family

*Method experiment, 23-24 Nov 2026. Scripts, arms and raw results: [`experiments/bakeoff-vehicle/`](../../experiments/bakeoff-vehicle/).
Exam run `20261123T204251`, run once by `harness/exam.py` for all 21 registered arms together. Second bake-off, after
[the River Bake-off](bakeoff-river.md).*

The River Bake-off found that, given the same simulated bank and budget, learner families converge, and that the
simulator is the lever. River's problem had weak physics and one input form. The Vehicle Bake-off tests the finding on
a problem with strong physics and two input forms. **The problem** is EnergyNet's: the energy (fuel or electricity) a
vehicle uses on a real trip, from the vehicle spec, the route and the weather. EnergyNet v2.1 (an ensemble of five MLPs
correcting a physics estimate, log(E / E_phys)) is the incumbent; the harness reproduced its published exam exactly on
every in-scope row.

Two tracks, two leaderboards:
- **Table track:** EnergyNet's 151 physics features, the same for every arm.
- **Trace track:** the 1 Hz trip trace (speed, grade, elevation, temperature, wind, air density, surface) plus the spec
  vector, read as shared 5-s block means.

Every arm got the same bank (EnergyNet v2.1's recipe on RoadSim physics v2; 28,800 simulated trips, one of v2.1's three
banks), the same targets, dev gate and exam, at most about 4 single-thread CPU-hours and 3 rungs, and zero real labels in
training. The exam covers VED trips (25,793 records; petrol, hybrid, Leaf, plug-in), EPA test cars (76,037 test legs),
Mersin buses, ICCT trucks and the 78 traced ZTBus legs of RoadData v1.

## Verdict

**The families converge, and they make the same mistakes.** On E16, the mean exam MdAPE over the 16 sets that form the
dev score D16, the seven networks and trees span 11.7-13.0 %. With 105k exam records most gaps are statistically
separable, but all are small. Per-trip log errors correlate at a median 0.92 between different families (0.95 within a
family). In 5 of 16 sets every learner's median bias has the same sign and exceeds 3 %: cold Leafs, plug-ins, both
Mersin bus fleets and the ICCT trucks.

**The levers are the physics and the bank, not the family.** Correcting the road-load estimate E_phys (22.9 % alone) is
worth 10-11 points to every learner; on the trace track, adding that anchor raised each network's simulated skill by
0.016-0.034. Data is next: the same MLP ensemble trained on 3x the bank (EnergyNet v2.1, 11.0 %) beats the retrained
control (12.1 %) by 1.2 points and beats every new learner by 0.7-2.0. Where RoadSim misses a fleet (the Mersin buses,
17-29 % under for every learner), a dev median beats the best learner by 10-13 points.

**Unlike River, the learners match the classical methods, and attention wins.** A transformer over 20-s patch tokens
(11.7 %) is the best new learner. It beats the GRU control by 0.4 points and a dev-fitted classical composite (MOVES-like
on VED, EPA regression, dev medians on heavy duty; 12.5 %) by 0.9; every other network or tree ties that composite. The
conformal layer improved CRPS for all 10 arms it wrapped, where River's made CRPS slightly worse in 14 of 18.

## Leaderboards (E16, mean exam MdAPE over 16 sets; paired cluster bootstrap, 2,000 resamples per set)

| Row | Track | E16 | vs EnergyNet v2.1 | vs control | cov90 raw → conformal | CRPS raw → conformal | Train CPU-h |
|---|---|---|---|---|---|---|---|
| EnergyNet v2.1 (as released) | table | **11.0 %** | | | 0.85 → 0.87 | 0.094 → 0.091 | |
| Transformer, 20-s patches | trace | 11.7 % | +0.7 loss | −0.4 win (GRU) | 0.75 → 0.89 | 0.101 → **0.090** | 1.64 |
| Bidirectional GRU (control) | trace | 12.0 % | +1.1 loss | | 0.77 → 0.88 | 0.102 → 0.092 | 1.65 |
| MLP, EnergyNet net retrained (control) | table | 12.1 % | +1.2 loss | | 0.71 → 0.86 | 0.104 → 0.094 | 0.62 |
| Classical composite (dev-fitted) | ref | 12.5 % | +1.6 loss | tie (MLP) | | 0.160 | |
| Sparse GP | table | 12.6 % | +1.6 loss | +0.4 loss (MLP) | 0.92 → 0.87 | 0.102 → 0.096 | 0.81 |
| Gradient-boosted trees | table | 12.6 % | +1.7 loss | +0.5 loss (MLP) | 0.75 → 0.83 | 0.099 → 0.094 | 0.24 |
| 1-D dilated CNN | trace | 12.7 % | +1.7 loss | +0.7 loss (GRU) | 0.72 → 0.82 | 0.108 → 0.098 | 1.57 |
| S4D state space | trace | 13.0 % | +2.0 loss | +0.9 loss (GRU) | 0.71 → 0.86 | 0.110 → 0.097 | 2.75 |
| Trip-energy law (symbolic) | both | 21.1 % | +10.1 loss | | 0.83 → 0.81 | 0.141 → 0.128 | 0.87 |
| E_phys alone / physics anchor | both | 22.9 % | +11.9 loss | | 0.83 → 0.82 | 0.164 → 0.154 | 0 |
| Climatology | both | 31.0 % | +20.1 loss | | 0.92 | 0.189 | |

TabPFN v2 (in context, a lens, not a release) and its conformal wrapper were excluded before the exam, for compute only.
Their pre-exam numbers stand: sim val skill 0.563, dev D16 12.5 %.

## Where each assumption wins

| Powertrain (mean MdAPE of its sets) | Best | Next |
|---|---|---|
| Conventional (VED ICE, 5 EPA tests, Mersin, ICCT) | classical composite 7.9 % | EnergyNet 10.4, transformer 10.7 |
| Hybrid (VED HEV, EPA hybrid FTP / HWY) | trees 9.0 % | EnergyNet 9.3, S4D / transformer 9.6 |
| Plug-in (VED PHEV) | EnergyNet 16.8 % | MLP 17.6, trees 18.1 (MOVES-like 41.3) |
| Battery-electric (VED Leaf, EPA BEV) | sparse GP 11.2 % | classical 11.8, trees 12.2, EnergyNet 12.5 |
| Heavy duty (Mersin, ICCT, ZTBus v1) | dev median 9.3 % | transformer / EnergyNet 13.5 |

By VED trip condition (all four powertrains pooled; read from inputs only), EnergyNet is lowest in city, highway, hot,
hilly and short trips, with the GRU statistically tied each time. The GRU leads cold trips (11.7 %), and the GP leads
long trips (9.4 %), tied with every learner. In each condition the best new learner is 1.3-2.8 points below the
dev-fitted MOVES-like model.

## Before the exam (from the ratchet ladders)

- The physics anchor was the biggest rung for every trace arm: +0.034 GRU, +0.028 S4D, +0.022 transformer, +0.016 CNN.
- More training on one bank did not substitute for more banks. The MLP at 5 x 180 epochs fell to 0.597 skill (0.668
  at 30 epochs), and 2x training was rejected for the GRU and the CNN.
- Monotone constraints on the trees cost simulated skill again (−0.005), as in River.
- The transformer needed more steps (+0.0145 from 24 more epochs, the River lesson); 10-s patches did not help.
- The formula's dynamometer-flag rung fixed EPA highway on dev (0.262 → 0.096) but was invisible to median simulated
  skill and was rejected. On the exam the released law scores 30.5 % on EPA highway.
- S4D had the best simulated skill of the trace arms (0.655) and was last on the exam. Overall, sim val skill ranks
  the arms' exam E16 with Spearman 0.60, and dev D16 with 0.62.

## LawFinder's fifth law

The formula arm's conventional-vehicle law, e = 3.406 p^0.835 + 0.385 x 1.184^(exp c) x m t exp(−0.058 m), scores
14.0 % MdAPE on VED petrol exam trips. That compares with 33.0 % for road-load, 13.0 % for the window label × distance,
10.3 % for MOVES-like and 8.6 % for EnergyNet. It is in `lawfinder_road/` for LawFinder (see MERGE_NOTES).

## What this does and does not show

- It shows, on one problem with strong physics, that the choice of family matters about ten times less than the
  physics anchor, and that different families share their errors. The next gain lies in the bank (Mersin buses,
  cold-weather BEVs, plug-in behaviour, dyno vs road), then in data volume.
- It does not rank families in general: networks of 49k-1M weights, one seed each, ≤ 4 CPU-hours, a third of
  EnergyNet's data, single networks on the trace track.
- ZTBus is the RoadData v1 subset (78 legs of 8 missions); EnergyNet's full-ZTBus rows are out of scope here. VED Leaf has
  two exam cars. E16 weights 16 very different sets equally.

## Rules and notes

Exam run once after `exam.py --check` passed; every arm re-hashed unchanged after scoring; 0 error rows. EnergyNet v2.1
and the classical baselines reproduced their published exam numbers. The TabPFN exclusion was decided before the exam
and recorded in `registry/excluded.json` with its reason; the documented mechanism needed a two-line harness fix
(`register.registered()` read `excluded.json` as an arm record). Zero real labels in training; dev only for rung gates
and conformal calibration. Page: the Vehicle Bake-off artifact (the same HTML is in `page/`).

## Supplementary exams (24-25 Nov 2026, in the repository, after the main exam)

Two dated exams, each run once, reported here beside the main board and never folded into it. Driver:
`harness/exam_supp.py` (exam.py's own worker and scoring, with its own lock and records); results in
`results/supp_tabpfn/` and `results/supp_ztbfull/`, each with a PRE_EXAM.md written before its run.

**TabPFN (handoff item 17; run 20261124T230626).** The two arms excluded before the main exam for compute, scored on
the main exam's records rebuilt here. The other 19 table-track rows were re-scored beside them and match the main exam
on all 1,306,017 units (prediction difference ≤ 1.2e-11). TabPFN's E16 is **13.3 %**, a loss to every table learner:
trees and the GP by 0.7 points, the classical composite by 0.8, the MLP control by 1.2, EnergyNet v2.1 by 2.3 (95 %
intervals all above 0). Its conformal wrapper lowers CRPS (0.105 to 0.097), not the point error. The rebuilt arm is not
bit-identical to the cloud's: dev D16 0.1240 here against 0.1249 at registration (library versions), recorded before
the run.

**Full ZTBus (handoff item 21; run 20261125T013250).** The Zurich trolleybus exam the cloud machine could not hold:
5,847 labelled legs of 1,115 missions. The records reproduce EnergyNet's published exam exactly (v2.1 legs 7.04 %,
missions 5.59 %; dev-median baseline 14.01 % and 10.66 %).

| Row | Legs MdAPE | vs EnergyNet v2.1 (legs) | Missions MdAPE (sum of legs) |
|---|---|---|---|
| EnergyNet v2.1 | **7.04 %** | | **5.22 %** |
| GRU (trace) | 7.15 % | +0.1 tie | 5.79 % |
| MLP control (table) | 7.23 % | +0.2 tie | 5.74 % |
| Transformer (trace) | 7.29 % | +0.3 tie | 5.74 % |
| 1-D CNN (trace) | 7.52 % | +0.5 loss | 6.53 % |
| Trees (table) | 8.25 % | +1.2 loss | 5.99 % |
| S4D (trace) | 8.62 % | +1.6 loss | 8.77 % |
| Trip-energy law (formula) | 9.29 % | +2.3 loss | 8.09 % |
| TabPFN (table) | 9.30 % | +2.3 loss | 8.20 % |
| Sparse GP (table) | 9.52 % | +2.5 loss | 6.84 % |
| Dev median (kWh/km x km) | 14.01 % | | 10.65 % |
| E_phys alone (physics anchor) | 21.5 % | | 21.3 % |

On missions EnergyNet beats every learner by 0.5 points or more. The pattern of the main board holds on a fleet it did
not include: the families sit within 2.5 points of each other, every learner halves the dev-median baseline, and the
physics anchor alone is 21 % low. Unlike the Mersin buses, these electric buses are not a shared miss. One new
weakness: the conformal layer, calibrated on dev strata, made EnergyNet's bands too narrow for these buses (90 %
coverage 0.96 raw, 0.62 after conformal; the GP 0.98 to 0.82). Full table: `results/supp_ztbfull/leaderboard.md`.
