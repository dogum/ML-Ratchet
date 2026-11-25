# Vehicle Bake-off round 2: pre-registration

*Written 24 Nov 2026, 23:00 EST, before any round-2 network was trained or any round-2 number was scored. Committed
before training. Later changes are dated amendments at the end; nothing above them is rewritten.*

Round 2 is handoff item 20 (Vehicle). The v1.0 bake-off (exam run 20261123T204251) found that learner families converge
(E16 11.7-13.0 %) and fail together (per-trip log errors correlate at a median 0.92 across families). In 5 of 16 sets
every learner's median exam bias has the same sign and exceeds 3 %: Mersin diesel and CNG buses (-17 to -29 %), the VED
Leafs (-16 to -27 %), VED plug-ins (-9 to -19 %) and the ICCT trucks (+6 to +27 %). EnergyNet v2.1 (5 MLPs on 3 banks)
beats the single-bank MLP control by 1.2 points. Round 2 asks: how much of the gap is data and ensembling, how much is
the simulator, does mixing families help, and can family disagreement flag the shared misses without labels.

## 0. Fixed for every rung

- **Rules.** No real labelled example is ever a training sample. Dev (VED 16 dev cars, EPA dev families, Mersin dev week,
  ICCT dev truck, 5 ZTBus dev legs) is for gates, physics checks and conformal calibration only. Exam units are never
  read before the round-2 exam, and the exam chooses nothing. Frozen splits are unchanged. Every rung is logged,
  rejected ones included, in `round2/STATUS.md` and the arm's `ratchet.json`.
- **Data.** Banks are rebuilt with the harness recipe (`harness/build_bank.py`, unchanged; driven by
  `round2/bank/build.py`): `val` = v2.1's frozen val2[h] (400 vehicles x 10 trips, seeds 202 / 777) and v2.1's three train
  banks `train101`, `train102`, `train103` (fleet train2 seed 201, trip seeds 101-103, 28,800 trips each, 86,400 in all).
  `val` and `train101` must equal `harness/bank/{val,train}/MANIFEST.json` byte for byte (sha256 of every file);
  if they do not, round 2 stops and reports. There is no manifest for 102 and 103: they are checked by row count
  (28,800) and by their dyno / idle-stop composition matching EnergyNet's (r14: 450 SC03 / cold-FTP rows per bank on
  average, r15: 2,964 idle-off rows per bank).
- **Scoring.** `harness/evaluate.py` functions, unchanged: full frozen sim val (3,985 trips, frozen tiers seed 777),
  skill = 1 - median |log error| / the dumb baseline's (dumb rates from train101 as in v1.0); dev **D16** = mean of the
  16 dev MdAPEs. Every incumbent is re-scored by the same code on the rebuilt val (it must reproduce its v1.0 numbers:
  GRU 0.6390 / 0.1089, transformer 0.6440 / 0.1135, MLP control 0.6682 / 0.1203, EnergyNet v2.1 0.6797 / 0.1133).
- **Inference** of every round-2 arm is numpy / scipy only (plus the harness's numpy `bake.e_phys_from_trace`, as the
  v1.0 trace arms). Training may use the GPU.
- **Ensemble combination (every ensemble in round 2, within a family and across families).** Each member returns 5
  log-quantiles (q05, q25, q50, q75, q95). The ensemble's log-quantile at each level is the equal-weight mean of the
  members' log-quantiles at that level (quantile averaging in log space); the point `pred` is the ensemble's q50 =
  exp(mean of the members' log q50). For the MLP family the member quantiles are its log-normal quantiles (members
  combined as in v1.0's tab_mlp: sigma = sqrt(mean member sigma^2 + spread of member means)). No member weights are fitted.
- **The data-rung gate (the bake-off's rule).** Keep a rung if its frozen sim val skill beats the incumbent's by
  **more than 0.01** AND its dev D16 is **not worse than the incumbent's + 0.01**. Both scored on the same frozen val
  by the same code.

## 1. Rungs, in order

| # | Rung | Change | Incumbent | Gate |
|---|---|---|---|---|
| D1 | GRU ensemble on 3 banks | v1.0 GRU r1 recipe (`arms_trace/train/configs/gru_r1.json`: same net, physics anchor, 12 epochs, same batching), 5 members (seeds 0-4), every member on all 86,400 trips | v1.0 `trace_gru` (r1) | data-rung gate |
| D2 | Transformer ensemble on 3 banks | v1.0 transformer r2 recipe (r1 phase: 12 epochs, peak lr 2e-3, data seed s; r2 phase: warm restart 24 epochs, peak lr 1e-3, data seed s + 100), 5 members (s = 0-4), all 86,400 trips | v1.0 `trace_transformer` (r2) | data-rung gate |
| D3 | MLP control on 3 banks | v1.0 tab_mlp r3 recipe (EnergyNet's net, 5 nets x 30 epochs, seeds 0-4), all 86,400 trips | v1.0 `tab_mlp` (r3) | data-rung gate |
| P0 | RoadSim physics / bank fix | section 2: candidates checked on dev only; a kept fix becomes preset `roadsim-physics-v4` (or a bank-recipe change) | RoadSim physics v2 | section 2 dev check |
| P1-P3 | GRU, transformer, MLP ensembles on the P0 banks | D1-D3's recipes, unchanged, on `val_v4` + `train101_v4` .. `train103_v4` (same draws, new physics) | the same family's round-2 entry after D1-D3 | physics-rung gate (section 2) |
| X | Cross-family ensemble | new arm (section 3) | its best member | section 3 |
| G | Gap detector | dev analysis only (section 4) | - | reported either way |
| C | Conformal wrappers | v1.0's Mondrian wrapper (`arms_phys/conformal`) on each round-2 arm, dev residuals only | - | registered as `<arm>_conformal` if made |

- **Diagnostics (not rungs, decide nothing):** member 0 of D1 and D2 alone (one network on 3 banks) is scored too, so the
  data effect (v1.0 single network, 1 bank -> member 0, 3 banks) and the ensembling effect (member 0 -> 5 members) are
  reported separately.
- **Round-2 entry of a family** = its last kept rung (P if kept, else D if kept, else the v1.0 arm). The GP family is
  not retrained in round 2 (budget): its entry is v1.0's `tab_gp` (r3, one bank).
- **Order:** banks -> PREREG commit -> D1, D2, D3 (GPU, may run concurrently) -> P0 (dev checks run while D trains) ->
  P banks -> P1-P3 -> X -> G -> C -> registration -> exam-2 driver `--check` and dev rehearsal. Stop.

## 2. The physics / bank rung (P0) and its gate

**Candidates** (in this order of attack; each one only if a dev set shows the miss):
1. Mersin buses too low (simulated diesel / CNG buses under-use fuel): bus auxiliaries, passenger load, stop / idle
   behaviour, CNG engine efficiency.
2. Cold-weather BEV losses (cabin heat, battery at low temperature): VED dev Leaf, EPA dev BEV cold data if any.
3. Plug-in charge-depleting behaviour: VED dev Volt-type and Energi cars.
4. Dynamometer vs road gap (ICCT trucks too high on the dyno; EPA dev vs VED dev road levels).

**Evidence first.** For each candidate the round-2 data rungs' dev predictions (D1-D3) and EnergyNet v2.1's are read
first. A candidate is pursued only if its dev set shows a same-sign median bias above 3 % in at least 3 of the 4 round-2
data-rung learners (D1, D2, D3, EnergyNet v2.1). If dev does not show the exam's miss, the candidate is logged as "not
visible on dev: not attempted" (dev cannot be used to fix what it cannot see).

**Dev check of a physics fix (P0 pass marks).** A fix is a RoadSim option change in a new named preset or a bank-recipe
change. It passes its dev check only if all hold:
- (a) on the target dev set, the median bias of RoadSim twins / the simulated quantity the fix acts on moves toward 0 by
  at least half of the dev bias it was aimed at, judged on dev units only;
- (b) every existing preset reproduces its golden (`pytest -q tests/test_roadsim* tests/test_vehicle*` passes) and the
  new preset gets its own golden;
- (c) guard: on the RoadSim checks it touches (EPA dev pool, VED dev twins, heavy-duty and ZTBus dev checks), no dev
  median error the change can reach moves away from 0 by more than 2 points.
A fix that fails is logged as rejected (with numbers) and the next candidate is tried. At most two fixes are combined
into `roadsim-physics-v4`.

**Physics-rung gate (P1-P3).** A physics rung changes the simulated targets, so sim val skill is not comparable across
physics (EnergyNet NOTEBOOK, 14 Nov 2026: rung r4 gained +0.097 skill on its own version while D2 got worse). A P rung
is kept only if BOTH hold:
- (i) the data-rung gate with both arms scored on the same frozen val: the P0 physics's own version of the frozen val
  (`val_v4`: val2's vehicles, trips, tiers and seeds re-simulated under the new physics; the incumbent re-scored on it):
  skill > incumbent + 0.01 AND D16 <= incumbent + 0.01;
- (ii) EnergyNet's physics-rung rule: dev D16 improves on the incumbent's by **more than 0.01**, and the rung's skill on
  its own val version is not more than 0.01 below the incumbent's skill on the v2 val.
Also reported, deciding nothing: both arms on the v2 frozen val, and the five shared-bias dev sets' bias before / after.

## 3. Cross-family ensemble arm (X)

- **Members:** the round-2 entries of the transformer, GRU, MLP and GP families (section 1), equal weights (1/4 each).
  It is a trace-track arm (the two trace members need the trace). The two table members' 151 inputs are recomputed
  from the trace-track batch with the harness's own functions (route vector from the trace, E_phys and the physics
  vector from the trace + rl, as `bake.e_phys_from_trace` and `tests/check_trace.py` do; the vehicle vector from spec
  with the label entries made relative again). Before use, the recomputed inputs are checked against the table batch on
  sim val and every dev set (pass mark: median |log difference| of the table members' predictions below 0.002). If the
  check fails, X is scored from the members' saved per-record predictions under the same rule instead, and the exam
  driver does the same. **Point and quantiles:** section 0's rule over the four members (mean log-quantile per level;
  point = exp(mean log q50)).
- **Its gate (pre-exam verdict, deciding only whether it is called a gain):** against its best member (highest sim val
  skill among the four): sim val skill > best member + 0.01 AND D16 <= best member + 0.01. It is registered for the
  exam either way, as a base rung (r0), unless its dev D16 is worse than its best member's by more than 0.01 (then it
  is logged as rejected and not examined).

## 4. Label-free gap detector (G): dev only

- **Spread** of a dev unit = the standard deviation, across the four family entries of X, of their log point
  predictions, at the scoring unit (VED trip, EPA test vehicle, Mersin trip, ICCT test, ZTBus leg).
- **Error** = |log(pred_X / y)| of the X ensemble on that unit.
- **Reported per dev set** (the 16 D16 sets + ZTBus legs): Spearman rho(spread, error) with a cluster bootstrap 95 % CI
  (clusters as `score.py`: car / vehicle / unit; 2,000 resamples, `default_rng(20261123)`).
- **Enrichment:** pool the units of the 16 D16 sets; weight each unit 1 / (units in its set), so every set counts
  equally; flag the top 10 % of weighted spread. Enrichment = share of flagged weight in the five shared-bias sets
  (VED_EV, VED_PHEV, Mersin_diesel, Mersin_cng, ICCT) / their share of all weight (5 / 16). Cluster bootstrap 95 % CI
  (clusters within sets, sets fixed).
- **Pass marks (fixed now):** the detector "flags the shared misses on dev" only if the enrichment CI lower bound is
  above 1.0 AND the median per-set Spearman over the 16 sets is above 0.10. Otherwise it "does not". Reported either way.
  The exam is never touched.

## 5. Round-2 exam plan (exam-2; written and checked now, run once by the owner)

- Driver `round2/exam2.py`, based on `harness/exam.py`, with its own lock and done files under `round2/results/`
  (`EXAM2_RUNNING.json`, `EXAM2_DONE.json`). v1.0's `results/EXAM_DONE.json`, `harness/registry` and its exam lock are
  never touched. `--check` and `--rehearsal` (dev) only, in this round; `--run` is the owner's.
- **Arms scored:** every arm in the round-2 registry (`round2/registry/`): D1-D3 and P1-P3 if kept, X, conformal
  wrappers. **Reference rows re-run in the same exam:** EnergyNet v2.1, climatology, E_phys, the classical baselines,
  and v1.0's `trace_gru`, `trace_transformer`, `tab_mlp`, `tab_gp` (deterministic: they must reproduce their v1.0 exam
  numbers on every set; that is the exam's reproduction check, as v1.0's EnergyNet check).
- **Sets:** v1.0's exam sets (E16's 16 sets: VED ICE / HEV / EV / PHEV trips, EPA conv FTP / HWY / US06 / cold FTP / SC03,
  hybrid FTP / HWY, BEV city / HWY, Mersin diesel / CNG, ICCT; plus car totals, VED conditions) and the ZTBus v1 legs
  (78) and missions (8). **Full ZTBus** (5,847 legs, 1,115 missions, EnergyNet's protocol incl. layovers) through a
  marked hook for the owner's `harness/exam_supp_ztbfull.py`.
- **Metrics:** as v1.0 (MdAPE, bias, cov50 / cov90, crps_log; E16; paired cluster bootstrap, 2,000 resamples, seed
  20261123; a win or loss only when the 95 % CI excludes 0).
- **Pre-registered comparisons (the exam chooses nothing; each is reported win / tie / loss):**
  1. D1 vs v1.0 GRU and D2 vs v1.0 transformer, D3 vs v1.0 MLP on E16 (data + ensembling on the exam);
  2. each kept P rung vs its D counterpart on E16 and on each of the five shared-bias sets (bias toward 0 and MdAPE);
  3. X vs each member and vs EnergyNet v2.1 on E16;
  4. every round-2 arm vs EnergyNet v2.1 on E16;
  5. cross-family per-trip error correlation (median over E16 sets) among round-2 entries, vs v1.0's 0.92;
  6. number of sets where every round-2 learner's median bias has the same sign and exceeds 3 % (v1.0: 5).

## 6. Budget and scale-down

About 8 hours of wall time; at most 8 CPU processes and under 5 GB of GPU memory. If a rung is clearly too slow, it is
scaled down by a dated amendment here, before its results exist (first choices: fewer epochs per member for D2 / P2, then
3 members instead of 5, then P rungs on the GRU and MLP only). Compute is logged per rung (GPU wall time and CPU time).

## Amendments

**A1 (24 Nov 2026, 23:50 EST; before any sim-val number on the rebuilt banks).** The rebuilt `val` bank's
`trace_v.npy` and `trace_x.npy` equal the v1.0 manifest byte for byte, but `rows.npz` (same size, 3,175,964 bytes) and
`fleet.json` (438,240 vs 438,245 bytes) do not. The npz members carry a fixed 1980 timestamp, so the difference is in
content: the generator's and RoadSim's floating-point results on this Windows machine vs the Linux cloud (fleet.json is
5 characters shorter: float reprs). Byte equality of these two files is therefore not reachable on this machine, and
section 0's "stop if not byte-equal" is replaced by a functional check of the same frozen set: (i) the trace files equal
byte for byte (val and train101); (ii) row counts equal; (iii) EnergyNet v2.1 scored on the rebuilt val reproduces its
frozen-set numbers (n 3,985, MdAPE 0.0746, skill 0.6797) to |diff| < 0.0005; (iv) v1.0's GRU, transformer and MLP
control re-scored on it reproduce their v1.0 sim val skill to |diff| < 0.002 (they were trained on the cloud train101).
If (i)-(iv) hold, round 2 continues on the rebuilt banks; every round-2 comparison is between arms scored on the same
rebuilt val by the same code. If not, round 2 stops and reports. Dev records were rebuilt too (harness/build_real.py);
EnergyNet v2.1's dev D16 on them is 0.1133 and the MLP / GP controls' 0.1203 / 0.1260, as v1.0 (already checked).

**A2 (24 Nov 2026, 23:52 EST; before any sim-val number on the rebuilt banks and before any training).** train101's
`trace_v.npy` equals the manifest, but its `trace_x.npy` does not (same size, 465,302,798 bytes): A1 (i) fails for
train101. trace_x holds the float16 weather and grade channels; synthetic routes and weather draws go through the same
platform math as fleet.json, so this is the same cross-platform float difference, now visible after float16 rounding.
It cannot be measured element by element (the cloud bank is not shipped). Rather than stop with the owner asleep, one
more functional check is added and decides, together with A1 (iii)-(iv): (v) v1.0's MLP-control recipe (tab_mlp r3:
5 nets x 30 epochs, seeds 0-4) retrained on the rebuilt train101 alone must reach the v1.0 tab_mlp's sim val skill
(0.6682) within 0.005 (the seed-to-seed spread of a 5-net set is about this size) and dev D16 (0.1203) within 0.005.
If (iii)-(v) hold, round 2 continues; the train-bank difference is reported. If not, round 2 stops and reports.
Composition of the rebuilt train banks (A1 / section 0): 28,800 rows each; SC03 + 20 F cold-FTP dyno rows 450 / 454 /
455 (EnergyNet: 1,359 over its three banks: 450 + 454 + 455 = 1,359, equal); idle-stop-off rows 2,964 each (EnergyNet:
8,892 = 3 x 2,964, equal).
