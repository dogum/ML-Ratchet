# Vehicle Bake-off round 2: STATUS

Pre-registration: `PREREG.md` (committed d1a98a4 before any training). Exam-2 is NOT run (the owner runs it once).

## Layout
- `paths.py` the path layer (import first): round-2 banks, work/, registry/, results/; points the harness's outputs
  (dev records, dev refs) at `work/` so v1.0's `results/` and `harness/registry` are never written.
- `bank/` bank builder (harness/build_bank.py driven per named bank; `run_banks.sh` = 8 workers + finalize).
- `trace/` GPU training of trace ensembles (`train_ens.py`, v1.0 nets and recipe imported unchanged; `dyncache.py`).
- `table/` MLP control on several banks (`train_mlp3.py`, v1.0's train_mlp.py imported).
- `arms/` round-2 arm folders (`common/enslib.py`: numpy ensemble inference).
- `eval.py` harness evaluate.py on round-2 banks (`--val val_v4` for the P0 physics's own val).
- `work/` (git-ignored): banks, dyn caches, checkpoints, eval outputs, dev records, logs.

## If the session restarts
- Bank build: `bash bank/run_banks.sh <banks...>` is resumable (finished chunks are skipped).
- Training: `train_ens.py <config> <member>` resumes from work/trace/<name>/m<k>/phase<j>.resume.pt.

## Log
- 24 Nov 22:40 EST. Worktree fast-forwarded to 5a4b6da (bake-off import). `shared/vehicle/data` is a junction to the
  main checkout's data (the tracked files there equal this commit's except CRLF line endings).
- 22:44 bank build started: val, train101, train102, train103 (8 workers, ~3 min per 600-trip chunk).
- 23:00 PREREG.md written and committed (d1a98a4) before any training.
- EnergyNet v2.1's own dev terms (training/ratchet2.json, r17), read for the P0 evidence rule: VED Leaf bias +4.6 %
  (1 dev car; exam -20 %: not visible on dev), VED plug-ins -7.3 %, Mersin CNG -17.0 %, Mersin diesel -3.1 %
  (dev is February only, no A/C trips), ICCT +19.0 % (3 cycles of one truck).
- 23:38 P0 dev check, candidate 4 (dynamometer vs road; bank recipe `hddyno`: heavy trucks never run on the dyno in
  the bank, ICCT records are dyno tests). RoadSim twin of the dev truck BIC-US (ref Class 8 sleeper at the 34.4 t test
  mass; `physics/icct_twin.py` -> `physics/icct_twin.json`): road run vs dyno run differ by -1.0 % (median of 3
  cycles), while the twin is +22.6 % (55 mph grade), +35.5 % (55 mph flat), -3.7 % (ARB transient) against the
  measured fuel. The dyno / road distinction cannot explain the +19 % dev bias: **rejected** by PREREG (a). The miss
  is the truck itself at cruise (RoadSim's reference Class 8 burns 25-35 % more than this best-in-class 2015 truck);
  one dev truck, labelled best-in-class, cannot ground a fleet-level truck change: not attempted.
- 23:47 P0 dev check, candidate 3 (plug-ins; bank recipe `phevcold`: a plug-in road trip with start charge > 5 % of
  the SOC window starts with engine and oil at ambient instead of the bank's warm 85 C engine). RoadSim's v2 VED dev
  twins on their own dev trips (`physics/ved_phev_twin.py` -> `.json`; run_trip of checks/check_ved_twin.py
  unchanged): trip median energy error, warm (the bank's rule; reproduces CALIBRATION.md H25: -12.6 / -10.7 %) ->
  soc rule: Energi (VED_11, 143 trips) +6.2 %, Volt-type (VED_537, 164 trips) +5.3 %. PREREG (a) passes by a hair
  (each |median| halves: 12.6 -> 6.2, 10.7 -> 5.3); (c) passes on the check's own medians (per car, both toward 0).
  Caveats, logged before any bank or network result: trips below 10 C overshoot (Energi -36.7 -> +12.8 %, Volt
  -27.0 -> +39.5 %), mean |error| gets worse (32.7 -> 35.7 %, 39.8 -> 50.7 %). This is H25a's finding again, with a
  SOC rule in place of the logged engine speed. Used for the P rungs only if the D rungs' dev evidence rule holds.
- 23:45-00:10 dev records rebuilt (harness/build_real.py -> work/real/dev: VED 1,498, EPA 10,345, HD 232 records);
  dev reference rows (eval.py --refs). Banks finalized 23:50 (8 workers, ~65 min wall). Manifest: val trace_v / trace_x
  EQUAL, rows.npz / fleet.json DIFFERENT; train101 trace_v EQUAL, trace_x / rows.npz / fleet.json DIFFERENT (same
  sizes; fleet.json 3-5 bytes shorter): cross-platform float results. PREREG amendments A1 / A2 (functional checks,
  written before the checks ran). Results: (iii) EnergyNet v2.1 on the rebuilt val: n 3,985, MdAPE 0.0746, skill
  0.6797 (= v1.0); dev D16 0.1133 (= v1.0). (iv) v1.0 arms on the rebuilt val / dev: GRU 0.6390 / 0.1089, transformer
  0.6442 / 0.1135, MLP 0.6682 / 0.1203, GP 0.5963 / 0.1260 (all = v1.0). (v) v1.0 MLP recipe retrained on the
  rebuilt train101 (GPU): skill 0.6644 (v1.0 0.6682, -0.0038 < 0.005), D16 0.1195 (0.1203): PASS. Composition:
  SC03 / cold-FTP dyno rows 450 / 454 / 455 (= EnergyNet's 1,359), idle-off rows 2,964 per bank (= 8,892 / 3).
  Round 2 continues on the rebuilt banks.
- 00:05 GPU training started: D1 GRU (m0 alone ~8 min/member at full speed; ~15 min with 5 jobs on the GPU),
  D2 transformer (~45 min/member, 3 queues), then D3 MLP (~1 min/net). dyn caches 13 s per bank.
- 00:15 **D3 (MLP control, 5 nets x 30 epochs, 3 banks, GPU) REJECTED**: sim val skill 0.67816 vs the v1.0 tab_mlp
  0.66823 = +0.00993, under the 0.01 margin (by 0.00007); dev D16 0.1118 vs 0.1203 (better by 0.0085). Same shape as
  EnergyNet's r11b (2x data, rejected by 0.001). The MLP family's round-2 entry stays v1.0 tab_mlp. Candidate folder
  moved to work/cand/r2_d3_mlp (not an exam arm). Dev bias of the five shared-bias sets (D3): Leaf +5.0 %, plug-ins
  -7.1 %, Mersin diesel -1.8 %, CNG -17.1 %, ICCT +20.5 %.
- 00:21 X input rebuild check (`arms/check_xinputs.py`, PREREG section 3): table inputs rebuilt from trace batches vs
  the table batch, member predictions median |log diff| 1e-6 to 8e-5 (sim val 400 trips, 200 records of each dev
  record set), max 0.0026: PASS. X runs as a pure trace-track arm.
- 00:20 `phevcold` banks: val_pc (134 of 290 plug-in rows changed), train101_pc (1,104 changed; median energy of a
  changed row +13.5 %, +31 % below 10 C); replay exact (5 re-simulated unchanged rows: max |rel diff| of the targets 0).
- 00:58 **P3 (MLP control recipe on the phevcold banks) REJECTED** (PREREG section 2 + A3). Incumbent v1.0 tab_mlp.
  (i) own-version val (`val_pc`): skill 0.6761 vs tab_mlp re-scored on val_pc 0.6630 (+0.0131 > 0.01) and D16 0.1133
  <= 0.1203 + 0.01: passes. (ii) D16 must improve by > 0.01: 0.1203 -> 0.1133 = -0.0070: FAILS. On the v2 val: skill
  0.6724 (tab_mlp 0.6682). Dev plug-in bias -7.1 % (D3, same 3-bank data on v2 physics) -> **+14.0 %** (VED_PHEV MdAPE
  0.169 -> 0.190): the bank rule overshoots on dev, as the twin's cold trips said it would. D3 vs P3 (deciding nothing):
  D16 0.1118 vs 0.1133. Other shared-bias dev sets barely move (Leaf +5.0 -> +5.3 %, CNG -17.1 -> -17.0 %, ICCT +20.5
  -> +19.4 %).
- 01:10 **D1 (GRU, 5 members, 3 banks) KEPT and released as `r2_d1_gru`**: sim val skill 0.6696 vs v1.0 trace_gru
  0.6390 (+0.0306 > 0.01), dev D16 0.1084 vs 0.1089. Diagnostic: member 0 alone 0.6547 / 0.1090, so data +0.0157 and
  ensembling +0.0149 skill; neither moves D16. Dev bias of the shared-bias sets: Leaf 0.0 %, plug-ins -15.4 %, Mersin
  diesel -2.9 %, CNG -18.9 %, ICCT +16.5 %. Registered in round2/registry (hash 6dffca65c342). **arms/common is now
  frozen** (enslib.py, xlib.py are hashed with every round-2 arm).
- 01:05 D2 subset (members 0-2; P2's incumbent per A3): skill 0.6693, D16 0.1061 (v1.0 transformer 0.6442 / 0.1135);
  plug-in dev bias -10.2 %.
- PREREG section 2 evidence rule for the plug-in candidate: dev plug-in bias D1 -15.4 %, D2(3) -10.2 %, D3 -7.1 %,
  EnergyNet v2.1 -7.3 %: 4 of 4 same sign above 3 %: P rungs pursued (already training; P3 above is rejected).
- 01:45 **P2 (transformer, 3 members, phevcold banks) REJECTED** vs its A3 incumbent D2 members 0-2. (i) val_pc:
  skill 0.6688 vs 0.6674 (+0.0014 < 0.01): fails. (ii) D16 0.1048 vs 0.1061 (-0.0013, needs < -0.01): fails. On v2
  val 0.6646 vs 0.6693. Dev plug-in bias -10.2 -> +5.8 % (VED_PHEV MdAPE 0.186 -> 0.168: for the transformer the rule
  helps the plug-ins; for the MLP it overshot); Leaf +2.8 -> +1.6 %, CNG -19.0 -> -17.8 %, ICCT +17.1 -> +16.1 %.
- 02:15 **D2 (transformer, 5 members, 3 banks) KEPT and released as `r2_d2_transformer`**: skill 0.6715 vs v1.0
  0.6442 (+0.0273), D16 0.1065 vs 0.1135. Member 0 alone 0.6603 / 0.1048 (data +0.0161, ensembling +0.0112). Dev bias:
  Leaf +2.4 %, plug-ins -9.9 %, Mersin diesel -1.5 %, CNG -18.8 %, ICCT +18.1 %. Registered (a83d255c2a24), with
  `r2_d2_transformer_conformal` (D16-mean cov50 / cov90 raw 0.44 / 0.76 -> cross-fit 0.50 / 0.86; CRPS 0.0940 ->
  0.0894). D1's wrapper `r2_d1_gru_conformal`: 0.42 / 0.76 -> 0.49 / 0.86; CRPS 0.0941 -> 0.0892.
- 02:25 **P1 (GRU, 5 members, phevcold banks) REJECTED** vs D1. (i) val_pc skill 0.6636 vs D1 on val_pc 0.6659
  (-0.0023): fails. (ii) D16 0.1082 vs 0.1084 (-0.0002): fails. v2 val 0.6612 vs 0.6696. Dev plug-in bias -15.4 ->
  +4.4 % (VED_PHEV MdAPE 0.197 -> 0.160); other shared-bias sets within 1 point. **P0 verdict:** the plug-in start
  rule moves the plug-in dev bias toward or past zero for all three learners (GRU -15 -> +4, transformer -10 -> +6,
  MLP -7 -> +14 %) and improves the plug-in dev error for the two trace learners, but it is worth at most 0.007 of D16
  (MLP) and costs 0.002-0.008 sim skill on the v2 val; no P rung passes. The families' round-2 entries stay D1, D2,
  v1.0 tab_mlp and v1.0 tab_gp. No physics change was made (RoadSim untouched); CALIBRATION.md has the dated R2 entries.
- 02:05 RoadSim / vehicle tests: `pytest -q tests/test_roadsim*.py tests/test_vehicle.py`: 70 passed (RoadSim unchanged).
- 02:10 debug: a partial exam-2 dev rehearsal (enet_v21, trace_gru, r2_d1_gru) ran end to end; its output was deleted
  (round2/results/*/preds and scores.parquet are git-ignored).
- Candidates 1 and 2 (not attempted, reasons fixed now): Mersin CNG (dev -17 %) has the same inputs as a diesel bus
  (build_real.hd writes diesel = 1 for both fuels), so no bank or physics change can move CNG without moving diesel
  (dev -3 %); the Mersin diesel exam miss is not visible on dev (February week, no A/C). Cold Leafs: the one dev Leaf
  is +4.6 % (EnergyNet), the exam -20 %: not visible on dev.
