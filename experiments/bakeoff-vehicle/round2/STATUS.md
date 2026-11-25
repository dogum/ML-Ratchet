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
- Candidates 1 and 2 (not attempted, reasons fixed now): Mersin CNG (dev -17 %) has the same inputs as a diesel bus
  (build_real.hd writes diesel = 1 for both fuels), so no bank or physics change can move CNG without moving diesel
  (dev -3 %); the Mersin diesel exam miss is not visible on dev (February week, no A/C). Cold Leafs: the one dev Leaf
  is +4.6 % (EnergyNet), the exam -20 %: not visible on dev.
